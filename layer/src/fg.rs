//! The ghostframe engine on raw Vulkan, running on the game's own device
//! and queue. Same shaders (compiled from WGSL at build time) and the same
//! pass structure as the wgpu engine in the main crate.

use crate::{log, DeviceData};
use ash::vk;
use ash::vk::Handle;
use std::sync::Arc;
use std::time::Instant;

const BLOCK: u32 = 8;
const MIN_LEVEL: u32 = 48;
const COARSE_RANGE: i32 = 6;
const LAMBDA: f32 = 0.002;
const RING: usize = 3;

fn spirv(bytes: &[u8]) -> Vec<u32> {
    bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

const PYRAMID: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/pyramid.spv"));
const MOTION: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/motion.spv"));
const MOTION_HW: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/motion_hw.spv"));
const COMPOSE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/compose.spv"));
const COMPOSE_HW: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/compose_hw.spv"));

#[repr(C)]
#[derive(Clone, Copy)]
struct MeParams {
    lsize: [i32; 2],
    grid: [i32; 2],
    cgrid: [i32; 2],
    mode: u32,
    fine: u32,
    lambda: f32,
    range: i32,
    sparse: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct ComposeParams {
    size: [i32; 2],
    grid: [i32; 2],
    t: f32,
    block: f32,
    vscale: f32,
    swap_rb: f32,
}

fn bytes_of<T: Copy>(v: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v as *const T as *const u8, std::mem::size_of::<T>()) }
}

#[derive(Clone, Copy)]
struct Level {
    w: u32,
    h: u32,
    gw: u32,
    gh: u32,
}

fn levels_for(w: u32, h: u32) -> Vec<Level> {
    let mut out = Vec::new();
    let (mut lw, mut lh) = (w, h);
    loop {
        out.push(Level { w: lw, h: lh, gw: lw.div_ceil(BLOCK), gh: lh.div_ceil(BLOCK) });
        if lw.min(lh) / 2 < MIN_LEVEL || out.len() == 6 {
            break;
        }
        lw = lw.div_ceil(2);
        lh = lh.div_ceil(2);
    }
    out
}

struct Img {
    image: vk::Image,
    mem: vk::DeviceMemory,
    view: vk::ImageView,
}

struct Buf {
    buf: vk::Buffer,
    mem: vk::DeviceMemory,
}

struct Slot {
    color: Img,
    luma: Vec<Img>,
}

struct Pass {
    pipeline: usize,
    sets: [vk::DescriptorSet; 2],
    gw: u32,
    gh: u32,
}

struct RingEntry {
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    acquired: vk::Semaphore,
    real_done: vk::Semaphore,
    gen_done: vk::Semaphore,
}

pub struct FrameGen {
    d: Arc<DeviceData>,
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    w: u32,
    h: u32,
    levels: Vec<Level>,
    slots: Vec<Slot>,
    fields: Vec<[Img; 2]>,
    dummy: Img,
    out: Img,
    samplers: Vec<vk::Sampler>,
    set_layouts: Vec<vk::DescriptorSetLayout>,
    pipe_layouts: Vec<vk::PipelineLayout>,
    pipelines: Vec<vk::Pipeline>, // luma, down, search, median, compose
    modules: Vec<vk::ShaderModule>,
    pool: vk::DescriptorPool,
    pyr_sets: [Vec<vk::DescriptorSet>; 2],
    passes: Vec<Pass>,
    compose_sets: [vk::DescriptorSet; 2],
    ubufs: Vec<Buf>,
    cmd_pool: vk::CommandPool,
    ring: Vec<RingEntry>,
    ring_i: usize,
    newest: usize,
    pushed: u64,
    generated: u64,
    skipped: u64,
    window_real: u64,
    spent: f64,
    last_log: Instant,
    dump: Option<(String, u64)>,
    swap_rb: bool,
}

fn check(r: vk::Result, what: &str) -> Result<(), String> {
    if r == vk::Result::SUCCESS {
        Ok(())
    } else {
        Err(format!("{what}: {r:?}"))
    }
}

impl FrameGen {
    pub fn image_count(&self) -> usize {
        self.images.len()
    }

    fn mem_type(&self, bits: u32, flags: vk::MemoryPropertyFlags) -> Option<u32> {
        let m = &self.d.mem;
        (0..m.memory_type_count).find(|&i| bits & (1 << i) != 0 && m.memory_types[i as usize].property_flags.contains(flags))
    }

    unsafe fn image(&self, w: u32, h: u32, format: vk::Format, usage: vk::ImageUsageFlags) -> Result<Img, String> {
        let fp = &self.d.fp;
        let dev = self.d.device;
        let ci = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D { width: w, height: h, depth: 1 })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let mut image = vk::Image::null();
        check((fp.create_image)(dev, &ci, std::ptr::null(), &mut image), "vkCreateImage")?;
        let mut req = vk::MemoryRequirements::default();
        (fp.get_image_memory_requirements)(dev, image, &mut req);
        let ty = self.mem_type(req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL).ok_or("no device-local memory type")?;
        let ai = vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(ty);
        let mut mem = vk::DeviceMemory::null();
        check((fp.allocate_memory)(dev, &ai, std::ptr::null(), &mut mem), "vkAllocateMemory")?;
        check((fp.bind_image_memory)(dev, image, mem, 0), "vkBindImageMemory")?;
        let vi = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(color_range());
        let mut view = vk::ImageView::null();
        check((fp.create_image_view)(dev, &vi, std::ptr::null(), &mut view), "vkCreateImageView")?;
        Ok(Img { image, mem, view })
    }

    unsafe fn host_buffer(&self, size: u64, usage: vk::BufferUsageFlags, data: Option<&[u8]>) -> Result<Buf, String> {
        let fp = &self.d.fp;
        let dev = self.d.device;
        let ci = vk::BufferCreateInfo::default().size(size).usage(usage).sharing_mode(vk::SharingMode::EXCLUSIVE);
        let mut buf = vk::Buffer::null();
        check((fp.create_buffer)(dev, &ci, std::ptr::null(), &mut buf), "vkCreateBuffer")?;
        let mut req = vk::MemoryRequirements::default();
        (fp.get_buffer_memory_requirements)(dev, buf, &mut req);
        let ty = self
            .mem_type(req.memory_type_bits, vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT)
            .ok_or("no host-visible memory type")?;
        let ai = vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(ty);
        let mut mem = vk::DeviceMemory::null();
        check((fp.allocate_memory)(dev, &ai, std::ptr::null(), &mut mem), "vkAllocateMemory")?;
        check((fp.bind_buffer_memory)(dev, buf, mem, 0), "vkBindBufferMemory")?;
        if let Some(bytes) = data {
            let mut ptr = std::ptr::null_mut();
            check((fp.map_memory)(dev, mem, 0, size, vk::MemoryMapFlags::empty(), &mut ptr), "vkMapMemory")?;
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr as *mut u8, bytes.len());
            (fp.unmap_memory)(dev, mem);
        }
        Ok(Buf { buf, mem })
    }

    unsafe fn set_layout(&mut self, types: &[vk::DescriptorType]) -> Result<vk::DescriptorSetLayout, String> {
        let bindings: Vec<vk::DescriptorSetLayoutBinding> = types
            .iter()
            .enumerate()
            .map(|(i, &t)| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i as u32)
                    .descriptor_type(t)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let ci = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let mut l = vk::DescriptorSetLayout::null();
        check((self.d.fp.create_descriptor_set_layout)(self.d.device, &ci, std::ptr::null(), &mut l), "vkCreateDescriptorSetLayout")?;
        self.set_layouts.push(l);
        Ok(l)
    }

    unsafe fn pipeline(&mut self, code: &[u8], entry: &std::ffi::CStr, layout: vk::DescriptorSetLayout) -> Result<(), String> {
        let fp = &self.d.fp;
        let dev = self.d.device;
        let words = spirv(code);
        let mci = vk::ShaderModuleCreateInfo::default().code(&words);
        let mut module = vk::ShaderModule::null();
        check((fp.create_shader_module)(dev, &mci, std::ptr::null(), &mut module), "vkCreateShaderModule")?;
        let layouts = [layout];
        let pli = vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts);
        let mut pl = vk::PipelineLayout::null();
        check((fp.create_pipeline_layout)(dev, &pli, std::ptr::null(), &mut pl), "vkCreatePipelineLayout")?;
        let stage = vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(entry);
        let ci = vk::ComputePipelineCreateInfo::default().stage(stage).layout(pl);
        let mut p = vk::Pipeline::null();
        check(
            (fp.create_compute_pipelines)(dev, vk::PipelineCache::null(), 1, &ci, std::ptr::null(), &mut p),
            "vkCreateComputePipelines",
        )?;
        self.modules.push(module);
        self.pipe_layouts.push(pl);
        self.pipelines.push(p);
        Ok(())
    }

    unsafe fn alloc_set(&self, layout: vk::DescriptorSetLayout) -> Result<vk::DescriptorSet, String> {
        let layouts = [layout];
        let ai = vk::DescriptorSetAllocateInfo::default().descriptor_pool(self.pool).set_layouts(&layouts);
        let mut s = vk::DescriptorSet::null();
        check((self.d.fp.allocate_descriptor_sets)(self.d.device, &ai, &mut s), "vkAllocateDescriptorSets")?;
        Ok(s)
    }

    /// Write bindings 0..n of `set`. Each entry is an image view (sampled or
    /// storage), a sampler, or a uniform buffer.
    unsafe fn write_set(&self, set: vk::DescriptorSet, entries: &[Bind]) {
        let infos: Vec<(Option<vk::DescriptorImageInfo>, Option<vk::DescriptorBufferInfo>)> = entries
            .iter()
            .map(|e| match *e {
                Bind::Sampled(v) | Bind::Storage(v) => (
                    Some(vk::DescriptorImageInfo { sampler: vk::Sampler::null(), image_view: v, image_layout: vk::ImageLayout::GENERAL }),
                    None,
                ),
                Bind::Sampler(s) => (
                    Some(vk::DescriptorImageInfo { sampler: s, image_view: vk::ImageView::null(), image_layout: vk::ImageLayout::UNDEFINED }),
                    None,
                ),
                Bind::Uniform(b) => (None, Some(vk::DescriptorBufferInfo { buffer: b, offset: 0, range: vk::WHOLE_SIZE })),
            })
            .collect();
        let writes: Vec<vk::WriteDescriptorSet> = entries
            .iter()
            .zip(&infos)
            .enumerate()
            .map(|(i, (e, (img, buf)))| {
                let w = vk::WriteDescriptorSet::default().dst_set(set).dst_binding(i as u32).descriptor_count(1);
                let ty = match e {
                    Bind::Sampled(_) => vk::DescriptorType::SAMPLED_IMAGE,
                    Bind::Storage(_) => vk::DescriptorType::STORAGE_IMAGE,
                    Bind::Sampler(_) => vk::DescriptorType::SAMPLER,
                    Bind::Uniform(_) => vk::DescriptorType::UNIFORM_BUFFER,
                };
                let mut w = w.descriptor_type(ty);
                if let Some(ii) = img {
                    w.p_image_info = ii as *const _;
                }
                if let Some(bi) = buf {
                    w.p_buffer_info = bi as *const _;
                }
                w
            })
            .collect();
        (self.d.fp.update_descriptor_sets)(self.d.device, writes.len() as u32, writes.as_ptr(), 0, std::ptr::null());
    }

    pub unsafe fn new(
        d: Arc<DeviceData>,
        queue: vk::Queue,
        family: u32,
        swapchain: vk::SwapchainKHR,
        format: vk::Format,
        extent: vk::Extent2D,
    ) -> Result<Self, String> {
        let (w, h) = (extent.width, extent.height);
        if w < 64 || h < 64 {
            return Err(format!("{w}x{h} is too small"));
        }
        let dev = d.device;
        let mut n = 0u32;
        check((d.swap.get_swapchain_images_khr)(dev, swapchain, &mut n, std::ptr::null_mut()), "vkGetSwapchainImagesKHR")?;
        let mut images = vec![vk::Image::null(); n as usize];
        check((d.swap.get_swapchain_images_khr)(dev, swapchain, &mut n, images.as_mut_ptr()), "vkGetSwapchainImagesKHR")?;

        let levels = levels_for(w, h);
        let finest = (if w.min(h) >= 360 { 1 } else { 0 }).min(levels.len() - 2);
        let (slot_format, swap_rb) = match format {
            vk::Format::B8G8R8A8_UNORM | vk::Format::B8G8R8A8_SRGB => (vk::Format::B8G8R8A8_UNORM, true),
            _ => (vk::Format::R8G8B8A8_UNORM, false),
        };
        let dump = std::env::var("GHOSTFRAME_DUMP").ok().map(|p| {
            let n = std::env::var("GHOSTFRAME_DUMP_FRAME").ok().and_then(|v| v.parse().ok()).unwrap_or(60);
            (p, n)
        });

        if let Some((p, n)) = &dump {
            log(&format!("will dump the generated frame after real frame {n} to {p}"));
        }
        let mut g = FrameGen {
            d: d.clone(),
            swapchain,
            images,
            w,
            h,
            levels: levels.clone(),
            slots: Vec::new(),
            fields: Vec::new(),
            dummy: Img { image: vk::Image::null(), mem: vk::DeviceMemory::null(), view: vk::ImageView::null() },
            out: Img { image: vk::Image::null(), mem: vk::DeviceMemory::null(), view: vk::ImageView::null() },
            samplers: Vec::new(),
            set_layouts: Vec::new(),
            pipe_layouts: Vec::new(),
            pipelines: Vec::new(),
            modules: Vec::new(),
            pool: vk::DescriptorPool::null(),
            pyr_sets: [Vec::new(), Vec::new()],
            passes: Vec::new(),
            compose_sets: [vk::DescriptorSet::null(); 2],
            ubufs: Vec::new(),
            cmd_pool: vk::CommandPool::null(),
            ring: Vec::new(),
            ring_i: 0,
            newest: 1,
            pushed: 0,
            generated: 0,
            skipped: 0,
            window_real: 0,
            spent: 0.0,
            last_log: Instant::now(),
            dump,
            swap_rb,
        };

        use vk::DescriptorType as T;
        use vk::ImageUsageFlags as U;
        // Images.
        for _ in 0..2 {
            let color = g.image(w, h, slot_format, U::SAMPLED | U::TRANSFER_DST)?;
            let mut luma = Vec::new();
            for lv in &levels {
                luma.push(g.image(lv.w, lv.h, vk::Format::R32_SFLOAT, U::SAMPLED | U::STORAGE)?);
            }
            g.slots.push(Slot { color, luma });
        }
        for lv in &levels {
            let a = g.image(lv.gw, lv.gh, vk::Format::R32G32B32A32_SFLOAT, U::SAMPLED | U::STORAGE)?;
            let b = g.image(lv.gw, lv.gh, vk::Format::R32G32B32A32_SFLOAT, U::SAMPLED | U::STORAGE)?;
            g.fields.push([a, b]);
        }
        g.dummy = g.image(1, 1, vk::Format::R32G32B32A32_SFLOAT, U::SAMPLED | U::STORAGE)?;
        g.out = g.image(w, h, vk::Format::R8G8B8A8_UNORM, U::STORAGE | U::TRANSFER_SRC)?;

        let sci = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .max_lod(0.0);
        let mut smp = vk::Sampler::null();
        check((d.fp.create_sampler)(dev, &sci, std::ptr::null(), &mut smp), "vkCreateSampler")?;
        g.samplers.push(smp);

        // Pipelines.
        let pyr_l = g.set_layout(&[T::SAMPLED_IMAGE, T::STORAGE_IMAGE])?;
        let me_l = g.set_layout(&[T::SAMPLED_IMAGE, T::SAMPLED_IMAGE, T::SAMPLED_IMAGE, T::SAMPLED_IMAGE, T::STORAGE_IMAGE, T::UNIFORM_BUFFER, T::SAMPLER])?;
        let co_l = g.set_layout(&[
            T::SAMPLED_IMAGE,
            T::SAMPLED_IMAGE,
            T::SAMPLED_IMAGE,
            T::SAMPLED_IMAGE,
            T::SAMPLED_IMAGE,
            T::SAMPLER,
            T::STORAGE_IMAGE,
            T::UNIFORM_BUFFER,
            T::SAMPLER,
        ])?;
        let hw = d.float_filter;
        g.pipeline(PYRAMID, c"luma", pyr_l)?;
        g.pipeline(PYRAMID, c"down", pyr_l)?;
        g.pipeline(if hw { MOTION_HW } else { MOTION }, c"search", me_l)?;
        g.pipeline(if hw { MOTION_HW } else { MOTION }, c"median", me_l)?;
        g.pipeline(if hw { COMPOSE_HW } else { COMPOSE }, c"compose", co_l)?;

        let sizes = [
            vk::DescriptorPoolSize { ty: T::SAMPLED_IMAGE, descriptor_count: 1024 },
            vk::DescriptorPoolSize { ty: T::STORAGE_IMAGE, descriptor_count: 256 },
            vk::DescriptorPoolSize { ty: T::UNIFORM_BUFFER, descriptor_count: 256 },
            vk::DescriptorPoolSize { ty: T::SAMPLER, descriptor_count: 256 },
        ];
        let pci = vk::DescriptorPoolCreateInfo::default().max_sets(256).pool_sizes(&sizes);
        check((d.fp.create_descriptor_pool)(dev, &pci, std::ptr::null(), &mut g.pool), "vkCreateDescriptorPool")?;

        // Pyramid sets.
        for s in 0..2 {
            let mut sets = Vec::new();
            for l in 0..levels.len() {
                let set = g.alloc_set(pyr_l)?;
                let src = if l == 0 { g.slots[s].color.view } else { g.slots[s].luma[l - 1].view };
                g.write_set(set, &[Bind::Sampled(src), Bind::Storage(g.slots[s].luma[l].view)]);
                sets.push(set);
            }
            g.pyr_sets[s] = sets;
        }

        // Motion passes, coarsest to finest (final field: [l][0], finest [finest][1]).
        let nl = levels.len();
        for l in (finest..nl).rev() {
            let lv = levels[l];
            let cg = if l + 1 < nl { (levels[l + 1].gw, levels[l + 1].gh) } else { (1, 1) };
            let coarse = if l + 1 < nl { g.fields[l + 1][0].view } else { g.dummy.view };
            let mut steps: Vec<(u32, vk::ImageView, vk::ImageView, usize)> = vec![
                (if l + 1 == nl { 0 } else { 1 }, g.dummy.view, g.fields[l][0].view, 2),
                (2, g.fields[l][0].view, g.fields[l][1].view, 2),
                (2, g.fields[l][1].view, g.fields[l][0].view, 2),
            ];
            if l == finest {
                steps.push((2, g.fields[l][0].view, g.fields[l][1].view, 3));
            }
            for (mode, prev, dst, pipe) in steps {
                let p = MeParams {
                    lsize: [lv.w as i32, lv.h as i32],
                    grid: [lv.gw as i32, lv.gh as i32],
                    cgrid: [cg.0 as i32, cg.1 as i32],
                    mode,
                    fine: (l == finest) as u32,
                    lambda: LAMBDA,
                    range: COARSE_RANGE,
                    sparse: (l <= finest + 1) as u32,
                    _pad: 0,
                };
                let ub = g.host_buffer(std::mem::size_of::<MeParams>() as u64, vk::BufferUsageFlags::UNIFORM_BUFFER, Some(bytes_of(&p)))?;
                let mut sets = [vk::DescriptorSet::null(); 2];
                for b in 0..2 {
                    let (sa, sb) = (1 - b, b);
                    let set = g.alloc_set(me_l)?;
                    g.write_set(
                        set,
                        &[
                            Bind::Sampled(g.slots[sa].luma[l].view),
                            Bind::Sampled(g.slots[sb].luma[l].view),
                            Bind::Sampled(prev),
                            Bind::Sampled(coarse),
                            Bind::Storage(dst),
                            Bind::Uniform(ub.buf),
                            Bind::Sampler(smp),
                        ],
                    );
                    sets[b] = set;
                }
                g.ubufs.push(ub);
                g.passes.push(Pass { pipeline: pipe, sets, gw: lv.gw, gh: lv.gh });
            }
        }

        let flv = levels[finest];
        let scale = (1u32 << finest) as f32;
        let cp = ComposeParams {
            size: [w as i32, h as i32],
            grid: [flv.gw as i32, flv.gh as i32],
            t: 0.5,
            block: BLOCK as f32 * scale,
            vscale: scale,
            swap_rb: if swap_rb { 1.0 } else { 0.0 },
        };
        let cub = g.host_buffer(std::mem::size_of::<ComposeParams>() as u64, vk::BufferUsageFlags::UNIFORM_BUFFER, Some(bytes_of(&cp)))?;
        for b in 0..2 {
            let (sa, sb) = (1 - b, b);
            let set = g.alloc_set(co_l)?;
            g.write_set(
                set,
                &[
                    Bind::Sampled(g.slots[sa].color.view),
                    Bind::Sampled(g.slots[sb].color.view),
                    Bind::Sampled(g.slots[sa].luma[0].view),
                    Bind::Sampled(g.slots[sb].luma[0].view),
                    Bind::Sampled(g.fields[finest][1].view),
                    Bind::Sampler(smp),
                    Bind::Storage(g.out.view),
                    Bind::Uniform(cub.buf),
                    Bind::Sampler(smp),
                ],
            );
            g.compose_sets[b] = set;
        }
        g.ubufs.push(cub);

        // Commands and sync.
        let cpi = vk::CommandPoolCreateInfo::default()
            .queue_family_index(family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        check((d.fp.create_command_pool)(dev, &cpi, std::ptr::null(), &mut g.cmd_pool), "vkCreateCommandPool")?;
        for _ in 0..RING {
            let cmd = g.command_buffer()?;
            let mut fence = vk::Fence::null();
            let fci = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
            check((d.fp.create_fence)(dev, &fci, std::ptr::null(), &mut fence), "vkCreateFence")?;
            let sem = || {
                let mut s = vk::Semaphore::null();
                check((d.fp.create_semaphore)(dev, &vk::SemaphoreCreateInfo::default(), std::ptr::null(), &mut s), "vkCreateSemaphore").map(|_| s)
            };
            g.ring.push(RingEntry { cmd, fence, acquired: sem()?, real_done: sem()?, gen_done: sem()? });
        }

        // Move every image of ours into GENERAL once.
        let mut all: Vec<vk::Image> = vec![g.dummy.image, g.out.image];
        for s in &g.slots {
            all.push(s.color.image);
            all.extend(s.luma.iter().map(|i| i.image));
        }
        for f in &g.fields {
            all.extend(f.iter().map(|i| i.image));
        }
        g.one_shot(queue, |g, cmd| {
            for img in &all {
                g.image_barrier(
                    cmd,
                    *img,
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::GENERAL,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::AccessFlags::empty(),
                    vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                );
            }
        })?;
        Ok(g)
    }

    unsafe fn command_buffer(&self) -> Result<vk::CommandBuffer, String> {
        let ai = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.cmd_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let mut cmd = vk::CommandBuffer::null();
        check((self.d.fp.allocate_command_buffers)(self.d.device, &ai, &mut cmd), "vkAllocateCommandBuffers")?;
        // Command buffers are dispatchable: the loader must stamp them.
        if let Some(set) = self.d.set_loader_data {
            let _ = set(self.d.device, cmd.as_raw() as *mut std::ffi::c_void);
        }
        Ok(cmd)
    }

    unsafe fn one_shot(&self, queue: vk::Queue, rec: impl FnOnce(&Self, vk::CommandBuffer)) -> Result<(), String> {
        let fp = &self.d.fp;
        let dev = self.d.device;
        let cmd = self.command_buffer()?;
        let bi = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        check((fp.begin_command_buffer)(cmd, &bi), "vkBeginCommandBuffer")?;
        rec(self, cmd);
        check((fp.end_command_buffer)(cmd), "vkEndCommandBuffer")?;
        let cmds = [cmd];
        let si = vk::SubmitInfo::default().command_buffers(&cmds);
        let mut fence = vk::Fence::null();
        check((fp.create_fence)(dev, &vk::FenceCreateInfo::default(), std::ptr::null(), &mut fence), "vkCreateFence")?;
        let r = (fp.queue_submit)(queue, 1, &si, fence);
        if r == vk::Result::SUCCESS {
            let _ = (fp.wait_for_fences)(dev, 1, &fence, vk::TRUE, u64::MAX);
        }
        (fp.destroy_fence)(dev, fence, std::ptr::null());
        (fp.free_command_buffers)(dev, self.cmd_pool, 1, &cmd);
        check(r, "vkQueueSubmit")
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn image_barrier(
        &self,
        cmd: vk::CommandBuffer,
        image: vk::Image,
        old: vk::ImageLayout,
        new: vk::ImageLayout,
        src_stage: vk::PipelineStageFlags,
        dst_stage: vk::PipelineStageFlags,
        src: vk::AccessFlags,
        dst: vk::AccessFlags,
    ) {
        let b = vk::ImageMemoryBarrier::default()
            .old_layout(old)
            .new_layout(new)
            .src_access_mask(src)
            .dst_access_mask(dst)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(color_range());
        (self.d.fp.cmd_pipeline_barrier)(cmd, src_stage, dst_stage, vk::DependencyFlags::empty(), 0, std::ptr::null(), 0, std::ptr::null(), 1, &b);
    }

    unsafe fn memory_barrier(&self, cmd: vk::CommandBuffer, src_stage: vk::PipelineStageFlags, dst_stage: vk::PipelineStageFlags, src: vk::AccessFlags, dst: vk::AccessFlags) {
        let b = vk::MemoryBarrier::default().src_access_mask(src).dst_access_mask(dst);
        (self.d.fp.cmd_pipeline_barrier)(cmd, src_stage, dst_stage, vk::DependencyFlags::empty(), 1, &b, 0, std::ptr::null(), 0, std::ptr::null());
    }

    unsafe fn compute_barrier(&self, cmd: vk::CommandBuffer) {
        self.memory_barrier(
            cmd,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::AccessFlags::SHADER_WRITE,
            vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
        );
    }

    unsafe fn dispatch(&self, cmd: vk::CommandBuffer, pipe: usize, set: vk::DescriptorSet, x: u32, y: u32) {
        let fp = &self.d.fp;
        (fp.cmd_bind_pipeline)(cmd, vk::PipelineBindPoint::COMPUTE, self.pipelines[pipe]);
        (fp.cmd_bind_descriptor_sets)(cmd, vk::PipelineBindPoint::COMPUTE, self.pipe_layouts[pipe], 0, 1, &set, 0, std::ptr::null());
        (fp.cmd_dispatch)(cmd, x, y, 1);
        self.compute_barrier(cmd);
    }

    unsafe fn copy_full(&self, cmd: vk::CommandBuffer, src: vk::Image, src_layout: vk::ImageLayout, dst: vk::Image, dst_layout: vk::ImageLayout) {
        let layers = vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 };
        let region = vk::ImageCopy {
            src_subresource: layers,
            src_offset: vk::Offset3D::default(),
            dst_subresource: layers,
            dst_offset: vk::Offset3D::default(),
            extent: vk::Extent3D { width: self.w, height: self.h, depth: 1 },
        };
        (self.d.fp.cmd_copy_image)(cmd, src, src_layout, dst, dst_layout, 1, &region);
    }

    /// Switched off in the tray: the next frames pass through untouched,
    /// and generation starts fresh (no stale previous frame) when it is back.
    pub fn pause(&mut self) {
        self.pushed = 0;
    }

    /// Replace one present by two: the generated in-between frame, then the
    /// real one. Returns the real present's result to the game and whether
    /// a frame was generated.
    pub unsafe fn present(&mut self, queue: vk::Queue, info: &vk::PresentInfoKHR<'_>) -> (vk::Result, bool) {
        let t0 = Instant::now();
        let fp = &self.d.fp;
        let sw = &self.d.swap;
        let dev = self.d.device;
        let ri = self.ring_i;
        self.ring_i = (self.ring_i + 1) % RING;
        let (cmd, fence, acquired, real_done, gen_done) = {
            let e = &self.ring[ri];
            (e.cmd, e.fence, e.acquired, e.real_done, e.gen_done)
        };
        let _ = (fp.wait_for_fences)(dev, 1, &fence, vk::TRUE, u64::MAX);

        let real_index = *info.p_image_indices;
        let real = self.images[real_index as usize];
        let nb = 1 - self.newest;

        // A second swapchain image for the generated frame. Don't block the
        // game for long: skip generation if none frees up in 50 ms.
        let mut gen_index = 0u32;
        let mut gen = false;
        if self.pushed >= 1 {
            let r = (sw.acquire_next_image_khr)(dev, self.swapchain, 50_000_000, acquired, vk::Fence::null(), &mut gen_index);
            gen = r == vk::Result::SUCCESS || r == vk::Result::SUBOPTIMAL_KHR;
            if !gen {
                self.skipped += 1;
            }
        }

        let _ = (fp.reset_fences)(dev, 1, &fence);
        let _ = (fp.reset_command_buffer)(cmd, vk::CommandBufferResetFlags::empty());
        let bi = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        let _ = (fp.begin_command_buffer)(cmd, &bi);
        use vk::AccessFlags as A;
        use vk::ImageLayout as L;
        use vk::PipelineStageFlags as S;

        // 1. Copy the game's frame into our slot.
        self.image_barrier(cmd, real, L::PRESENT_SRC_KHR, L::TRANSFER_SRC_OPTIMAL, S::ALL_COMMANDS, S::TRANSFER, A::MEMORY_WRITE, A::TRANSFER_READ);
        self.memory_barrier(cmd, S::COMPUTE_SHADER, S::TRANSFER, A::SHADER_READ, A::TRANSFER_WRITE);
        self.copy_full(cmd, real, L::TRANSFER_SRC_OPTIMAL, self.slots[nb].color.image, L::GENERAL);
        self.image_barrier(cmd, real, L::TRANSFER_SRC_OPTIMAL, L::PRESENT_SRC_KHR, S::TRANSFER, S::BOTTOM_OF_PIPE, A::TRANSFER_READ, A::empty());
        self.memory_barrier(cmd, S::TRANSFER, S::COMPUTE_SHADER, A::TRANSFER_WRITE, A::SHADER_READ);

        // 2. Its luma pyramid.
        for (l, lv) in self.levels.iter().enumerate() {
            self.dispatch(cmd, if l == 0 { 0 } else { 1 }, self.pyr_sets[nb][l], lv.w.div_ceil(16), lv.h.div_ceil(16));
        }

        // 3. Motion search and the in-between frame, into a swapchain image.
        if gen {
            for p in &self.passes {
                self.dispatch(cmd, p.pipeline, p.sets[nb], p.gw.div_ceil(8), p.gh.div_ceil(8));
            }
            self.dispatch(cmd, 4, self.compose_sets[nb], self.w.div_ceil(16), self.h.div_ceil(16));
            let target = self.images[gen_index as usize];
            self.memory_barrier(cmd, S::COMPUTE_SHADER, S::TRANSFER, A::SHADER_WRITE, A::TRANSFER_READ);
            self.image_barrier(cmd, target, L::UNDEFINED, L::TRANSFER_DST_OPTIMAL, S::TRANSFER, S::TRANSFER, A::empty(), A::TRANSFER_WRITE);
            self.copy_full(cmd, self.out.image, L::GENERAL, target, L::TRANSFER_DST_OPTIMAL);
            self.image_barrier(cmd, target, L::TRANSFER_DST_OPTIMAL, L::PRESENT_SRC_KHR, S::TRANSFER, S::BOTTOM_OF_PIPE, A::TRANSFER_WRITE, A::empty());
            self.memory_barrier(cmd, S::TRANSFER, S::COMPUTE_SHADER, A::TRANSFER_READ, A::SHADER_WRITE);
        }
        let _ = (fp.end_command_buffer)(cmd);

        // Wait for the game's rendering (its present semaphores) and for the
        // acquired image; signal one semaphore per present.
        let mut waits: Vec<vk::Semaphore> =
            (0..info.wait_semaphore_count as usize).map(|i| *info.p_wait_semaphores.add(i)).collect();
        let mut stages = vec![S::ALL_COMMANDS; waits.len()];
        let mut signals = vec![real_done];
        if gen {
            waits.push(acquired);
            stages.push(S::TRANSFER);
            signals.push(gen_done);
        }
        let cmds = [cmd];
        let si = vk::SubmitInfo::default()
            .wait_semaphores(&waits)
            .wait_dst_stage_mask(&stages)
            .command_buffers(&cmds)
            .signal_semaphores(&signals);
        let r = (fp.queue_submit)(queue, 1, &si, fence);
        if r != vk::Result::SUCCESS {
            log(&format!("vkQueueSubmit failed: {r:?}"));
            return (r, false);
        }

        let swapchains = [self.swapchain];
        if gen {
            let idx = [gen_index];
            let sems = [gen_done];
            let pi = vk::PresentInfoKHR::default().wait_semaphores(&sems).swapchains(&swapchains).image_indices(&idx);
            let _ = (sw.queue_present_khr)(queue, &pi);
            self.generated += 1;
        }
        let sems = [real_done];
        let mut real_info = *info;
        real_info.wait_semaphore_count = 1;
        real_info.p_wait_semaphores = sems.as_ptr();
        let result = (sw.queue_present_khr)(queue, &real_info);

        self.newest = nb;
        self.pushed += 1;
        self.window_real += 1;
        if gen {
            if let Some((path, n)) = self.dump.clone() {
                if self.pushed >= n {
                    self.dump_output(queue, fence, &path);
                    self.dump = None;
                }
            }
        }
        self.spent += t0.elapsed().as_secs_f64() * 1000.0;
        if self.last_log.elapsed().as_secs_f64() >= 5.0 {
            let secs = self.last_log.elapsed().as_secs_f64();
            log(&format!(
                "{:.0} real fps -> {:.0} presented fps ({} generated, {} skipped, {:.2} ms/frame in the layer on the CPU)",
                self.window_real as f64 / secs,
                (self.window_real + self.generated) as f64 / secs,
                self.generated,
                self.skipped,
                self.spent / self.window_real.max(1) as f64
            ));
            self.last_log = Instant::now();
            self.window_real = 0;
            self.generated = 0;
            self.skipped = 0;
            self.spent = 0.0;
        }
        (result, gen)
    }

    /// Debugging and end-to-end tests: write the last generated frame to a
    /// file (header + raw pixels) so it can be compared with ground truth.
    unsafe fn dump_output(&mut self, queue: vk::Queue, fence: vk::Fence, path: &str) {
        let d = self.d.clone();
        let fp = &d.fp;
        let dev = d.device;
        let _ = (fp.wait_for_fences)(dev, 1, &fence, vk::TRUE, u64::MAX);
        let size = (self.w * self.h * 4) as u64;
        let buf = match self.host_buffer(size, vk::BufferUsageFlags::TRANSFER_DST, None) {
            Ok(b) => b,
            Err(e) => return log(&format!("dump failed: {e}")),
        };
        let (w, h) = (self.w, self.h);
        let out = self.out.image;
        let ok = self.one_shot(queue, |g, cmd| {
            g.memory_barrier(cmd, vk::PipelineStageFlags::ALL_COMMANDS, vk::PipelineStageFlags::TRANSFER, vk::AccessFlags::MEMORY_WRITE, vk::AccessFlags::TRANSFER_READ);
            let region = vk::BufferImageCopy {
                buffer_offset: 0,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_subresource: vk::ImageSubresourceLayers { aspect_mask: vk::ImageAspectFlags::COLOR, mip_level: 0, base_array_layer: 0, layer_count: 1 },
                image_offset: vk::Offset3D::default(),
                image_extent: vk::Extent3D { width: w, height: h, depth: 1 },
            };
            (g.d.fp.cmd_copy_image_to_buffer)(cmd, out, vk::ImageLayout::GENERAL, buf.buf, 1, &region);
        });
        if let Err(e) = &ok {
            log(&format!("dump failed: {e}"));
        }
        if ok.is_ok() {
            let mut ptr = std::ptr::null_mut();
            if (fp.map_memory)(dev, buf.mem, 0, size, vk::MemoryMapFlags::empty(), &mut ptr) == vk::Result::SUCCESS {
                let px = std::slice::from_raw_parts(ptr as *const u8, size as usize);
                let mut file = Vec::with_capacity(size as usize + 32);
                file.extend_from_slice(b"GFDUMP1\n");
                file.extend_from_slice(&w.to_le_bytes());
                file.extend_from_slice(&h.to_le_bytes());
                file.extend_from_slice(&(self.swap_rb as u32).to_le_bytes());
                file.extend_from_slice(&self.pushed.to_le_bytes());
                file.extend_from_slice(px);
                match std::fs::write(path, file) {
                    Ok(()) => log(&format!("dumped generated frame between real frames {} and {} to {path}", self.pushed - 2, self.pushed - 1)),
                    Err(e) => log(&format!("dump failed: {e}")),
                }
                (fp.unmap_memory)(dev, buf.mem);
            }
        }
        (fp.destroy_buffer)(dev, buf.buf, std::ptr::null());
        (fp.free_memory)(dev, buf.mem, std::ptr::null());
    }

    pub fn destroy(self) {
        unsafe {
            let fp = &self.d.fp;
            let dev = self.d.device;
            let _ = (fp.device_wait_idle)(dev);
            for e in &self.ring {
                (fp.destroy_fence)(dev, e.fence, std::ptr::null());
                for s in [e.acquired, e.real_done, e.gen_done] {
                    (fp.destroy_semaphore)(dev, s, std::ptr::null());
                }
            }
            (fp.destroy_command_pool)(dev, self.cmd_pool, std::ptr::null());
            (fp.destroy_descriptor_pool)(dev, self.pool, std::ptr::null());
            for b in &self.ubufs {
                (fp.destroy_buffer)(dev, b.buf, std::ptr::null());
                (fp.free_memory)(dev, b.mem, std::ptr::null());
            }
            for &p in &self.pipelines {
                (fp.destroy_pipeline)(dev, p, std::ptr::null());
            }
            for &l in &self.pipe_layouts {
                (fp.destroy_pipeline_layout)(dev, l, std::ptr::null());
            }
            for &m in &self.modules {
                (fp.destroy_shader_module)(dev, m, std::ptr::null());
            }
            for &l in &self.set_layouts {
                (fp.destroy_descriptor_set_layout)(dev, l, std::ptr::null());
            }
            for &s in &self.samplers {
                (fp.destroy_sampler)(dev, s, std::ptr::null());
            }
            let mut imgs: Vec<&Img> = vec![&self.dummy, &self.out];
            for s in &self.slots {
                imgs.push(&s.color);
                imgs.extend(s.luma.iter());
            }
            for f in &self.fields {
                imgs.extend(f.iter());
            }
            for i in imgs {
                (fp.destroy_image_view)(dev, i.view, std::ptr::null());
                (fp.destroy_image)(dev, i.image, std::ptr::null());
                (fp.free_memory)(dev, i.mem, std::ptr::null());
            }
        }
    }
}

enum Bind {
    Sampled(vk::ImageView),
    Storage(vk::ImageView),
    Sampler(vk::Sampler),
    Uniform(vk::Buffer),
}

fn color_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 }
}
