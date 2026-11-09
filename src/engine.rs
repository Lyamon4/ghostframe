//! The frame-generation engine: push real frames in, ask for in-between
//! frames at any time t in (0, 1). Everything runs in compute shaders.

use anyhow::{bail, Result};
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

const BLOCK: u32 = 8;
/// Coarsest pyramid level is kept at least this many pixels on its short side.
const MIN_LEVEL: u32 = 48;
/// Exhaustive search radius at the coarsest level, in that level's pixels.
const COARSE_RANGE: i32 = 6;
const LAMBDA: f32 = 0.002;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MeParams {
    lsize: [i32; 2],
    grid: [i32; 2],
    cgrid: [i32; 2],
    mode: u32,
    fine: u32,
    lambda: f32,
    range: i32,
    sparse: u32,
    _pad: [u32; 1],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ComposeParams {
    size: [i32; 2],
    grid: [i32; 2],
    t: f32,
    block: f32,
    vscale: f32,
    swap_rb: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct Level {
    pub w: u32,
    pub h: u32,
    pub gw: u32,
    pub gh: u32,
}

struct Slot {
    color: wgpu::Texture,
    color_view: wgpu::TextureView,
    luma: Vec<(wgpu::Texture, wgpu::TextureView)>,
}

struct Pass {
    pipeline: usize, // 0 = search, 1 = median
    groups: [wgpu::BindGroup; 2], // indexed by which slot holds frame B
    gw: u32,
    gh: u32,
}

pub struct Engine {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub width: u32,
    pub height: u32,
    pub levels: Vec<Level>,
    /// Finest pyramid level on which motion is searched (0 = full resolution).
    pub finest: usize,
    slots: [Slot; 2],
    newest: usize,
    pushed: usize,
    out: wgpu::Texture,
    out_view: wgpu::TextureView,
    pyr_luma: wgpu::ComputePipeline,
    pyr_down: wgpu::ComputePipeline,
    me: [wgpu::ComputePipeline; 2],
    compose: wgpu::ComputePipeline,
    pyr_groups: [Vec<wgpu::BindGroup>; 2], // per slot: luma, then each down level
    passes: Vec<Pass>,
    compose_groups: [wgpu::BindGroup; 2],
    compose_params: wgpu::Buffer,
    readback: wgpu::Buffer,
    padded_row: u32,
}

fn tex(device: &wgpu::Device, label: &str, w: u32, h: u32, format: wgpu::TextureFormat, usage: wgpu::TextureUsages) -> (wgpu::Texture, wgpu::TextureView) {
    let t = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    });
    let v = t.create_view(&Default::default());
    (t, v)
}

fn tex_entry(binding: u32, filterable: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn storage_entry(binding: u32, format: wgpu::TextureFormat) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}

fn sampler_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

/// Swap the manual bilinear `lum()` for one hardware sample when the GPU
/// can filter R32Float.
fn with_lum(src: &str, hw: bool, size: &str) -> String {
    if !hw {
        return src.to_string();
    }
    let (a, rest) = src.split_once("//LUM_BEGIN").expect("LUM_BEGIN marker");
    let (_, b) = rest.split_once("//LUM_END").expect("LUM_END marker");
    format!(
        "{a}fn lum(t: texture_2d<f32>, p: vec2f) -> f32 {{\n    return textureSampleLevel(t, lsmp, p / vec2f({size}), 0.0).r;\n}}{b}"
    )
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    }
}

fn pipeline(device: &wgpu::Device, bgl: &wgpu::BindGroupLayout, module: &wgpu::ShaderModule, entry: &str) -> wgpu::ComputePipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(entry),
        bind_group_layouts: &[Some(bgl)],
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry),
        layout: Some(&layout),
        module,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn bind(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, entries: &[wgpu::BindingResource]) -> wgpu::BindGroup {
    let entries: Vec<wgpu::BindGroupEntry> =
        entries.iter().enumerate().map(|(i, r)| wgpu::BindGroupEntry { binding: i as u32, resource: r.clone() }).collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout, entries: &entries })
}

fn div_up(a: u32, b: u32) -> u32 {
    a.div_ceil(b)
}

pub fn pyramid_levels(w: u32, h: u32) -> Vec<Level> {
    let mut levels = Vec::new();
    let (mut lw, mut lh) = (w, h);
    loop {
        levels.push(Level { w: lw, h: lh, gw: div_up(lw, BLOCK), gh: div_up(lh, BLOCK) });
        if lw.min(lh) / 2 < MIN_LEVEL || levels.len() == 6 {
            break;
        }
        lw = div_up(lw, 2);
        lh = div_up(lh, 2);
    }
    levels
}

/// The pyramid level motion is searched down to: the smallest one whose short
/// side is still at least max(180, short side / 6) pixels (720p: 180,
/// 1080p: 270, 1440p: 360). Measured with `bench`: searching finer is 2-3x
/// slower and no better; the frame itself is always built at full resolution.
pub fn search_level(levels: &[Level]) -> usize {
    let short = |l: &Level| l.w.min(l.h);
    let need = 180.max(short(&levels[0]) / 6);
    let deepest = levels.iter().rposition(|l| short(l) >= need).unwrap_or(0);
    deepest.min(levels.len().saturating_sub(2))
}

/// Create a device on the best available adapter (a software one if there
/// is no GPU, so tests still run in CI).
pub fn gpu() -> Result<(wgpu::Instance, wgpu::Adapter, wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
        apply_limit_buckets: false,
    }))
    .or_else(|_| {
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::None,
            force_fallback_adapter: true,
            compatible_surface: None,
            apply_limit_buckets: false,
        }))
    })?;
    // Hardware bilinear filtering of 32-bit float luma is a big speed-up for
    // block matching; desktop GPUs have it, the code falls back without it.
    let features = adapter.features() & wgpu::Features::FLOAT32_FILTERABLE;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("ghostframe"),
        required_features: features,
        required_limits: adapter.limits(),
        experimental_features: Default::default(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: Default::default(),
    }))?;
    Ok((instance, adapter, device, queue))
}

impl Engine {
    /// `finest`: search motion down to this pyramid level (0 = full
    /// resolution, 1 = half). The frame itself is always built at full
    /// resolution with per-pixel vector selection.
    pub fn new(device: wgpu::Device, queue: wgpu::Queue, width: u32, height: u32, finest: usize) -> Result<Self> {
        if width < 64 || height < 64 {
            bail!("frames must be at least 64x64");
        }
        let levels = pyramid_levels(width, height);
        let finest = finest.min(levels.len() - 2);
        use wgpu::TextureFormat as F;
        use wgpu::TextureUsages as U;

        let hw = device.features().contains(wgpu::Features::FLOAT32_FILTERABLE);
        let pyr_src = include_str!("shaders/pyramid.wgsl");
        let me_src = with_lum(include_str!("shaders/motion.wgsl"), hw, "P.lsize");
        let co_src = with_lum(include_str!("shaders/compose.wgsl"), hw, "P.size");
        let module = |label: &str, src: &str| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(label), source: wgpu::ShaderSource::Wgsl(src.into()) })
        };
        let (pyr_mod, me_mod, co_mod) = (module("pyramid", pyr_src), module("motion", &me_src), module("compose", &co_src));

        let bgl_luma = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("luma"),
            entries: &[tex_entry(0, true), storage_entry(1, F::R32Float)],
        });
        let bgl_down = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("down"),
            entries: &[tex_entry(0, false), storage_entry(1, F::R32Float)],
        });
        let bgl_me = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("motion"),
            entries: &[
                tex_entry(0, hw),
                tex_entry(1, hw),
                tex_entry(2, false),
                tex_entry(3, false),
                storage_entry(4, F::Rgba32Float),
                uniform_entry(5),
                sampler_entry(6),
            ],
        });
        let bgl_co = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("compose"),
            entries: &[
                tex_entry(0, true),
                tex_entry(1, true),
                tex_entry(2, hw),
                tex_entry(3, hw),
                tex_entry(4, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                storage_entry(6, F::Rgba8Unorm),
                uniform_entry(7),
                sampler_entry(8),
            ],
        });

        let pyr_luma = pipeline(&device, &bgl_luma, &pyr_mod, "luma");
        let pyr_down = pipeline(&device, &bgl_down, &pyr_mod, "down");
        let me = [pipeline(&device, &bgl_me, &me_mod, "search"), pipeline(&device, &bgl_me, &me_mod, "median")];
        let compose = pipeline(&device, &bgl_co, &co_mod, "compose");

        let make_slot = |i: usize| {
            let (color, color_view) = tex(&device, &format!("color{i}"), width, height, F::Rgba8Unorm, U::TEXTURE_BINDING | U::COPY_DST);
            let luma = levels
                .iter()
                .enumerate()
                .map(|(l, lv)| tex(&device, &format!("luma{i}.{l}"), lv.w, lv.h, F::R32Float, U::TEXTURE_BINDING | U::STORAGE_BINDING))
                .collect();
            Slot { color, color_view, luma }
        };
        let slots = [make_slot(0), make_slot(1)];

        let fields: Vec<[(wgpu::Texture, wgpu::TextureView); 2]> = levels
            .iter()
            .enumerate()
            .map(|(l, lv)| {
                let u = U::TEXTURE_BINDING | U::STORAGE_BINDING;
                [tex(&device, &format!("field{l}a"), lv.gw, lv.gh, F::Rgba32Float, u), tex(&device, &format!("field{l}b"), lv.gw, lv.gh, F::Rgba32Float, u)]
            })
            .collect();
        let (_dummy, dummy_view) = tex(&device, "dummy", 1, 1, F::Rgba32Float, U::TEXTURE_BINDING);
        let (out, out_view) = tex(&device, "out", width, height, F::Rgba8Unorm, U::STORAGE_BINDING | U::TEXTURE_BINDING | U::COPY_SRC);
        let lsampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        // Pyramid bind groups, per slot.
        let pyr_groups = [0, 1].map(|s| {
            let slot = &slots[s];
            let mut g = vec![bind(
                &device,
                &bgl_luma,
                &[wgpu::BindingResource::TextureView(&slot.color_view), wgpu::BindingResource::TextureView(&slot.luma[0].1)],
            )];
            for l in 1..levels.len() {
                g.push(bind(
                    &device,
                    &bgl_down,
                    &[wgpu::BindingResource::TextureView(&slot.luma[l - 1].1), wgpu::BindingResource::TextureView(&slot.luma[l].1)],
                ));
            }
            g
        });

        // Motion passes, coarsest to finest. Final field of level l > 0 is
        // fields[l][0]; level 0 ends in fields[0][1] after the median.
        let mut passes = Vec::new();
        let nl = levels.len();
        for l in (finest..nl).rev() {
            let lv = levels[l];
            let cg = if l + 1 < nl { (levels[l + 1].gw, levels[l + 1].gh) } else { (1, 1) };
            let coarse_view = if l + 1 < nl { &fields[l + 1][0].1 } else { &dummy_view };
            let mk = |mode: u32| {
                let p = MeParams {
                    lsize: [lv.w as i32, lv.h as i32],
                    grid: [lv.gw as i32, lv.gh as i32],
                    cgrid: [cg.0 as i32, cg.1 as i32],
                    mode,
                    fine: (l == finest) as u32,
                    lambda: LAMBDA,
                    range: COARSE_RANGE,
                    sparse: (l <= finest + 1) as u32,
                    _pad: [0; 1],
                };
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("me params"),
                    contents: bytemuck::bytes_of(&p),
                    usage: wgpu::BufferUsages::UNIFORM,
                })
            };
            let steps: [(u32, &wgpu::TextureView, &wgpu::TextureView, usize); 3] = [
                (if l + 1 == nl { 0 } else { 1 }, &dummy_view, &fields[l][0].1, 0),
                (2, &fields[l][0].1, &fields[l][1].1, 0),
                (2, &fields[l][1].1, &fields[l][0].1, 0),
            ];
            for (mode, prev, dst, pipe) in steps {
                let buf = mk(mode);
                let groups = [0, 1].map(|b| {
                    let (sa, sb) = (&slots[1 - b], &slots[b]);
                    bind(
                        &device,
                        &bgl_me,
                        &[
                            wgpu::BindingResource::TextureView(&sa.luma[l].1),
                            wgpu::BindingResource::TextureView(&sb.luma[l].1),
                            wgpu::BindingResource::TextureView(prev),
                            wgpu::BindingResource::TextureView(coarse_view),
                            wgpu::BindingResource::TextureView(dst),
                            buf.as_entire_binding(),
                            wgpu::BindingResource::Sampler(&lsampler),
                        ],
                    )
                });
                passes.push(Pass { pipeline: pipe, groups, gw: lv.gw, gh: lv.gh });
            }
            if l == finest {
                let buf = mk(2);
                let groups = [0, 1].map(|b| {
                    let (sa, sb) = (&slots[1 - b], &slots[b]);
                    bind(
                        &device,
                        &bgl_me,
                        &[
                            wgpu::BindingResource::TextureView(&sa.luma[l].1),
                            wgpu::BindingResource::TextureView(&sb.luma[l].1),
                            wgpu::BindingResource::TextureView(&fields[l][0].1),
                            wgpu::BindingResource::TextureView(&dummy_view),
                            wgpu::BindingResource::TextureView(&fields[l][1].1),
                            buf.as_entire_binding(),
                            wgpu::BindingResource::Sampler(&lsampler),
                        ],
                    )
                });
                passes.push(Pass { pipeline: 1, groups, gw: lv.gw, gh: lv.gh });
            }
        }

        let compose_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("compose params"),
            size: std::mem::size_of::<ComposeParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let compose_groups = [0, 1].map(|b| {
            let (sa, sb) = (&slots[1 - b], &slots[b]);
            bind(
                &device,
                &bgl_co,
                &[
                    wgpu::BindingResource::TextureView(&sa.color_view),
                    wgpu::BindingResource::TextureView(&sb.color_view),
                    wgpu::BindingResource::TextureView(&sa.luma[0].1),
                    wgpu::BindingResource::TextureView(&sb.luma[0].1),
                    wgpu::BindingResource::TextureView(&fields[finest][1].1),
                    wgpu::BindingResource::Sampler(&sampler),
                    wgpu::BindingResource::TextureView(&out_view),
                    compose_params.as_entire_binding(),
                    wgpu::BindingResource::Sampler(&lsampler),
                ],
            )
        });

        let padded_row = (width * 4).div_ceil(256) * 256;
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (padded_row * height) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Ok(Self {
            device,
            queue,
            width,
            height,
            levels,
            finest,
            slots,
            newest: 1,
            pushed: 0,
            out,
            out_view,
            pyr_luma,
            pyr_down,
            me,
            compose,
            pyr_groups,
            passes,
            compose_groups,
            compose_params,
            readback,
            padded_row,
        })
    }

    /// Upload the next real frame (tightly packed RGBA8, sRGB-encoded).
    pub fn push_rgba(&mut self, rgba: &[u8]) {
        assert_eq!(rgba.len(), (self.width * self.height * 4) as usize, "frame size mismatch");
        self.newest = 1 - self.newest;
        self.pushed += 1;
        let slot = &self.slots[self.newest];
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &slot.color, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            rgba,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(self.width * 4), rows_per_image: Some(self.height) },
            wgpu::Extent3d { width: self.width, height: self.height, depth_or_array_layers: 1 },
        );
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            let groups = &self.pyr_groups[self.newest];
            for (l, lv) in self.levels.iter().enumerate() {
                pass.set_pipeline(if l == 0 { &self.pyr_luma } else { &self.pyr_down });
                pass.set_bind_group(0, &groups[l], &[]);
                pass.dispatch_workgroups(div_up(lv.w, 16), div_up(lv.h, 16), 1);
            }
        }
        self.queue.submit([enc.finish()]);
    }

    pub fn ready(&self) -> bool {
        self.pushed >= 2
    }

    /// Create the frame at time `t` between the two most recent pushed
    /// frames (0 = older, 1 = newer). Returns the output texture view.
    pub fn interpolate(&mut self, t: f32) -> &wgpu::TextureView {
        assert!(self.ready(), "push two frames first");
        let b = self.newest;
        let lv = self.levels[self.finest];
        let scale = (1u32 << self.finest) as f32;
        let p = ComposeParams {
            size: [self.width as i32, self.height as i32],
            grid: [lv.gw as i32, lv.gh as i32],
            t: t.clamp(0.0, 1.0),
            block: BLOCK as f32 * scale,
            vscale: scale,
            swap_rb: 0.0,
        };
        self.queue.write_buffer(&self.compose_params, 0, bytemuck::bytes_of(&p));
        let mut enc = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            for ps in &self.passes {
                pass.set_pipeline(&self.me[ps.pipeline]);
                pass.set_bind_group(0, &ps.groups[b], &[]);
                // search: one workgroup per block; median: 8x8 blocks per group
                if ps.pipeline == 0 {
                    pass.dispatch_workgroups(ps.gw, ps.gh, 1);
                } else {
                    pass.dispatch_workgroups(div_up(ps.gw, 8), div_up(ps.gh, 8), 1);
                }
            }
            pass.set_pipeline(&self.compose);
            pass.set_bind_group(0, &self.compose_groups[b], &[]);
            pass.dispatch_workgroups(div_up(self.width, 16), div_up(self.height, 16), 1);
        }
        self.queue.submit([enc.finish()]);
        &self.out_view
    }

    /// The texture the last interpolated frame was written to.
    pub fn output_view(&self) -> &wgpu::TextureView {
        &self.out_view
    }

    /// Block until the GPU is idle.
    pub fn wait(&self) {
        let _ = self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
    }

    /// Copy the last interpolated frame back to the CPU (tightly packed RGBA8).
    pub fn read_output(&self) -> Vec<u8> {
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: &self.out, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.readback,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(self.padded_row), rows_per_image: Some(self.height) },
            },
            wgpu::Extent3d { width: self.width, height: self.height, depth_or_array_layers: 1 },
        );
        self.queue.submit([enc.finish()]);
        let slice = self.readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.wait();
        let data = slice.get_mapped_range().expect("readback buffer mapped");
        let mut out = Vec::with_capacity((self.width * self.height * 4) as usize);
        for row in 0..self.height {
            let s = (row * self.padded_row) as usize;
            out.extend_from_slice(&data[s..s + (self.width * 4) as usize]);
        }
        drop(data);
        self.readback.unmap();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::{blend, psnr, Kind, Scene};

    #[test]
    fn pyramid_stops_at_min_level() {
        let l = pyramid_levels(1920, 1080);
        assert_eq!((l[0].w, l[0].h, l[0].gw, l[0].gh), (1920, 1080, 240, 135));
        assert!(l.last().unwrap().h >= MIN_LEVEL);
        assert!(l.windows(2).all(|p| p[1].w * 2 >= p[0].w));
    }

    /// The real test: regenerate a dropped frame and compare it with the
    /// ground truth. Skipped when no GPU (or software adapter) exists.
    #[test]
    fn interpolation_beats_crossfade() {
        let Ok((_i, _a, device, queue)) = gpu() else {
            eprintln!("no GPU adapter, skipping");
            return;
        };
        for kind in [Kind::Pan, Kind::Sprites] {
            let (w, h) = (640, 360);
            let sc = Scene::new(kind, w, h);
            let mut eng = Engine::new(device.clone(), queue.clone(), w, h, 0).unwrap();
            let (a, truth, b) = (sc.render(0.0), sc.render(1.0), sc.render(2.0));
            eng.push_rgba(&a);
            eng.push_rgba(&b);
            eng.interpolate(0.5);
            let ours = psnr(&eng.read_output(), &truth);
            let base = psnr(&blend(&a, &b), &truth);
            assert!(ours > base + 6.0, "{kind:?}: ghostframe {ours:.2} dB vs blend {base:.2} dB");
        }
    }
}
