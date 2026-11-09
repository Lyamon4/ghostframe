//! A tiny Vulkan "game" for testing the layer end to end: renders the
//! synthetic scene at a fixed frame rate through wgpu's Vulkan backend, so
//! the ghostframe layer sees a real Vulkan swapchain and real presents.
//! Frame k shows scene time 2k, so the frame the layer generates between
//! k and k+1 has an exact ground truth at scene time 2k+1.

use crate::scene::{Kind, Scene};
use anyhow::{bail, Context, Result};
use std::sync::Arc;
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

struct Gfx {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::RenderPipeline,
    tex: wgpu::Texture,
    group: wgpu::BindGroup,
}

struct Game {
    w: u32,
    h: u32,
    fps: f64,
    frames: Option<u64>,
    scene: Scene,
    gfx: Option<Gfx>,
    start: Instant,
    shown: u64,
    err: Option<anyhow::Error>,
}

impl Game {
    fn init(&mut self, el: &ActiveEventLoop) -> Result<()> {
        let attrs = Window::default_attributes()
            .with_title("ghostframe test game (Vulkan)")
            .with_inner_size(PhysicalSize::new(self.w, self.h))
            .with_resizable(false)
            .with_active(false);
        let window = Arc::new(el.create_window(attrs)?);
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
        desc.backends = wgpu::Backends::VULKAN;
        let instance = wgpu::Instance::new(desc);
        let surface = instance.create_surface(window.clone())?;
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
            apply_limit_buckets: false,
        }))
        .context("no Vulkan adapter (the test game needs Vulkan)")?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))?;
        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats.iter().copied().find(|f| !f.is_srgb()).unwrap_or(caps.formats[0]);
        let size = window.inner_size();
        if (size.width, size.height) != (self.w, self.h) {
            bail!("window came out {}x{} instead of {}x{} (display scaling?)", size.width, size.height, self.w, self.h);
        }
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: self.w,
            height: self.h,
            present_mode: wgpu::PresentMode::Fifo,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            color_space: wgpu::SurfaceColorSpace::Auto,
        };
        surface.configure(&device, &config);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("blit"),
            source: wgpu::ShaderSource::Wgsl(crate::demo::BLIT.into()),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("blit"),
            layout: None,
            vertex: wgpu::VertexState { module: &module, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[] },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(format.into())],
            }),
            multiview_mask: None,
            cache: None,
        });
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("frame"),
            size: wgpu::Extent3d { width: self.w, height: self.h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        // Nearest sampling: the window shows the scene's pixels 1:1.
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor::default());
        let view = tex.create_view(&Default::default());
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler) },
            ],
        });
        println!("test game: {}x{} {:?} via {:?} on {}", self.w, self.h, format, adapter.get_info().backend, adapter.get_info().name);
        self.gfx = Some(Gfx { window, surface, config, device, queue, pipeline, tex, group });
        self.start = Instant::now();
        Ok(())
    }

    fn frame(&mut self, el: &ActiveEventLoop) -> Result<()> {
        let due = (self.start.elapsed().as_secs_f64() * self.fps) as u64;
        if due < self.shown {
            return Ok(());
        }
        let k = self.shown;
        let pixels = self.scene.render((2 * k) as f32);
        let g = self.gfx.as_ref().unwrap();
        g.queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &g.tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            &pixels,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(self.w * 4), rows_per_image: Some(self.h) },
            wgpu::Extent3d { width: self.w, height: self.h, depth_or_array_layers: 1 },
        );
        let frame = match g.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f) | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            other => bail!("could not get a swapchain image: {:?}", std::mem::discriminant(&other)),
        };
        let view = frame.texture.create_view(&Default::default());
        let mut enc = g.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&g.pipeline);
            pass.set_bind_group(0, &g.group, &[]);
            pass.draw(0..3, 0..1);
        }
        g.queue.submit([enc.finish()]);
        g.queue.present(frame);
        self.shown += 1;
        if self.frames.is_some_and(|n| self.shown >= n) {
            println!("test game: presented {} frames in {:.1}s", self.shown, self.start.elapsed().as_secs_f64());
            el.exit();
        }
        Ok(())
    }
}

impl ApplicationHandler for Game {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.gfx.is_none() {
            if let Err(e) = self.init(el) {
                self.err = Some(e);
                el.exit();
            }
        }
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::RedrawRequested => {
                if let Err(e) = self.frame(el) {
                    self.err = Some(e);
                    el.exit();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, el: &ActiveEventLoop) {
        let next = self.start + Duration::from_secs_f64(self.shown as f64 / self.fps);
        el.set_control_flow(ControlFlow::WaitUntil(next));
        if let Some(g) = &self.gfx {
            g.window.request_redraw();
            let _ = &g.config;
        }
    }
}

pub fn run(w: u32, h: u32, fps: f64, frames: Option<u64>, kind: Kind) -> Result<()> {
    let event_loop = EventLoop::new().context("cannot open a window")?;
    let mut game = Game { w, h, fps, frames, scene: Scene::new(kind, w, h), gfx: None, start: Instant::now(), shown: 0, err: None };
    event_loop.run_app(&mut game)?;
    match game.err {
        Some(e) => Err(e),
        None => Ok(()),
    }
}

/// Compare a frame dumped by the layer with the ground truth. The layer's
/// frame counter can reset when a swapchain is recreated, so every
/// in-between position is tried and the best match is reported.
pub fn verify_dump(path: &std::path::Path, kind: Kind) -> Result<()> {
    let data = std::fs::read(path).with_context(|| format!("cannot read {}", path.display()))?;
    if !data.starts_with(b"GFDUMP1\n") || data.len() < 28 {
        bail!("{} is not a ghostframe dump", path.display());
    }
    let u32_at = |o: usize| u32::from_le_bytes(data[o..o + 4].try_into().unwrap());
    let (w, h, swap_rb) = (u32_at(8), u32_at(12), u32_at(16) == 1);
    let pushed = u64::from_le_bytes(data[20..28].try_into().unwrap());
    let mut px = data[28..].to_vec();
    if px.len() != (w * h * 4) as usize {
        bail!("dump size does not match {w}x{h}");
    }
    if swap_rb {
        for p in px.chunks_exact_mut(4) {
            p.swap(0, 2);
        }
    }
    let scene = Scene::new(kind, w, h);
    let mut best = (f64::MIN, 0u64);
    for k in 0..pushed + 4 {
        let p = crate::scene::psnr(&px, &scene.render((2 * k + 1) as f32));
        if p > best.0 {
            best = (p, k);
        }
    }
    let k = best.1;
    let (a, truth, b) = (scene.render((2 * k) as f32), scene.render((2 * k + 1) as f32), scene.render((2 * k + 2) as f32));
    let blend = crate::scene::blend(&a, &b);
    println!("generated frame from the layer: {w}x{h}, best match between game frames {k} and {}", k + 1);
    println!("  repeat frame   {:.2} dB", crate::scene::psnr(&a, &truth));
    println!("  cross-fade     {:.2} dB", crate::scene::psnr(&blend, &truth));
    println!("  ghostframe     {:.2} dB   (in-game, through the Vulkan layer)", best.0);
    image::save_buffer(path.with_extension("png"), &px, w, h, image::ColorType::Rgba8)?;
    println!("  saved as {}", path.with_extension("png").display());
    Ok(())
}
