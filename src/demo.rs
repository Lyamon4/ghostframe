//! Side-by-side live demo: the left half is a "game" running at 30 fps,
//! the right half is the same game at 60 fps with every other frame made
//! by ghostframe.

use crate::engine::{self, Engine};
use crate::scene::{Kind, Scene};
use anyhow::{Context, Result};
use std::sync::Arc;
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

const OUT_HZ: f64 = 60.0;

pub(crate) const BLIT: &str = r#"
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var smp: sampler;
struct V { @builtin(position) pos: vec4f, @location(0) uv: vec2f }
@vertex fn vs(@builtin(vertex_index) i: u32) -> V {
    let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
    var o: V;
    o.pos = vec4f(p * 2.0 - 1.0, 0.0, 1.0);
    o.uv = vec2f(p.x, 1.0 - p.y);
    return o;
}
@fragment fn fs(v: V) -> @location(0) vec4f { return textureSample(tex, smp, v.uv); }
"#;

struct Gfx {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    queue: wgpu::Queue,
    engine: Engine,
    pipeline: wgpu::RenderPipeline,
    source: wgpu::Texture,
    left_group: wgpu::BindGroup,
    right_real: wgpu::BindGroup,
    right_ghost: wgpu::BindGroup,
}

struct App {
    w: u32,
    h: u32,
    kind_i: usize,
    scene: Scene,
    gfx: Option<Gfx>,
    start: Instant,
    paused_at: Option<Instant>,
    last_tick: i64,
    current: Vec<u8>, // source frame k (shown on the left)
    gen_ms: f64,
    err: Option<anyhow::Error>,
}

impl App {
    fn title(&self) -> String {
        format!(
            "ghostframe demo · {} · left: game at 30 fps · right: ghostframe at 60 fps · ghost frame {:.1} ms · [Tab] scene  [Space] pause  [Esc] quit",
            self.scene.kind.name(),
            self.gen_ms
        )
    }

    fn init(&mut self, el: &ActiveEventLoop) -> Result<()> {
        let attrs = Window::default_attributes()
            .with_title("ghostframe demo")
            .with_inner_size(PhysicalSize::new(self.w * 2, self.h));
        let attrs = match el.primary_monitor() {
            Some(m) => {
                let ms = m.size();
                let x = (ms.width as i32 - (self.w * 2) as i32).max(0) / 2;
                let y = (ms.height as i32 - self.h as i32).max(0) / 2;
                attrs.with_position(winit::dpi::PhysicalPosition::new(x, y))
            }
            None => attrs,
        };
        let window = Arc::new(el.create_window(attrs)?);
        let (instance, adapter, device, queue) = engine::gpu()?;
        let surface = instance.create_surface(window.clone())?;
        let caps = surface.get_capabilities(&adapter);
        // Our pixels are already sRGB-encoded: present them as-is.
        let format = caps.formats.iter().copied().find(|f| !f.is_srgb()).unwrap_or(caps.formats[0]);
        let size = window.inner_size();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            color_space: wgpu::SurfaceColorSpace::Auto,
        };
        surface.configure(&device, &config);

        let engine = Engine::new(device.clone(), queue.clone(), self.w, self.h, crate::default_finest(self.w, self.h))?;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("blit"), source: wgpu::ShaderSource::Wgsl(BLIT.into()) });
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
        let source = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("source"),
            size: wgpu::Extent3d { width: self.w, height: self.h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let layout = pipeline.get_bind_group_layout(0);
        let group = |view: &wgpu::TextureView| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler) },
                ],
            })
        };
        let source_view = source.create_view(&Default::default());
        let left_group = group(&source_view);
        let right_real = group(&source_view);
        let ghost_view = engine.output_view();
        let right_ghost = group(ghost_view);
        self.gfx = Some(Gfx { window, surface, config, device, queue, engine, pipeline, source, left_group, right_real, right_ghost });
        self.reset_stream();
        Ok(())
    }

    /// Start the source stream over (new scene or first launch).
    fn reset_stream(&mut self) {
        self.start = Instant::now();
        self.last_tick = -1;
        self.current = self.scene.render(0.0);
        let g = self.gfx.as_mut().unwrap();
        g.engine.push_rgba(&self.current);
        g.engine.push_rgba(&self.current);
    }

    fn elapsed(&self) -> f64 {
        self.paused_at.unwrap_or_else(Instant::now).duration_since(self.start).as_secs_f64()
    }

    fn render(&mut self) -> Result<()> {
        let tick = (self.elapsed() * OUT_HZ).floor() as i64;
        if tick == self.last_tick {
            return Ok(());
        }
        self.last_tick = tick;
        let k = tick / 2;
        let odd = tick % 2 == 1;
        // Source frame k lives at scene time 2k (scene speeds are per 60 fps frame).
        if !odd {
            self.current = self.scene.render((2 * k) as f32);
            let next = self.scene.render((2 * k + 2) as f32);
            let g = self.gfx.as_mut().unwrap();
            g.queue.write_texture(
                wgpu::TexelCopyTextureInfo { texture: &g.source, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                &self.current,
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(self.w * 4), rows_per_image: Some(self.h) },
                wgpu::Extent3d { width: self.w, height: self.h, depth_or_array_layers: 1 },
            );
            // Engine pair (k, k+1): the in-between frame for the next tick.
            g.engine.push_rgba(&self.current);
            g.engine.push_rgba(&next);
        } else {
            let g = self.gfx.as_mut().unwrap();
            g.engine.wait(); // time only the frame generation, not earlier work
            let t0 = Instant::now();
            g.engine.interpolate(0.5);
            g.engine.wait();
            self.gen_ms = self.gen_ms * 0.9 + t0.elapsed().as_secs_f64() * 1000.0 * 0.1;
        }

        let g = self.gfx.as_ref().unwrap();
        // Minimised or tiny window: nothing to draw into.
        if g.config.width < 16 || g.config.height < 2 {
            return Ok(());
        }
        let frame = match g.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f) | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            _ => {
                g.surface.configure(&g.device, &g.config);
                return Ok(());
            }
        };
        let view = frame.texture.create_view(&Default::default());
        let mut enc = g.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("blit"),
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
            let (fw, fh) = (g.config.width as f32, g.config.height as f32);
            let gap = 4.0;
            let half = (fw - gap) / 2.0;
            pass.set_pipeline(&g.pipeline);
            pass.set_viewport(0.0, 0.0, half, fh, 0.0, 1.0);
            pass.set_bind_group(0, &g.left_group, &[]);
            pass.draw(0..3, 0..1);
            pass.set_viewport(half + gap, 0.0, half, fh, 0.0, 1.0);
            pass.set_bind_group(0, if odd { &g.right_ghost } else { &g.right_real }, &[]);
            pass.draw(0..3, 0..1);
        }
        g.queue.submit([enc.finish()]);
        g.queue.present(frame);
        if tick % 30 == 0 {
            g.window.set_title(&self.title());
        }
        Ok(())
    }
}

impl ApplicationHandler for App {
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
            WindowEvent::Resized(size) => {
                if let Some(g) = &mut self.gfx {
                    g.config.width = size.width.max(1);
                    g.config.height = size.height.max(1);
                    g.surface.configure(&g.device, &g.config);
                }
            }
            WindowEvent::KeyboardInput { event, .. } if event.state == ElementState::Pressed => match event.logical_key {
                Key::Named(NamedKey::Escape) => el.exit(),
                Key::Named(NamedKey::Space) => {
                    self.paused_at = match self.paused_at {
                        Some(p) => {
                            self.start += p.elapsed();
                            None
                        }
                        None => Some(Instant::now()),
                    };
                }
                Key::Named(NamedKey::Tab) => {
                    self.kind_i = (self.kind_i + 1) % Kind::ALL.len();
                    self.scene = Scene::new(Kind::ALL[self.kind_i], self.w, self.h);
                    self.reset_stream();
                }
                _ => {}
            },
            WindowEvent::RedrawRequested => {
                if let Err(e) = self.render() {
                    self.err = Some(e);
                    el.exit();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, el: &ActiveEventLoop) {
        let next = self.start + Duration::from_secs_f64((self.last_tick + 1) as f64 / OUT_HZ);
        el.set_control_flow(ControlFlow::WaitUntil(next));
        if let Some(g) = &self.gfx {
            g.window.request_redraw();
        }
    }
}

pub fn run(w: u32, h: u32) -> Result<()> {
    let event_loop = EventLoop::new().context("cannot open a window")?;
    let mut app = App {
        w,
        h,
        kind_i: 1,
        scene: Scene::new(Kind::Sprites, w, h),
        gfx: None,
        start: Instant::now(),
        paused_at: None,
        last_tick: -1,
        current: Vec::new(),
        gen_ms: 0.0,
        err: None,
    };
    event_loop.run_app(&mut app)?;
    match app.err {
        Some(e) => Err(e),
        None => Ok(()),
    }
}
