//! Synthetic, game-like test scenes rendered on the CPU at any (fractional)
//! time. Because a scene can be rendered at t + 0.5, every interpolated
//! frame has an exact ground truth to compare against.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// The camera pans across a detailed world.
    Pan,
    /// A static background with textured objects moving at different
    /// speeds, overlapping each other (occlusions).
    Sprites,
    /// Fast camera motion plus fast objects.
    Fast,
    /// The camera zooms in.
    Zoom,
}

impl Kind {
    pub const ALL: [Kind; 4] = [Kind::Pan, Kind::Sprites, Kind::Fast, Kind::Zoom];
    pub fn name(self) -> &'static str {
        match self {
            Kind::Pan => "pan",
            Kind::Sprites => "sprites",
            Kind::Fast => "fast",
            Kind::Zoom => "zoom",
        }
    }
    pub fn about(self) -> &'static str {
        match self {
            Kind::Pan => "camera pan, 6 px/frame",
            Kind::Sprites => "8 objects, up to 14 px/frame, occlusions",
            Kind::Fast => "camera 22 px/frame + objects 30 px/frame",
            Kind::Zoom => "zoom in 1.5%/frame",
        }
    }
}

fn hash(x: i32, y: i32, seed: u32) -> f32 {
    let mut h = (x as u32).wrapping_mul(0x8da6_b343) ^ (y as u32).wrapping_mul(0xd816_3841) ^ seed.wrapping_mul(0xcb1a_b31f);
    h ^= h >> 13;
    h = h.wrapping_mul(0x5bd1_e995);
    h ^= h >> 15;
    (h & 0x00ff_ffff) as f32 / 16_777_215.0
}

fn smooth(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

fn value_noise(x: f32, y: f32, scale: f32, seed: u32) -> f32 {
    let (fx, fy) = (x / scale, y / scale);
    let (ix, iy) = (fx.floor() as i32, fy.floor() as i32);
    let (tx, ty) = (smooth(fx - fx.floor()), smooth(fy - fy.floor()));
    let a = hash(ix, iy, seed);
    let b = hash(ix + 1, iy, seed);
    let c = hash(ix, iy + 1, seed);
    let d = hash(ix + 1, iy + 1, seed);
    let top = a + (b - a) * tx;
    let bot = c + (d - c) * tx;
    top + (bot - top) * ty
}

/// A detailed, colourful world texture: soft large shapes, mid-size
/// tiles with sharp edges, and fine grain (like foliage or gravel).
fn world(x: f32, y: f32, seed: u32) -> [f32; 3] {
    let large = value_noise(x, y, 96.0, seed);
    let mid = value_noise(x, y, 24.0, seed + 1);
    let fine = value_noise(x, y, 5.0, seed + 2);
    let tile = hash((x / 40.0).floor() as i32, (y / 40.0).floor() as i32, seed + 3);
    let edge = if tile > 0.72 { 0.25 } else { 0.0 };
    let base = 0.25 + 0.45 * large + 0.2 * mid + 0.12 * fine + edge;
    [
        (base * (0.75 + 0.5 * value_noise(x, y, 160.0, seed + 4))).clamp(0.0, 1.0),
        (base * (0.85 + 0.3 * mid)).clamp(0.0, 1.0),
        (base * (0.65 + 0.6 * value_noise(x, y, 130.0, seed + 5))).clamp(0.0, 1.0),
    ]
}

struct Sprite {
    x0: f32,
    y0: f32,
    vx: f32,
    vy: f32,
    w: f32,
    h: f32,
    round: bool,
    seed: u32,
}

pub struct Scene {
    pub kind: Kind,
    pub width: u32,
    pub height: u32,
    sprites: Vec<Sprite>,
}

impl Scene {
    pub fn new(kind: Kind, width: u32, height: u32) -> Self {
        let n = match kind {
            Kind::Sprites => 8,
            Kind::Fast => 5,
            _ => 0,
        };
        let speed = if kind == Kind::Fast { 30.0 } else { 14.0 };
        let (w, h) = (width as f32, height as f32);
        let sprites = (0..n)
            .map(|i| {
                let r = |k: u32| hash(i as i32, k as i32, 77);
                let size = (0.12 + 0.18 * r(1)) * h;
                Sprite {
                    x0: r(2) * w,
                    y0: r(3) * h,
                    vx: (r(4) - 0.5) * 2.0 * speed,
                    vy: (r(5) - 0.5) * speed,
                    w: size * (0.8 + 0.8 * r(6)),
                    h: size,
                    round: r(7) > 0.5,
                    seed: 100 + i,
                }
            })
            .collect();
        Self { kind, width, height, sprites }
    }

    /// Render the scene at (fractional) frame index `t` as RGBA8.
    pub fn render(&self, t: f32) -> Vec<u8> {
        let (w, h) = (self.width as f32, self.height as f32);
        let (cam_x, cam_y, zoom) = match self.kind {
            Kind::Pan => (6.0 * t, 1.5 * t, 1.0),
            Kind::Sprites => (0.0, 0.0, 1.0),
            Kind::Fast => (22.0 * t, -4.0 * t, 1.0),
            Kind::Zoom => (0.0, 0.0, 1.015f32.powf(t)),
        };
        // Sprite positions wrap around a slightly larger area so they keep
        // entering and leaving the frame.
        let pos: Vec<(f32, f32)> = self
            .sprites
            .iter()
            .map(|s| {
                let span_x = w + s.w * 2.0;
                let span_y = h + s.h * 2.0;
                let x = (s.x0 + s.vx * t).rem_euclid(span_x) - s.w;
                let y = (s.y0 + s.vy * t).rem_euclid(span_y) - s.h;
                (x, y)
            })
            .collect();

        let mut out = vec![0u8; (self.width * self.height * 4) as usize];
        let row_bytes = (self.width * 4) as usize;
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        let rows_per = (self.height as usize).div_ceil(threads);
        std::thread::scope(|s| {
            for (chunk_i, chunk) in out.chunks_mut(rows_per * row_bytes).enumerate() {
                let pos = &pos;
                s.spawn(move || self.render_rows(chunk, chunk_i * rows_per, pos, cam_x, cam_y, zoom));
            }
        });
        out
    }

    fn render_rows(&self, out: &mut [u8], y0: usize, pos: &[(f32, f32)], cam_x: f32, cam_y: f32, zoom: f32) {
        let (w, h) = (self.width as f32, self.height as f32);
        let rows = out.len() / (self.width as usize * 4);
        for ry in 0..rows {
            let py = (y0 + ry) as u32;
            for px in 0..self.width {
                let (fx, fy) = (px as f32 + 0.5, py as f32 + 0.5);
                let wx = (fx - w * 0.5) / zoom + w * 0.5 + cam_x;
                let wy = (fy - h * 0.5) / zoom + h * 0.5 + cam_y;
                let mut c = world(wx, wy, 1);
                for (s, &(sx, sy)) in self.sprites.iter().zip(pos) {
                    let (u, v) = (fx - sx, fy - sy);
                    let inside = if s.round {
                        let (dx, dy) = (u / s.w - 0.5, v / s.h - 0.5);
                        dx * dx + dy * dy < 0.25
                    } else {
                        u >= 0.0 && v >= 0.0 && u < s.w && v < s.h
                    };
                    if inside {
                        let t = world(u * 1.3, v * 1.3, s.seed);
                        c = [t[0] * 0.9 + 0.1, t[1] * 0.6, t[2] * 0.5 + 0.15];
                    }
                }
                let i = (ry * self.width as usize + px as usize) * 4;
                out[i] = (c[0] * 255.0) as u8;
                out[i + 1] = (c[1] * 255.0) as u8;
                out[i + 2] = (c[2] * 255.0) as u8;
                out[i + 3] = 255;
            }
        }
    }
}

/// Peak signal-to-noise ratio over RGB, in dB (higher is better).
pub fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let mut se = 0.0f64;
    let mut n = 0usize;
    for (pa, pb) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        for k in 0..3 {
            let d = pa[k] as f64 - pb[k] as f64;
            se += d * d;
        }
        n += 3;
    }
    let mse = se / n as f64;
    if mse == 0.0 {
        return 99.0;
    }
    10.0 * (255.0f64 * 255.0 / mse).log10()
}

/// 50/50 cross-fade: what you get with no motion compensation at all.
pub fn blend(a: &[u8], b: &[u8]) -> Vec<u8> {
    a.iter().zip(b).map(|(&x, &y)| ((x as u16 + y as u16 + 1) / 2) as u8).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn psnr_of_identical_frames_is_capped() {
        let a = vec![10u8, 20, 30, 255, 40, 50, 60, 255];
        assert_eq!(psnr(&a, &a), 99.0);
    }

    #[test]
    fn psnr_drops_with_error() {
        let a = vec![100u8; 16];
        let mut b = a.clone();
        b[0] = 110;
        let mut c = a.clone();
        c[0] = 160;
        assert!(psnr(&a, &b) > psnr(&a, &c));
    }

    #[test]
    fn scenes_render_deterministically_and_move() {
        let s = Scene::new(Kind::Pan, 64, 48);
        assert_eq!(s.render(3.0), s.render(3.0));
        assert_ne!(s.render(0.0), s.render(1.0));
    }
}
