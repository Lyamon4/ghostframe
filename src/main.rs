//! ghostframe: open-source frame generation.

mod demo;
mod engine;
mod install;
mod scene;
mod testgame;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use engine::Engine;
use scene::{Kind, Scene};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Parser)]
#[command(name = "ghostframe", version, about = "Open-source frame generation: double your frame rate with GPU motion-compensated interpolation.")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Measure quality against ground truth on synthetic game-like scenes.
    Bench {
        /// Frame size as WIDTHxHEIGHT.
        #[arg(long, default_value = "960x540")]
        size: String,
        /// Interpolated frames per scene.
        #[arg(long, default_value_t = 8)]
        pairs: usize,
        /// Save example frames (truth / ghostframe / blend) here.
        #[arg(long)]
        save: Option<PathBuf>,
        /// Finest motion-search level: 0 = full resolution, 1 = half.
        #[arg(long)]
        finest: Option<usize>,
    },
    /// Time the GPU pipeline at a given resolution.
    Speed {
        #[arg(long, default_value = "1920x1080")]
        size: String,
        #[arg(long, default_value_t = 100)]
        iters: usize,
        #[arg(long)]
        finest: Option<usize>,
    },
    /// Raise the frame rate of a PNG sequence (frames sorted by name).
    Interp {
        input: PathBuf,
        output: PathBuf,
        /// 2 = double, 3 = triple, 4 = quadruple the frame rate.
        #[arg(long, default_value_t = 2)]
        factor: u32,
    },
    /// Live side-by-side window: a game at 30 fps vs. ghostframe at 60 fps.
    Demo {
        /// Size of each half of the window.
        #[arg(long, default_value = "800x450")]
        size: String,
    },
    /// Install the Vulkan layer for games (off until a game asks for it).
    Install,
    /// Remove the Vulkan layer.
    Uninstall,
    /// Show whether the layer is installed.
    Status,
    /// Start a game with frame generation: ghostframe run game.exe [args]
    Run {
        #[arg(trailing_var_arg = true, allow_hyphen_values = true, required = true)]
        command: Vec<String>,
    },
    /// A small Vulkan test game for checking the layer end to end.
    TestGame {
        #[arg(long, default_value = "960x540")]
        size: String,
        #[arg(long, default_value_t = 30.0)]
        fps: f64,
        /// Quit after this many frames.
        #[arg(long)]
        frames: Option<u64>,
    },
    /// Compare a frame dumped by the layer (GHOSTFRAME_DUMP) with ground truth.
    VerifyDump { file: PathBuf },
    /// Show which GPU and backend ghostframe uses.
    Info,
}

fn parse_size(s: &str) -> Result<(u32, u32)> {
    let (w, h) = s.split_once(['x', 'X']).context("size must look like 1920x1080")?;
    Ok((w.trim().parse()?, h.trim().parse()?))
}

fn save_png(path: &Path, w: u32, h: u32, rgba: &[u8]) -> Result<()> {
    image::save_buffer(path, rgba, w, h, image::ColorType::Rgba8).with_context(|| format!("writing {}", path.display()))
}

/// Default search depth: full resolution for small frames, half
/// resolution from 360p up (motion is estimated on 2x2-downscaled frames,
/// the output is still built per pixel at full resolution).
pub fn default_finest(w: u32, h: u32) -> usize {
    if w.min(h) >= 360 { 1 } else { 0 }
}

fn new_engine(w: u32, h: u32, finest: Option<usize>) -> Result<(Engine, String)> {
    let (_i, adapter, device, queue) = engine::gpu()?;
    let info = adapter.get_info();
    let label = format!("{} ({:?})", info.name, info.backend);
    let eng = Engine::new(device, queue, w, h, finest.unwrap_or_else(|| default_finest(w, h)))?;
    Ok((eng, label))
}

fn bench(size: &str, pairs: usize, save: Option<PathBuf>, finest: Option<usize>) -> Result<()> {
    let (w, h) = parse_size(size)?;
    let (mut eng, gpu) = new_engine(w, h, finest)?;
    println!("ghostframe bench · {w}x{h} · {pairs} frames per scene · motion search down to level {} · {gpu}", eng.finest);
    println!("each test drops every other frame of a 60 fps sequence, regenerates it from its neighbours, and compares to the real one\n");
    println!("{:<9} {:<42} {:>10} {:>10} {:>12} {:>9}", "scene", "motion", "repeat dB", "blend dB", "ghostframe", "ms/frame");
    let mut sum = [0.0f64; 3];
    for kind in Kind::ALL {
        let sc = Scene::new(kind, w, h);
        let mut p = [0.0f64; 3];
        let mut ms = 0.0;
        let mut prev = sc.render(0.0);
        eng.push_rgba(&prev);
        for i in 0..pairs {
            let t0 = (2 * i) as f32;
            let truth = sc.render(t0 + 1.0);
            let next = sc.render(t0 + 2.0);
            eng.push_rgba(&next);
            eng.wait();
            let start = Instant::now();
            eng.interpolate(0.5);
            eng.wait();
            ms += start.elapsed().as_secs_f64() * 1000.0;
            let ours = eng.read_output();
            let mix = scene::blend(&prev, &next);
            p[0] += scene::psnr(&prev, &truth);
            p[1] += scene::psnr(&mix, &truth);
            p[2] += scene::psnr(&ours, &truth);
            if i == 0 {
                if let Some(dir) = &save {
                    std::fs::create_dir_all(dir)?;
                    let n = kind.name();
                    save_png(&dir.join(format!("{n}_truth.png")), w, h, &truth)?;
                    save_png(&dir.join(format!("{n}_ghostframe.png")), w, h, &ours)?;
                    save_png(&dir.join(format!("{n}_blend.png")), w, h, &mix)?;
                }
            }
            prev = next;
        }
        let n = pairs as f64;
        println!(
            "{:<9} {:<42} {:>10.2} {:>10.2} {:>12.2} {:>9.2}",
            kind.name(),
            kind.about(),
            p[0] / n,
            p[1] / n,
            p[2] / n,
            ms / n
        );
        for k in 0..3 {
            sum[k] += p[k] / n;
        }
    }
    let k = Kind::ALL.len() as f64;
    println!("{:<9} {:<42} {:>10.2} {:>10.2} {:>12.2}", "mean", "", sum[0] / k, sum[1] / k, sum[2] / k);
    println!("\nrepeat = show the previous frame again (no frame generation) · blend = 50/50 cross-fade · higher dB is better");
    if let Some(dir) = save {
        println!("example frames saved to {}", dir.display());
    }
    Ok(())
}

fn speed(size: &str, iters: usize, finest: Option<usize>) -> Result<()> {
    let (w, h) = parse_size(size)?;
    let (mut eng, gpu) = new_engine(w, h, finest)?;
    let sc = Scene::new(Kind::Sprites, w, h);
    eng.push_rgba(&sc.render(0.0));
    eng.push_rgba(&sc.render(2.0));
    for _ in 0..10 {
        eng.interpolate(0.5);
    }
    eng.wait();
    let start = Instant::now();
    for _ in 0..iters {
        eng.interpolate(0.5);
    }
    eng.wait();
    let ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    println!("ghostframe speed · {w}x{h} · motion search down to level {} · {gpu}", eng.finest);
    println!("  {ms:.2} ms per generated frame  ({:.0} generated frames/s)", 1000.0 / ms);
    println!("  pyramid levels: {}", eng.levels.iter().map(|l| format!("{}x{}", l.w, l.h)).collect::<Vec<_>>().join(" → "));
    Ok(())
}

fn interp(input: &Path, output: &Path, factor: u32) -> Result<()> {
    if !(2..=8).contains(&factor) {
        bail!("factor must be between 2 and 8");
    }
    let mut files: Vec<PathBuf> = std::fs::read_dir(input)?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("png")))
        .collect();
    files.sort();
    if files.len() < 2 {
        bail!("need at least two PNG frames in {}", input.display());
    }
    let load = |p: &Path| -> Result<(u32, u32, Vec<u8>)> {
        let img = image::open(p).with_context(|| format!("reading {}", p.display()))?.to_rgba8();
        Ok((img.width(), img.height(), img.into_raw()))
    };
    let (w, h, first) = load(&files[0])?;
    let (mut eng, gpu) = new_engine(w, h, None)?;
    std::fs::create_dir_all(output)?;
    println!("ghostframe · {} frames → {} frames · {w}x{h} · {gpu}", files.len(), (files.len() - 1) * factor as usize + 1);
    let mut n = 0usize;
    let mut write = |rgba: &[u8]| -> Result<()> {
        save_png(&output.join(format!("{n:06}.png")), w, h, rgba)?;
        n += 1;
        Ok(())
    };
    eng.push_rgba(&first);
    write(&first)?;
    let start = Instant::now();
    for f in &files[1..] {
        let (fw, fh, next) = load(f)?;
        if (fw, fh) != (w, h) {
            bail!("{} has a different size", f.display());
        }
        eng.push_rgba(&next);
        for k in 1..factor {
            eng.interpolate(k as f32 / factor as f32);
            write(&eng.read_output())?;
        }
        write(&next)?;
    }
    println!("done in {:.1}s → {}", start.elapsed().as_secs_f64(), output.display());
    Ok(())
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Bench { size, pairs, save, finest } => bench(&size, pairs, save, finest),
        Cmd::Speed { size, iters, finest } => speed(&size, iters, finest),
        Cmd::Interp { input, output, factor } => interp(&input, &output, factor),
        Cmd::Demo { size } => {
            let (w, h) = parse_size(&size)?;
            demo::run(w, h)
        }
        Cmd::Install => install::install(),
        Cmd::Uninstall => install::uninstall(),
        Cmd::Status => install::status(),
        Cmd::Run { command } => install::run(&command),
        Cmd::TestGame { size, fps, frames } => {
            let (w, h) = parse_size(&size)?;
            testgame::run(w, h, fps, frames, Kind::Sprites)
        }
        Cmd::VerifyDump { file } => testgame::verify_dump(&file, Kind::Sprites),
        Cmd::Info => {
            let (_i, adapter, _d, _q) = engine::gpu()?;
            let info = adapter.get_info();
            println!("{} · {:?} · driver {}", info.name, info.backend, info.driver);
            Ok(())
        }
    }
}
