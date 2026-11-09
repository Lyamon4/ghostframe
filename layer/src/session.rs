//! The link between a running game and the tray app.
//!
//! Every second the layer writes `sessions/<pid>.json` (game, resolution,
//! frame rates) into the ghostframe data folder, and re-reads `games.cfg`,
//! where the tray app stores per-game settings (`Game.exe=off`). So the tray
//! menu always lists the games that are running right now, and toggling a
//! game there takes effect within a second, without restarting it.

use std::path::PathBuf;
use std::time::{Instant, SystemTime};

pub fn data_dir() -> Option<PathBuf> {
    if cfg!(windows) {
        std::env::var_os("LOCALAPPDATA").map(|p| PathBuf::from(p).join("ghostframe"))
    } else {
        std::env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")))
            .map(|p| p.join("ghostframe"))
    }
}

pub fn exe_name() -> String {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
        .unwrap_or_else(|| "unknown".into())
}

/// Generation is off until switched on in the tray, except for programs
/// started with `ghostframe run` (ENABLE_GHOSTFRAME=1).
fn default_on() -> bool {
    std::env::var("ENABLE_GHOSTFRAME").is_ok_and(|v| v == "1")
}

/// `enabled` for `exe` from a games.cfg text.
pub fn enabled_in(cfg: &str, exe: &str, default: bool) -> bool {
    for line in cfg.lines() {
        let line = line.trim();
        if line.starts_with('#') {
            continue;
        }
        if let Some((name, value)) = line.rsplit_once('=') {
            if name.trim().eq_ignore_ascii_case(exe) {
                return !value.trim().eq_ignore_ascii_case("off");
            }
        }
    }
    default
}

fn json_escape(s: &str) -> String {
    s.chars()
        .flat_map(|c| match c {
            '"' => "\\\"".chars().collect::<Vec<_>>(),
            '\\' => "\\\\".chars().collect(),
            c if (c as u32) < 0x20 => format!("\\u{:04x}", c as u32).chars().collect(),
            c => vec![c],
        })
        .collect()
}

pub struct Session {
    status_path: PathBuf,
    cfg_path: PathBuf,
    cfg_mtime: Option<SystemTime>,
    exe: String,
    pub enabled: bool,
    tick: Instant,
    real: u64,
    shown: u64,
}

impl Session {
    pub fn new() -> Option<Self> {
        let dir = data_dir()?;
        let sessions = dir.join("sessions");
        std::fs::create_dir_all(&sessions).ok()?;
        let mut s = Session {
            status_path: sessions.join(format!("{}.json", std::process::id())),
            cfg_path: dir.join("games.cfg"),
            cfg_mtime: None,
            exe: exe_name(),
            enabled: default_on(),
            tick: Instant::now(),
            real: 0,
            shown: 0,
        };
        s.poll_config();
        Some(s)
    }

    fn poll_config(&mut self) {
        let mtime = std::fs::metadata(&self.cfg_path).and_then(|m| m.modified()).ok();
        if mtime == self.cfg_mtime {
            return;
        }
        self.cfg_mtime = mtime;
        let cfg = std::fs::read_to_string(&self.cfg_path).unwrap_or_default();
        let enabled = enabled_in(&cfg, &self.exe, default_on());
        if enabled != self.enabled {
            crate::log(&format!("frame generation {} for {} (from the tray)", if enabled { "on" } else { "off" }, self.exe));
        }
        self.enabled = enabled;
    }

    /// Call on every real present; `generated` is how many extra frames
    /// were shown with it; every second this also re-reads games.cfg.
    pub fn frame(&mut self, generated: u64, width: u32, height: u32) {
        self.real += 1;
        self.shown += 1 + generated;
        let secs = self.tick.elapsed().as_secs_f64();
        if secs < 1.0 {
            return;
        }
        self.poll_config();
        let json = format!(
            "{{\"pid\":{},\"exe\":\"{}\",\"width\":{},\"height\":{},\"real_fps\":{:.1},\"shown_fps\":{:.1},\"enabled\":{}}}",
            std::process::id(),
            json_escape(&self.exe),
            width,
            height,
            self.real as f64 / secs,
            self.shown as f64 / secs,
            self.enabled
        );
        // Write-then-rename so the tray never reads half a file.
        let tmp = self.status_path.with_extension("tmp");
        if std::fs::write(&tmp, json).is_ok() {
            let _ = std::fs::rename(&tmp, &self.status_path);
        }
        self.tick = Instant::now();
        self.real = 0;
        self.shown = 0;
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.status_path);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn config_lookup() {
        let cfg = "# comment\nGame.exe=off\nOther.exe = on\n";
        assert!(!super::enabled_in(cfg, "game.exe", true));
        assert!(super::enabled_in(cfg, "Other.exe", false));
        assert!(!super::enabled_in(cfg, "missing.exe", false));
        assert!(super::enabled_in(cfg, "missing.exe", true));
    }
}
