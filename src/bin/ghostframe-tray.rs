//! ghostframe in the notification area.
//!
//! Click the icon to see the games running with ghostframe right now, with
//! their frame rates and a per-game on/off switch that applies within a
//! second. With no game running it says so. The layer and this app talk
//! through files in %LOCALAPPDATA%\ghostframe (see layer/src/session.rs).

#![cfg_attr(windows, windows_subsystem = "windows")]

#[cfg(not(windows))]
fn main() {
    eprintln!("the ghostframe tray app is Windows-only for now; on Linux use `ghostframe status`");
}

#[cfg(windows)]
fn main() {
    app::run();
}

#[cfg(windows)]
mod app {
    use std::collections::BTreeMap;
    use std::os::windows::process::CommandExt;
    use std::path::PathBuf;
    use std::time::{Duration, Instant, SystemTime};
    use tray_icon::menu::{CheckMenuItem, Menu, MenuEvent, MenuId, MenuItem, PredefinedMenuItem};
    use tray_icon::{Icon, TrayIcon, TrayIconBuilder};
    use winit::application::ApplicationHandler;
    use winit::event::{StartCause, WindowEvent};
    use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
    use winit::window::WindowId;

    const NO_WINDOW: u32 = 0x0800_0000;
    const RUN_KEY: &str = r"HKCU\Software\Microsoft\Windows\CurrentVersion\Run";

    struct Text {
        no_game: &'static str,
        hint: &'static str,
        generation: &'static str,
        off_now: &'static str,
        autostart: &'static str,
        quit: &'static str,
        tip_idle: &'static str,
    }

    const RU: Text = Text {
        no_game: "Игра не запущена",
        hint: "Запусти Vulkan-игру, и она появится здесь",
        generation: "Генерация кадров",
        off_now: "выключено",
        autostart: "Запускать вместе с Windows",
        quit: "Выход",
        tip_idle: "игра не запущена",
    };
    const EN: Text = Text {
        no_game: "No game running",
        hint: "Start a Vulkan game and it shows up here",
        generation: "Frame generation",
        off_now: "off",
        autostart: "Start with Windows",
        quit: "Quit",
        tip_idle: "no game running",
    };

    #[derive(Clone, PartialEq)]
    struct Game {
        pid: u64,
        exe: String,
        width: u64,
        height: u64,
        real: f64,
        shown: f64,
        enabled: bool,
    }

    fn data_dir() -> PathBuf {
        PathBuf::from(std::env::var_os("LOCALAPPDATA").unwrap_or_default()).join("ghostframe")
    }

    /// Games whose layer wrote a status in the last few seconds.
    fn running_games() -> Vec<Game> {
        let dir = data_dir().join("sessions");
        let mut games = Vec::new();
        let Ok(entries) = std::fs::read_dir(&dir) else { return games };
        for e in entries.flatten() {
            let p = e.path();
            if p.extension().is_none_or(|x| x != "json") {
                continue;
            }
            let age = e.metadata().and_then(|m| m.modified()).ok().and_then(|t| SystemTime::now().duration_since(t).ok());
            match age {
                Some(a) if a > Duration::from_secs(60) => {
                    let _ = std::fs::remove_file(&p); // the game crashed or was killed
                    continue;
                }
                Some(a) if a > Duration::from_secs(3) => continue,
                _ => {}
            }
            let Ok(text) = std::fs::read_to_string(&p) else { continue };
            let Ok(v) = serde_json::from_str::<serde_json::Value>(&text) else { continue };
            games.push(Game {
                pid: v["pid"].as_u64().unwrap_or(0),
                exe: v["exe"].as_str().unwrap_or("?").to_string(),
                width: v["width"].as_u64().unwrap_or(0),
                height: v["height"].as_u64().unwrap_or(0),
                real: v["real_fps"].as_f64().unwrap_or(0.0),
                shown: v["shown_fps"].as_f64().unwrap_or(0.0),
                enabled: v["enabled"].as_bool().unwrap_or(true),
            });
        }
        games.sort_by(|a, b| a.exe.cmp(&b.exe).then(a.pid.cmp(&b.pid)));
        games
    }

    /// Store `exe=on|off` in games.cfg, which the layer re-reads every second.
    fn set_enabled(exe: &str, on: bool) {
        let path = data_dir().join("games.cfg");
        let old = std::fs::read_to_string(&path).unwrap_or_default();
        let mut lines: Vec<String> = old
            .lines()
            .filter(|l| l.rsplit_once('=').is_none_or(|(n, _)| !n.trim().eq_ignore_ascii_case(exe)))
            .map(str::to_string)
            .collect();
        if lines.is_empty() {
            lines.push("# ghostframe per-game settings (written by the tray app)".into());
        }
        lines.push(format!("{exe}={}", if on { "on" } else { "off" }));
        let _ = std::fs::create_dir_all(data_dir());
        let _ = std::fs::write(&path, lines.join("\n") + "\n");
    }

    fn reg(args: &[&str]) -> bool {
        std::process::Command::new("reg").args(args).creation_flags(NO_WINDOW).output().is_ok_and(|o| o.status.success())
    }

    fn autostart_on() -> bool {
        reg(&["query", RUN_KEY, "/v", "ghostframe"])
    }

    pub fn set_autostart(on: bool) {
        if on {
            let exe = std::env::current_exe().map(|p| p.display().to_string()).unwrap_or_default();
            reg(&["add", RUN_KEY, "/v", "ghostframe", "/t", "REG_SZ", "/d", &format!("\"{exe}\""), "/f"]);
        } else {
            reg(&["delete", RUN_KEY, "/v", "ghostframe", "/f"]);
        }
    }

    /// The logo mark: a ghost with its in-between frame trailing behind it
    /// (assets/tray-icon.png, generated with the other logo files).
    fn icon() -> Icon {
        let png = image::load_from_memory(include_bytes!("../../assets/tray-icon.png")).expect("valid icon").into_rgba8();
        let (w, h) = png.dimensions();
        Icon::from_rgba(png.into_raw(), w, h).expect("valid icon")
    }

    struct Tray {
        t: &'static Text,
        tray: Option<TrayIcon>,
        games: Vec<Game>,
        shape: Vec<(u64, String, bool)>,
        fps_items: BTreeMap<u64, MenuItem>,
        toggles: Vec<(MenuId, String)>,
        autostart_id: Option<MenuId>,
        quit_id: Option<MenuId>,
        next_refresh: Instant,
    }

    impl Tray {
        fn label(g: &Game, t: &Text) -> String {
            if g.enabled {
                format!("\u{25b6}  {}   {:.0} \u{2192} {:.0} FPS   {}\u{00d7}{}", g.exe, g.real, g.shown, g.width, g.height)
            } else {
                format!("\u{25b6}  {}   {:.0} FPS ({})   {}\u{00d7}{}", g.exe, g.real, t.off_now, g.width, g.height)
            }
        }

        fn tooltip(&self) -> String {
            match self.games.as_slice() {
                [] => format!("ghostframe \u{00b7} {}", self.t.tip_idle),
                [g] if g.enabled => format!("ghostframe \u{00b7} {} {:.0} \u{2192} {:.0} FPS", g.exe, g.real, g.shown),
                [g] => format!("ghostframe \u{00b7} {} ({})", g.exe, self.t.off_now),
                gs => format!("ghostframe \u{00b7} {} games", gs.len()),
            }
        }

        /// Rebuild the menu when the set of games changes; otherwise only
        /// update the frame-rate labels in place.
        fn refresh(&mut self) {
            self.games = running_games();
            let shape: Vec<(u64, String, bool)> = self.games.iter().map(|g| (g.pid, g.exe.clone(), g.enabled)).collect();
            if shape != self.shape || self.tray.is_none() {
                self.shape = shape;
                self.rebuild();
            } else {
                for g in &self.games {
                    if let Some(item) = self.fps_items.get(&g.pid) {
                        item.set_text(Self::label(g, self.t));
                    }
                }
            }
            if let Some(tray) = &self.tray {
                let _ = tray.set_tooltip(Some(self.tooltip()));
            }
        }

        fn rebuild(&mut self) {
            let t = self.t;
            let menu = Menu::new();
            let _ = menu.append(&MenuItem::new("ghostframe", false, None));
            let _ = menu.append(&PredefinedMenuItem::separator());
            self.fps_items.clear();
            self.toggles.clear();
            if self.games.is_empty() {
                let _ = menu.append(&MenuItem::new(t.no_game, false, None));
                let _ = menu.append(&MenuItem::new(t.hint, false, None));
            }
            for (i, g) in self.games.iter().enumerate() {
                if i > 0 {
                    let _ = menu.append(&PredefinedMenuItem::separator());
                }
                let info = MenuItem::new(Self::label(g, t), false, None);
                let toggle = CheckMenuItem::new(format!("      {}", t.generation), true, g.enabled, None);
                let _ = menu.append(&info);
                let _ = menu.append(&toggle);
                self.toggles.push((toggle.id().clone(), g.exe.clone()));
                self.fps_items.insert(g.pid, info);
            }
            let _ = menu.append(&PredefinedMenuItem::separator());
            let autostart = CheckMenuItem::new(t.autostart, true, autostart_on(), None);
            let quit = MenuItem::new(t.quit, true, None);
            let _ = menu.append(&autostart);
            let _ = menu.append(&quit);
            self.autostart_id = Some(autostart.id().clone());
            self.quit_id = Some(quit.id().clone());

            match &self.tray {
                Some(tray) => tray.set_menu(Some(Box::new(menu))),
                None => {
                    self.tray = TrayIconBuilder::new()
                        .with_icon(icon())
                        .with_tooltip(self.tooltip())
                        .with_menu(Box::new(menu))
                        .with_menu_on_left_click(true)
                        .build()
                        .ok();
                }
            }
        }

        fn handle_menu(&mut self, el: &ActiveEventLoop) {
            while let Ok(ev) = MenuEvent::receiver().try_recv() {
                if Some(&ev.id) == self.quit_id.as_ref() {
                    el.exit();
                    return;
                }
                if Some(&ev.id) == self.autostart_id.as_ref() {
                    set_autostart(!autostart_on());
                    self.shape.clear(); // rebuild to show the new state
                    continue;
                }
                if let Some((_, exe)) = self.toggles.iter().find(|(id, _)| *id == ev.id).cloned() {
                    let now = self.games.iter().find(|g| g.exe == exe).is_none_or(|g| g.enabled);
                    set_enabled(&exe, !now);
                    // The layer confirms through its next status file.
                    self.next_refresh = Instant::now() + Duration::from_millis(1200);
                }
            }
        }
    }

    impl ApplicationHandler for Tray {
        fn new_events(&mut self, _el: &ActiveEventLoop, cause: StartCause) {
            if cause == StartCause::Init {
                self.refresh();
            }
        }

        fn resumed(&mut self, _el: &ActiveEventLoop) {}

        fn window_event(&mut self, _el: &ActiveEventLoop, _id: WindowId, _e: WindowEvent) {}

        fn about_to_wait(&mut self, el: &ActiveEventLoop) {
            self.handle_menu(el);
            if Instant::now() >= self.next_refresh {
                self.refresh();
                self.next_refresh = Instant::now() + Duration::from_secs(1);
            }
            el.set_control_flow(ControlFlow::WaitUntil(Instant::now() + Duration::from_millis(150)));
        }
    }

    fn already_running() -> bool {
        use windows_sys::Win32::Foundation::{GetLastError, ERROR_ALREADY_EXISTS};
        use windows_sys::Win32::System::Threading::CreateMutexW;
        let name: Vec<u16> = "Local\\ghostframe-tray".encode_utf16().chain(Some(0)).collect();
        unsafe {
            let h = CreateMutexW(std::ptr::null(), 0, name.as_ptr());
            !h.is_null() && GetLastError() == ERROR_ALREADY_EXISTS
        }
    }

    pub fn run() {
        if already_running() {
            return;
        }
        let ru = sys_locale::get_locale().is_some_and(|l| l.to_lowercase().starts_with("ru"));
        let mut app = Tray {
            t: if ru { &RU } else { &EN },
            tray: None,
            games: Vec::new(),
            shape: vec![(u64::MAX, String::new(), false)],
            fps_items: BTreeMap::new(),
            toggles: Vec::new(),
            autostart_id: None,
            quit_id: None,
            next_refresh: Instant::now(),
        };
        if let Ok(el) = EventLoop::new() {
            let _ = el.run_app(&mut app);
        }
    }
}
