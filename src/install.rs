//! One-command install of the Vulkan layer.
//!
//! The layer is registered as an *implicit* layer, so the Vulkan loader puts
//! it into every Vulkan game without any launch options. It stays passive
//! (frames pass through untouched) until frame generation is switched on for
//! that game in the tray app, or the game is started with
//! `ENABLE_GHOSTFRAME=1` (`ghostframe run`). `DISABLE_GHOSTFRAME=1` keeps it
//! out of a program entirely.
//! Windows: a JSON manifest + HKCU\Software\Khronos\Vulkan\ImplicitLayers
//! Linux:   ~/.local/share/vulkan/implicit_layer.d/ghostframe_layer.json

use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};

pub const LAYER_NAME: &str = "VK_LAYER_GHOSTFRAME_framegen";
pub const ENABLE_VAR: &str = "ENABLE_GHOSTFRAME";

fn lib_file() -> &'static str {
    if cfg!(windows) {
        "ghostframe_layer.dll"
    } else if cfg!(target_os = "macos") {
        "libghostframe_layer.dylib"
    } else {
        "libghostframe_layer.so"
    }
}

/// The layer library built next to this executable.
fn built_layer() -> Result<PathBuf> {
    let exe = std::env::current_exe()?;
    let dir = exe.parent().context("no executable directory")?;
    let p = dir.join(lib_file());
    if !p.exists() {
        bail!(
            "{} not found next to {} - build it first: cargo build --release --workspace",
            lib_file(),
            exe.display()
        );
    }
    Ok(p)
}

fn data_dir() -> Result<PathBuf> {
    if cfg!(windows) {
        let base = std::env::var_os("LOCALAPPDATA").context("LOCALAPPDATA is not set")?;
        Ok(PathBuf::from(base).join("ghostframe"))
    } else {
        let base = std::env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")))
            .context("neither XDG_DATA_HOME nor HOME is set")?;
        Ok(base.join("ghostframe"))
    }
}

/// Where the manifest goes so the Vulkan loader finds it.
fn manifest_path() -> Result<PathBuf> {
    if cfg!(windows) {
        Ok(data_dir()?.join("ghostframe_layer.json"))
    } else {
        let base = std::env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/share")))
            .context("neither XDG_DATA_HOME nor HOME is set")?;
        Ok(base.join("vulkan/implicit_layer.d/ghostframe_layer.json"))
    }
}

pub fn manifest(lib: &Path) -> String {
    let lib = lib.display().to_string().replace('\\', "\\\\");
    format!(
        r#"{{
  "file_format_version": "1.2.0",
  "layer": {{
    "name": "{LAYER_NAME}",
    "type": "GLOBAL",
    "library_path": "{lib}",
    "api_version": "1.3.0",
    "implementation_version": "1",
    "description": "ghostframe: open-source frame generation",
    "functions": {{
      "vkNegotiateLoaderLayerInterfaceVersion": "vkNegotiateLoaderLayerInterfaceVersion"
    }},
    "disable_environment": {{ "DISABLE_GHOSTFRAME": "1" }}
  }}
}}
"#
    )
}

const REG_KEY: &str = r"HKCU\Software\Khronos\Vulkan\ImplicitLayers";

fn reg(args: &[&str]) -> Result<bool> {
    let out = std::process::Command::new("reg").args(args).output().context("could not run reg.exe")?;
    Ok(out.status.success())
}

pub fn install() -> Result<()> {
    let src = built_layer()?;
    let dir = data_dir()?;
    std::fs::create_dir_all(&dir).with_context(|| format!("cannot create {}", dir.display()))?;
    let lib = dir.join(lib_file());
    std::fs::copy(&src, &lib).with_context(|| format!("cannot copy the layer to {} (is a game using it right now?)", lib.display()))?;
    let man = manifest_path()?;
    if let Some(p) = man.parent() {
        std::fs::create_dir_all(p)?;
    }
    std::fs::write(&man, manifest(&lib))?;
    if cfg!(windows) && !reg(&["add", REG_KEY, "/v", &man.display().to_string(), "/t", "REG_DWORD", "/d", "0", "/f"])? {
        bail!("could not register the layer in {REG_KEY}");
    }
    println!("ghostframe layer installed");
    println!("  library   {}", lib.display());
    println!("  manifest  {}", man.display());
    println!();
    println!("Just start a Vulkan game. Frame generation stays off until you switch it on:");
    if cfg!(windows) {
        println!("  click the ghostframe icon in the notification area -> the game -> Frame generation");
    } else {
        println!("  Steam launch options:  {ENABLE_VAR}=1 %command%");
    }
    println!("  or start it with:      ghostframe run <game executable> [args]");
    println!("Remove it again with: ghostframe uninstall");
    if cfg!(windows) {
        match start_tray(true) {
            Ok(()) => println!("
The ghostframe icon is now in the notification area: click it to see running games and switch frame generation per game."),
            Err(e) => println!("
(tray app not started: {e})"),
        }
    }
    Ok(())
}

const RUN_KEY: &str = r"HKCU\Software\Microsoft\Windows\CurrentVersion\Run";

fn tray_exe() -> Result<PathBuf> {
    let exe = std::env::current_exe()?;
    let p = exe.parent().context("no executable directory")?.join("ghostframe-tray.exe");
    if !p.exists() {
        bail!("{} not found - build it with: cargo build --release --workspace", p.display());
    }
    Ok(p)
}

/// Start the notification-area app (it exits at once if it is already
/// running). With `autostart`, it also starts with Windows from now on.
pub fn start_tray(autostart: bool) -> Result<()> {
    if !cfg!(windows) {
        bail!("the tray app is Windows-only for now; use `ghostframe status`");
    }
    let exe = tray_exe()?;
    if autostart {
        let _ = reg(&["add", RUN_KEY, "/v", "ghostframe", "/t", "REG_SZ", "/d", &format!("\"{}\"", exe.display()), "/f"]);
    }
    use std::process::Stdio;
    // Our console handles are inheritable; a child that outlives us would
    // keep a pipe like `ghostframe install | more` open forever.
    #[cfg(windows)]
    unsafe {
        use windows_sys::Win32::Foundation::{SetHandleInformation, HANDLE_FLAG_INHERIT};
        use windows_sys::Win32::System::Console::{GetStdHandle, STD_ERROR_HANDLE, STD_INPUT_HANDLE, STD_OUTPUT_HANDLE};
        for h in [STD_INPUT_HANDLE, STD_OUTPUT_HANDLE, STD_ERROR_HANDLE] {
            SetHandleInformation(GetStdHandle(h), HANDLE_FLAG_INHERIT, 0);
        }
    }
    std::process::Command::new(&exe)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn().with_context(|| format!("could not start {}", exe.display()))?;
    Ok(())
}

/// Status files the layer writes from inside running games.
fn running_sessions() -> Vec<String> {
    let Ok(dir) = data_dir() else { return vec![] };
    let Ok(entries) = std::fs::read_dir(dir.join("sessions")) else { return vec![] };
    let fresh = |e: &std::fs::DirEntry| {
        e.metadata().and_then(|m| m.modified()).ok().and_then(|t| t.elapsed().ok()).is_some_and(|a| a.as_secs() < 3)
    };
    entries
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == "json") && fresh(e))
        .filter_map(|e| std::fs::read_to_string(e.path()).ok())
        .collect()
}

pub fn uninstall() -> Result<()> {
    let man = manifest_path()?;
    if cfg!(windows) {
        let _ = reg(&["delete", REG_KEY, "/v", &man.display().to_string(), "/f"]);
    }
    let _ = std::fs::remove_file(&man);
    let lib = data_dir()?.join(lib_file());
    let _ = std::fs::remove_file(&lib);
    if cfg!(windows) {
        let _ = reg(&["delete", RUN_KEY, "/v", "ghostframe", "/f"]);
        let _ = std::process::Command::new("taskkill").args(["/IM", "ghostframe-tray.exe", "/F"]).output();
    }
    println!("ghostframe layer removed");
    Ok(())
}

pub fn status() -> Result<()> {
    let man = manifest_path()?;
    let lib = data_dir()?.join(lib_file());
    let registered = if cfg!(windows) {
        reg(&["query", REG_KEY, "/v", &man.display().to_string()])?
    } else {
        man.exists()
    };
    println!("layer      {LAYER_NAME}");
    println!("library    {} {}", lib.display(), if lib.exists() { "(present)" } else { "(missing)" });
    println!("manifest   {} {}", man.display(), if man.exists() { "(present)" } else { "(missing)" });
    println!("registered {}", if registered { "yes" } else { "no" });
    println!("loaded     into every Vulkan game; generation per game from the tray, or {ENABLE_VAR}=1");
    let games = running_sessions();
    if games.is_empty() {
        println!("running    no game is using ghostframe right now");
    }
    for g in games {
        println!("running    {g}");
    }
    Ok(())
}

/// Launch a program with frame generation enabled.
pub fn run(cmd: &[String]) -> Result<()> {
    let (prog, args) = cmd.split_first().context("usage: ghostframe run <program> [args]")?;
    let status = std::process::Command::new(prog)
        .args(args)
        .env(ENABLE_VAR, "1")
        .status()
        .with_context(|| format!("could not start {prog}"))?;
    if !status.success() {
        std::process::exit(status.code().unwrap_or(1));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn manifest_is_valid_json_shape() {
        let m = super::manifest(std::path::Path::new(r"C:\x\ghostframe_layer.dll"));
        assert!(m.contains(r#""library_path": "C:\\x\\ghostframe_layer.dll""#));
        assert!(!m.contains("enable_environment"));
        assert!(m.contains("disable_environment"));
    }
}
