//! One-command install of the Vulkan layer.
//!
//! The layer is registered as an *implicit* layer that stays inactive unless
//! `ENABLE_GHOSTFRAME=1` is set, so nothing changes for other programs:
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
    "enable_environment": {{ "{ENABLE_VAR}": "1" }},
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
    println!("It stays off until you ask for it, per game:");
    println!("  ghostframe run <game executable> [args]");
    println!("  Steam launch options:  {ENABLE_VAR}=1 %command%");
    println!("  anything else:         set {ENABLE_VAR}=1 in the game's environment");
    println!("Remove it again with: ghostframe uninstall");
    Ok(())
}

pub fn uninstall() -> Result<()> {
    let man = manifest_path()?;
    if cfg!(windows) {
        let _ = reg(&["delete", REG_KEY, "/v", &man.display().to_string(), "/f"]);
    }
    let _ = std::fs::remove_file(&man);
    let lib = data_dir()?.join(lib_file());
    let _ = std::fs::remove_file(&lib);
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
    println!("enabled    only for programs started with {ENABLE_VAR}=1 (now: {})", std::env::var(ENABLE_VAR).unwrap_or_else(|_| "unset".into()));
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
        assert!(m.contains(r#""enable_environment": { "ENABLE_GHOSTFRAME": "1" }"#));
        assert!(m.contains("disable_environment"));
    }
}
