// Windows: give ghostframe.exe and ghostframe-tray.exe the logo as their icon.
fn main() {
    println!("cargo:rerun-if-changed=assets/ghostframe.rc");
    println!("cargo:rerun-if-changed=assets/ghostframe.ico");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        embed_resource::compile("assets/ghostframe.rc", embed_resource::NONE).manifest_optional().ok();
    }
}
