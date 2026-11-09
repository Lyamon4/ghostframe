//! Compile the engine's WGSL shaders (shared with the wgpu app) to SPIR-V,
//! so the layer runs exactly the same algorithm on the game's own device.

use std::path::PathBuf;

fn with_lum(src: &str, hw: bool, size: &str) -> String {
    if !hw {
        return src.to_string();
    }
    let (a, rest) = src.split_once("//LUM_BEGIN").expect("LUM_BEGIN marker");
    let (_, b) = rest.split_once("//LUM_END").expect("LUM_END marker");
    format!("{a}fn lum(t: texture_2d<f32>, p: vec2f) -> f32 {{\n    return textureSampleLevel(t, lsmp, p / vec2f({size}), 0.0).r;\n}}{b}")
}

fn compile(name: &str, src: &str, out: &PathBuf) {
    let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(src)));
    let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::default())
        .validate(&module)
        .unwrap_or_else(|e| panic!("{name}: {e:?}"));
    let mut opts = naga::back::spv::Options::default();
    opts.flags.remove(naga::back::spv::WriterFlags::DEBUG);
    let words = naga::back::spv::write_vec(&module, &info, &opts, None).unwrap_or_else(|e| panic!("{name}: {e}"));
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    std::fs::write(out.join(format!("{name}.spv")), bytes).unwrap();
}

fn main() {
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("../src/shaders");
    for f in ["pyramid.wgsl", "motion.wgsl", "compose.wgsl"] {
        println!("cargo:rerun-if-changed={}", dir.join(f).display());
    }
    let read = |f: &str| std::fs::read_to_string(dir.join(f)).unwrap();
    compile("pyramid", &read("pyramid.wgsl"), &out);
    let motion = read("motion.wgsl");
    compile("motion", &with_lum(&motion, false, "P.lsize"), &out);
    compile("motion_hw", &with_lum(&motion, true, "P.lsize"), &out);
    let compose = read("compose.wgsl");
    compile("compose", &with_lum(&compose, false, "P.size"), &out);
    compile("compose_hw", &with_lum(&compose, true, "P.size"), &out);
}
