// Build the in-between frame. Per pixel, the block vectors around it are
// candidates; the one whose two motion-compensated samples agree best on a
// small luma patch wins (pixel-accurate object edges), then the two colours
// are blended by time.

struct Params {
    size: vec2i,
    grid: vec2i,
    t: f32,
    block: f32,   // block size of the motion field in output pixels
    vscale: f32,  // field vectors are in pixels of a smaller level
    _pad: f32,
}

@group(0) @binding(0) var ca: texture_2d<f32>;
@group(0) @binding(1) var cb: texture_2d<f32>;
@group(0) @binding(2) var la: texture_2d<f32>;
@group(0) @binding(3) var lb: texture_2d<f32>;
@group(0) @binding(4) var field: texture_2d<f32>;
@group(0) @binding(5) var smp: sampler;
@group(0) @binding(6) var dst: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(7) var<uniform> P: Params;
@group(0) @binding(8) var lsmp: sampler;


fn vec_at(g: vec2i) -> vec2f {
    return textureLoad(field, clamp(g, vec2i(0), P.grid - 1), 0).rg * P.vscale;
}

//LUM_BEGIN
fn lum(t: texture_2d<f32>, p: vec2f) -> f32 {
    let q = clamp(p - 0.5, vec2f(0.0), vec2f(P.size - 1));
    let i = vec2i(floor(q));
    let f = q - floor(q);
    let j = min(i + 1, P.size - 1);
    let a = textureLoad(t, i, 0).r;
    let b = textureLoad(t, vec2i(j.x, i.y), 0).r;
    let c = textureLoad(t, vec2i(i.x, j.y), 0).r;
    let d = textureLoad(t, j, 0).r;
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
//LUM_END

fn mismatch(p: vec2f, v: vec2f) -> f32 {
    let da = -P.t * v;
    let db = (1.0 - P.t) * v;
    // 5-point cross instead of a full 3x3 patch.
    var e = abs(lum(la, p + da) - lum(lb, p + db));
    e += abs(lum(la, p + da + vec2f(1.0, 0.0)) - lum(lb, p + db + vec2f(1.0, 0.0)));
    e += abs(lum(la, p + da - vec2f(1.0, 0.0)) - lum(lb, p + db - vec2f(1.0, 0.0)));
    e += abs(lum(la, p + da + vec2f(0.0, 1.0)) - lum(lb, p + db + vec2f(0.0, 1.0)));
    e += abs(lum(la, p + da - vec2f(0.0, 1.0)) - lum(lb, p + db - vec2f(0.0, 1.0)));
    return e;
}

@compute @workgroup_size(16, 16)
fn compose(@builtin(global_invocation_id) id: vec3u) {
    if (i32(id.x) >= P.size.x || i32(id.y) >= P.size.y) { return; }
    let p = vec2f(id.xy) + 0.5;

    // Smooth field: bilinear between the four surrounding block centres.
    let gp = p / P.block - 0.5;
    let g0 = vec2i(floor(gp));
    let f = gp - floor(gp);
    let v00 = vec_at(g0);
    let v10 = vec_at(g0 + vec2i(1, 0));
    let v01 = vec_at(g0 + vec2i(0, 1));
    let v11 = vec_at(g0 + vec2i(1, 1));
    let vsmooth = mix(mix(v00, v10, f.x), mix(v01, v11, f.x), f.y);

    var v = vsmooth;
    // Only where the surrounding vectors disagree (object edges) is a
    // per-pixel choice worth its cost; elsewhere the smooth field is exact.
    let spread = max(max(length(v00 - v11), length(v10 - v01)), max(length(v00 - v10), length(v00 - v01)));
    if (spread > 0.75) {
        var e = mismatch(p, vsmooth) * 0.85;   // prefer the smooth field slightly
        let cands = array<vec2f, 4>(v00, v10, v01, v11);
        for (var i = 0; i < 4; i++) {
            let ei = mismatch(p, cands[i]);
            if (ei < e) { e = ei; v = cands[i]; }
        }
    }

    let inv = 1.0 / vec2f(P.size);
    let a = textureSampleLevel(ca, smp, (p - P.t * v) * inv, 0.0);
    let b = textureSampleLevel(cb, smp, (p + (1.0 - P.t) * v) * inv, 0.0);
    textureStore(dst, vec2i(id.xy), vec4f(mix(a.rgb, b.rgb, P.t), 1.0));
}
