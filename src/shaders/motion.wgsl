// Bilateral (symmetric) block motion estimation on the grid of the frame we
// want to create. For a block centred at p we look for the displacement v
// such that frame A at p - v/2 matches frame B at p + v/2. Searching on the
// in-between grid means every output block gets exactly one vector: no
// holes and no overlaps, unlike forward-projecting a flow field.
//
// Coarse-to-fine over a luma pyramid. Per level:
//   mode 0  (coarsest) exhaustive search in a +-RANGE window
//   mode 1  seed from the coarser level, then local refinement
//   mode 2  candidate propagation from the 8 neighbours (3D recursive search)

struct Params {
    lsize: vec2i,     // luma size at this level
    grid: vec2i,      // block grid size at this level
    cgrid: vec2i,     // block grid size one level coarser
    mode: u32,
    fine: u32,        // 1 on the full-resolution level: also try half-pixel steps
    lambda: f32,      // smoothness weight
    range: i32,       // exhaustive search radius (mode 0)
    sparse: u32,      // 1 = checkerboard-subsample the matching window
}

@group(0) @binding(0) var la: texture_2d<f32>;
@group(0) @binding(1) var lb: texture_2d<f32>;
@group(0) @binding(2) var prev: texture_2d<f32>;    // this level, previous iteration
@group(0) @binding(3) var coarse: texture_2d<f32>;  // one level coarser
@group(0) @binding(4) var dst: texture_storage_2d<rg32float, write>;
@group(0) @binding(5) var<uniform> P: Params;
@group(0) @binding(6) var lsmp: sampler;

const BS: i32 = 8;       // block size in pixels at every level
const HALF: f32 = 4.0;

//LUM_BEGIN
fn lum(t: texture_2d<f32>, p: vec2f) -> f32 {
    let q = clamp(p - 0.5, vec2f(0.0), vec2f(P.lsize - 1));
    let i = vec2i(floor(q));
    let f = q - floor(q);
    let j = min(i + 1, P.lsize - 1);
    let a = textureLoad(t, i, 0).r;
    let b = textureLoad(t, vec2i(j.x, i.y), 0).r;
    let c = textureLoad(t, vec2i(i.x, j.y), 0).r;
    let d = textureLoad(t, j, 0).r;
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
//LUM_END

// Mean absolute luma difference over a 10x10 window around the block
// (1 px of context on each side makes matching robust on flat blocks).
// On the large levels every other pixel is skipped in a checkerboard:
// the vector there is only being refined, so half the samples are plenty.
fn sad(center: vec2f, v: vec2f) -> f32 {
    var s = 0.0;
    var n = 0.0;
    let h = 0.5 * v;
    for (var y = -5; y < 5; y++) {
        for (var x = -5; x < 5; x++) {
            if (P.sparse == 1u && ((x + y) & 1) != 0) { continue; }
            let q = center + vec2f(f32(x) + 0.5, f32(y) + 0.5);
            s += abs(lum(la, q - h) - lum(lb, q + h));
            n += 1.0;
        }
    }
    return s / n;
}

fn fetch(t: texture_2d<f32>, g: vec2i, size: vec2i) -> vec2f {
    return textureLoad(t, clamp(g, vec2i(0), size - 1), 0).rg;
}

var<private> best_v: vec2f;
var<private> best_c: f32;
var<private> center: vec2f;
var<private> anchor: vec2f;

fn total_cost(v: vec2f) -> f32 {
    // Smoothness: stay close to what the neighbourhood / coarser level says.
    // Tiny zero bias: static content should not wander.
    var c = sad(center, v) + P.lambda * length(v - anchor);
    if (all(v == vec2f(0.0))) { c *= 0.97; }
    return c;
}

fn try_vec(v: vec2f) {
    // Neighbouring blocks usually share a vector: don't pay for it twice.
    if (best_c < 1e8 && all(abs(v - best_v) < vec2f(1e-3))) { return; }
    let c = total_cost(v);
    if (c < best_c) {
        best_c = c;
        best_v = v;
    }
}

fn refine(step: f32) {
    let base = best_v;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            if (x != 0 || y != 0) {
                try_vec(base + vec2f(f32(x), f32(y)) * step);
            }
        }
    }
}

@compute @workgroup_size(8, 8)
fn search(@builtin(global_invocation_id) id: vec3u) {
    let g = vec2i(id.xy);
    if (g.x >= P.grid.x || g.y >= P.grid.y) { return; }
    center = vec2f(g * BS) + HALF;
    best_c = 1e9;
    best_v = vec2f(0.0);

    if (P.mode == 0u) {
        anchor = vec2f(0.0);
        for (var y = -P.range; y <= P.range; y++) {
            for (var x = -P.range; x <= P.range; x++) {
                try_vec(vec2f(f32(x), f32(y)));
            }
        }
        refine(0.5);
    } else if (P.mode == 1u) {
        // Seed from the coarser level: the covering block and its neighbours,
        // scaled to this level's pixels.
        let cg = vec2i(floor((center * 0.5) / f32(BS)));
        let up = fetch(coarse, cg, P.cgrid) * 2.0;
        anchor = up;
        try_vec(up);
        try_vec(vec2f(0.0));
        try_vec(fetch(coarse, cg + vec2i(1, 0), P.cgrid) * 2.0);
        try_vec(fetch(coarse, cg + vec2i(-1, 0), P.cgrid) * 2.0);
        try_vec(fetch(coarse, cg + vec2i(0, 1), P.cgrid) * 2.0);
        try_vec(fetch(coarse, cg + vec2i(0, -1), P.cgrid) * 2.0);
        refine(1.0);
        refine(1.0);
        if (P.fine == 1u) { refine(0.5); refine(0.25); }
    } else {
        // Propagation: own vector, 8 neighbours, coarse seed.
        let own = fetch(prev, g, P.grid);
        var n: array<vec2f, 8>;
        var k = 0;
        for (var y = -1; y <= 1; y++) {
            for (var x = -1; x <= 1; x++) {
                if (x != 0 || y != 0) {
                    n[k] = fetch(prev, g + vec2i(x, y), P.grid);
                    k++;
                }
            }
        }
        // Anchor to the component-wise median of the neighbourhood.
        var xs = array<f32, 9>(own.x, n[0].x, n[1].x, n[2].x, n[3].x, n[4].x, n[5].x, n[6].x, n[7].x);
        var ys = array<f32, 9>(own.y, n[0].y, n[1].y, n[2].y, n[3].y, n[4].y, n[5].y, n[6].y, n[7].y);
        for (var i = 0; i < 9; i++) {
            for (var j = i + 1; j < 9; j++) {
                if (xs[j] < xs[i]) { let t = xs[i]; xs[i] = xs[j]; xs[j] = t; }
                if (ys[j] < ys[i]) { let t = ys[i]; ys[i] = ys[j]; ys[j] = t; }
            }
        }
        anchor = vec2f(xs[4], ys[4]);
        try_vec(own);
        for (var i = 0; i < 8; i++) { try_vec(n[i]); }
        try_vec(anchor);
        refine(1.0);
        if (P.fine == 1u) { refine(0.5); refine(0.25); }
    }
    textureStore(dst, g, vec4f(best_v, best_c, 0.0));
}

// 3x3 vector median: the neighbour vector with the smallest summed distance
// to all others. Kills isolated wrong vectors without blurring edges.
@compute @workgroup_size(8, 8)
fn median(@builtin(global_invocation_id) id: vec3u) {
    let g = vec2i(id.xy);
    if (g.x >= P.grid.x || g.y >= P.grid.y) { return; }
    var v: array<vec2f, 9>;
    var k = 0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            v[k] = fetch(prev, g + vec2i(x, y), P.grid);
            k++;
        }
    }
    var best = v[4];
    var bd = 1e9;
    for (var i = 0; i < 9; i++) {
        var d = 0.0;
        for (var j = 0; j < 9; j++) { d += length(v[i] - v[j]); }
        if (d < bd) { bd = d; best = v[i]; }
    }
    textureStore(dst, g, vec4f(best, 0.0, 0.0));
}
