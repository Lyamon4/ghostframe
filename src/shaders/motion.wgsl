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
@group(0) @binding(4) var dst: texture_storage_2d<rgba32float, write>;
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

// One workgroup of 64 threads per block, as 8 groups of 8: each round checks
// up to 8 candidate vectors at once, each group of 8 threads splitting that
// candidate's matching window. A refinement step (8 neighbours) is one round
// instead of 8 sequential ones. All threads see the same sums
// (workgroupUniformLoad) and make the same choices, so control flow stays
// uniform.

var<workgroup> wg_part: array<f32, 64>;
var<workgroup> wg_sums: array<f32, 8>;

var<private> lid: u32;
var<private> best_v: vec2f;
var<private> best_c: f32;
var<private> center: vec2f;
var<private> anchor: vec2f;

// Pixel k of the matching window: 10x10 around the block (1 px of context
// on each side makes matching robust on flat blocks), or a checkerboard half
// of it on the large levels, where the vector is only being refined.
fn window_px(k: u32) -> vec2f {
    let i = i32(k);
    if (P.sparse == 1u) {
        let yr = i / 5;
        let xr = 2 * (i % 5) + (yr & 1);
        return vec2f(f32(xr - 5) + 0.5, f32(yr - 5) + 0.5);
    }
    return vec2f(f32(i % 10 - 5) + 0.5, f32(i / 10 - 5) + 0.5);
}

fn fetch(t: texture_2d<f32>, g: vec2i, size: vec2i) -> vec2f {
    return textureLoad(t, clamp(g, vec2i(0), size - 1), 0).rg;
}

// Mean absolute luma difference of candidates c[0..m] (m <= 8), then keep
// the best by total cost.
fn eval_batch(c: array<vec2f, 8>, m: i32) {
    let slot = lid / 8u;
    let lane = lid % 8u;
    let npix = select(100u, 50u, P.sparse == 1u);
    var s = 0.0;
    if (i32(slot) < m) {
        let h = 0.5 * c[slot];
        for (var k = lane; k < npix; k += 8u) {
            let q = center + window_px(k);
            s += abs(lum(la, q - h) - lum(lb, q + h));
        }
    }
    wg_part[lid] = s;
    workgroupBarrier();
    if (lane == 0u) {
        var t = 0.0;
        for (var j = 0u; j < 8u; j++) { t += wg_part[lid + j]; }
        wg_sums[slot] = t;
    }
    let sums = workgroupUniformLoad(&wg_sums);
    let inv_n = 1.0 / f32(npix);
    for (var i = 0; i < m; i++) {
        let v = c[i];
        // Smoothness: stay close to what the neighbourhood / coarser level
        // says. Tiny zero bias: static content should not wander.
        var cost = sums[i] * inv_n + P.lambda * length(v - anchor);
        if (all(v == vec2f(0.0))) { cost *= 0.97; }
        if (cost < best_c) {
            best_c = cost;
            best_v = v;
        }
    }
}

fn refine(step: f32) {
    let b = best_v;
    var c: array<vec2f, 8>;
    c[0] = b + vec2f(-step, -step);
    c[1] = b + vec2f(0.0, -step);
    c[2] = b + vec2f(step, -step);
    c[3] = b + vec2f(-step, 0.0);
    c[4] = b + vec2f(step, 0.0);
    c[5] = b + vec2f(-step, step);
    c[6] = b + vec2f(0.0, step);
    c[7] = b + vec2f(step, step);
    eval_batch(c, 8);
}

// Seed vectors, each distinct one once (neighbouring blocks usually share a
// vector), in at most two rounds.
fn try_seeds(seeds: array<vec2f, 10>, n: i32) {
    var list: array<vec2f, 10>;
    var k = 0;
    for (var i = 0; i < n; i++) {
        var dup = false;
        for (var j = 0; j < k; j++) {
            if (all(abs(list[j] - seeds[i]) < vec2f(1e-3))) { dup = true; }
        }
        if (!dup) { list[k] = seeds[i]; k++; }
    }
    var a: array<vec2f, 8>;
    var b: array<vec2f, 8>;
    for (var i = 0; i < 8; i++) { a[i] = list[i]; }
    b[0] = list[8];
    b[1] = list[9];
    eval_batch(a, min(k, 8));
    if (n > 8) { eval_batch(b, k - 8); }
}

@compute @workgroup_size(64)
fn search(@builtin(workgroup_id) wid: vec3u, @builtin(local_invocation_index) li: u32) {
    let g = vec2i(wid.xy);
    lid = li;
    center = vec2f(g * BS) + HALF;
    best_c = 1e9;
    best_v = vec2f(0.0);
    var seeds: array<vec2f, 10>;

    if (P.mode == 0u) {
        // Exhaustive search in a +-range window, 8 vectors per round.
        anchor = vec2f(0.0);
        let side = 2 * P.range + 1;
        let total = side * side;
        for (var base = 0; base < total; base += 8) {
            var c: array<vec2f, 8>;
            for (var j = 0; j < 8; j++) {
                let i = base + j;
                c[j] = vec2f(f32(i % side - P.range), f32(i / side - P.range));
            }
            eval_batch(c, min(8, total - base));
        }
        refine(0.5);
    } else if (P.mode == 1u) {
        // Seed from the coarser level: the covering block and its neighbours,
        // scaled to this level's pixels.
        let cg = vec2i(floor((center * 0.5) / f32(BS)));
        let up = fetch(coarse, cg, P.cgrid) * 2.0;
        anchor = up;
        seeds[0] = up;
        seeds[1] = vec2f(0.0);
        seeds[2] = fetch(coarse, cg + vec2i(1, 0), P.cgrid) * 2.0;
        seeds[3] = fetch(coarse, cg + vec2i(-1, 0), P.cgrid) * 2.0;
        seeds[4] = fetch(coarse, cg + vec2i(0, 1), P.cgrid) * 2.0;
        seeds[5] = fetch(coarse, cg + vec2i(0, -1), P.cgrid) * 2.0;
        try_seeds(seeds, 6);
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
        for (var a = 0; a < 9; a++) {
            for (var b = a + 1; b < 9; b++) {
                if (xs[b] < xs[a]) { let t = xs[a]; xs[a] = xs[b]; xs[b] = t; }
                if (ys[b] < ys[a]) { let t = ys[a]; ys[a] = ys[b]; ys[b] = t; }
            }
        }
        anchor = vec2f(xs[4], ys[4]);
        seeds[0] = own;
        for (var a = 0; a < 8; a++) { seeds[a + 1] = n[a]; }
        seeds[9] = anchor;
        try_seeds(seeds, 10);
        refine(1.0);
        if (P.fine == 1u) { refine(0.5); refine(0.25); }
    }
    if (lid == 0u) {
        textureStore(dst, g, vec4f(best_v, best_c, 0.0));
    }
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
