// Luma extraction and 2x2 box downsampling for the motion-search pyramid.

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<r32float, write>;

// Rec. 709 luma of the (gamma-encoded) colour: perceptual enough for matching.
@compute @workgroup_size(16, 16)
fn luma(@builtin(global_invocation_id) id: vec3u) {
    let size = textureDimensions(dst);
    if (id.x >= size.x || id.y >= size.y) { return; }
    let c = textureLoad(src, vec2i(id.xy), 0).rgb;
    textureStore(dst, vec2i(id.xy), vec4f(dot(c, vec3f(0.2126, 0.7152, 0.0722)), 0.0, 0.0, 0.0));
}

@compute @workgroup_size(16, 16)
fn down(@builtin(global_invocation_id) id: vec3u) {
    let size = textureDimensions(dst);
    if (id.x >= size.x || id.y >= size.y) { return; }
    let s = vec2i(textureDimensions(src)) - 1;
    let p = vec2i(id.xy) * 2;
    let v = textureLoad(src, min(p, s), 0).r
          + textureLoad(src, min(p + vec2i(1, 0), s), 0).r
          + textureLoad(src, min(p + vec2i(0, 1), s), 0).r
          + textureLoad(src, min(p + vec2i(1, 1), s), 0).r;
    textureStore(dst, vec2i(id.xy), vec4f(v * 0.25, 0.0, 0.0, 0.0));
}
