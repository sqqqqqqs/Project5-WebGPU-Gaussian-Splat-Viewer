struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    //TODO: information passed from vertex shader to fragment shader
    @location(0) color: vec4<f32>,
    @location(1) d: vec2<f32>,  // splat center minus this pixel, in pixels
    @location(2) @interpolate(flat) conic_opacity: vec4<f32>,  // same for the whole quad
};

struct Splat {
    //TODO: information defined in preprocess compute shader
    pos: u32,  // NDC center (x, y), packed as 2 x f16
    size: u32,  // NDC quad width and height, packed as 2 x f16
    color: array<u32, 2>,  // RGBA, packed as 4 x f16
    conic_opacity: array<u32, 2>,  // inverse 2D covariance (xx, xy, yy) and opacity, packed as 4 x f16
};

@group(0) @binding(0)
var<storage, read> splats: array<Splat>;
@group(0) @binding(1)
var<storage, read> sort_indices: array<u32>;
@group(0) @binding(2)
var<uniform> camera: CameraUniforms;

@vertex
fn vs_main(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32,
) -> VertexOutput {
    //TODO: reconstruct 2D quad based on information from splat, pass 
    var out: VertexOutput;

    // splat center was already projected to NDC by the preprocess shader
    let splat = splats[sort_indices[instance_index]];  // i-th visible splat after sorting
    let center = unpack2x16float(splat.pos);

    // two triangles forming a quad, corners in [-1, 1]
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1., -1.), vec2<f32>(1., -1.), vec2<f32>(1., 1.),
        vec2<f32>(-1., -1.), vec2<f32>(1., 1.), vec2<f32>(-1., 1.),
    );
    let half_size = unpack2x16float(splat.size) * 0.5;
    let offset = corners[vertex_index] * half_size;

    out.position = vec4<f32>(center + offset, 0., 1.);  // already NDC, so w = 1
    out.color = vec4<f32>(unpack2x16float(splat.color[0]), unpack2x16float(splat.color[1]));

    // center - corner in pixels: NDC y points up but pixel y points down, so only x flips sign
    out.d = vec2<f32>(-offset.x, offset.y) * camera.viewport * 0.5;
    out.conic_opacity = vec4<f32>(unpack2x16float(splat.conic_opacity[0]), unpack2x16float(splat.conic_opacity[1]));
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // d is linear across the quad, so the interpolated value is exact for every pixel
    let d = in.d;
    let conic = in.conic_opacity.xyz;
    let opacity = in.conic_opacity.w;

    // gaussian exponent -1/2 * d^T * conic * d (conic is symmetric, so the xy term counts twice)
    let power = -0.5 * (conic.x * d.x * d.x + conic.z * d.y * d.y) - conic.y * d.x * d.y;
    if (power > 0.) {  // only possible from numerical error
        discard;
    }

    // opacity decays exponentially away from the center, capped below 1 as in the reference
    let alpha = min(0.99, opacity * exp(power));
    if (alpha < 1. / 255.) {  // invisible, skip blending work
        discard;
    }
    return vec4<f32>(in.color.rgb, alpha);
}