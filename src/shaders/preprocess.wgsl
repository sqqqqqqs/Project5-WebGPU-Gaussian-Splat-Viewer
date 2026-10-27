const SH_C0: f32 = 0.28209479177387814;
const SH_C1 = 0.4886025119029199;
const SH_C2 = array<f32,5>(
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
);
const SH_C3 = array<f32,7>(
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
);

override workgroupSize: u32;
override sortKeyPerThread: u32;
override frustumCulling: bool = true;  // switch for performance analysis

struct DispatchIndirect {
    dispatch_x: atomic<u32>,
    dispatch_y: u32,
    dispatch_z: u32,
}

struct SortInfos {
    keys_size: atomic<u32>,  // instance_count in DrawIndirect
    //data below is for info inside radix sort 
    padded_size: u32, 
    passes: u32,
    even_pass: u32,
    odd_pass: u32,
}

struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct RenderSettings {
    gaussian_scaling: f32,
    sh_deg: f32,
}

struct Gaussian {
    pos_opacity: array<u32,2>,
    rot: array<u32,2>,
    scale: array<u32,2>
};

struct Splat {
    //TODO: store information for 2D splat rendering
    pos: u32,  // NDC center (x, y), packed as 2 x f16
    size: u32,  // NDC quad width and height, packed as 2 x f16
    color: array<u32, 2>,  // RGBA, packed as 4 x f16
    conic_opacity: array<u32, 2>,  // inverse 2D covariance (xx, xy, yy) and opacity, packed as 4 x f16
};

//TODO: bind your data here
@group(0) @binding(0)
var<uniform> camera: CameraUniforms;
@group(0) @binding(1)
var<uniform> render_settings: RenderSettings;

@group(1) @binding(0)
var<storage, read> gaussians: array<Gaussian>;
@group(1) @binding(1)
var<storage, read_write> splats: array<Splat>;
@group(1) @binding(2)
var<storage, read> sh_buffer: array<u32>;  // f16 pairs; per gaussian 16 coefs x RGB = 48 f16

@group(2) @binding(0)
var<storage, read_write> sort_infos: SortInfos;
@group(2) @binding(1)
var<storage, read_write> sort_depths : array<u32>;
@group(2) @binding(2)
var<storage, read_write> sort_indices : array<u32>;
@group(2) @binding(3)
var<storage, read_write> sort_dispatch: DispatchIndirect;

// reads the i-th f16 value from sh_buffer (two f16 per u32)
fn read_sh_f16(i: u32) -> f32 {
    return unpack2x16float(sh_buffer[i / 2u])[i % 2u];
}

/// reads the ith sh coef from the storage buffer 
fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f32> {
    //TODO: access your binded sh_coeff, see load.ts for how it is stored
    let base = splat_idx * 48u + c_idx * 3u;  // R, G, B are stored next to each other
    return vec3<f32>(read_sh_f16(base), read_sh_f16(base + 1u), read_sh_f16(base + 2u));
}

// spherical harmonics evaluation with Condon–Shortley phase
fn computeColorFromSH(dir: vec3<f32>, v_idx: u32, sh_deg: u32) -> vec3<f32> {
    var result = SH_C0 * sh_coef(v_idx, 0u);

    if sh_deg > 0u {

        let x = dir.x;
        let y = dir.y;
        let z = dir.z;

        result += - SH_C1 * y * sh_coef(v_idx, 1u) + SH_C1 * z * sh_coef(v_idx, 2u) - SH_C1 * x * sh_coef(v_idx, 3u);

        if sh_deg > 1u {

            let xx = dir.x * dir.x;
            let yy = dir.y * dir.y;
            let zz = dir.z * dir.z;
            let xy = dir.x * dir.y;
            let yz = dir.y * dir.z;
            let xz = dir.x * dir.z;

            result += SH_C2[0] * xy * sh_coef(v_idx, 4u) + SH_C2[1] * yz * sh_coef(v_idx, 5u) + SH_C2[2] * (2.0 * zz - xx - yy) * sh_coef(v_idx, 6u) + SH_C2[3] * xz * sh_coef(v_idx, 7u) + SH_C2[4] * (xx - yy) * sh_coef(v_idx, 8u);

            if sh_deg > 2u {
                result += SH_C3[0] * y * (3.0 * xx - yy) * sh_coef(v_idx, 9u) + SH_C3[1] * xy * z * sh_coef(v_idx, 10u) + SH_C3[2] * y * (4.0 * zz - xx - yy) * sh_coef(v_idx, 11u) + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_coef(v_idx, 12u) + SH_C3[4] * x * (4.0 * zz - xx - yy) * sh_coef(v_idx, 13u) + SH_C3[5] * z * (xx - yy) * sh_coef(v_idx, 14u) + SH_C3[6] * x * (xx - 3.0 * yy) * sh_coef(v_idx, 15u);
            }
        }
    }
    result += 0.5;

    return  max(vec3<f32>(0.), result);
}

// 3D covariance from rotation and scale: Sigma = R * S * S^T * R^T
fn compute_cov3d(rot: vec4<f32>, scale: vec3<f32>) -> mat3x3<f32> {
    let q = normalize(rot);
    let r = q.x;  // quaternion stored as (w, x, y, z)
    let x = q.y;
    let y = q.z;
    let z = q.w;

    // column-major, same layout as the reference CUDA code (glm), so this is R^T
    let R = mat3x3<f32>(
        1. - 2. * (y * y + z * z), 2. * (x * y - r * z), 2. * (x * z + r * y),
        2. * (x * y + r * z), 1. - 2. * (x * x + z * z), 2. * (y * z - r * x),
        2. * (x * z - r * y), 2. * (y * z + r * x), 1. - 2. * (x * x + y * y)
    );
    let S = mat3x3<f32>(
        scale.x, 0., 0.,
        0., scale.y, 0.,
        0., 0., scale.z
    );

    let M = S * R;
    return transpose(M) * M;
}

// 2D (screen space, pixels) covariance: J * W * Sigma * W^T * J^T, returned as (xx, xy, yy)
fn compute_cov2d(pos_view: vec3<f32>, cov3d: mat3x3<f32>) -> vec3<f32> {
    // clamp to 1.3x the FOV so gaussians near the edge don't blow up the Jacobian
    let lim = 1.3 * camera.viewport / (2. * camera.focal);  // 1.3 * tan(fov / 2)
    let tz = pos_view.z;
    let txy = clamp(pos_view.xy / tz, -lim, lim) * tz;

    // Jacobian of the perspective projection, column-major as in the reference
    let J = mat3x3<f32>(
        camera.focal.x / tz, 0., -(camera.focal.x * txy.x) / (tz * tz),
        0., camera.focal.y / tz, -(camera.focal.y * txy.y) / (tz * tz),
        0., 0., 0.
    );
    let W = transpose(mat3x3<f32>(camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz));
    let T = W * J;
    let cov = transpose(T) * cov3d * T;

    // low-pass filter: make every splat at least about one pixel wide
    return vec3<f32>(cov[0][0] + 0.3, cov[0][1], cov[1][1] + 0.3);
}

@compute @workgroup_size(workgroupSize,1,1)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) wgs: vec3<u32>) {
    let idx = gid.x + gid.y * wgs.x * workgroupSize;  // workgroups wrap into rows when one row isn't enough
    //TODO: set up pipeline as described in instruction
    if (idx >= arrayLength(&gaussians)) {
        return;
    }

    // unpack the gaussian center (stored as f16)
    let gaussian = gaussians[idx];
    let a = unpack2x16float(gaussian.pos_opacity[0]);
    let b = unpack2x16float(gaussian.pos_opacity[1]);
    let pos_world = vec4<f32>(a.x, a.y, b.x, 1.);

    // world -> view -> clip -> NDC
    let pos_view = camera.view * pos_world;
    let pos_clip = camera.proj * pos_view;
    let pos_ndc = pos_clip.xyz / pos_clip.w;

    // behind the camera or outside near/far: always skipped, the math below would be wrong
    if (pos_clip.w <= 0. || pos_ndc.z < 0. || pos_ndc.z > 1.) {
        return;
    }
    // view frustum culling: keep 1.2x the screen so edge gaussians aren't cut off
    if (frustumCulling && (abs(pos_ndc.x) > 1.2 || abs(pos_ndc.y) > 1.2)) {
        return;
    }

    // rotation (w, x, y, z) and log-space scale, stored as f16
    let rot = vec4<f32>(unpack2x16float(gaussian.rot[0]), unpack2x16float(gaussian.rot[1]));
    let scale = exp(vec3<f32>(unpack2x16float(gaussian.scale[0]), unpack2x16float(gaussian.scale[1]).x));
    if (dot(rot, rot) == 0.) {  // a zero quaternion can't be normalized (would give NaN)
        return;
    }

    let cov3d = compute_cov3d(rot, scale * render_settings.gaussian_scaling);
    let cov2d = compute_cov2d(pos_view.xyz, cov3d);

    let det = cov2d.x * cov2d.z - cov2d.y * cov2d.y;
    if (det <= 0.) {  // not invertible (or numerically broken), no valid ellipse
        return;
    }

    // radius = 3 sigma along the longer axis (larger eigenvalue), in pixels
    let mid = 0.5 * (cov2d.x + cov2d.z);
    let lambda = mid + sqrt(max(0.1, mid * mid - det));
    let radius = ceil(3. * sqrt(lambda));

    // pixels -> NDC (the viewport spans 2 NDC units), then radius -> full width
    let size = 2. * radius * 2. / camera.viewport;

    splats[idx].pos = pack2x16float(pos_ndc.xy);
    splats[idx].size = pack2x16float(size);
    // view-dependent color: direction from the camera to the gaussian
    let cam_pos = camera.view_inv[3].xyz;
    let dir = normalize(pos_world.xyz - cam_pos);
    let color = computeColorFromSH(dir, idx, u32(render_settings.sh_deg));
    splats[idx].color = array<u32, 2>(pack2x16float(color.rg), pack2x16float(vec2<f32>(color.b, 1.)));

    // conic = inverse of the 2D covariance; the file stores opacity before sigmoid
    let conic = vec3<f32>(cov2d.z, -cov2d.y, cov2d.x) / det;
    let opacity = 1. / (1. + exp(-b.y));
    splats[idx].conic_opacity = array<u32, 2>(pack2x16float(conic.xy), pack2x16float(vec2<f32>(conic.z, opacity)));

    // reserve a slot in the sort list; keys_size ends up as the visible count
    let key = atomicAdd(&sort_infos.keys_size, 1u);
    sort_indices[key] = idx;
    sort_depths[key] = bitcast<u32>(100. - pos_view.z);  // far plane is 100; ascending sort -> back to front

    let keys_per_dispatch = workgroupSize * sortKeyPerThread; 
    // increment DispatchIndirect.dispatchx each time you reach limit for one dispatch of keys
    if (key % keys_per_dispatch == 0u) {
        atomicAdd(&sort_dispatch.dispatch_x, 1u);
    }
}