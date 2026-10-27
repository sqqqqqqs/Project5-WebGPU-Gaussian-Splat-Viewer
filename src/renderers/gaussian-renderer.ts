import { PointCloud } from '../utils/load';
import preprocessWGSL from '../shaders/preprocess.wgsl';
import renderWGSL from '../shaders/gaussian.wgsl';
import { get_sorter,c_histogram_block_rows,C } from '../sort/sort';
import { Renderer } from './renderer';
import { GpuTimer } from '../utils/gpu-timer';

export interface GaussianRenderer extends Renderer {
  set_gaussian_multiplier: (value: number) => void,
  read_visible_count: () => void,  // call after submit; result goes to visible_count
  visible_count: number,
}

// switches for performance analysis
export interface GaussianOptions {
  workgroup_size: number,  // preprocess workgroup size, must divide keys per sort dispatch (3840)
  frustum_culling: boolean,
}

// Utility to create GPU buffers
const createBuffer = (
  device: GPUDevice,
  label: string,
  size: number,
  usage: GPUBufferUsageFlags,
  data?: ArrayBuffer | ArrayBufferView
) => {
  const buffer = device.createBuffer({ label, size, usage });
  if (data) device.queue.writeBuffer(buffer, 0, data);
  return buffer;
};

export default function get_renderer(
  pc: PointCloud,
  device: GPUDevice,
  presentation_format: GPUTextureFormat,
  camera_buffer: GPUBuffer,
  options: GaussianOptions = { workgroup_size: C.histogram_wg_size, frustum_culling: true },
): GaussianRenderer {

  const sorter = get_sorter(pc.num_points, device);
  // timestamps: 0 preprocess start, 1 sort start (preprocess end), 2 render start, 3 render end
  const timer = new GpuTimer(device, ['preprocess', 'sort', 'render']);

  // ===============================================
  //            Initialize GPU Buffers
  // ===============================================

  const nulling_data = new Uint32Array([0]);

  // [vertexCount, instanceCount, firstVertex, firstInstance]: 6 vertices per quad
  // instanceCount is overwritten every frame with the visible count from preprocess
  const indirect_draw_data = new Uint32Array([6, 0, 0, 0]);
  const indirect_draw_buffer = createBuffer(
    device,
    'indirect draw',
    indirect_draw_data.byteLength,
    GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST,
    indirect_draw_data
  );

  // output of preprocess, input of render; size must match the WGSL Splat struct
  const c_size_splat = 24;
  const splat_buffer = createBuffer(
    device,
    'splats',
    pc.num_points * c_size_splat,
    GPUBufferUsage.STORAGE
  );

  // matches RenderSettings in preprocess.wgsl: [gaussian_scaling, sh_deg]
  const render_settings_data = new Float32Array([1.0, pc.sh_deg]);
  const render_settings_buffer = createBuffer(
    device,
    'render settings',
    render_settings_data.byteLength,
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    render_settings_data
  );

  // reads back keys_size (number of gaussians that passed culling) for the GUI
  const visible_count_buffer = createBuffer(
    device,
    'visible count read',
    4,
    GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
  );
  let visible_count = 0;
  let visible_count_copied = false;
  let visible_count_reading = false;

  // ===============================================
  //    Create Compute Pipeline and Bind Groups
  // ===============================================
  const preprocess_pipeline = device.createComputePipeline({
    label: 'preprocess',
    layout: 'auto',
    compute: {
      module: device.createShaderModule({ code: preprocessWGSL }),
      entryPoint: 'preprocess',
      constants: {
        workgroupSize: options.workgroup_size,
        // keep workgroupSize * sortKeyPerThread equal to the keys per sort dispatch
        sortKeyPerThread: C.histogram_wg_size * c_histogram_block_rows / options.workgroup_size,
        frustumCulling: options.frustum_culling ? 1 : 0,
      },
    },
  });

  const sort_bind_group = device.createBindGroup({
    label: 'sort',
    layout: preprocess_pipeline.getBindGroupLayout(2),
    entries: [
      { binding: 0, resource: { buffer: sorter.sort_info_buffer } },
      { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_depths_buffer } },
      { binding: 2, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },
      { binding: 3, resource: { buffer: sorter.sort_dispatch_indirect_buffer } },
    ],
  });

  const camera_bind_group = device.createBindGroup({
    label: 'preprocess camera',
    layout: preprocess_pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: camera_buffer } },
      { binding: 1, resource: { buffer: render_settings_buffer } },
    ],
  });

  const gaussian_bind_group = device.createBindGroup({
    label: 'preprocess gaussians',
    layout: preprocess_pipeline.getBindGroupLayout(1),
    entries: [
      { binding: 0, resource: { buffer: pc.gaussian_3d_buffer } },
      { binding: 1, resource: { buffer: splat_buffer } },
      { binding: 2, resource: { buffer: pc.sh_buffer } },
    ],
  });


  // ===============================================
  //    Create Render Pipeline and Bind Groups
  // ===============================================
  const render_shader = device.createShaderModule({ code: renderWGSL });
  const render_pipeline = device.createRenderPipeline({
    label: 'gaussian render',
    layout: 'auto',
    vertex: {
      module: render_shader,
      entryPoint: 'vs_main',
    },
    fragment: {
      module: render_shader,
      entryPoint: 'fs_main',
      targets: [{
        format: presentation_format,
        // back-to-front "over" blending: result = src * alpha + dst * (1 - alpha)
        blend: {
          color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
          alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
        },
      }],
    },
    primitive: {
      topology: 'triangle-list',
    },
  });

  const splat_bind_group = device.createBindGroup({
    label: 'render splats',
    layout: render_pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: splat_buffer } },
      { binding: 1, resource: { buffer: sorter.ping_pong[0].sort_indices_buffer } },  // sorted result
      { binding: 2, resource: { buffer: camera_buffer } },
    ],
  });

  // ===============================================
  //    Command Encoder Functions
  // ===============================================
  const preprocess = (encoder: GPUCommandEncoder) => {
    const pass = encoder.beginComputePass({ label: 'preprocess', timestampWrites: timer.writes(0, 1) });
    pass.setPipeline(preprocess_pipeline);
    pass.setBindGroup(0, camera_bind_group);
    pass.setBindGroup(1, gaussian_bind_group);
    pass.setBindGroup(2, sort_bind_group);
    // one thread per gaussian; a single dimension allows at most 65535 workgroups, so wrap into rows
    const workgroups = Math.ceil(pc.num_points / options.workgroup_size);
    const max_per_row = device.limits.maxComputeWorkgroupsPerDimension;
    pass.dispatchWorkgroups(Math.min(workgroups, max_per_row), Math.ceil(workgroups / max_per_row));
    pass.end();
  };

  const render = (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
    const pass = encoder.beginRenderPass({
      label: 'gaussian render',
      timestampWrites: timer.writes(2, 3),
      colorAttachments: [
        {
          view: texture_view,
          loadOp: 'clear',
          storeOp: 'store',
        }
      ],
    });
    pass.setPipeline(render_pipeline);
    pass.setBindGroup(0, splat_bind_group);
    pass.drawIndirect(indirect_draw_buffer, 0);  // instance count comes from the buffer
    pass.end();
  };

  // ===============================================
  //    Return Render Object
  // ===============================================
  return {
    frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => {
      // clear the counters that preprocess fills with atomicAdd
      device.queue.writeBuffer(sorter.sort_info_buffer, 0, nulling_data);  // keys_size = 0
      device.queue.writeBuffer(sorter.sort_dispatch_indirect_buffer, 0, nulling_data);  // dispatch_x = 0

      preprocess(encoder);
      // instanceCount (byte offset 4) = keys_size (byte offset 0)
      encoder.copyBufferToBuffer(sorter.sort_info_buffer, 0, indirect_draw_buffer, 4, 4);
      sorter.sort(encoder);
      render(encoder, texture_view);
      timer.resolve(encoder);

      if (!visible_count_reading) {
        encoder.copyBufferToBuffer(sorter.sort_info_buffer, 0, visible_count_buffer, 0, 4);  // keys_size
        visible_count_copied = true;
      }
    },
    read_visible_count: async () => {
      if (!visible_count_copied || visible_count_reading) return;
      visible_count_copied = false;
      visible_count_reading = true;
      try {
        await visible_count_buffer.mapAsync(GPUMapMode.READ);
        visible_count = new Uint32Array(visible_count_buffer.getMappedRange())[0];
        visible_count_buffer.unmap();
      } catch {
        // buffer destroyed or device lost: keep the last value
      } finally {
        visible_count_reading = false;  // always allow the next read
      }
    },
    get visible_count() { return visible_count; },
    set_gaussian_multiplier: (value: number) => {
      render_settings_data[0] = value;
      device.queue.writeBuffer(render_settings_buffer, 0, render_settings_data);
    },
    camera_buffer,
    timer,
    destroy: () => {
      timer.destroy();
      const buffers = [
        indirect_draw_buffer, splat_buffer, render_settings_buffer, visible_count_buffer,
        sorter.sort_info_buffer, sorter.sort_dispatch_indirect_buffer,
        ...sorter.ping_pong.flatMap((p) => [p.sort_indices_buffer, p.sort_depths_buffer]),
      ];
      buffers.forEach((b) => b.destroy());  // the sorter's small internal histogram buffer is left to GC
    },
  };
}
