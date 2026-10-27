import { load, PointCloud } from '../utils/load';
import { Pane } from 'tweakpane';
import * as TweakpaneFileImportPlugin from 'tweakpane-plugin-file-import';
import { default as get_renderer_gaussian, GaussianRenderer } from './gaussian-renderer';
import { default as get_renderer_pointcloud } from './point-cloud-renderer';
import { Camera, load_camera_presets} from '../camera/camera';
import { CameraControl } from '../camera/camera-control';
import { time, timeReturn } from '../utils/simple-console';
import { GpuTimer } from '../utils/gpu-timer';

export interface Renderer {
  frame: (encoder: GPUCommandEncoder, texture_view: GPUTextureView) => void,
  camera_buffer: GPUBuffer,
  timer: GpuTimer,
  destroy: () => void,  // frees the GPU buffers this renderer created
}

export default async function init(
  canvas: HTMLCanvasElement,
  context: GPUCanvasContext,
  device: GPUDevice
) {
  let ply_file_loaded = false; 
  let cam_file_loaded = false; 
  let renderers: { pointcloud?: Renderer, gaussian?: Renderer } = {};
  let gaussian_renderer: GaussianRenderer | undefined; 
  let pointcloud_renderer: Renderer | undefined; 
  let renderer: Renderer | undefined; 
  let cameras;
  let pc: PointCloud | undefined;  // kept so the gaussian renderer can be rebuilt without reloading
  
  const camera = new Camera(canvas, device);
  const control = new CameraControl(camera);

  const observer = new ResizeObserver(() => {
    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientHeight;

    camera.on_update_canvas();
  });
  observer.observe(canvas);
  
  const presentation_format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device,
    format: presentation_format,
    alphaMode: 'opaque',
  });
  

  // Tweakpane: easily adding tweak control for parameters.
  const params = {
    fps: 0.0,
    gaussian_multiplier: 1,
    renderer: 'pointcloud',
    ply_file: '',
    cam_file: '',
    // GPU time per frame in ms, from timestamp queries
    gpu_preprocess: 0.0,
    gpu_sort: 0.0,
    gpu_render: 0.0,
    gpu_total: 0.0,
    visible: 0,  // gaussians drawn this frame (after culling)
    // performance analysis switches
    workgroup_size: 256,
    frustum_culling: true,
  };

  // (re)build the gaussian renderer with the current GUI options
  function create_gaussian_renderer() {
    if (!pc) return;
    gaussian_renderer?.destroy();  // don't wait for garbage collection to free the old buffers
    gaussian_renderer = get_renderer_gaussian(pc, device, presentation_format, camera.uniform_buffer, {
      workgroup_size: params.workgroup_size,
      frustum_culling: params.frustum_culling,
    });
    gaussian_renderer.set_gaussian_multiplier(params.gaussian_multiplier);  // keep current slider value
    renderers.gaussian = gaussian_renderer;
    renderer = renderers[params.renderer];
  }

  const pane = new Pane({
    title: 'Config',
    expanded: true,
  });
  pane.registerPlugin(TweakpaneFileImportPlugin);
  {
    pane.addMonitor(params, 'fps', {
      readonly:true
    });
  }
  if (device.features.has('timestamp-query')) {
    for (const key of ['gpu_preprocess', 'gpu_sort', 'gpu_render', 'gpu_total'] as const) {
      pane.addMonitor(params, key, { format: (v: number) => `${v.toFixed(3)} ms` });
    }
  }
  {
    pane.addMonitor(params, 'visible', { format: (v: number) => v.toFixed(0) });
  }
  {
    pane.addInput(params, 'renderer', {
      options: {
        pointcloud: 'pointcloud',
        gaussian: 'gaussian',
      }
    }).on('change', (e) => {
      renderer = renderers[e.value];
    });
  }
  {
    pane.addInput(params, 'ply_file', {
      view: 'file-input',
      lineCount: 3,
      filetypes: ['.ply'],
      invalidFiletypeMessage: "We can't accept those filetypes!"
    })
    .on('change', async (file) => {
      const uploadedFile = file.value;
      if (uploadedFile) {
        const new_pc = await load(uploadedFile, device);
        // free the previous scene (the old gaussian renderer is freed in create_gaussian_renderer)
        pointcloud_renderer?.destroy();
        pc?.gaussian_3d_buffer.destroy();
        pc?.sh_buffer.destroy();
        pc = new_pc;
        pointcloud_renderer = get_renderer_pointcloud(pc, device, presentation_format, camera.uniform_buffer);
        renderers = {
          pointcloud: pointcloud_renderer,
        };
        create_gaussian_renderer();  // also adds it to renderers and picks the active renderer
        ply_file_loaded = true;
      }else{
        ply_file_loaded = false;
      }
    });
  }
  {
    pane.addInput(params, 'cam_file', {
      view: 'file-input',
      lineCount: 3,
      filetypes: ['.json'],
      invalidFiletypeMessage: "We can't accept those filetypes!"
    })
    .on('change', async (file) => {
      const uploadedFile = file.value;
      if (uploadedFile) {
        cameras=await load_camera_presets(file.value);
        camera.set_preset(cameras[0]);
        cam_file_loaded = true;
      }else{
        cam_file_loaded = false;
      }
    });
  }
  {
    pane.addInput(
      params,
      'gaussian_multiplier',
      {min: 0, max: 1.5}
    ).on('change', (e) => {
      //TODO: Bind constants to the gaussian renderer.
      gaussian_renderer?.set_gaussian_multiplier(e.value);  // no-op before a ply is loaded
    });
  }
  {
    // performance analysis: changing these rebuilds the gaussian renderer, no file reload needed
    pane.addInput(params, 'workgroup_size', {
      options: { 32: 32, 64: 64, 128: 128, 256: 256 },
    }).on('change', create_gaussian_renderer);
    pane.addInput(params, 'frustum_culling').on('change', create_gaussian_renderer);
  }

  document.addEventListener('keydown', (event) => {
    switch(event.key) {
      case '0':
      case '1':
      case '2':
      case '3':
      case '4':
      case '5':
      case '6':
      case '7':
      case '8':
      case '9':
        const i = parseInt(event.key);
        console.log(`set to camera preset ${i}`);
        camera.set_preset(cameras[i]);
        break;
    }
  });

  function frame() {
    if (ply_file_loaded && cam_file_loaded) {
      params.fps=1.0/timeReturn()*1000.0;
      time();
      const encoder = device.createCommandEncoder();
      const texture_view = context.getCurrentTexture().createView();
      renderer.frame(encoder, texture_view);
      device.queue.submit([encoder.finish()]);

      renderer.timer.read();
      const ms = renderer.timer.ms;
      params.gpu_preprocess = ms.preprocess ?? 0;  // point cloud has only the render stage
      params.gpu_sort = ms.sort ?? 0;
      params.gpu_render = ms.render ?? 0;
      params.gpu_total = params.gpu_preprocess + params.gpu_sort + params.gpu_render;

      if (gaussian_renderer && renderer === gaussian_renderer) {
        gaussian_renderer.read_visible_count();
        params.visible = gaussian_renderer.visible_count;
      } else {
        params.visible = pc?.num_points ?? 0;  // the point cloud draws every point
      }
    }
    requestAnimationFrame(frame);
  }

  requestAnimationFrame(frame);
}
