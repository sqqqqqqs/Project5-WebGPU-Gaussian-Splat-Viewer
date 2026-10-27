// Measures GPU time of each stage with timestamp queries (vsync does not affect it).
// Timestamp i marks the start of stage labels[i]; the last timestamp marks the end of the last stage.
export class GpuTimer {
  private readonly enabled: boolean;
  readonly ms: Record<string, number> = {};  // smoothed duration of each stage, in milliseconds

  private query_set?: GPUQuerySet;
  private resolve_buffer?: GPUBuffer;
  private read_buffer?: GPUBuffer;
  private copied = false;   // read_buffer holds new data waiting to be mapped
  private reading = false;  // read_buffer is being mapped, don't copy into it

  constructor(device: GPUDevice, private readonly labels: string[]) {
    this.enabled = device.features.has('timestamp-query');
    if (!this.enabled) return;

    const count = labels.length + 1;
    this.query_set = device.createQuerySet({ type: 'timestamp', count });
    this.resolve_buffer = device.createBuffer({
      label: 'timestamp resolve',
      size: count * 8,  // one u64 (nanoseconds) per timestamp
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    this.read_buffer = device.createBuffer({
      label: 'timestamp read',
      size: count * 8,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
  }

  // timestampWrites for a pass that writes timestamp `begin` at its start and `end` at its end
  writes(begin: number, end: number): GPUComputePassTimestampWrites | undefined {
    if (!this.enabled) return undefined;
    return { querySet: this.query_set!, beginningOfPassWriteIndex: begin, endOfPassWriteIndex: end };
  }

  // call after all passes of the frame are recorded
  resolve(encoder: GPUCommandEncoder) {
    if (!this.enabled || this.reading) return;
    encoder.resolveQuerySet(this.query_set!, 0, this.labels.length + 1, this.resolve_buffer!, 0);
    encoder.copyBufferToBuffer(this.resolve_buffer!, 0, this.read_buffer!, 0, this.resolve_buffer!.size);
    this.copied = true;
  }

  // call after the frame is submitted; results arrive a few frames later
  async read() {
    if (!this.copied || this.reading) return;
    this.copied = false;
    this.reading = true;

    let t: BigUint64Array;
    try {
      await this.read_buffer!.mapAsync(GPUMapMode.READ);
      t = new BigUint64Array(this.read_buffer!.getMappedRange().slice(0));
      this.read_buffer!.unmap();
    } catch {
      return;  // buffer destroyed or device lost: skip this sample
    } finally {
      this.reading = false;  // always allow the next read
    }

    this.labels.forEach((label, i) => {
      const sample = Number(t[i + 1] - t[i]) / 1e6;  // ns -> ms
      const prev = this.ms[label];
      this.ms[label] = prev === undefined ? sample : prev * 0.95 + sample * 0.05;  // smooth out noise
    });
  }

  destroy() {
    this.query_set?.destroy();
    this.resolve_buffer?.destroy();
    this.read_buffer?.destroy();
  }
}
