from __future__ import annotations

import os
import queue
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import stitch_pipeline
from data import create_dataset
from models import create_model
from options.test_options import TestOptions

REPO_ROOT = Path(__file__).resolve().parent
IMAGE_ROOT = stitch_pipeline.IMAGE_ROOT
TEST_ROOT = stitch_pipeline.TEST_ROOT
MODEL_NAME = stitch_pipeline.MODEL_NAME
MODEL = stitch_pipeline.MODEL
DIRECTION = stitch_pipeline.DIRECTION
PHASE = stitch_pipeline.PHASE
NET_G = stitch_pipeline.NET_G
NETWORK_SIZE = stitch_pipeline.NETWORK_SIZE

BATCH_SIZES = [192]
NUM_THREADS = 12
SAVE_THREADS = 8
OUTPUT_SIZE = 128
USE_AMP = True
STREAM_STITCH = True

class _GpuUtilizationSampler:

    def __init__(self, interval_ms: int = 100):
        self.interval_ms = interval_ms
        self.samples: list[int] = []
        self.process = None
        self.thread = None

    def start(self) -> None:
        if not torch.cuda.is_available():
            return
        try:
            self.process = subprocess.Popen(
                [
                    "nvidia-smi",
                    "--id=0",
                    "--query-gpu=utilization.gpu",
                    "--format=csv,noheader,nounits",
                    f"--loop-ms={self.interval_ms}",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
        except (FileNotFoundError, OSError):
            self.process = None
            return

        def collect() -> None:
            assert self.process is not None and self.process.stdout is not None
            for line in self.process.stdout:
                try:
                    self.samples.append(int(line.strip()))
                except ValueError:
                    continue

        self.thread = threading.Thread(target=collect, daemon=True)
        self.thread.start()

    def stop(self) -> tuple[float, int] | None:
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            if self.thread is not None:
                self.thread.join(timeout=1)
            self.process = None
        if not self.samples:
            return None
        return sum(self.samples) / len(self.samples), max(self.samples)

class _AsyncImageSaver:

    def __init__(self, image_dir: Path, workers: int = SAVE_THREADS, max_pending: int = 512):
        self.image_dir = image_dir
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="png-writer")
        self.pending = set()
        self.max_pending = max_pending
        self.closed = False

    @staticmethod
    def _save(image_tensor: torch.Tensor, output_path: Path) -> None:
        image = Image.fromarray(image_tensor.numpy(), mode="RGB")
        if image.size != (OUTPUT_SIZE, OUTPUT_SIZE):
            image = image.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.BILINEAR)
        image.save(output_path)

    def submit_batch(self, batch_tensor: torch.Tensor, image_paths: list[str]) -> None:

        images = (
            ((batch_tensor.detach() + 1.0) * 127.5)
            .clamp_(0, 255)
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .cpu()
        )
        for index, image_path in enumerate(image_paths):
            output_path = self.image_dir / f"{Path(image_path).stem}_fake_B.png"
            self.pending.add(self.executor.submit(self._save, images[index], output_path))
            if len(self.pending) >= self.max_pending:
                done, self.pending = wait(self.pending, return_when=FIRST_COMPLETED)
                for future in done:
                    future.result()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            if self.pending:
                done, _ = wait(self.pending)
                for future in done:
                    future.result()
                self.pending.clear()
        finally:
            self.executor.shutdown(wait=True)

class _StreamingMosaic:

    def __init__(
        self,
        grid_rows: int,
        grid_cols: int,
        step_size: int,
        output_path: Path,
        original_size: tuple[int, int],
    ):
        from stitch.stitch_images import _feather_mask

        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.step_size = step_size
        self.output_path = output_path
        self.original_width, self.original_height = original_size
        self.canvas_width = (grid_cols - 1) * step_size + OUTPUT_SIZE
        self.canvas_height = (grid_rows - 1) * step_size + OUTPUT_SIZE
        self.acc = np.zeros((self.canvas_height, self.canvas_width, 3), dtype=np.float32)
        self.weight = np.zeros((self.canvas_height, self.canvas_width, 1), dtype=np.float32)
        feather_px = max(1, (OUTPUT_SIZE - step_size) // 2)
        self.base_mask = _feather_mask(OUTPUT_SIZE, OUTPUT_SIZE, feather_px)
        self.feather_px = feather_px
        self.items = queue.Queue(maxsize=3)
        self.error = None
        self.closed = False
        self.worker = threading.Thread(target=self._consume, daemon=True, name="mosaic-worker")
        self.worker.start()

    def _consume(self) -> None:
        try:
            while True:
                item = self.items.get()
                try:
                    if item is None:
                        return
                    images, image_paths = item
                    for image, image_path in zip(images, image_paths):
                        tile_index = int(Path(image_path).stem) - 1
                        row = tile_index // self.grid_cols
                        col = tile_index % self.grid_cols
                        if row >= self.grid_rows:
                            continue
                        x0 = col * self.step_size
                        y0 = row * self.step_size
                        x1 = min(x0 + OUTPUT_SIZE, self.canvas_width)
                        y1 = min(y0 + OUTPUT_SIZE, self.canvas_height)
                        mask = self.base_mask.copy()
                        if col == 0:
                            mask[:, :self.feather_px] = 1.0
                        if col == self.grid_cols - 1:
                            mask[:, -self.feather_px:] = 1.0
                        if row == 0:
                            mask[:self.feather_px, :] = 1.0
                        if row == self.grid_rows - 1:
                            mask[-self.feather_px:, :] = 1.0
                        mask = mask[:y1 - y0, :x1 - x0]
                        tile = image[:y1 - y0, :x1 - x0].numpy().astype(np.float32, copy=False)
                        self.acc[y0:y1, x0:x1] += tile * mask[:, :, None]
                        self.weight[y0:y1, x0:x1, 0] += mask
                finally:
                    self.items.task_done()
        except BaseException as error:
            self.error = error

    def submit_batch(self, batch_tensor: torch.Tensor, image_paths: list[str]) -> None:
        resized = F.interpolate(
            batch_tensor.detach(),
            size=(OUTPUT_SIZE, OUTPUT_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        images = (
            ((resized + 1.0) * 127.5)
            .clamp_(0, 255)
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .cpu()
        )
        self.items.put((images, image_paths))
        if self.error is not None:
            raise RuntimeError("Background mosaic worker failed") from self.error

    def close(self) -> Path:
        if self.closed:
            return self.output_path
        self.closed = True
        self.items.put(None)
        self.items.join()
        self.worker.join()
        if self.error is not None:
            raise RuntimeError("Background mosaic worker failed") from self.error

        np.divide(self.acc, self.weight, out=self.acc, where=self.weight > 0)
        del self.weight
        np.rint(self.acc, out=self.acc)
        np.clip(self.acc, 0, 255, out=self.acc)
        cropped = self.acc[:self.original_height, :self.original_width].astype(np.uint8)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(cropped, mode="RGB").save(self.output_path)
        del cropped
        del self.acc
        print(
            f"[3/3] Streamed mosaic cropped to {self.original_width}x{self.original_height} "
            f"and saved to: {self.output_path}"
        )
        return self.output_path

    def abort(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.items.put(None)
        self.items.join()
        self.worker.join()
        del self.acc
        del self.weight

def run_crop() -> tuple[int, list[Path], int, int]:

    return stitch_pipeline.run_crop()

def _count_pngs(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    return sum(1 for path in directory.iterdir() if path.is_file() and path.suffix.lower() == ".png")

def _validate_test_data(num_test: int, combined_dirs: list[Path]) -> None:

    if not combined_dirs:
        raise FileNotFoundError("No pix2pix combined directory was found")
    if _count_pngs(Path(combined_dirs[0])) < num_test:
        raise FileNotFoundError(
            f"Insufficient pix2pix data: expected {num_test} images, "
            f"but {combined_dirs[0]} contains {_count_pngs(Path(combined_dirs[0]))}"
        )

def _make_streaming_mosaic(grid_rows: int, grid_cols: int) -> _StreamingMosaic:
    config_path = TEST_ROOT / "processing_config.json"
    with config_path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    step_size = int(config["processing_parameters"]["step_size"])
    patch_dimensions = config["output_statistics"]["patch_dimensions"]
    if not patch_dimensions:
        raise ValueError(f"No patch dimensions found in {config_path}")
    first_patch_info = next(iter(patch_dimensions.values()))
    original_size = first_patch_info["original_image_size"]
    original_width = int(original_size["width"])
    original_height = int(original_size["height"])
    stem = IMAGE_ROOT.stem
    output_stem = stem.replace("DRUM", "Hp") if "DRUM" in stem else f"{stem}_Hp"
    output_path = (TEST_ROOT / f"{output_stem}.png").resolve()
    print(
        f"[2/3] Streaming stitch enabled: grid={grid_rows}x{grid_cols}, "
        f"step={step_size}, no intermediate fake_B PNG files"
    )
    return _StreamingMosaic(
        grid_rows,
        grid_cols,
        step_size,
        output_path,
        (original_width, original_height),
    )

def _make_options(num_test: int):

    model = MODEL
    model_name = MODEL_NAME
    phase = PHASE

    argv = [
        "test_stitch_batch.py",
        "--dataroot", str(TEST_ROOT),
        "--name", model_name,
        "--phase", phase,
        "--model", model,
        "--direction", DIRECTION,
        "--num_test", str(num_test),
        "--netG", NET_G,
        "--load_size", str(NETWORK_SIZE),
        "--crop_size", str(NETWORK_SIZE),
    ]
    old_argv = sys.argv
    try:
        sys.argv = argv
        opt = TestOptions().parse()
    finally:
        sys.argv = old_argv

    opt.num_threads = NUM_THREADS

    opt.checkpoints_dir = (REPO_ROOT / "checkpoints").as_posix()
    opt.results_dir = (REPO_ROOT.parent / "results").as_posix()
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1
    opt.max_dataset_size = num_test
    opt.inference_input_only = True
    opt.adjust_input = stitch_pipeline.ENABLE_CROP_BC_ADJUSTMENT
    return opt

def initialize_model():

    opt = _make_options(1)
    print("[Init] Preloading model before crop; model will wait for the first prepared batch ...")
    initialization_started = time.perf_counter()
    model = create_model(opt)
    model.setup(opt)
    model.eval()
    initialization_elapsed = time.perf_counter() - initialization_started
    print(f"[Init] Model ready in {initialization_elapsed:.3f}s")
    return model, opt, initialization_elapsed

def run_test(
    num_test: int,
    combined_dirs: list[Path],
    grid_rows: int,
    grid_cols: int,
    model,
    opt,
) -> Path | None:

    _validate_test_data(num_test, combined_dirs)
    opt.max_dataset_size = num_test

    result_images = Path(opt.results_dir) / opt.name / f"{opt.phase}_{opt.epoch}" / "images"
    successful_batches: list[int] = []
    streamed_mosaic_path = None

    for batch_size in BATCH_SIZES:
        opt.batch_size = batch_size
        dataset = create_dataset(opt)
        processed = 0
        inference_elapsed = 0.0
        use_streaming_stitch = STREAM_STITCH
        output_sink = (
            _make_streaming_mosaic(grid_rows, grid_cols)
            if use_streaming_stitch
            else _AsyncImageSaver(result_images)
        )
        output_completed = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        started = time.perf_counter()
        gpu_sampler = _GpuUtilizationSampler()
        gpu_sampler.start()

        try:
            with torch.inference_mode():
                for data in dataset:
                    model.set_input(data)
                    if torch.cuda.is_available():
                        inference_start = torch.cuda.Event(enable_timing=True)
                        inference_end = torch.cuda.Event(enable_timing=True)
                        inference_start.record()
                    else:
                        inference_started = time.perf_counter()
                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.float16,
                        enabled=USE_AMP and torch.cuda.is_available(),
                    ):
                        model.test()
                    if torch.cuda.is_available():
                        inference_end.record()
                        inference_end.synchronize()
                        inference_elapsed += inference_start.elapsed_time(inference_end) / 1000.0
                    else:
                        inference_elapsed += time.perf_counter() - inference_started
                    paths = list(model.get_image_paths())
                    output_sink.submit_batch(model.fake_B, paths)
                    processed += len(paths)

            completed_output = output_sink.close()
            output_completed = True
            if use_streaming_stitch:
                streamed_mosaic_path = completed_output

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            utilization = gpu_sampler.stop()
            if torch.cuda.is_available():
                total_gpu_bytes = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
                peak_allocated_bytes = torch.cuda.max_memory_allocated()
                peak_reserved_bytes = torch.cuda.max_memory_reserved()
                peak_gb = peak_allocated_bytes / 1024**3
                peak_memory_percent = peak_allocated_bytes / total_gpu_bytes * 100
                peak_reserved_gb = peak_reserved_bytes / 1024**3
                peak_reserved_percent = peak_reserved_bytes / total_gpu_bytes * 100
            else:
                peak_gb = peak_memory_percent = 0.0
                peak_reserved_gb = peak_reserved_percent = 0.0
            speed = processed / elapsed if elapsed else float("inf")
            successful_batches.append(batch_size)
            utilization_text = (
                f", GPU utilization avg={utilization[0]:.1f}%, peak={utilization[1]}%"
                if utilization is not None
                else ", GPU utilization=N/A"
            )
            print(
                f"[2/3] batch={batch_size}: {processed} images, {elapsed:.3f}s, "
                f"{speed:.2f} images/s, "
                f"peak GPU memory allocated={peak_gb:.2f} GiB ({peak_memory_percent:.1f}%), "
                f"reserved={peak_reserved_gb:.2f} GiB ({peak_reserved_percent:.1f}%)"
                f"{utilization_text}"
            )
            inference_speed = processed / inference_elapsed if inference_elapsed else float("inf")
            print(
                f"[Timing] Pure GPU inference: {inference_elapsed:.3f}s, "
                f"{inference_speed:.2f} images/s; data + transfer + output: "
                f"{max(0.0, elapsed - inference_elapsed):.3f}s"
            )
        except torch.cuda.OutOfMemoryError:
            print(f"[2/3] batch={batch_size}: CUDA OOM, skipped")
            torch.cuda.empty_cache()
        finally:
            gpu_sampler.stop()
            if not output_completed:
                if use_streaming_stitch:
                    output_sink.abort()
                else:
                    output_sink.close()

    if not successful_batches:
        raise RuntimeError("All batch sizes failed")
    print(f"[2/3] Finished. Successful batch sizes: {successful_batches}")
    return streamed_mosaic_path

def run_stitch(grid_rows: int, grid_cols: int) -> None:

    stitch_pipeline.run_stitch(grid_rows, grid_cols, generated_at_stitch_size=True)

def main() -> None:
    pipeline_started = time.perf_counter()

    model, opt, initialization_elapsed = initialize_model()

    crop_started = time.perf_counter()
    total_patches, combined_dirs, grid_rows, grid_cols = run_crop()
    crop_elapsed = time.perf_counter() - crop_started
    print(f"[Timing] Crop: {crop_elapsed:.3f}s")

    streamed_mosaic_path = run_test(
        total_patches, combined_dirs, grid_rows, grid_cols, model, opt
    )

    if streamed_mosaic_path is None:
        stitch_started = time.perf_counter()
        run_stitch(grid_rows, grid_cols)
        stitch_elapsed = time.perf_counter() - stitch_started
    else:
        stitch_elapsed = 0.0
        print("[Timing] Stitch: integrated with inference (no intermediate PNG files)")
    pipeline_elapsed = time.perf_counter() - pipeline_started - initialization_elapsed
    if streamed_mosaic_path is None:
        print(f"[Timing] Stitch: {stitch_elapsed:.3f}s")
    print(f"[Timing] Total pipeline (excluding model initialization): {pipeline_elapsed:.3f}s")

if __name__ == "__main__":
    os.chdir(REPO_ROOT)
    main()
