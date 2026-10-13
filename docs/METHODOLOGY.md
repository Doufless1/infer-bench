# Methodology and limits

This page says exactly what `infer-bench` measures, how, and where the numbers can mislead you.
Every statement here corresponds to code in `infer_bench/` and, where it matters, to a test in `tests/`.

## 1. What a "run" is

```
warm-up (discarded)  ->  timed region  ->  statistics  ->  one JSON results file
```

* **Warm-up.** The first calls pay for lazy initialisation, memory allocation and cold caches, so
  `--warmup` calls (default 50) are made and thrown away. The count is fixed, not adaptive: if it is too
  short for your model the drift check (section 3) is what tells you.
* **Timed region.** Each call is timed on its own with `time.perf_counter_ns()`. Only the call is inside
  the stopwatch; picking the input and appending the sample are outside it. `infer-bench selfcheck` prints
  the cost and the resolution of the clock on your machine.
* **Garbage collector.** Python's GC is paused during the timed region, so a collection cannot land in the
  middle of a measurement. This does not affect pauses inside ONNX Runtime or the operating system.
* **Inputs.** A small pool of different inputs is cycled (8 synthetic ones by default, or up to 64 pictures with
  `--images`), so the model does not see the exact same tensor every time. They are built *before* the stopwatch
  starts. A small pool is friendlier to the CPU cache than real traffic would be.
  * Synthetic inputs are smooth random textures, not white noise (noise makes every network output look the
    same, which would make any agreement metric trivially perfect).
  * `--images DIR` uses your own pictures instead. Decoding a JPEG/PNG file is **outside** the timed region;
    resizing and normalising is inside it in `pipeline` mode. When the folder holds more pictures than
    `--max-images` (benchmark) or `--samples` (quantisation, accuracy), a **random, seeded** sample is taken
    (same `--seed`, same pictures). Put varied, representative pictures in the folder: calibrating on 100
    consecutive video frames gives a poor calibration set.
* **Throughput** is `items / wall-clock time of the whole timed loop`, not the sum of the latencies. The loop
  itself adds a little time per call outside the stopwatch (measured at roughly 0.5-0.9 microseconds in the
  development sandbox), so `1 / p50` is slightly higher than the reported throughput. That only matters for
  models that run in well under a millisecond.

### Modes

| Mode | What is timed |
|---|---|
| `model` (default) | only the call into the runtime |
| `pipeline` | pre-processing + model + post-processing, **each stage timed separately** and also as a total |

Pipeline post-processing is `--task classification` (softmax + top-5) or `--task detection` (YOLO-style decode,
confidence threshold, class-aware NMS, mapping boxes back to the source image). It is plain NumPy, so it
is a realistic *lower bound* on what a real application pays, not a tuned production implementation.

### Several streams (`--streams N`)

`N` Python threads share one loaded session and run for a fixed time window (`--duration`, default 5 s). A
barrier makes every stream finish its warm-up before the window opens, so all streams are busy for the whole
window. Reported: total items/s, items/s per stream, and latency percentiles over all calls.

* ONNX Runtime releases the GIL while a model runs, so threads can overlap. Unless you set `--threads`, the CPU
  cores are divided between the streams (otherwise `N` streams would each try to use every core and fight).
* In `pipeline` mode, pre- and post-processing hold the GIL, so scaling is limited. A warning says so.
* This simulates "several camera streams on one device" only roughly. It is not a multi-process or
  multi-device benchmark.

## 2. Percentiles and the confidence interval

* **p50 / p90 / p95 / p99** use NumPy's linear interpolation. With few samples the tail percentiles are noise:
  below 100 samples p99 is essentially the maximum, below 1000 it rests on fewer than ten samples. The tool
  warns in both cases. Use `--iterations 1000` or more when you care about p99.
* **95 % confidence interval of the median.** Consecutive latencies are not independent (a busy moment on the
  machine slows several calls in a row), so a plain bootstrap would give an interval that is too narrow. A
  *moving-block bootstrap* with blocks of about `n^(1/3)` samples (1000 resamples, fixed seed, so the result is
  deterministic) is used instead.
* What the interval does **not** cover: differences between runs. It describes how well this one run pins down
  its own median. Another run at another time (thermal state, other programs) can land outside it. For a claim
  that matters, repeat the run and look at the spread of the medians.

### What `compare` concludes

`infer-bench compare A B` prints the ratios B / A and a verdict:

* the two confidence intervals **do not overlap** -> "median latency is x.xx times lower/higher";
* they **overlap** -> "within measurement noise: do not claim a difference".

This is a conservative rule of thumb, not a hypothesis test. Overlap does not prove the two are equal; it says
this data cannot separate them. It also warns when the two runs differ in CPU model or architecture, Python or
package versions (ONNX Runtime and others that affect speed), batch size, mode, number of streams, thread settings,
graph-optimisation level or execution providers, or when a run has too few samples, because then the comparison
is not like for like.

## 3. Warnings: what triggers them and what they miss

| Warning | Trigger | Known limits |
|---|---|---|
| few samples | fewer than 1000 timed samples | - |
| drift | median of the first half differs from the second half by more than 5 % | A slowdown that covers less than about the last quarter of the run does not move the second-half median and is **not** caught here; it shows up as outliers and in p95/p99. In multi-stream runs the check is made per stream. |
| outliers | more than 5 % of samples outside the Tukey fences (Q1 - 1.5 IQR, Q3 + 1.5 IQR) | Latency distributions are skewed, so a healthy run on a noisy machine can trip this. Treat it as a hint to look at the machine, not as proof. Outliers are **counted, never removed**. |
| wide interval | CI of the median wider than 5 % of the median | - |
| environment | CPU governor is not `performance`, on battery, CPU already more than 25 % busy, running in a container | Read from `/sys` and `psutil`; not available on every OS. |
| synthetic inputs | no `--images` given | Fine for speed, not meaningful for accuracy. |

## 4. Accuracy: it is *agreement*, not accuracy

`infer-bench accuracy REFERENCE CANDIDATE` (or `run --reference`) runs both models on the same inputs, one
sample at a time, and reports how closely the candidate follows the reference (normally FP32 vs. INT8):

* cosine similarity and relative L2 error of the raw outputs (all outputs concatenated);
* classification: **top-1 agreement** (same top class) and how often the reference's top class is in the
  candidate's top 5;
* detection (YOLOv8-style output `(1, 4 + classes, anchors)`): the reference model's detections are treated as
  ground truth and the candidate's boxes are scored against them (same class, IoU >= 0.5, greedy matching by
  score), giving precision, recall and **F1**. Both outputs go through the same decoding (confidence 0.25,
  NMS IoU 0.45 unless changed).

What this is good for: noticing that quantisation damaged a model. What it cannot do: tell you the FP32 model
is any good, or give you mAP or top-1 *accuracy*. Those need labelled data and a proper evaluation. The
results file says so in its `notes`.

Cases where the numbers would be meaningless are detected and written into the results:

* synthetic inputs were used;
* the reference output is almost zero on these inputs (for example a text detector shown pictures without text);
* the reference gives almost the same output for every input (agreement is then trivial);
* the reference finds no objects, or the candidate finds none although the reference does;
* the candidate produced NaN or infinity (those samples are counted and excluded).

## 5. Quantisation

`infer-bench quantize` wraps ONNX Runtime's **static post-training quantisation** in QDQ format:
8-bit unsigned activations, 8-bit signed weights, per-channel weights (per tensor for old operator sets, see
pitfall 3), MinMax calibration by default
(`--method entropy|percentile`, `--reduce-range`, `--op-types`, `--exclude-nodes` are available). The activation
ranges are measured once on a calibration set and baked in, so **calibrate on pictures that look like your
real data** (`--images`). Calibrating on the synthetic textures only makes sense for the demo models.

Five pitfalls were met while testing, and the tool now diagnoses each of them:

1. **Mixed-scale outputs.** A YOLO-style output tensor holds pixel-scale box values and 0..1 class scores. If that
   tensor is quantised, both share one scale and the scores round to zero. On the demo detector (a constructed
   example with random weights, ONNX Runtime 1.29) the naive INT8 model keeps a cosine similarity of 0.99999 to FP32
   and still finds **no** objects (F1 = 0). Quantising only convolutions (`--op-types Conv`) fixes it (F1 about
   0.95 on the demo). How badly the naive version fails depends on the model and the runtime version, and it has
   not been tried on a real YOLO export here. Cosine similarity alone would have hidden the problem, which is why
   the detection F1, the "no detections" note, and a hint for detector-shaped outputs without a task exist.
2. **Weights stored as `Constant` nodes.** Some exporters (observed with PaddlePaddle exports) store weights as
   `Constant` nodes instead of ONNX initializers. With ONNX Runtime 1.29 its own pre-processing left them alone,
   the weights were not quantised, and the "INT8" model came out larger, slower and wrong. ONNX Runtime 1.31
   converts them during pre-processing and the problem disappears. `quantize` checks the model that is actually
   quantised (after pre-processing) and warns about large `Constant` tensors, unfolded `BatchNormalization`
   nodes, and an output file that is hardly smaller than its input.
3. **Old operator sets and per-channel weights.** Per-channel INT8 needs operator set 13 or higher. On a real OCR
   model with operator set 12, ONNX Runtime 1.18 stops with an error, while 1.31 **writes a file it then cannot
   load** (found only by the next command). `quantize` therefore picks per-tensor weights automatically for an
   old opset (and says so), and it loads every INT8 file it writes: if ONNX Runtime cannot, the command fails with
   the reason and a hint instead of printing "Wrote ...". `--per-channel` forces the old behaviour.
   The per-tensor fallback has a price that depends on the model. On MobileNetV2 from the ONNX Model Zoo
   (opset 12), top-1 agreement with FP32 on 300 real pictures was 81.0 % per tensor and 87.7 % after converting
   the model to opset 13 with `onnx.version_converter` and quantising per channel (ResNet-50: 93.3 % and 95.7 %).
   Depthwise convolutions, whose channels have very different weight ranges, suffer most from a shared scale.
4. **INT8 is not automatically faster, or good.** Speed-ups depend on the CPU (integer dot-product instructions),
   on how many operators actually become integer operators, and on the cost of the extra quantise/dequantise
   steps. In the development sandbox (a small cloud VM) the tiny demo classifier ran *slower* as INT8 than as
   FP32 (1.16 to 1.26 times the median latency in repeated runs), and for the real OCR text detector the INT8 and FP32 medians were
   within noise of each other (the `compare` verdict said so and refused to call it a speed-up). On top of that,
   that INT8 detector, calibrated naively on a dozen photos without any text, produced all-zero output on some
   images: its cosine similarity to FP32 averaged 0.01 over 13 real pictures. That is exactly why the harness
   exists: measure speed and agreement instead of assuming either.
5. **`--method entropy` is MinMax.** ONNX Runtime's `quantize_static` runs its entropy calibrator with 128 histogram
   bins and 128 quantised bins and has no option to change either, so the KL-divergence search
   (`range(num_quantized_bins // 2, num_bins // 2 + 1)`) has exactly one candidate threshold: the full min/max range.
   On MobileNetV2 with ONNX Runtime 1.31 the two methods wrote byte-identical files, the entropy run taking 282 s
   instead of 15 s. A real search with 2048 bins (ranges passed through `TensorQuantOverrides`) was tried and gave
   *lower* agreement than MinMax on that model (83.7 % vs 87.7 %), so `quantize` warns instead of re-implementing
   the calibrator, and `test_entropy_calibration_is_diagnosed_as_identical_to_minmax` fails if a future ONNX Runtime
   makes the two outputs differ.

## 6. Power and energy

| Source | How | What it covers |
|---|---|---|
| `rapl` (automatic, Linux) | Intel/AMD energy counters in `/sys/class/powercap/intel-rapl:*`; the counter wrap-around is handled | **CPU package only.** Not the whole machine, not a GPU, not an accelerator card. |
| `manual` | `--power-watts` (and optionally `--idle-watts`) from a wall-socket meter or smart plug | Whatever your meter covers; the tool only does the arithmetic and labels the source as user-supplied. |

* Reading RAPL usually needs root, or `sudo chmod -R a+r /sys/class/powercap/intel-rapl*`. If the counters exist but
  cannot be read, the tool says so instead of silently reporting nothing.
* Before the run, RAPL idle power is measured for `--idle-seconds` (default 2 s) while the process sleeps. Reported:
  total power, **net** power (total minus idle), energy per item, and items/s per watt both ways. The net figure is
  only as good as the idle measurement; two seconds on a machine with background activity is a rough baseline.
* If the average power is not above idle, net values are not reported, with a note.
* On macOS, Windows and most VMs there is no RAPL. Use `--power-watts`.
* **The RAPL reading code is tested against a fake `/sys` tree only** (counter wrap-around, permissions, missing
  domains). It has not been run against real RAPL hardware.

## 7. What is stored, and reproducibility

Every run writes one self-describing JSON file (`schema_version` 1.0) with: tool version, run id, the model
(file name, **SHA-256**, size, precision, opset, node count, operators, input/output shapes), the backend and its
options, the complete configuration, the environment (CPU model, cores, memory, OS, Python and package versions,
CPU governor, container, battery, CPU load at start), latency and throughput, per-stage results, streams, power,
accuracy, the warnings, and **the raw per-call latencies** (rounded to a nanosecond; thinned evenly only above
100 000 samples, in which case `truncated` is set). Anyone can recompute the statistics instead of trusting the
summary.

Deliberately **not** stored: host name, user name, folders, the Python executable path. Only the model's file name
is kept. Still, look through a results file before you publish it.

Runs are seeded (`--seed`) for the synthetic data and the bootstrap, so the same file gives the same statistics
every time. The timing itself is of course not reproducible to the nanosecond.

## 8. Checking the harness itself

* `infer-bench selfcheck` benchmarks a fake model that busy-waits for exactly 1 ms and 5 ms and checks that the
  harness reports that latency within a small tolerance. If this fails on your machine, nothing else it measures
  can be trusted there.
* The test-suite includes the same kind of test (`test_the_harness_measures_a_known_latency_accurately`) and
  tests for: percentile and bootstrap maths, drift and outlier detection, GC state handling, multi-stream
  fairness and failure handling (a failing stream raises instead of hanging), RAPL counter wrap-around, results
  validation, quantisation of the demo models, and the dashboard.
* Eight tests depend on wall-clock timing and are marked `timing`; deselect them on a busy machine with
  `pytest -m "not timing"`.

## 9. What this tool does not do

* No vendor-accelerator support. The only real backend is ONNX Runtime. Another runtime (an accelerator vendor's
  own SDK, for example) means writing one `Backend` subclass (see `infer_bench/backends/base.py`); no such
  subclass exists in this repository.
* Execution providers other than the CPU are passed through to ONNX Runtime but were **not tested**. An
  unavailable provider is refused instead of silently falling back to the CPU. Host-to-device copies are inside
  the timed call.
* No energy measurement of accelerators or whole systems without an external meter.
* No labelled-dataset accuracy (mAP, top-1 on ImageNet).
* Tested on Linux and Windows 11, not on macOS (the RAPL tests are skipped on Windows, which has no RAPL). Console
  output is plain ASCII on purpose, for Windows code pages.
* Latency here is the latency of a call into the runtime from Python. It is not end-to-end latency of a camera
  pipeline.
