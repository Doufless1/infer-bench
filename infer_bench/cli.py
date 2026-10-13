"""Command-line interface: ``infer-bench <command> ...``

Commands
    run         benchmark a model and save a results file
    quantize    INT8 post-training quantisation
    accuracy    agreement of a candidate model with a reference model
    compare     compare two results files
    summary     table of all results in a folder
    info        describe a model file
    env         show the machine/software environment and power-meter status
    selfcheck   check the harness' own timing against a backend with a known latency
    demo-model  write a small demo ONNX model
    dashboard   open the results dashboard in a browser

Output is plain ASCII on purpose, so it prints correctly in any terminal.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .accuracy import TASKS as ACCURACY_TASKS
from .accuracy import compare_models
from .backends import BACKENDS, create_backend
from .backends.base import Backend, TensorSpec
from .compare import compare_results
from .demo_model import KINDS as DEMO_KINDS
from .demo_model import build_demo_model
from .env import capture_environment, noise_warnings
from .inputs import (
    image_feeds,
    load_images,
    model_image_size,
    parse_shape_override,
    parse_size,
    synthetic_feeds,
)
from .modelinfo import describe_model
from .power import RaplMeter
from .preprocess import PreprocessConfig, list_images
from .quantize import quantize_int8
from .results import (
    default_output_path,
    flatten_result,
    iter_results,
    load_result,
    save_result,
)
from .runner import MODES, TASKS, RunConfig, run_benchmark


# ----------------------------------------------------------------- formatting

def _num(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    rule = "  ".join("-" * widths[i] for i in range(len(headers)))
    body = ["  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) for row in rows]
    return "\n".join([line, rule, *body])


def format_run_summary(result: dict[str, Any]) -> str:
    lat, thr, cfg = result["latency"], result["throughput"], result["config"]
    model, backend = result["model"], result["backend"]
    lo, hi = lat["median_ci95_ms"]
    drift = lat["drift"]
    lines = [
        f"Run:        {result['label']}",
        f"Model:      {model.get('file', '-')} ({model.get('precision', '?')})   "
        f"Backend: {backend.get('name')} {backend.get('version', '')} {backend.get('active_providers', '')}".rstrip(),
        f"Setup:      mode={cfg['mode']} batch={cfg['batch_size']} streams={cfg['streams']} "
        f"warmup={cfg['warmup']} samples={lat['n']} inputs={cfg.get('input_source')}",
        f"Latency:    p50 {lat['p50_ms']:.3f} ms (95% CI {lo:.3f}-{hi:.3f})   p90 {lat['p90_ms']:.3f}   "
        f"p95 {lat['p95_ms']:.3f}   p99 {lat['p99_ms']:.3f}   max {lat['max_ms']:.3f}",
        f"            mean {lat['mean_ms']:.3f} +/- {lat['std_ms']:.3f} ms   "
        f"outliers {lat['outliers']['fraction'] * 100:.1f}%   "
        + (
            f"drift {drift['relative_change'] * 100:+.1f}% ({'stable' if drift['stable'] else 'DRIFTING'})"
            if drift.get("checked")
            else "drift not checked"
        ),
        f"Throughput: {thr['items_per_second']:.1f} items/s   ({thr['n_calls']} calls in {thr['wall_time_s']:.2f} s)",
    ]
    if result.get("stages"):
        st = result["stages"]
        lines.append(
            "Stages:     p50 ms  preprocess {:.3f}   inference {:.3f}   postprocess {:.3f}".format(
                st["preprocess"]["p50_ms"], st["inference"]["p50_ms"], st["postprocess"]["p50_ms"]
            )
        )
    if result.get("streams"):
        per = ", ".join(f"{v:.1f}" for v in result["streams"]["per_stream_items_per_second"])
        lines.append(f"Streams:    {result['streams']['n_streams']} streams, items/s per stream: {per}")
    power = result.get("power") or {}
    if power.get("avg_power_w") is not None:
        text = f"Power:      {power['avg_power_w']:.2f} W ({power['source']}, {power['scope']})"
        if power.get("items_per_s_per_w") is not None:
            text += f"   {power['items_per_s_per_w']:.2f} items/s per W   {_num(power.get('energy_per_item_mj'), 3)} mJ/item"
        if power.get("net_power_w") is not None:
            text += f"   net of idle: {power['net_power_w']:.2f} W"
        lines.append(text)
    elif power.get("reason"):
        lines.append(f"Power:      not measured ({power['reason']})")
    acc = result.get("accuracy")
    if acc:
        parts = [f"cosine {_num(acc.get('cosine_similarity_mean'), 5)}"]
        if acc.get("top1_agreement") is not None:
            parts.append(f"top-1 agreement {acc['top1_agreement'] * 100:.1f}%")
        if acc.get("detection"):
            parts.append(f"detection F1@0.5 {_num(acc['detection'].get('f1_iou50'), 3)}")
        lines.append(f"Accuracy:   vs reference ({acc.get('n_compared')} samples, {acc.get('input_source')}): " + "   ".join(parts))
    if result.get("warnings"):
        lines.append("Warnings:")
        lines.extend(f"  - {w}" for w in result["warnings"])
    return "\n".join(lines)


# ------------------------------------------------------------- shared helpers

def _add_input_arguments(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("inputs")
    g.add_argument("--images", metavar="DIR", help="folder of real images (jpg/png/bmp/webp); default: synthetic inputs")
    g.add_argument("--max-images", type=int, default=64, help="use at most this many images (default 64)")
    g.add_argument("--input-size", metavar="HxW", help="height x width for models with a dynamic image size, e.g. 640x640")
    g.add_argument("--input-shape", action="append", default=[], metavar="NAME=1,3,H,W", help="full shape for a dynamic input (repeatable)")
    g.add_argument("--preprocess", choices=("letterbox", "resize"), help="image resizing (default: letterbox for detection, else resize)")
    g.add_argument("--scale", type=float, default=1.0 / 255.0, help="multiply pixels by this (default 1/255)")
    g.add_argument("--mean", default="0,0,0", help="per-channel mean after scaling, e.g. 0.485,0.456,0.406")
    g.add_argument("--std", default="1,1,1", help="per-channel std after scaling, e.g. 0.229,0.224,0.225")
    g.add_argument("--seed", type=int, default=0, help="random seed for synthetic data and bootstrap (default 0)")


def _triple(text: str) -> tuple[float, float, float]:
    parts = [float(x) for x in text.split(",")]
    if len(parts) != 3:
        raise ValueError(f"expected three comma-separated numbers, got '{text}'")
    return parts[0], parts[1], parts[2]


def _input_size(args: argparse.Namespace) -> tuple[int, int] | None:
    return parse_size(args.input_size) if args.input_size else None


def _shape_overrides(args: argparse.Namespace) -> dict[str, tuple[int, ...]]:
    return dict(parse_shape_override(t) for t in args.input_shape)


def _make_preprocess(args: argparse.Namespace, spec: TensorSpec, task: str) -> PreprocessConfig:
    height, width = model_image_size(spec, _input_size(args))
    mode = args.preprocess or ("letterbox" if task == "detection" else "resize")
    return PreprocessConfig(
        height=height, width=width, mode=mode, scale=args.scale,
        mean=_triple(args.mean), std=_triple(args.std),
    )


def _load_image_arrays(args: argparse.Namespace, limit: int | None = None) -> list[np.ndarray] | None:
    if not args.images:
        return None
    return load_images(list_images(args.images, limit=limit or args.max_images, seed=args.seed))


def _make_feeds(
    backend: Backend, args: argparse.Namespace, count: int, task: str
) -> tuple[list[dict[str, np.ndarray]], str]:
    """Batch-1 feeds for calibration or accuracy: real images if given, else synthetic."""
    specs = backend.inputs
    if args.images:
        if len(specs) != 1:
            raise ValueError("--images needs a model with exactly one input")
        images = load_images(list_images(args.images, limit=count, seed=args.seed))
        return image_feeds(images, specs[0], _make_preprocess(args, specs[0], task), 1), "images"
    feeds = synthetic_feeds(
        specs, 1, count=count, seed=args.seed, overrides=_shape_overrides(args), image_size=_input_size(args)
    )
    return feeds, "synthetic"


def _ort_options(args: argparse.Namespace, streams: int = 1) -> dict[str, Any]:
    intra = getattr(args, "threads", 0) or 0
    if not intra and streams > 1:
        # Several streams each running a multi-threaded model would fight for the cores.
        intra = max(1, (os.cpu_count() or 1) // streams)
    return {
        "providers": [p.strip() for p in args.providers.split(",") if p.strip()],
        "intra_op_threads": intra,
        "inter_op_threads": getattr(args, "inter_threads", 0) or 0,
        "graph_optimization": args.graph_opt,
    }


def _add_backend_arguments(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("backend (onnxruntime)")
    g.add_argument("--providers", default="CPUExecutionProvider", help="comma-separated execution providers (default CPUExecutionProvider)")
    g.add_argument("--threads", type=int, default=0, help="intra-op threads (0 = ONNX Runtime default)")
    g.add_argument("--inter-threads", type=int, default=0, help="inter-op threads (0 = default)")
    g.add_argument("--graph-opt", choices=("disable", "basic", "extended", "all"), default="all", help="graph optimisation level (default all)")


def _open_backend(name: str, model: str | None, args: argparse.Namespace, streams: int = 1) -> Backend:
    if name == "sleep":
        backend = create_backend("sleep", latency_ms=args.sleep_ms, spin=not args.sleep_yield)
        backend.load(None)
        return backend
    if not model:
        raise ValueError("a model file is required for this backend")
    if not Path(model).is_file():
        raise FileNotFoundError(f"model file not found: {model}")
    backend = create_backend(name, **_ort_options(args, streams))
    backend.load(model)
    return backend


# ----------------------------------------------------------------------- run

def cmd_run(args: argparse.Namespace) -> int:
    backend = _open_backend(args.backend, args.model, args, args.streams)
    model_info = describe_model(args.model) if args.backend != "sleep" else {"file": "(none)", "precision": "n/a"}
    images = _load_image_arrays(args)
    input_size = _input_size(args)

    pre_cfg = None
    if images or args.mode == "pipeline":
        if args.backend == "sleep":
            pre_cfg = PreprocessConfig(32, 32, mode=args.preprocess or "resize")
        else:
            pre_cfg = _make_preprocess(args, backend.inputs[0], args.task)

    cfg = RunConfig(
        mode=args.mode, task=args.task, batch_size=args.batch, warmup=args.warmup,
        iterations=args.iterations, duration_s=args.duration, streams=args.streams, seed=args.seed,
        input_size=input_size, shape_overrides=_shape_overrides(args), preprocess=pre_cfg,
        source_size=parse_size(args.source_size), conf_threshold=args.conf, iou_threshold=args.iou,
        measure_power=not args.no_power, power_watts=args.power_watts, idle_watts=args.idle_watts,
        idle_seconds=args.idle_seconds,
    )
    cfg.validate()

    env = capture_environment()
    meter = RaplMeter() if (not args.no_power and args.power_watts is None) else None
    label = args.label or "-".join(
        [Path(args.model).stem if args.model else "sleep", model_info.get("precision", "n/a"), args.mode,
         f"b{args.batch}"] + ([f"s{args.streams}"] if args.streams > 1 else [])
    )

    print(f"Benchmarking '{label}' ...", flush=True)
    result = run_benchmark(backend, cfg, label=label, model_info=model_info, images=images, meter=meter, env=env)

    if args.reference:
        print("Comparing with the reference model ...", flush=True)
        ref = _open_backend("onnxruntime", args.reference, args)
        task = args.accuracy_task
        if task == "auto" and args.task in ("classification", "detection"):
            task = args.task  # the pipeline task says what kind of model this is
        feeds, source = _make_feeds(backend, args, args.accuracy_samples, "detection" if task == "detection" else "none")
        acc = compare_models(
            ref, backend, feeds, task=task, conf_threshold=args.conf, iou_threshold=args.iou, input_source=source
        )
        acc["reference"] = describe_model(args.reference)
        result["accuracy"] = acc

    print()
    print(format_run_summary(result))
    if not args.no_save:
        out = Path(args.out) if args.out else default_output_path(args.results_dir, label, result["run_id"])
        save_result(result, out)
        print(f"\nSaved: {out}")
    return 0


# ------------------------------------------------------------------ quantize

def cmd_quantize(args: argparse.Namespace) -> int:
    backend = _open_backend("onnxruntime", args.model, args)
    feeds, source = _make_feeds(backend, args, args.samples, args.task)
    print(f"Calibrating on {len(feeds)} {'synthetic' if source == 'synthetic' else 'real image'} sample(s) ...", flush=True)
    t0 = time.perf_counter()
    summary = quantize_int8(
        args.model, args.output, feeds,
        per_channel=args.per_channel, activation_type=args.activation_type, weight_type=args.weight_type,
        method=args.method, reduce_range=args.reduce_range, pre_process=not args.no_preprocess,
        op_types=[t.strip() for t in args.op_types.split(",")] if args.op_types else None,
        nodes_to_exclude=[n.strip() for n in args.exclude_nodes.split(",")] if args.exclude_nodes else None,
    )
    took = time.perf_counter() - t0
    shrink = summary["input_size_bytes"] / max(1, summary["output_size_bytes"])
    print(f"Wrote {summary['output']}")
    print(f"  size {summary['input_size_bytes'] / 1e6:.2f} MB -> {summary['output_size_bytes'] / 1e6:.2f} MB ({shrink:.1f}x smaller), {took:.1f} s")
    print(f"  {summary['format']} activations={summary['activation_type']} weights={summary['weight_type']} "
          f"per_channel={summary['per_channel']} method={summary['calibration_method']}")
    print(f"  {summary['pre_processing']}")
    if source == "synthetic":
        print("  note: calibrated on synthetic inputs; for a real model use --images with representative pictures")
    for note in summary["notes"]:
        print(f"  note: {note}")
    for warning in summary["warnings"]:
        print(f"  WARNING: {warning}")
    print("Next: check what it cost in accuracy with:  infer-bench accuracy", args.model, args.output)
    return 0


# ------------------------------------------------------------------ accuracy

def cmd_accuracy(args: argparse.Namespace) -> int:
    ref = _open_backend("onnxruntime", args.reference, args)
    cand = _open_backend("onnxruntime", args.candidate, args)
    feeds, source = _make_feeds(cand, args, args.samples, "detection" if args.task == "detection" else "none")
    acc = compare_models(ref, cand, feeds, task=args.task, conf_threshold=args.conf, iou_threshold=args.iou, input_source=source)
    acc["reference"] = describe_model(args.reference)
    print(f"Candidate: {args.candidate}\nReference: {args.reference}\nSamples: {acc['n_compared']} ({source}), task: {acc['task']}")
    print(f"  cosine similarity   mean {_num(acc['cosine_similarity_mean'], 6)}   min {_num(acc['cosine_similarity_min'], 6)}")
    print(f"  relative L2 error   mean {_num(acc['relative_l2_error_mean'], 5)}   max {_num(acc['relative_l2_error_max'], 5)}")
    if acc["top1_agreement"] is not None:
        print(f"  top-1 agreement     {acc['top1_agreement'] * 100:.1f}%   reference top-1 inside candidate top-5: {acc['ref_top1_in_cand_top5'] * 100:.1f}%")
    det = acc.get("detection")
    if det:
        print(f"  detection agreement (reference = ground truth, IoU>=0.5): precision {_num(det['precision'], 3)}  "
              f"recall {_num(det['recall'], 3)}  F1 {_num(det['f1_iou50'], 3)}  "
              f"[{det['reference_detections']} reference / {det['candidate_detections']} candidate boxes]")
    for note in acc["notes"]:
        print(f"  note: {note}")
    if args.into:
        result = load_result(args.into)
        result["accuracy"] = acc
        save_result(result, args.into)
        print(f"Stored in {args.into}")
    return 0


# ------------------------------------------------------- compare / summary

def cmd_compare(args: argparse.Namespace) -> int:
    base, other = load_result(args.base), load_result(args.other)
    cmp = compare_results(base, other)
    print(f"Base : {cmp['base']}\nOther: {cmp['other']}   (ratios are other / base)")
    ratio = cmp["throughput_ratio"]
    print(f"Throughput ratio: {_num(ratio, 3)}x   ({_num(base['throughput']['items_per_second'], 1)} -> {_num(other['throughput']['items_per_second'], 1)} items/s)")
    lr = cmp["latency_ratio"]
    print("Latency ratio:    " + "   ".join(f"{q} {_num(v, 3)}" for q, v in lr.items()) + "   (below 1 = faster)")
    print(f"Verdict: {cmp['verdict']}")
    if cmp["power"]:
        print(f"Efficiency ratio (items/s per W): {_num(cmp['power']['items_per_s_per_w_ratio'], 3)}x  "
              f"(power {cmp['power']['avg_power_w'][0]:.2f} W -> {cmp['power']['avg_power_w'][1]:.2f} W; sources {cmp['power']['sources']})")
    if cmp["accuracy"]:
        a = cmp["accuracy"]
        print(f"Accuracy of '{cmp['other']}' vs its reference: cosine {_num(a['cosine_similarity_mean'], 5)}, "
              f"top-1 agreement {_num(a['top1_agreement'], 3)}, detection F1 {_num(a['detection_f1'], 3)}")
    if cmp["warnings"]:
        print("Warnings:")
        for w in cmp["warnings"]:
            print(f"  - {w}")
    return 0


def _collect_rows(paths: list[str], sort: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for p in paths or ["results"]:
        path = Path(p)
        if path.is_dir():
            for file, result, error in iter_results(path):
                if result is None:
                    print(f"skipping {file}: {error}", file=sys.stderr)
                else:
                    rows.append(flatten_result(result, file))
        else:
            rows.append(flatten_result(load_result(path), path))
    keys = {"fps": lambda r: -(r["fps"] or 0), "p50": lambda r: r["p50_ms"] or 0, "label": lambda r: r["label"] or "",
            "time": lambda r: r["created_utc"] or ""}
    rows.sort(key=keys[sort])
    return rows


_SUMMARY_COLUMNS = (
    ("label", "run"), ("precision", "prec"), ("mode", "mode"), ("batch", "b"), ("streams", "str"),
    ("fps", "items/s"), ("p50_ms", "p50 ms"), ("p95_ms", "p95 ms"), ("p99_ms", "p99 ms"),
    ("avg_power_w", "W"), ("fps_per_w", "items/s/W"), ("top1_agree", "top1 agr"), ("cosine", "cosine"), ("det_f1", "det F1"),
)


def cmd_summary(args: argparse.Namespace) -> int:
    rows = _collect_rows(args.paths, args.sort)
    if not rows:
        print("no results found")
        return 1
    digits = {"fps": 1, "p50_ms": 3, "p95_ms": 3, "p99_ms": 3, "avg_power_w": 2, "fps_per_w": 2,
              "top1_agree": 3, "cosine": 5, "det_f1": 3}
    cells = [[_num(r[key], digits.get(key, 2)) for key, _ in _SUMMARY_COLUMNS] for r in rows]
    headers = [h for _, h in _SUMMARY_COLUMNS]
    if args.format == "csv":
        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        writer.writerow([k for k in rows[0].keys()])
        for r in rows:
            writer.writerow(["" if v is None else v for v in r.values()])
        print(buf.getvalue(), end="")
    elif args.format == "markdown":
        print("| " + " | ".join(headers) + " |")
        print("|" + "|".join("---" for _ in headers) + "|")
        for row in cells:
            print("| " + " | ".join(row) + " |")
    else:
        print(format_table(headers, cells))
    return 0


# ------------------------------------------------------- info / env / misc

def cmd_info(args: argparse.Namespace) -> int:
    info = describe_model(args.model)
    backend = _open_backend("onnxruntime", args.model, args)
    print(f"{info['file']}  {info['size_bytes'] / 1e6:.2f} MB  precision={info['precision']}  opset={info['opset']}  "
          f"nodes={info['n_nodes']}  parameters={info['n_parameters']:,}")
    print(f"sha256 {info['sha256']}")
    for s in backend.inputs:
        print(f"input   {s.name}: {list(s.shape)} {s.dtype}")
    for s in backend.outputs:
        print(f"output  {s.name}: {list(s.shape)} {s.dtype}")
    print("top operators: " + ", ".join(f"{k} x{v}" for k, v in info["top_operators"].items()))
    return 0


def cmd_env(args: argparse.Namespace) -> int:
    env = capture_environment()
    print(json.dumps(env, indent=2))
    meter = RaplMeter()
    print()
    if meter.available:
        print(f"Power: RAPL package counters readable: {[d.name for d in meter.domains]} (CPU package only)")
    else:
        print(f"Power: RAPL not available ({meter.reason_unavailable}). Use --power-watts with a wall-meter reading.")
    warnings = noise_warnings(env)
    if warnings:
        print("\nNoise warnings:")
        for w in warnings:
            print(f"  - {w}")
    return 0


def _timer_overhead_ns(samples: int = 20000) -> tuple[float, int]:
    """Median cost of one clock read, and the smallest non-zero step of the clock."""
    clock = time.perf_counter_ns
    deltas = np.empty(samples, dtype=np.int64)
    for i in range(samples):
        a = clock()
        b = clock()
        deltas[i] = b - a
    nonzero = deltas[deltas > 0]
    return float(np.median(deltas)), int(nonzero.min()) if nonzero.size else 0


def cmd_selfcheck(args: argparse.Namespace) -> int:
    median_ns, step_ns = _timer_overhead_ns()
    print(f"Timer: perf_counter_ns call costs ~{median_ns:.0f} ns, smallest clock step {step_ns} ns")
    ok = True
    for expected in (1.0, 5.0):
        backend = create_backend("sleep", latency_ms=expected, spin=True)
        backend.load(None)
        cfg = RunConfig(iterations=args.iterations, warmup=20, measure_power=False, bootstrap_resamples=200)
        result = run_benchmark(backend, cfg, label="selfcheck", env={"timestamp_utc": None})
        p50 = result["latency"]["p50_ms"]
        error = p50 - expected
        tolerance = max(0.02 * expected, 0.05)
        passed = abs(error) <= tolerance
        ok = ok and passed
        print(f"Fake model with a known {expected:.1f} ms latency: measured p50 {p50:.4f} ms "
              f"(error {error:+.4f} ms, allowed +/-{tolerance:.3f})  {'PASS' if passed else 'FAIL'}")
    print("The harness timing is trustworthy on this machine." if ok else
          "The harness timing is NOT accurate here (heavy background load or a very coarse clock).")
    return 0 if ok else 1


def cmd_demo_model(args: argparse.Namespace) -> int:
    path = build_demo_model(args.kind, args.output)
    print(f"Wrote {path} ({Path(path).stat().st_size / 1e3:.0f} kB). Random weights: for trying the tool, not for real results.")
    if args.kind == "detector":
        print("Detector-style output (batch, 4+classes, anchors): use  --task detection  in pipeline mode / accuracy.")
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    try:
        from .dashboard import create_app
    except ImportError as exc:
        raise RuntimeError(
            "the dashboard needs Dash and Plotly: pip install dash plotly  (or  pip install \"infer-bench[dashboard]\")"
        ) from exc
    app = create_app(args.results)
    print(f"Dashboard for '{args.results}' at http://{args.host}:{args.port}  (Ctrl+C to stop)")
    app.run(host=args.host, port=args.port, debug=args.dash_debug)
    return 0


# --------------------------------------------------------------------- parser

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="infer-bench",
        description="Benchmark inference: latency percentiles, throughput, accuracy vs a reference, power.",
    )
    parser.add_argument("--version", action="version", version=f"infer-bench {__version__}")
    parser.add_argument("--debug", action="store_true", help="show full tracebacks on errors")
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")

    # run
    p = sub.add_parser("run", help="benchmark a model", description="Benchmark a model and save a results file.")
    p.add_argument("model", nargs="?", help="path to an .onnx model (not needed for --backend sleep)")
    p.add_argument("--backend", choices=BACKENDS, default="onnxruntime")
    p.add_argument("--label", help="name of this run (default: derived from the model and settings)")
    p.add_argument("--mode", choices=MODES, default="model", help="model = time the model call only; pipeline = pre + model + post")
    p.add_argument("--task", choices=TASKS, default="none", help="pipeline post-processing: classification or detection (YOLO-style)")
    p.add_argument("--batch", type=int, default=1, help="batch size (default 1)")
    p.add_argument("--warmup", type=int, default=50, help="discarded warm-up calls (default 50)")
    p.add_argument("--iterations", type=int, default=500, help="timed calls (default 500; use 1000+ for a stable p99)")
    p.add_argument("--duration", type=float, help="run for this many seconds instead of a fixed number of calls")
    p.add_argument("--streams", type=int, default=1, help="concurrent streams (threads); default 1")
    p.add_argument("--conf", type=float, default=0.25, help="detection confidence threshold (default 0.25)")
    p.add_argument("--iou", type=float, default=0.45, help="detection NMS IoU threshold (default 0.45)")
    p.add_argument("--source-size", default="480x640", metavar="HxW", help="size of synthetic source images in pipeline mode")
    _add_backend_arguments(p)
    _add_input_arguments(p)
    g = p.add_argument_group("power")
    g.add_argument("--no-power", action="store_true", help="do not try to measure power")
    g.add_argument("--power-watts", type=float, help="average power in watts you measured yourself (wall meter)")
    g.add_argument("--idle-watts", type=float, help="idle power in watts you measured yourself")
    g.add_argument("--idle-seconds", type=float, default=2.0, help="seconds of RAPL idle baseline before the run (default 2, 0 = skip)")
    g = p.add_argument_group("accuracy")
    g.add_argument("--reference", metavar="REF.onnx", help="also measure agreement with this reference model (e.g. the FP32 original)")
    g.add_argument("--accuracy-samples", type=int, default=100)
    g.add_argument("--accuracy-task", choices=ACCURACY_TASKS, default="auto")
    g = p.add_argument_group("output")
    g.add_argument("--results-dir", default="results", help="where results files go (default ./results)")
    g.add_argument("--out", help="exact output file (overrides --results-dir)")
    g.add_argument("--no-save", action="store_true", help="print only, do not write a file")
    g = p.add_argument_group("sleep backend (for testing the harness)")
    g.add_argument("--sleep-ms", type=float, default=5.0, help="latency of the fake backend")
    g.add_argument("--sleep-yield", action="store_true", help="use time.sleep (releases the CPU) instead of busy-waiting")
    p.set_defaults(func=cmd_run)

    # quantize
    p = sub.add_parser("quantize", help="INT8 post-training quantisation", description="Quantise an FP32 ONNX model to INT8 (QDQ).")
    p.add_argument("model")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--samples", type=int, default=100, help="calibration samples (default 100)")
    p.add_argument("--task", choices=("none", "detection"), default="none", help="only affects the default image resizing")
    p.add_argument("--method", choices=("minmax", "entropy", "percentile"), default="minmax")
    p.add_argument("--activation-type", choices=("int8", "uint8"), default="uint8")
    p.add_argument("--weight-type", choices=("int8", "uint8"), default="int8")
    p.add_argument("--per-channel", action=argparse.BooleanOptionalAction, default=None,
                   help="per-channel weight scales (default: yes if the model's opset is 13 or higher, otherwise per tensor)")
    p.add_argument("--reduce-range", action="store_true", help="7-bit weights: avoids saturation on CPUs without VNNI")
    p.add_argument("--no-preprocess", action="store_true", help="skip ONNX Runtime's quantisation pre-processing")
    p.add_argument("--op-types", help="quantise only these operator types, e.g. Conv,Gemm (default: all supported)")
    p.add_argument("--exclude-nodes", help="comma-separated node names to leave in FP32")
    _add_backend_arguments(p)
    _add_input_arguments(p)
    p.set_defaults(func=cmd_quantize)

    # accuracy
    p = sub.add_parser("accuracy", help="agreement of a candidate with a reference model")
    p.add_argument("reference", help="the reference model (usually the FP32 original)")
    p.add_argument("candidate", help="the model to check (usually the INT8 version)")
    p.add_argument("--samples", type=int, default=100)
    p.add_argument("--task", choices=ACCURACY_TASKS, default="auto", help="classification, detection (YOLO-style) or none")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--into", metavar="RESULT.json", help="store the accuracy block in an existing results file")
    _add_backend_arguments(p)
    _add_input_arguments(p)
    p.set_defaults(func=cmd_accuracy)

    # compare
    p = sub.add_parser("compare", help="compare two results files")
    p.add_argument("base", help="results file of the baseline run")
    p.add_argument("other", help="results file to compare against the baseline")
    p.set_defaults(func=cmd_compare)

    # summary
    p = sub.add_parser("summary", help="table of results")
    p.add_argument("paths", nargs="*", help="results files or folders (default ./results)")
    p.add_argument("--format", choices=("table", "markdown", "csv"), default="table")
    p.add_argument("--sort", choices=("fps", "p50", "label", "time"), default="time")
    p.set_defaults(func=cmd_summary)

    # info
    p = sub.add_parser("info", help="describe a model file")
    p.add_argument("model")
    _add_backend_arguments(p)
    p.set_defaults(func=cmd_info)

    # env
    p = sub.add_parser("env", help="show environment and power-meter status")
    p.set_defaults(func=cmd_env)

    # selfcheck
    p = sub.add_parser("selfcheck", help="check the harness timing against a fake model of known latency")
    p.add_argument("--iterations", type=int, default=300)
    p.set_defaults(func=cmd_selfcheck)

    # demo-model
    p = sub.add_parser("demo-model", help="write a small demo ONNX model (random weights)")
    p.add_argument("kind", choices=tuple(DEMO_KINDS))
    p.add_argument("-o", "--output", required=True)
    p.set_defaults(func=cmd_demo_model)

    # dashboard
    p = sub.add_parser("dashboard", help="open the results dashboard")
    p.add_argument("--results", default="results", help="folder with results files (default ./results)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8050)
    p.add_argument("--dash-debug", action="store_true", help="Dash debug mode (auto-reload, dev tools)")
    p.set_defaults(func=cmd_dashboard)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - turn any failure into a one-line message
        if args.debug:
            raise
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
