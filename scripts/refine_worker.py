#!/usr/bin/env python3
"""Resident DI2FIX refine worker: load the model once, serve many runs.

Why: the refine stage of one demo run measured 161 s on 2026-09-09, of which
the single denoising step on two images is a few seconds -- the rest is
DifixPipeline.from_pretrained + .to("cuda") in a fresh process every run.
Keeping one process alive with the model loaded turns that stage into
seconds for every run after the first.

Protocol (plain files, no sockets -- trivially inspectable, survives either
side restarting):
  <dir>/heartbeat                     touched every second while alive
  <dir>/requests/<id>.json            {"render_t0","clean_render","image_t1",
                                       "config","output_dir"} (paths)
  <dir>/results/<id>.done | <id>.err  written when finished / on failure

The worker stays loaded for ONE config's refine section; a request with a
different refine section reloads the model. Run in the `difix3d` env:

  python scripts/refine_worker.py --dir results/tidy_demo/refine_worker
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--idle-exit-s", type=float, default=0.0,
                    help="exit after this long without requests (0 = never)")
    args = ap.parse_args()
    d = args.dir
    (d / "requests").mkdir(parents=True, exist_ok=True)
    (d / "results").mkdir(parents=True, exist_ok=True)
    (d / "pid").write_text(str(__import__("os").getpid()))

    # The heartbeat must be touched by its OWN thread, not by the request
    # loop: the first request loads the model (~33 s), and a loop that only
    # touches the heartbeat between requests goes silent for that whole time.
    # The client then declares the worker dead and runs the slow subprocess
    # while the worker is still working -- both then write the same output
    # files. Observed on the first live run, 2026-09-10.
    import threading

    def beat():
        while True:
            (d / "heartbeat").touch()
            time.sleep(1.0)

    threading.Thread(target=beat, daemon=True).start()

    from ocmask_pipeline.config import load_config
    from ocmask_pipeline.refine import load_refiner, refine_with

    pipe, loaded_for = None, None
    last_request = time.time()
    print(f"refine worker: waiting for requests in {d}", flush=True)
    while True:
        reqs = sorted((d / "requests").glob("*.json"))
        if not reqs:
            if args.idle_exit_s and time.time() - last_request > args.idle_exit_s:
                print("refine worker: idle, exiting", flush=True)
                return 0
            time.sleep(1.0)
            continue
        req = reqs[0]
        rid = req.stem
        try:
            r = json.loads(req.read_text())
            config = load_config(r["config"])
            key = json.dumps(config["refine"], sort_keys=True)
            if pipe is None or key != loaded_for:
                t0 = time.time()
                pipe, loaded_for = load_refiner(config), key
                print(f"refine worker: model loaded in {time.time() - t0:.0f}s", flush=True)
            imgs = [np.asarray(Image.open(r[k]).convert("RGB")) for k in ("render_t0", "clean_render", "image_t1")]
            t0 = time.time()
            fixed_t0, fixed_clean = refine_with(pipe, *imgs, config)
            out = Path(r["output_dir"]); out.mkdir(parents=True, exist_ok=True)
            Image.fromarray(fixed_t0).save(out / "render_t0.png")
            Image.fromarray(fixed_clean).save(out / "clean_render.png")
            (d / "results" / f"{rid}.done").write_text(json.dumps({"seconds": time.time() - t0}))
            print(f"refine worker: {rid} refined in {time.time() - t0:.1f}s -> {out}", flush=True)
        except Exception:
            (d / "results" / f"{rid}.err").write_text(traceback.format_exc())
            print(f"refine worker: {rid} FAILED\n{traceback.format_exc()}", flush=True)
        finally:
            req.unlink(missing_ok=True)
            last_request = time.time()


if __name__ == "__main__":
    sys.exit(main())
