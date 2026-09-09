# Running the final PASLCD evaluation (v10_no_dino) on another machine

The shipped configuration is `configs/ablate_v10_no_dino.yaml`: every section
except `three_image_comparison` is identical to `configs/pipeline.yaml`; the
detection stage adds SAM3 text-prompt ceiling/sky suppression, depth-ordered
occlusion merging, and disables DINOv2 (`use_dino_features: false`, neither
computed nor consulted). DI²FIX refinement is on (`refine.enabled: true`).

## 1. Install / update the code

The shipped solution lives on the `v10_final` branch of
https://github.com/pullover00/corgi (its `master` carries an older, unrelated
history of the evaluator and was left untouched).

```bash
git clone -b v10_final https://github.com/pullover00/corgi.git change_pipeline && cd change_pipeline
# or, in an existing clone:  git fetch origin v10_final && git checkout v10_final
```

Then follow `SETUP.md` once per machine for the three conda envs
(`vggt-omega`, `difix3d`, `goldilocs`), the checkpoints and `data/PASLCD`.
Quick verification that everything the final run needs is present:

```bash
conda env list | grep -E "vggt-omega|difix3d|goldilocs"
ls data/PASLCD | wc -l                                # 10 datasets
ls ~/.cache/huggingface/hub/models--facebook--sam3/snapshots/*/sam3.pt
ls ~/sam3 | head -2                                   # SAM3_SOURCE
grep -n "vos_optimized" configs/ablate_v10_no_dino.yaml
```

The config carries this machine's absolute paths (VGGT-Omega checkpoint/root,
MASt3R root, DI²FIX root) and the SAM2 `vos_optimized` flag. On another
machine, port exactly those keys from a config that already works there —
nothing else is touched, and every changed key is printed and written as a
header comment into the output:

```bash
python scripts/make_machine_config.py --base configs/ablate_v10_no_dino.yaml \
    --machine configs/pipeline_remote.yaml --out configs/ablate_v10_no_dino_remote.yaml
```

`sam2.vos_optimized` must be `false` on Blackwell (sm_120) GPUs — see the
comment in `scripts/detect_batch.py`. Changing it changes the SAM2 predictor
class and shifts PASLCD numbers by about 0.03 mIoU (measured on 30 queries
between the laptop with `true` and the office PC with `false`), so a paper
table must not mix machines with different settings. The ported config's
path is recorded in `provenance/command.txt` by the launcher.

## 2. Smoke test (one query, ~5 min)

```bash
conda activate vggt-omega
export SAM3_SOURCE=~/sam3
export SAM3_IMAGE_CHECKPOINT=$(ls ~/.cache/huggingface/hub/models--facebook--sam3/snapshots/*/sam3.pt | head -1)
python scripts/run_paslcd_scene.py --dataset Cantina --instance Instance_1 --limit 1 \
    --output-root results/paslcd_final_smoke --config configs/ablate_v10_no_dino.yaml
```

Expect `results/paslcd_final_smoke/Cantina_Instance_1/intermediate/<stem>/{reconstruction,refined,detect}`
and one row in `metrics.csv`. `inference.json` must show
`"use_dino_features": false` under `settings`.

## 3. Launch the full benchmark (500 queries; ~35-40 h on an RTX 4090 laptop)

```bash
tmux new -d -s paslcd_final 'bash scripts/run_paslcd_final_v10.sh results/paslcd_final_v10_no_dino configs/ablate_v10_no_dino_remote.yaml'
tail -f results/paslcd_final_v10_no_dino/launcher.log
```

(On the authoring laptop the second argument is simply `configs/ablate_v10_no_dino.yaml`.)

The script is resumable — rerun the same command after any interruption. It
writes `results/paslcd_final_v10_no_dino/{provenance/,reference_scenes/,
<scene>/metrics.csv,run_logs/}` and finishes with
`PASLCD_FINAL_V10_NO_DINO.md` + `final_summary.json` from
`scripts/summarize_paslcd_final.py`.

## 4. Report

`scripts/summarize_paslcd_final.py --root results/paslcd_final_v10_no_dino`
prints and writes: query totals, success/failure with reasons, mean-of-scene
and pooled mIoU/F1/precision/recall, summed TP/FP/FN pixels, ADDED/REMOVED/
MOVED/REPLACED decision counts, per-scene and per-query tables, runtime.
