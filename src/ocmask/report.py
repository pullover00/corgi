from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .cache import load_reconstruction
from .changesim import MetricAccumulator, load_manifest, normalize_target
from .geometry_artifacts import save_extended_geometry_artifacts
from .io import load_rgb, save_image, save_json
from .metrics import evaluate_arrays
from .ssim import colorize_ssim, compute_ssim_dissimilarity, heatmap_overlay
from .types import Label
from .visualization import COLORS, colorize, overlay


def _relative(path: Path, base: Path) -> str:
    """Return a browser-friendly path relative to the generated index."""
    return Path(__import__("os").path.relpath(path.resolve(), base.resolve())).as_posix()


def _error_map(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Visualize correct changes, false positives, misses, and wrong classes."""
    pred_changed = prediction != Label.UNCHANGED
    target_changed = target != Label.UNCHANGED
    correct_change = pred_changed & target_changed & (prediction == target)
    false_positive = pred_changed & ~target_changed
    false_negative = ~pred_changed & target_changed
    wrong_class = pred_changed & target_changed & (prediction != target)
    result = np.full((*prediction.shape, 3), (35, 40, 48), dtype=np.uint8)
    result[correct_change] = (42, 190, 105)
    result[false_positive] = (235, 70, 70)
    result[false_negative] = (55, 125, 235)
    result[wrong_class] = (240, 185, 45)
    return result


def _object_instances(artifact_dir: Path, shape: tuple[int, int]) -> np.ndarray:
    """Color every saved object mask distinctly to expose SAM2 proposals."""
    output = np.zeros((*shape, 3), dtype=np.uint8)
    objects_path = artifact_dir / "objects" / "objects.json"
    if not objects_path.exists():
        return output
    records = json.loads(objects_path.read_text(encoding="utf-8"))
    for index, record in enumerate(records):
        mask = np.asarray(Image.open(objects_path.parent / record["mask"])) > 0
        if mask.shape != shape:
            mask = np.asarray(
                Image.fromarray(mask).resize(shape[::-1], Image.Resampling.NEAREST)
            )
        # A deterministic high-contrast palette makes overlapping instances
        # visible without implying semantic meaning.
        color = (
            60 + (index * 83) % 196,
            60 + (index * 137) % 196,
            60 + (index * 191) % 196,
        )
        output[mask] = color
    return output


def _save_geometry_diagnostics(artifact_dir: Path, report_pair_dir: Path) -> None:
    """Render binary geometry masks stored in NPZ into browser-viewable PNGs."""
    geometry_path = artifact_dir / "geometry.npz"
    if not geometry_path.exists():
        return
    with np.load(geometry_path) as data:
        for key in ("keep0", "keep1", "coverage01", "clean_coverage"):
            if key in data:
                save_image(report_pair_dir / f"{key}.png", data[key].astype(np.uint8) * 255)


def _ensure_ssim_artifacts(artifact_dir: Path, reconstruction) -> None:
    """Backfill correct SSIM float and heatmap artifacts for older runs."""
    raw_path = artifact_dir / "ssim_dissimilarity.npy"
    config = json.loads((artifact_dir / "config.json").read_text(encoding="utf-8"))
    if raw_path.exists():
        dissimilarity = np.load(raw_path)
    else:
        reference = np.asarray(Image.open(artifact_dir / "render_0_to_1.png"))
        dissimilarity = compute_ssim_dissimilarity(
            reference, reconstruction.images[1], config["ssim"]
        )
        np.save(raw_path, dissimilarity)
    heatmap = colorize_ssim(dissimilarity)
    save_image(artifact_dir / "ssim_heatmap.png", heatmap)
    save_image(
        artifact_dir / "ssim_heatmap_overlay.png",
        heatmap_overlay(reconstruction.images[1], heatmap),
    )


def _metric_rows(metrics: dict) -> str:
    rows = []
    for name, values in metrics["multiclass"].items():
        rows.append(
            "<tr>"
            f"<td>{html.escape(name.title())}</td>"
            f"<td>{values['support']:,}</td>"
            f"<td>{values['precision']:.3f}</td>"
            f"<td>{values['recall']:.3f}</td>"
            f"<td>{values['f1']:.3f}</td>"
            f"<td>{values['iou']:.3f}</td>"
            "</tr>"
        )
    return "".join(rows)


def _table3(metrics: dict) -> str:
    """Format IoU percentages in the exact column layout of paper Table 3."""
    multi = metrics["multiclass"]
    binary = metrics["binary"]
    percent = lambda value: f"{100 * value:.1f}"
    return f"""
    <div class="table3">
      <h3>Table 3 protocol · IoU (%)</h3>
      <p>Computed over all evaluated pixels. Binary collapses every non-static prediction
      into Changed. Multiclass follows Added, Removed, Moved, Replaced, and Unchanged.</p>
      <div class="table-wrap"><table>
        <thead><tr><th rowspan="2">Method</th><th colspan="3">Binary</th><th colspan="6">Multiclass</th></tr>
        <tr><th>Changed</th><th>Unchanged</th><th>mIoU</th><th>Added</th><th>Removed</th>
        <th>Moved</th><th>Replaced</th><th>Unchanged</th><th>mIoU</th></tr></thead>
        <tbody><tr><td>Our reproduction</td>
        <td>{percent(binary['changed']['iou'])}</td>
        <td>{percent(binary['unchanged']['iou'])}</td>
        <td>{percent(metrics['binary_miou'])}</td>
        <td>{percent(multi['added']['iou'])}</td>
        <td>{percent(multi['removed']['iou'])}</td>
        <td>{percent(multi['moved']['iou'])}</td>
        <td>{percent(multi['replaced']['iou'])}</td>
        <td>{percent(multi['unchanged']['iou'])}</td>
        <td>{percent(metrics['multiclass_miou'])}</td></tr>
        <tr class="paper"><td>Paper · full ChangeSim</td><td>37.3</td><td>92.5</td><td>64.9</td>
        <td>25.4</td><td>20.9</td><td>7.7</td><td>21.4</td><td>92.5</td><td>33.6</td></tr>
        </tbody>
      </table></div>
      <p class="caveat">The paper row uses all 8,212 pairs. This report uses
      the selected smoke-test subset, so the comparison is diagnostic rather than statistically equivalent.</p>
    </div>
    """


def _table3_values(metrics: dict) -> dict:
    """Return the Table 3 IoU layout as percentages for external analysis."""
    return {
        "binary": {
            "changed": metrics["binary"]["changed"]["iou"] * 100,
            "unchanged": metrics["binary"]["unchanged"]["iou"] * 100,
            "miou": metrics["binary_miou"] * 100,
        },
        "multiclass": {
            **{
                name: metrics["multiclass"][name]["iou"] * 100
                for name in ("added", "removed", "moved", "replaced", "unchanged")
            },
            "miou": metrics["multiclass_miou"] * 100,
        },
        "paper_full_changesim": {
            "binary": {"changed": 37.3, "unchanged": 92.5, "miou": 64.9},
            "multiclass": {
                "added": 25.4,
                "removed": 20.9,
                "moved": 7.7,
                "replaced": 21.4,
                "unchanged": 92.5,
                "miou": 33.6,
            },
        },
    }


def _figure(src: str, title: str, caption: str) -> str:
    return (
        '<figure class="panel">'
        f'<a href="{html.escape(src)}"><img loading="lazy" src="{html.escape(src)}" alt="{html.escape(title)}"></a>'
        f"<figcaption><strong>{html.escape(title)}</strong><span>{html.escape(caption)}</span></figcaption>"
        "</figure>"
    )


def _build_metrics_only_report(
    evaluation_dir: Path, evaluation: dict, output_path: Path
) -> Path:
    """Build an HTML summary when evaluation intentionally saved no images."""
    aggregate = evaluation["metrics"]
    save_json(output_path.parent / "table3.json", _table3_values(aggregate))

    pair_rows = []
    for record in evaluation["pairs"]:
        accumulator = MetricAccumulator()
        accumulator.add_confusion(record["confusion"])
        metrics = accumulator.compute()
        changed = metrics["binary"]["changed"]
        timing = record.get("timings", {})
        pair_rows.append(
            "<tr>"
            f"<td>{html.escape(record['id'])}</td>"
            f"<td>{100 * changed['iou']:.1f}</td>"
            f"<td>{100 * changed['precision']:.1f}</td>"
            f"<td>{100 * changed['recall']:.1f}</td>"
            f"<td>{100 * metrics['binary_miou']:.1f}</td>"
            f"<td>{100 * metrics['multiclass_miou']:.1f}</td>"
            f"<td>{sum(timing.values()):.1f}</td>"
            "</tr>"
        )

    failure_rows = "".join(
        "<tr>"
        f"<td>{html.escape(failure['id'])}</td>"
        f"<td>{html.escape(failure.get('type', 'Error'))}</td>"
        f"<td>{html.escape(failure.get('message', ''))}</td>"
        "</tr>"
        for failure in evaluation.get("failures", [])
    )
    failures = (
        f"""<section class="card"><h2>Failures</h2><div class="table-wrap"><table>
        <thead><tr><th>Pair</th><th>Type</th><th>Message</th></tr></thead>
        <tbody>{failure_rows}</tbody></table></div></section>"""
        if failure_rows
        else '<section class="card"><h2>Failures</h2><p>None.</p></section>'
    )
    protocol = evaluation["protocol"]
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GOLDILOCS Metrics Report</title>
<style>
:root{{--bg:#0d1117;--card:#151b23;--line:#2a3441;--text:#edf2f7;--muted:#9ba8b6;--accent:#79c0ff}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 Inter,system-ui,sans-serif}}
main{{max-width:1250px;margin:auto;padding:46px 26px 90px}} h1{{font-size:clamp(2.5rem,7vw,5rem);line-height:.95;margin:.15em 0 .35em}}
h2{{margin:0 0 12px}} h3{{margin:0 0 8px}} p{{color:var(--muted)}} .eyebrow{{color:var(--accent);text-transform:uppercase;letter-spacing:.16em;font-size:.72rem;font-weight:700}}
.summary{{display:flex;gap:12px;flex-wrap:wrap;margin:24px 0 34px}} .summary span,.card,.table3{{background:var(--card);border:1px solid var(--line);border-radius:12px}}
.summary span{{padding:11px 15px;color:var(--muted)}} .summary b{{display:block;color:var(--accent);font-size:1.25rem}}
.card,.table3{{padding:22px;margin:22px 0}} .table-wrap{{overflow:auto}} table{{width:100%;border-collapse:collapse}}
th,td{{padding:10px 12px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}}
th:first-child,td:first-child{{text-align:left}} th{{color:var(--muted);font-size:.72rem;text-transform:uppercase}}
tr.paper td{{color:var(--muted)}} .caveat{{font-size:.82rem}} a{{color:var(--accent)}}
</style></head><body><main>
<p class="eyebrow">Metrics-only evaluation</p>
<h1>GOLDILOCS<br>{html.escape(evaluation_dir.name)}</h1>
<p>This run intentionally saved no pair images or debug geometry. The report is
computed from compact per-pair confusion matrices.</p>
<div class="summary">
<span><b>{protocol['pairs_succeeded']}</b>successful pairs</span>
<span><b>{100 * aggregate['binary_miou']:.1f}%</b>binary mIoU</span>
<span><b>{100 * aggregate['multiclass_miou']:.1f}%</b>multiclass mIoU</span>
<span><b>{evaluation['elapsed_seconds']:.1f}s</b>elapsed</span>
</div>
{_table3(aggregate)}
<section class="card"><h2>Aggregate class metrics</h2>
<div class="table-wrap"><table><thead><tr><th>Class</th><th>GT pixels</th>
<th>Precision</th><th>Recall</th><th>F1</th><th>IoU</th></tr></thead>
<tbody>{_metric_rows(aggregate)}</tbody></table></div></section>
<section class="card"><h2>Per-pair diagnostics</h2>
<div class="table-wrap"><table><thead><tr><th>Pair</th><th>Changed IoU %</th>
<th>Changed precision %</th><th>Changed recall %</th><th>Binary mIoU %</th>
<th>Multiclass mIoU %</th><th>Runtime s</th></tr></thead>
<tbody>{''.join(pair_rows)}</tbody></table></div></section>
{failures}
</main></body></html>"""
    output_path.write_text(document, encoding="utf-8")
    return output_path


def build_evaluation_report(
    evaluation_dir: str | Path,
    manifest_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Build a static, self-contained pipeline walkthrough for an evaluation."""
    evaluation_dir = Path(evaluation_dir)
    output_path = Path(output_path) if output_path else evaluation_dir / "index.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    assets_root = output_path.parent / "report-assets"
    assets_root.mkdir(parents=True, exist_ok=True)
    manifest = {pair.pair_id: pair for pair in load_manifest(manifest_path)}
    evaluation = json.loads((evaluation_dir / "report.json").read_text(encoding="utf-8"))
    if not any((evaluation_dir / "pairs").glob("*/inputs.json")):
        return _build_metrics_only_report(evaluation_dir, evaluation, output_path)

    # Metrics-only progress records intentionally omit artifact paths. Full
    # diagnostic runs still save ``inputs.json`` in each hashed pair directory,
    # so index those files to support reports created by newer evaluators.
    artifacts_by_inputs: dict[tuple[Path, Path], Path] = {}
    for inputs_path in (evaluation_dir / "pairs").glob("*/inputs.json"):
        inputs = json.loads(inputs_path.read_text(encoding="utf-8"))
        artifacts_by_inputs[
            (Path(inputs["image0"]).resolve(), Path(inputs["image1"]).resolve())
        ] = inputs_path.parent.resolve()

    pair_sections = []
    for pair_result in evaluation["pairs"]:
        pair_id = pair_result["id"]
        pair = manifest[pair_id]
        if "artifacts" in pair_result:
            artifact_dir = Path(pair_result["artifacts"])
            if not artifact_dir.is_absolute():
                artifact_dir = (Path.cwd() / artifact_dir).resolve()
        else:
            key = (pair.image0.resolve(), pair.image1.resolve())
            try:
                artifact_dir = artifacts_by_inputs[key]
            except KeyError as exc:
                raise FileNotFoundError(
                    f"No full artifacts found for report pair {pair_id}"
                ) from exc
        # Older runs predate the expanded artifact set. Recreate the requested
        # pointmaps/clouds/renders from cached MASt3R arrays without inference.
        reconstruction = load_reconstruction(artifact_dir / "reconstruction.npz")
        if not (artifact_dir / "render_1_to_0.png").exists():
            with np.load(artifact_dir / "geometry.npz") as geometry:
                save_extended_geometry_artifacts(
                    artifact_dir,
                    reconstruction,
                    geometry["keep0"],
                    geometry["keep1"],
                    json.loads((artifact_dir / "config.json").read_text())["geometry"],
                )
        _ensure_ssim_artifacts(artifact_dir, reconstruction)
        prediction = np.asarray(Image.open(artifact_dir / "labels.png"), dtype=np.uint8)
        target = normalize_target(pair.target)
        if target.shape != prediction.shape:
            target = np.asarray(
                Image.fromarray(target).resize(prediction.shape[::-1], Image.Resampling.NEAREST)
            )
        image0 = load_rgb(pair.image0, prediction.shape[::-1])
        image1 = load_rgb(pair.image1, prediction.shape[::-1])
        metrics = evaluate_arrays(prediction, target)

        pair_assets = assets_root / pair_id
        pair_assets.mkdir(parents=True, exist_ok=True)
        save_image(pair_assets / "gt_color.png", colorize(target))
        save_image(pair_assets / "gt_overlay.png", overlay(image1, target))
        save_image(pair_assets / "prediction_overlay.png", overlay(image1, prediction))
        save_image(pair_assets / "error_map.png", _error_map(prediction, target))
        save_image(pair_assets / "instances.png", _object_instances(artifact_dir, prediction.shape))
        _save_geometry_diagnostics(artifact_dir, pair_assets)

        rel = lambda path: _relative(Path(path), output_path.parent)
        sam_debug_html = ""
        sam_debug_index = artifact_dir / "sam_debug" / "index.json"
        if sam_debug_index.exists():
            sam_calls = json.loads(sam_debug_index.read_text(encoding="utf-8"))
            call_sections = []
            for call in sam_calls:
                title = call["stage"].replace("_", " ").title()
                if call["kind"] == "generate":
                    figures = (
                        _figure(
                            rel(artifact_dir / "sam_debug" / call["input"]),
                            "SAM input",
                            "Exact image supplied to automatic mask generation",
                        )
                        + _figure(
                            rel(artifact_dir / "sam_debug" / call["output"]),
                            f"{call['mask_count']} proposals",
                            "Numbered class-agnostic SAM masks",
                        )
                    )
                    grid_class = "two"
                    detail = (
                        f"Automatic generation retained {call['mask_count']} masks "
                        "after the minimum-area filter."
                    )
                else:
                    figures = (
                        _figure(
                            rel(artifact_dir / "sam_debug" / call["source"]),
                            "Propagation source",
                            "Green IDs tracked successfully · red IDs failed",
                        )
                        + _figure(
                            rel(artifact_dir / "sam_debug" / call["target"]),
                            "Propagation target",
                            "Successful target masks retain their source IDs",
                        )
                        + _figure(
                            rel(artifact_dir / "sam_debug" / call["failed"]),
                            "Failed source masks",
                            "Red masks became change candidates at this step",
                        )
                    )
                    grid_class = "three"
                    detail = (
                        f"{call['tracked_count']} of {call['input_count']} masks "
                        f"tracked; {call['failed_count']} failed."
                    )
                call_sections.append(
                    f"""
                    <div class="sam-call">
                      <h4>{html.escape(title)}</h4>
                      <p>{html.escape(detail)}</p>
                      <div class="grid {grid_class}">{figures}</div>
                    </div>
                    """
                )
            sam_debug_html = (
                '<div class="sam-debug"><h3>Every SAM2 call</h3>'
                '<p>Instance numbers correspond between each propagation source '
                'and target. Green means the track survived; red means SAM2 returned '
                'no mask above the minimum-area threshold.</p>'
                + "".join(call_sections)
                + "</div>"
            )
        timing = pair_result.get("timings", {})
        pair_sections.append(
            f"""
            <article class="pair" id="{html.escape(pair_id)}">
              <header class="pair-header">
                <div><p class="eyebrow">Evaluated pair</p><h2>{html.escape(pair_id)}</h2></div>
                <div class="scores">
                  <span><b>{metrics['binary_miou']:.3f}</b> binary mIoU</span>
                  <span><b>{metrics['multiclass_miou']:.3f}</b> multiclass mIoU</span>
                  <span><b>{sum(timing.values()):.1f}s</b> measured stages</span>
                </div>
              </header>

              <section class="stage">
                <div class="stage-copy"><span>01</span><div><h3>Temporal inputs</h3>
                <p><em>I₀</em> is the matched pre-change reference supplied by ChangeSim. <em>I₁</em>
                is the later query image. They observe the same scene from slightly different poses;
                direct pixel subtraction would therefore confuse parallax with change.</p></div></div>
                <div class="grid two">
                  {_figure(rel(pair.image0), "I₀ · Before", "Reference image at T₀")}
                  {_figure(rel(pair.image1), "I₁ · After", "Query image at T₁")}
                </div>
              </section>

              <section class="stage">
                <div class="stage-copy"><span>02</span><div><h3>MASt3R reconstruction and viewpoint alignment</h3>
                <p>MASt3R estimates both cameras and a dense XYZ point for every pixel. The false-color
                previews encode camera-space depth; blue is nearer and red is farther after robust
                per-view normalization. Use the PLY links to inspect actual XYZ and RGB values.</p></div></div>
                <div class="grid two">
                  {_figure(rel(artifact_dir / "pointmap_0.png"), "Pointmap P₀ · depth preview", "Camera-space depth visualization—not an SSIM heatmap")}
                  {_figure(rel(artifact_dir / "pointmap_1.png"), "Pointmap P₁ · depth preview", "Camera-space depth visualization—not an SSIM heatmap")}
                </div>
                <p class="downloads"><a href="{html.escape(rel(artifact_dir / 'pointmap_0.ply'))}">Download P₀ PLY</a>
                <a href="{html.escape(rel(artifact_dir / 'pointmap_1.ply'))}">Download P₁ PLY</a></p>
              </section>

              <section class="stage">
                <div class="stage-copy"><span>03</span><div><h3>Reverse-depth conflict filtering</h3>
                <p>Each pointmap is projected into the opposing camera. A point lying in front of
                that camera's observed depth is temporally inconsistent and removed. The surviving
                T₀ and T₁ surfaces form P*, the static canonical scene.</p></div></div>
                <div class="grid four">
                  {_figure(rel(pair_assets / "keep0.png"), "Kept T₀ geometry", "White survives the T₀→T₁ depth test")}
                  {_figure(rel(pair_assets / "keep1.png"), "Kept T₁ geometry", "White survives the T₁→T₀ depth test")}
                  {_figure(rel(artifact_dir / "clean_pointcloud_0.png"), "Clean pointcloud P₀ clean", "I₀ appearance retained after conflict filtering")}
                  {_figure(rel(artifact_dir / "clean_pointcloud_1.png"), "Clean pointcloud P₁ clean", "I₁ appearance retained after conflict filtering")}
                </div>
                <p class="downloads"><a href="{html.escape(rel(artifact_dir / 'clean_pointcloud_0.ply'))}">Download clean P₀ PLY</a>
                <a href="{html.escape(rel(artifact_dir / 'clean_pointcloud_1.ply'))}">Download clean P₁ PLY</a>
                <a href="{html.escape(rel(artifact_dir / 'canonical.ply'))}">Download union P* PLY</a></p>
              </section>

              <section class="stage">
                <div class="stage-copy"><span>04</span><div><h3>All viewpoint-aligned renders</h3>
                <p>R₀,₁ projects the old pointmap into camera 1; R₁,₀ performs the reverse projection.
                R*,₁ and R*,₀ render the union of both cleaned clouds from each camera. Black pixels
                are unsupported by the point rasterizer and are an important source of downstream error.</p></div></div>
                <div class="grid four">
                  {_figure(rel(artifact_dir / "render_0_to_1.png"), "R₀,₁", "P₀ rendered through camera C₁")}
                  {_figure(rel(artifact_dir / "render_1_to_0.png"), "R₁,₀", "P₁ rendered through camera C₀")}
                  {_figure(rel(artifact_dir / "render_clean_to_1.png"), "R*,₁", "Canonical clean union rendered through C₁")}
                  {_figure(rel(artifact_dir / "render_clean_to_0.png"), "R*,₀", "Canonical clean union rendered through C₀")}
                </div>
              </section>

              <section class="stage">
                <div class="stage-copy"><span>05</span><div><h3>SAM2 segmentation and tracking</h3>
                <p>SAM2 proposes class-agnostic object masks and propagates them between R₀,₁,
                R*,₁, and I₁. Failure to survive the clean rendering identifies a changed object;
                subsequent tracking distinguishes moved from added or removed.</p></div></div>
                <div class="grid two">
                  {_figure(rel(pair_assets / "instances.png"), "Retained object instances", "Arbitrary colors distinguish saved object masks")}
                  {_figure(rel(artifact_dir / "labels_color.png"), "Rigid-change labels", "Semantic colors after tracking and visibility filtering")}
                </div>
                {sam_debug_html}
              </section>

              <section class="stage">
                <div class="stage-copy"><span>06</span><div><h3>Non-rigid change with SSIM</h3>
                <p>SSIM compares viewpoint-aligned R₀,₁ against I₁. Static SAM2 masks whose mean
                dissimilarity exceeds the object-score mean plus one standard deviation are labeled
                warped. Bright regions indicate stronger structural disagreement.</p></div></div>
                <div class="grid three">
                  {_figure(rel(artifact_dir / "ssim_heatmap.png"), "SSIM heatmap", "Blue≈0 similar · cyan/yellow/red increasingly dissimilar · red≥1")}
                  {_figure(rel(artifact_dir / "ssim_heatmap_overlay.png"), "SSIM heatmap over I₁", "False-color dissimilarity blended over the target view")}
                  {_figure(rel(artifact_dir / "overlay.png"), "Final prediction", "Predicted change labels over I₁")}
                </div>
                <div class="heat-legend"><span>0 · similar</span><i></i><span>≥1 · dissimilar</span></div>
              </section>

              <section class="stage verdict">
                <div class="stage-copy"><span>07</span><div><h3>Ground truth versus result</h3>
                <p>This is the diagnostic view. Green in the error map is a correctly classified
                changed pixel; red is a false positive, blue is a missed change, yellow is a detected
                change assigned the wrong class, and dark gray is correctly unchanged.</p></div></div>
                <div class="grid four">
                  {_figure(rel(pair_assets / "gt_overlay.png"), "Ground truth", "Official ChangeSim labels over I₁")}
                  {_figure(rel(pair_assets / "prediction_overlay.png"), "Prediction", "GOLDILOCS labels over I₁")}
                  {_figure(rel(pair_assets / "error_map.png"), "Error map", "Green correct · red FP · blue FN · yellow wrong class")}
                  {_figure(rel(pair.image1), "I₁ reference", "Use appearance to judge annotation boundaries")}
                </div>
                <div class="table-wrap"><table><thead><tr><th>Class</th><th>GT pixels</th><th>Precision</th><th>Recall</th><th>F1</th><th>IoU</th></tr></thead>
                <tbody>{_metric_rows(metrics)}</tbody></table></div>
              </section>
            </article>
            """
        )

    aggregate = evaluation["metrics"]
    save_json(output_path.parent / "table3.json", _table3_values(aggregate))
    navigation = "".join(
        f'<a href="#{html.escape(pair["id"])}">{html.escape(pair["id"])}</a>'
        for pair in evaluation["pairs"]
    )
    legend = "".join(
        f'<span><i style="background:rgb{COLORS[label]}"></i>{label.name.title()}</span>'
        for label in (Label.ADDED, Label.REMOVED, Label.MOVED, Label.WARPED, Label.REPLACED)
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GOLDILOCS Pipeline Diagnostic</title>
<style>
:root{{--bg:#0d1117;--card:#151b23;--line:#2a3441;--text:#edf2f7;--muted:#9ba8b6;--accent:#79c0ff}}
*{{box-sizing:border-box}} html{{scroll-behavior:smooth}} body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 Inter,system-ui,sans-serif}}
main{{max-width:1500px;margin:auto;padding:42px 28px 100px}} h1{{font-size:clamp(2.4rem,6vw,5rem);line-height:.95;margin:.15em 0}} h2{{margin:0;font-size:1.65rem}} h3{{margin:0 0 8px;font-size:1.25rem}} p{{margin:0;color:var(--muted)}} .eyebrow{{color:var(--accent);text-transform:uppercase;letter-spacing:.16em;font-size:.72rem;font-weight:700}}
.hero{{padding:34px 0 44px;border-bottom:1px solid var(--line)}} .summary{{display:flex;gap:12px;flex-wrap:wrap;margin-top:24px}} .summary span,.scores span{{background:var(--card);border:1px solid var(--line);padding:10px 14px;border-radius:10px}} .summary b,.scores b{{color:var(--accent);font-size:1.15rem}}
nav{{position:sticky;top:0;z-index:4;display:flex;gap:8px;overflow:auto;background:rgba(13,17,23,.93);backdrop-filter:blur(12px);padding:14px 0}} nav a{{white-space:nowrap;color:var(--text);text-decoration:none;border:1px solid var(--line);border-radius:999px;padding:6px 12px}}
.legend{{display:flex;flex-wrap:wrap;gap:16px;margin-top:18px}} .legend span{{display:flex;align-items:center;gap:7px;color:var(--muted)}} .legend i{{width:12px;height:12px;border-radius:3px}} .pair{{padding-top:62px}} .pair-header{{display:flex;align-items:end;justify-content:space-between;gap:20px;border-bottom:1px solid var(--line);padding-bottom:20px}} .scores{{display:flex;gap:8px;flex-wrap:wrap}} .scores span{{display:flex;flex-direction:column;color:var(--muted);font-size:.72rem}}
.sam-debug{{margin-top:30px;padding-top:26px;border-top:1px solid var(--line)}} .sam-debug>p{{margin-bottom:22px}} .sam-call{{margin:22px 0 34px;padding:20px;background:#10161e;border:1px solid var(--line);border-radius:12px}} .sam-call h4{{margin:0 0 4px;font-size:1.05rem}} .sam-call>p{{margin-bottom:16px}}
.stage{{padding:34px 0;border-bottom:1px solid var(--line)}} .stage-copy{{display:grid;grid-template-columns:46px minmax(0,800px);gap:16px;margin-bottom:20px}} .stage-copy>span{{font-size:1.5rem;font-weight:800;color:var(--accent)}} .grid{{display:grid;gap:14px}} .grid.two{{grid-template-columns:repeat(2,1fr)}} .grid.three{{grid-template-columns:repeat(3,1fr)}} .grid.four{{grid-template-columns:repeat(4,1fr)}} figure{{margin:0}} .panel{{background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden}} .panel img{{display:block;width:100%;aspect-ratio:4/3;object-fit:contain;background:#05070a}} figcaption{{display:flex;flex-direction:column;padding:11px 13px}} figcaption span{{font-size:.78rem;color:var(--muted)}} .verdict{{background:linear-gradient(180deg,transparent,rgba(121,192,255,.035))}}
.downloads{{display:flex;gap:12px;flex-wrap:wrap;margin-top:14px}} a{{color:var(--accent)}} .table3{{margin:34px 0;padding:22px;background:var(--card);border:1px solid var(--line);border-radius:12px}} .caveat{{margin-top:12px;font-size:.82rem}} tr.paper td{{color:var(--muted)}}
.heat-legend{{display:flex;align-items:center;gap:10px;margin-top:12px;color:var(--muted);font-size:.78rem}} .heat-legend i{{display:block;width:min(420px,60vw);height:12px;border-radius:8px;background:linear-gradient(90deg,#000080,#00d4ff,#f5f500,#ff0000)}}
.table-wrap{{overflow:auto;margin-top:18px}} table{{width:100%;border-collapse:collapse;background:var(--card)}} th,td{{padding:10px 12px;text-align:right;border-bottom:1px solid var(--line)}} th:first-child,td:first-child{{text-align:left}} th{{color:var(--muted);font-size:.75rem;text-transform:uppercase}} @media(max-width:900px){{.grid.three,.grid.four{{grid-template-columns:repeat(2,1fr)}}.pair-header{{align-items:start;flex-direction:column}}}} @media(max-width:580px){{main{{padding:24px 14px 70px}}.grid.two,.grid.three,.grid.four{{grid-template-columns:1fr}}}}
</style></head><body><main>
<section class="hero"><p class="eyebrow">Reproduction diagnostic</p><h1>GOLDILOCS,<br>step by step.</h1>
<p>A visual audit of every saved stage, ending with ground truth, predictions, and exact errors.</p>
<div class="summary"><span><b>{evaluation['protocol']['pairs_succeeded']}</b> successful pairs</span>
<span><b>{aggregate['binary_miou']:.3f}</b> aggregate binary mIoU</span>
<span><b>{aggregate['multiclass_miou']:.3f}</b> aggregate multiclass mIoU</span></div>
<div class="legend">{legend}</div>{_table3(aggregate)}</section><nav>{navigation}</nav>
{''.join(pair_sections)}
</main></body></html>"""
    output_path.write_text(document, encoding="utf-8")
    return output_path
