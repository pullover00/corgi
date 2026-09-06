# Method

Given two time-separated captures of the same physical scene, each as a short
walkthrough video (or an unordered set of photos) rather than a single fixed
before/after photo pair, the pipeline localizes and classifies every
object-level change (added, removed, or moved) between them. It runs in four
stages: multi-view reconstruction, canonical-scene cleaning, optional render
refinement, and object-level change resolution.

## 1. Multi-view reconstruction

Rather than reconstructing the scene from a single photo per time step (as a
two-view method such as MASt3R would), we sample *N* frames (N=10 in our
runs) from each of the two captures and reconstruct them **jointly** with
VGGT-Omega, a feed-forward multi-view transformer that predicts, for every
input frame in one forward pass, a camera pose (extrinsics and intrinsics)
and a per-pixel depth map, all expressed in one shared world coordinate
frame. Because both time steps are reconstructed together, their two point
clouds arrive already registered to each other with no separate alignment
step.

Each frame's depth map is unprojected into that shared world frame, giving a
colored world-space point per pixel per frame. Points are kept only where
VGGT-Omega's own predicted depth confidence clears a percentile threshold
computed over the batch (20th percentile in our runs). Pooling ten views
this way gives dense, largely hole-free coverage of each time step, in
contrast to a single photo, which only ever samples the scene from one
vantage point and leaves anything outside its field of view or defeated by
its stereo matching (a common failure on reflective or textureless surfaces,
e.g. a glass stovetop) as an unrecoverable gap.

One frame from each time step is designated the *reference frame*: the T1
(after) reference frame's camera is the one everything is ultimately
rendered into and becomes the pipeline's real-photo output (`image_t1`); the
T0 (before) reference frame supplies the "opposing view" depth used by the
cleaning step below. In practice we pick whichever frame each dataset's own
annotation already treats as most representative of the full scene, since
that is usually the widest, most complete establishing shot.

## 2. Canonical-scene cleaning (mutual depth-conflict filter)

A raw before/after point-cloud union is not by itself a clean canonical
reconstruction: it still contains both the old and new state of every
changed object superimposed. We separate genuine background from add/remove
transients with a **mutual depth-conflict filter**: every point from the T0
cloud is reprojected into the T1 reference camera and compared against that
camera's own observed depth at the same pixel; if the T0 point sits *closer*
to the camera than what T1 actually saw there, it must be occluding
something transient rather than depicting the true, still-present
background, and it is dropped. The same check runs in the opposite
direction — T1's points checked against the T0 reference frame's own
observed depth. Points that survive both directions are pooled into one
canonical point cloud with add/remove transients pruned. Because the check
requires reliable *opposing* evidence to reject a point, and defaults to
keeping a point when the opposing view offers none (out of frustum, or no
valid depth there), an object that is genuinely new and therefore invisible
to the opposing time step's own reconstruction is not incorrectly kept as
"background" by this step alone — it is the downstream object-matching stage
that ultimately resolves it as added.

## 3. Rendering

The uncleaned T0 cloud and the cleaned canonical cloud are each splatted
(z-buffered point rendering with a small footprint per point and
conservative single-pixel hole filling) into the T1 reference camera,
producing two synthetic renders — `render_t0` and `clean_render` — on
exactly the same pixel grid as the real T1 photo (`image_t1`, used directly,
no rendering needed). These three images are the pipeline's fixed contract
with the object-resolution stage below, regardless of how they were
produced.

## 4. Render refinement (optional)

`render_t0` and `clean_render` still carry residual splatting artifacts from
underconstrained regions of the reconstruction (speckle noise, small holes).
Each is independently passed through DI²FIX/Difix (`nvidia/difix_ref`), a
single-step diffusion model trained to remove exactly this class of
artifact, conditioned on a reference image — here, the real `image_t1`
photo. This measurably improves downstream object-count and change-decision
quality in our tests (Section "Results" below).

This step can hallucinate plausible-but-fabricated content into a genuine
gap left by the depth-conflict filter (observed once: a fabricated
liquid-filled container filling a removed-object gap in `clean_render`). We
have not seen this propagate into a wrong final decision, because of how
`clean_render`'s objects are used in stage 5 below — but this has only been
checked on a small number of cases and is not architecturally guaranteed.

## 5. Object-level change resolution

### Per-frame object inventory

SAM3's automatic mask generator proposes class-agnostic object masks
independently in each of the three images. Raw proposals are filtered down
to a compact inventory: masks below a minimum area or above a maximum area
fraction are dropped, masks touching the image border are treated as
probable background and dropped, and near-duplicate or nested proposals are
merged (the largest enclosing mask wins, so a lid, button, or face does not
become a second object beside the whole instance it belongs to). For every
retained mask, appearance is captured twice, independently: a pooled dense
SAM3 image-encoder feature and a pooled dense DINOv2 feature, each averaged
over the mask's footprint in that model's own feature map and L2-normalized.

### Bidirectional mask tracking

Every object's mask is also tracked with SAM2 (mask-prompted, not
video-frame propagation) into each of the other two images — `t0<->t1`,
`t0<->clean`, `t1<->clean` — giving, for every object, a candidate location
(or "not found") in the other two frames.

### Matching and classification

An object in `render_t0` and an object in `image_t1` are accepted as one
identity only when there is agreement across independent signals: either a
reliable bidirectional SAM2 track between them plus both SAM3 and DINOv2
cosine similarity above threshold, or, when tracking is unavailable, mutual
nearest-neighbor agreement between the two feature systems alone (a
track-free fallback so identity is not solely gated on a good track). A
matched pair is then classified unchanged or moved by the spatial IoU
between its two masks. When a direct match fails — typically because
`render_t0` is a synthetic render and `image_t1` a real photo, which can
widen the render/photo appearance gap enough to break a direct feature
comparison — a track-only candidate pair can still be validated by checking
both endpoints' identity inside the shared `clean_render` feature domain
instead, using it as a bridge.

### Recovering proposal-generation misses

SAM3's automatic mask generator is not guaranteed to independently propose a
mask for the same object in both frames — one frame's grid can simply miss
it. Naively, every such case looks identical to a genuine removal or
addition. Before finalizing any unmatched object as removed/added, we
revisit it using the SAM2 track already computed for it (regardless of
whether that track ever found a matching *proposed* object): if the track
was accepted, the opposite frame's own dense SAM3 and DINOv2 feature maps
are pooled under the tracked mask directly (not tied to any pre-existing
proposal there) and checked against the same identity thresholds used
everywhere above. If they agree, the object is reclassified
unchanged/moved — absorbing any independently-proposed but wrongly-orphaned
object at the same location, if one exists — instead of being called
removed/added. This one change reduced the changed-pixel fraction from
23.3% to 14.3% on our development pair, roughly halving both false removals
and false additions (Section "Results").

### Output

Everything still unmatched after recovery is genuinely removed (present
only in `render_t0`) or added (present only in `image_t1`). The final
per-pixel label map is rasterized aligned to `image_t1`, in priority order
added > removed > moved where object masks overlap (added is the weakest
explanation and yields to any more specific one).

An earlier version also classified same-location, confidently-different-
identity pairs as a fifth `replaced` class. It was removed: a "replacement"
is not a distinct physical event from the pipeline's own evidence -- it is
just a removed object and an added object that happen to occupy the same
mask slot -- and treating it specially added a class with no clear
downstream use and no PASLCD/SceneDiff ground-truth analogue. Such pairs now
simply fall through to independent removed/added decisions.

A `replaced` class (`Label.REPLACED`) was reintroduced later, deliberately,
for a different mechanism and a different reason: a dense chromaticity-
residual pass over `render_t0`/`image_t1` directly (see
`color_residual.py`), catching same-footprint color swaps that never get
segmented as their own object in either frame at all (e.g. a red block
replaced by a blue one, absorbed into a much larger surface's SAM3 proposal
in both frames) -- there is no removed/added *pair* here for it to fall
through to, since no per-object comparison ever ran on the region in the
first place. The "no clear downstream use" objection above no longer
applies for the same reason binary evaluation never cared about the label
value: PASLCD's own metric only asks changed-vs-unchanged, so this class
exists purely for internal diagnostics and visualization, same as `moved`
vs. `added`/`removed` already did.

## Data preparation note

Applying this to the SceneDiff benchmark surfaced two dataset-specific
issues worth recording because they are easy to miss silently: (1) the
benchmark's `video1.mp4`/`video2.mp4` are not raw footage — some objects are
repainted with flat synthetic colors, apparently a review-tool
visualization artifact; the true footage is `original_video{1,2}.*`, and
using the wrong one biases both reconstruction and appearance matching. (2)
Those original files carry sensor-orientation metadata that OpenCV's
`VideoCapture` does not apply automatically, producing sideways frames
unless `CAP_PROP_ORIENTATION_AUTO` is set explicitly.

## Results (development pair, `kitchen_2_kitchen_3`)

Ablated on one SceneDiff pair, each row cumulative on the previous. Recorded
when the pipeline still had the `replaced` class (since removed, see
above) and the border-touching proposal filter (since removed, see PASLCD
notes) -- kept here as the historical record of that run, not as current
behavior.

| Configuration | unchanged | moved | replaced | removed | added | changed px fraction |
|---|---|---|---|---|---|---|
| Reconstruction only (no tracking recovery) | 3 | 6 | 1 | 13 | 17 | 23.3% |
| + tracking-recovery (Section 5) | 13 | 7 | 1 | 10 | 9 | 14.3% |
| + DI²FIX render refinement (Section 4) | 18 | 8 | 2 | 9 | 5 | 12.4% |

This is one pair; treat it as a development signal, not a validated result —
broader evaluation against the benchmark's own held-out split and official
metric (see below) is the natural next step.

## Evaluation protocol (not yet wired up)

SceneDiff's own paper (arXiv 2512.16908) evaluates with Average Precision
over point-in-box matching (a prediction is reduced to its mask centroid; a
ground-truth object is a true positive if any detection point falls inside
its box), reported both **per-view** (`obj/im AP`) and, its distinguishing
contribution, **per-scene** (`obj/sc AP`): each changed object counted once
across the whole video pair rather than once per frame, with duplicate
detections of the same object treated as false positives. A pixel-level
`px/im IoU` is also reported. The authors' own evaluator lives at
`github.com/yuqunw/scene_diff` (`scripts/evaluate_multiview.py`); using it
directly, once our output is reshaped to its expected prediction format, is
preferable to re-deriving the metric from the paper's prose.
