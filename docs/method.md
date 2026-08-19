# Method: object-consistent full target masks

The full method is all 11 pipeline stages (see README.md's architecture
table); this page documents the last one in detail. This is the decision
procedure implemented in `src/ocmask/stages/slot_inconsistency.py` and run
by `scripts/run_slot_inconsistency_replacement_experiment.py` (stage 11). It
refines the prediction produced by stages 1-10 of this same pipeline
("R4", `r4_no_geometry_ablation`) rather than predicting from scratch --
stages 1-10 are this repository's own base pipeline, not a third-party or
externally supplied baseline.

## Procedure

```text
R4 prediction + old source hypotheses + target SAM inventory
                         │
                         ▼
              match the old and target slots
                         │
                         ▼
       compare aligned DINO, SAM, DINO identity, and color
                         │
                         ▼
             identity-mismatch candidate only
                         │
                         ▼
               target-object plausibility
       ├─ compact object
       ├─ or coherent fragment with strong direct alignment
       └─ non-weak slot match
             │                         │
          accepted                   rejected
             │                         │
             │              do not paint REPLACED here
             │
             ▼
  search for an adjacent same-object companion
  require proximity + same depth + SAM + DINO + color
             │
             ▼
  compare target depth with each overlapping REMOVED component
  ├─ replacement closer → replacement owns overlap; cleanup may proceed
  ├─ removed closer / tie / unknown → keep red and protect component
  └─ exception: 3 mismatches + one-to-one identity
     → matched source is obsolete history; replacement owns only that pair
             │
             ▼
  unify ADDED/MOVED fragments only within replacement-owned pixels
             │
             ▼
  arbitrate every coherent, independently changed target object
  ├─ ADDED + REMOVED in one object → REPLACED
  ├─ REPLACED majority + MOVED island → REPLACED
  └─ otherwise require ≥75% dominant-class support
  (relabel existing changed pixels only; never grow binary support)
             │
             ▼
  remove ADDED components that are predominantly fitted floor
             │
             ▼
              final one-class-per-pixel raster
```

### Object plausibility

Identity mismatch establishes that the old appearance disappeared, but it
does not prove that a fragmented shelf, pole, or exposed background patch is
a new object. A replacement candidate is rasterized only if:

- its target mask compactness is at least `0.45`; or
- compactness is at least `0.33` and aligned IoU is at least `0.45`;
- its slot score is at least `0.33`; or
- it already passed the stricter cleanup checks.

On the 25-pair evaluation set, 14 of 26 identity-mismatch candidates pass
this object gate; 12 are rejected before rasterization.

### Compatible companion expansion

A confirmed replacement can include a neighboring target proposal only when
all of the following hold:

- mask gap at most 6 pixels;
- area ratio `0.5-2.0x`;
- robust target-depth difference at most `0.08 m`;
- target-to-target SAM cosine at least `0.80`;
- target-to-target DINO cosine at least `0.80`;
- color-histogram intersection at least `0.50`; and
- companion compactness at least `0.45`.

### Old-footprint cleanup

Replacement acceptance and source-footprint deletion are separate decisions.
After a target has passed object plausibility, its matched old `REMOVED`
footprint is erased outside the visible new target only when:

- all three identity cues (SAM, DINO, and color) indicate a mismatch; and
- the source and target have one-to-one slot geometry, or the target passes
  the compact-foreground override.

Two additional general checks cover asymmetric and fragmented masks. A
smaller replacement can clean a larger old envelope when containment is at
least `0.85`, the target/source area ratio is at most `0.65`, the slot score
is at least `0.45`, and the object has one dominant component. Cleanup also
completes a nearby connected `REMOVED` component only when at least half of
it is covered by the matched source or the local replacement neighborhood,
its area is at most three times the directly matched red support, and either
overlap depth places the replacement in front or the component belongs to a
strong identity-scoped source pair. Source-front, tied, and unknown-depth red
components remain protected for weak/asymmetric pairs. Conversely, when a
target proposal is rejected and supplies no new-object support, the aligned
source footprint is treated as a retained `REMOVED` foreground object. Only
already-changed pixels are relabelled, so binary support cannot expand.

### Dataset-wide target-object arbitration

This stage examines every consolidated target SAM object, not only a
rejected replacement candidate. It acts only if the object is coherent, at
least 75% of its mask is already changed, at least 32 changed pixels support
the decision, and at least one constituent proposal has independent frozen
target-to-clean `object_absent` evidence (this last check prevents stable
shelves, walls, and floors with noisy baseline colors from being forced into
a change class).

Within an eligible object:

- simultaneous `ADDED` and `REMOVED` evidence means `REPLACED`;
- when the only labels are `MOVED` and `REPLACED`, a majority `REPLACED`
  region absorbs the implausible interior `MOVED` island; and
- all other mixtures require at least 75% support for one dominant class.

Only pixels already marked changed are relabeled, so this stage cannot
enlarge the binary change mask.

### Floor plausibility

A floor plane is fit from reconstruction points already retained as static.
An `ADDED` connected component is suppressed only when it has at least 32
pixels and at least 70% of the entire component lies within `0.05 m` of the
fitted floor plane. Incidental object-floor contact is retained.

## Worked failure cases

These are drawn from the 25-pair evaluation and illustrate where the method
succeeds and where it deliberately declines to act.

**`Warehouse_9_Seq_0_944`** — the clean success case. One compact barrel mask
is identified; an adjacent barrel is recovered as a compatible companion
(depth difference `0.0051 m`, SAM cosine `0.858`, DINO cosine `0.990`, color
intersection `0.817`). Both barrels are written uniformly as `REPLACED`
instead of the baseline's fragmented purple/orange/green stripes. Three
floor-dominant `ADDED` components (6,036 pixels) are reset to `UNCHANGED`.
Pair multiclass mIoU improves `0.470 → 0.612`; binary mIoU `0.777 → 0.845`;
REPLACED IoU `0.135 → 0.582`.

**`Warehouse_8_Seq_0_148`** — depth protection blocking a plausible cleanup.
Strong SAM/color mismatch evidence exists, and an asymmetric-envelope rule
recognizes the correspondence, but depth places both overlapping red pixels
in front of the replacement, so the entire red component stays protected and
zero pixels are deleted (pair mIoU moves only `0.3112 → 0.3117`). This is the
method choosing not to act rather than acting incorrectly.

**`Warehouse_7_Seq_0_220`** — reached through the dataset-wide arbitration
path, not the replacement-candidate path, because this pair produces no
slot-level replacement candidate. A consolidated target hypothesis (969
`MOVED` + 1,420 `REPLACED` pixels, one constituent proposal independently
`object_absent`) has its interior orange band folded into the majority
`REPLACED` barrel mask. Eight objects qualify in this pair, rewriting 1,590
pixels without changing a single binary changed/unchanged decision. Small
red/orange regions beside the barrel are deliberately *not* absorbed: they
sit outside this hypothesis and at different depths, and merging them would
recreate the exact old-mask overlay error this method was built to avoid.

**`Warehouse_9_Seq_1_917`** — three identity-mismatch candidates all fail
object plausibility and are correctly prevented from painting new `REPLACED`
regions; two of them still have their minority-class fragments folded into
an existing dominant-class consensus (403 pixels), while the candidate with
no changed-pixel support at all is left untouched.

## Remaining limitations

- A SAM object may legitimately contain multiple benchmark change types; the
  consensus gate is deliberately high (75%) because forcing every proposal
  to one class would be incorrect.
- A severely fragmented target proposal can be rejected even when its
  identity evidence is correct.
- Companion expansion is intentionally strict and may miss a group whose
  objects differ in color or size.
- Floor suppression depends on a reliable reconstruction and robust floor
  fit; if no floor is found, it abstains rather than guessing.
- `MOVED` is a separate problem from replacement consistency and is
  essentially unaffected by this method (`IoU 0.1762 → 0.1779` combined).
