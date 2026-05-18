# Orthogonal postfix slots — structural fix for K=1 collapse

Companion to [`archive/bench/postfix/initial_postfix_problems.md`](../../archive/bench/postfix/initial_postfix_problems.md).
That note documented the K=1 collapse on the trained `anima_postfix.safetensors`
checkpoint and landed soft fixes (per-slot bias `slot_embed`, inter-caption
contrastive). The follow-up checkpoint still behaves as a domain prior — slot
symmetry is partially broken but `cond_mlp` output remains low-rank in
practice. This proposal swaps the soft fixes for a **structural** orthogonality
constraint, mirroring the move that made OrthoHydraLoRA work.

## Why this is the next move (and not before)

We have three converging signals that "K orthogonal slots, by construction,
not by regularization" is the right shape:

1. **Soft fixes hit a ceiling.** `slot_embed_init_std=0.02` and
   `contrastive_weight=0.1` together break exact slot equivalence and pressure
   `cond_mlp` toward caption-varying outputs, but neither prevents the
   loss-minimum from being a low-rank attractor. The contrastive term decays as
   the buffer drifts; `slot_embed` is a fixed bias that the rest of the
   pipeline can route around. Both are gradient-pressure tools, not
   architectural constraints.

2. **OrthoHydraLoRA succeeded with exactly this move.** `OrthoHydraLoRAModule`
   (`networks/lora_modules/ortho.py:166`) replaced soft orthogonality
   regularization with a Cayley-parameterized rotation of *disjoint
   per-expert SVD-derived bases*. Per the docstring: "Because the SVD columns
   are mutually orthonormal, experts are **structurally orthogonal in output
   space** ... regardless of training dynamics." That's the exact pattern we
   want to port: replace "K slots that *might* end up orthogonal under
   regularization" with "K slots that are orthogonal by construction at every
   gradient step."

3. **Existing prior art (`archive/inversion/invert_reference.py`) is K=8
   prefix-slot inversion, never characterized as useful auxiliary supervision.**
   The verifier flagged this as "the closest in-tree analog to per-image
   inversion-into-postfix-slots, built but not converted into a teacher
   signal." Re-reading it confirms: it inverts K=8 free token vectors against
   FM-MSE, with no orthogonality, no symmetry-breaking. If the K=1 collapse
   afflicts that path too (it should — same symmetric splice into padding),
   that explains why it never produced anything richer than a single
   subject-direction embedding.

The downstream consequence: orthogonal postfix is also the precondition for
the inversion-as-teacher idea (Z-Image-style self-distillation) being
non-degenerate. The verifier red-flagged that idea on the basis that
per-image inversions into the postfix slot would collapse for the same
symmetric-splice reason. Hard orthogonality removes that degeneracy at
inversion time too — the teacher signal becomes K-rank instead of effective-1.

## Design (v1, default postfix mode only)

Scope this first pass to `mode="postfix"` (the simplest path: a single
`postfix_embeds: nn.Parameter(K, D)` tensor). Cond / cond-timestep are
deferred — see "Out of scope" below. Default postfix is where the K=1
collapse first manifested and is the cleanest test of the structural fix.

### Parameterization

Replace `self.postfix_embeds = nn.Parameter(torch.randn(K, D) * init_std)`
with a Cayley-rotated frozen basis:

```python
# Frozen orthonormal basis for the K-dim subspace the K slots live in.
# Choice of basis is a tunable knob (see below) — default to random
# orthonormal for v1, swap for SVD-of-T5-cache in v1.5 if v1 underperforms.
self.register_buffer("postfix_basis", _make_orthonormal_basis(K, D))  # (K, D)

# Trainable: K×K skew-symmetric seed → K×K orthogonal Cayley(S).
# Init S=0 → R=I → postfix = postfix_basis (orthonormal at step 0).
self.S = nn.Parameter(torch.zeros(K, K))

# Trainable per-slot scale (analogous to lambda_layer in OrthoHydraLoRA).
# Zero-init so postfix_embeds = 0 at step 0 and training starts from
# baseline behavior.
self.lambda_slot = nn.Parameter(torch.zeros(K))
```

Effective postfix at every forward:

```
A = S - S.T                             # skew-symmetric
R = solve(I + A, I - A)                 # K×K orthogonal, by Cayley
postfix_embeds = (R @ postfix_basis) * lambda_slot.unsqueeze(-1)  # (K, D)
```

This guarantees `postfix_embeds @ postfix_embeds.T` is diagonal with values
`lambda_slot[k]²` — K orthogonal directions of independently-tunable
magnitude — at every gradient step.

Trainable param count: `K(K-1)/2 + K` ≈ 528 at K=32 (vs `K*D = 32768` for
the current free parameterization). Capacity is intentionally smaller; the
hypothesis is that the current parameterization is over-parameterized into
a degenerate basin and a smaller-but-orthogonal parameterization will reach
a better optimum.

### Basis choice (one knob to ablate)

Three candidates for `postfix_basis`:

- **Random orthonormal** (v1 default) — `torch.randn(K, D)` then QR.
  Simplest; no inductive bias toward any particular subspace.
- **SVD of cached T5 embeddings** — top-K right singular vectors of the
  cached `_anima_te.safetensors` corpus across the dataset. Aligns the
  postfix subspace with "directions T5 actually uses for this corpus."
- **SVD of cross-attention K-projection** — top-K right singular vectors
  of stacked K-proj weights across DiT blocks. Aligns with "directions the
  pretrained DiT looks for through cross-attention." Closest analog to
  OrthoHydraLoRA's SVD-of-W init.

Default v1: random orthonormal. Run the SVD variants only if v1 caps out
on capacity at saturation (i.e. the `lambda_slot` magnitudes converge but
the qualitative effect is still a domain prior).

### Config surface

`configs/methods/postfix.toml` gets a new variant block:

```toml
# ── Postfix (ortho — structurally orthogonal K slots) ───────────────────
# K orthonormal slot vectors via Cayley rotation of a frozen basis.
# Replaces slot_embed_init_std + contrastive_weight (both inert here).
#
# network_dim = 32
# network_args = [
#     "mode=postfix",
#     "splice_position=end_of_sequence",
#     "ortho=true",
#     "ortho_basis=random",  # random | svd_te | svd_kproj
# ]
# output_name = "anima_postfix_ortho"
# max_train_epochs = 2
# checkpointing_epochs = 2
```

Plus a parallel `configs/gui-methods/postfix_ortho.toml` for the GUI variant
picker.

## Where this helps

### 1. Postfix capacity (primary, the experiment's own merit)

Direct: K=32 effective rank instead of K=1. Even if the result is "K=32
orthogonal directions of one shared style prior," that's still a strict
improvement on the current "K copies of the same single direction." Whether
the freed capacity gets used productively is what `analyze_cond_postfix.py`
will measure — but the structural fix is independent of whether downstream
pressure exists to fill it.

### 2. Cleaner story than the soft-fix stack

If v1 passes the validation criteria below, it obsoletes both
`slot_embed_init_std` and `contrastive_weight` for the postfix variant —
both become inert under structural orthogonality. The codebase loses two
hyperparameters and gains one architectural commitment. (`slot_embed` and
the contrastive loss stay live for cond / cond-timestep until v2 ports
the constraint over there.)

### 3. Diagnostic for the verifier's deeper concern

The verifier flagged that even with non-degenerate slot vectors, the
**positional symmetry of the splice itself** (all K slots land in
interchangeable zero-padding positions, no positional encoding distinguishes
them) might still cause attention to read them as effectively rank-1. v1
cleanly tests this: if the trained orthogonal-postfix checkpoint
*still* produces a domain-prior-only inference behavior despite measured
K-rank `postfix_embeds`, the problem is downstream of the parameter tensor
and the splice-position symmetry is the real culprit. That result kills the
"orthogonalize and pray" pathway and points to splice-position changes
(e.g., learnable per-slot positional offsets, scattered placement) as the
real next move.

If v1 *does* produce caption-conditional, K-rank inference behavior, the
splice-symmetry concern was over-stated and we have a real method.

### 4. Conditional unlock for inversion-as-teacher (downstream)

The Z-Image-style self-distillation idea (text+inverted-postfix as teacher,
text-only as student, alignment loss) was red-flagged primarily because
per-image inversions into the postfix slot share the K=1 collapse — teacher
and student features end up near-identical, alignment loss has nothing to
teach. With orthogonal slots, per-image inversions can carry K-rank
image-distinctive signal. The circularity concern (FM-MSE-coupled teacher,
no outside knowledge) survives and is independent — but at least the
degeneracy concern goes away and the experiment becomes worth running on
its merits.

This is **conditional** value — orthogonal postfix is worth doing for
reasons 1–3 alone. The unlock for inversion-as-teacher is a bonus, not the
justification.

### 5. Diagnostic for whether the K=8 prefix inversion was working

`archive/inversion/invert_reference.py` left K=8 reference-image
inversion unanalyzed. Adding an `--ortho` flag to that script (Cayley-
rotated frozen basis around a free K vectors of textual-inversion target)
re-runs the same workload with the symmetry broken. If the orthogonal
version produces meaningfully different (richer? more image-distinctive?)
inversions, that's evidence the original K=8 path was capacity-limited
all along, and resurrecting it as a real reference-tuning method becomes
viable.

## Risks / open questions

### A. Basis subspace limits expressivity

Frozen `postfix_basis` means postfix lives in a K-dim subspace of R^D
chosen at init. If the loss-optimal postfix points outside this subspace,
the Cayley-rotated parameterization cannot find it — the `lambda_slot`
magnitudes will grow trying to compensate, but the geometry is wrong.

Mitigation path: full-Stiefel parameterization via Householder reflections
(K*D - K(K+1)/2 ≈ 32K params at K=32, D=1024). Defer to v2 unless v1
clearly underperforms a free K*D parameterization at any point.

### B. Splice-position symmetry may dominate

If the K=1 collapse is genuinely caused by all K slots landing in
positionally-interchangeable padding slots — and the cross-attention
softmax averages over them regardless of slot vector identity — then
orthogonal vectors will not produce K-rank attention outputs. The
parameter tensor is K-rank; the cross-attention output is still rank-1.

This is the diagnostic-value flip side of point #3 above. The result is
informative either way, but is the dominant risk for v1 producing a flat
result. If it triggers, splice-position changes (scattered placement
across the sequence, learnable per-slot positional offsets, or one slot
per cross-attention layer instead of K-in-padding) become the next
proposal.

### C. cond / cond-timestep are not addressed in v1

The `cond_mlp` output is structurally rank-1 across slots — it produces
K*D scalars from a single D-dim pooled input through a 2-layer MLP, so
the K rows of its output are necessarily linearly dependent (same input,
same hidden, output decomposes into K linear projections of the hidden
state). Orthogonalizing `slot_embed` alone doesn't fix this; the full fix
needs either runtime QR on `cond_mlp` output rows or factoring the MLP's
last layer to enforce orthogonality of K output projections.

v2 if v1 succeeds. Until then, cond / cond-timestep stay on the soft-fix
stack.

### D. `lora_path` warm-start is incompatible

The current `postfix.toml` warms postfix training from a base LoRA
checkpoint via `lora_path`. The new orthogonal-postfix module has a
different state dict layout (`S`, `lambda_slot`, `postfix_basis` buffer)
and cannot consume legacy `postfix_embeds` weights directly. v1 starts
cold; if cold-start regresses vs warm-start by enough to matter, write
a one-shot conversion script that projects the legacy `postfix_embeds`
onto the new `postfix_basis` to seed `S` (recover the rotation) and
`lambda_slot` (recover the magnitudes).

## Validation criteria

Re-run **both** existing analyzers on the trained orthogonal-postfix
checkpoint:

`archive/bench/postfix/analyze_sigma_tokens.py`:
- `slot-symmetry max |diff|` ≫ 0 — by construction (K orthonormal rows
  cannot equal each other), so this is a sanity check, not a discriminator.
- `SVD effective DoF @ 90% energy` **= K** (or close to it) — direct check
  that orthogonality survives bf16 save/load roundtrip.
- `top-k nearest T5 tokens cosine` — should now **vary across k**, not
  collapse to identical-top-k for every slot.

`archive/bench/postfix/analyze_cond_postfix.py` (run despite mode mismatch
— the script reads `postfix_embeds` directly so it works on this variant
too):
- `pairwise cos across captions` — N/A in default `postfix` mode (the
  postfix is caption-independent). Skip this check; cond v2 will revive it.

New script `archive/bench/postfix/analyze_ortho_postfix.py`:
- `‖postfix_embeds @ postfix_embeds.T - diag(λ²)‖_F` < 1e-4 — orthogonality
  preserved end-to-end.
- Per-slot magnitude distribution `|lambda_slot|`: pass if the
  distribution is non-degenerate (max/min ratio < 100, no slot pinned at
  zero by the optimizer).
- A/B inference comparison: same prompts, base DiT vs DiT + orthogonal
  postfix vs DiT + legacy collapsed postfix. Qualitative pass = the
  orthogonal version is at least as good as the legacy one and produces
  visibly different outputs (not the same domain-prior look across all
  prompts).

Bench script lives at `bench/postfix_ortho/` per the
`bench/<method>/results/` convention from `CONTRIBUTING.md`.

## Implementation sketch

### Code paths

- `networks/methods/postfix.py::PostfixNetwork` — add `ortho` and
  `ortho_basis` kwargs; in `__init__`, if `ortho=True` and
  `mode="postfix"`, replace `self.postfix_embeds` parameter with the
  Cayley parameterization (`S`, `lambda_slot`, `postfix_basis` buffer).
  Override `append_postfix` to compute effective postfix on the fly.
  Save/load methods serialize `S` + `lambda_slot` + `postfix_basis`
  under new keys (`ortho_S`, `ortho_lambda`, `ortho_basis`); metadata
  records `ss_ortho=true` so `create_network_from_weights` dispatches
  back to this branch.
- `networks/methods/postfix.py::create_network` — wire the new kwargs.
- `configs/methods/postfix.toml` — add the ortho variant block (commented
  out; enable explicitly to switch).
- `configs/gui-methods/postfix_ortho.toml` — clean GUI variant.
- `scripts/tasks/training.py` (and `_common.py` if needed) — surface the
  variant under `make exp-postfix-ortho` (or accept the variant via
  `make exp-postfix VARIANT=ortho`; pick whichever is more consistent
  with existing variants).
- `scripts/experimental_tasks/inference.py` — make sure
  `make exp-test-postfix` works with the new variant; the spliced-into-
  crossattn flow is unchanged, only the parameterization differs.
- `archive/bench/postfix/analyze_ortho_postfix.py` — new diagnostic.
- `bench/postfix_ortho/run_bench.py` + `bench/postfix_ortho/README.md` —
  per `CONTRIBUTING.md` Tier 1.5 (efficiency / numerics revision to an
  existing method).

### Memory entries to add after v1 result lands

- `project_postfix_ortho_<verdict>.md` — the result of the v1 experiment.
  Either "structural orthogonality fixed K=1 collapse, slot_embed +
  contrastive deprecated for postfix mode" or "K=1 collapse persists
  despite K-rank parameter tensor; splice-position symmetry confirmed as
  deeper cause."
- Update `project_postfix_slot_collapse.md` with the v1 result either way.

## Out of scope (v2 candidates)

- **cond / cond-timestep orthogonalization** — needs runtime QR on
  `cond_mlp` output rows or factored last-layer parameterization.
  Defer until v1 settles.
- **Full-Stiefel parameterization** (Householder, K*D - K(K+1)/2 params).
  Defer until v1 caps out on capacity.
- **Inversion-as-teacher self-distillation** — independently red-flagged
  on circularity grounds (verifier). v1 only addresses the degeneracy
  concern, not the FM-MSE-coupled-teacher concern. Decide separately
  whether to swap teacher objective.
- **Splice-position fixes** — only worth proposing if v1 confirms (via
  point #3 in "Where this helps") that splice symmetry is the dominant
  remaining cause.
- **Reference-image inversion port** — adding `--ortho` to
  `archive/inversion/invert_reference.py` is a small follow-up if v1
  succeeds, low-priority.

## Estimated cost

Implementation: ~1 day (the Cayley parameterization mirrors
`OrthoLoRAModule` almost line-for-line; the only postfix-specific
work is the basis-choice surface and save/load round-trip for the new
keys).

Training: same as current postfix runs (~2 epochs at the existing
preset). Cold-start may need 1 extra epoch if the warm-start path is
deferred.

Evaluation: re-run existing analyzers + new ortho-specific analyzer on
the trained checkpoint (~1 hour of bench script time). Qualitative A/B
inference comparison: a few hours of human eyeballing.

Total: <1 week including a couple of training reruns.
