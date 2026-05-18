# postfix_residual — image-conditional postfix as DirectEdit's ψ_src residual carrier

Companion proposal to [`img2emb_plan.md`](img2emb_plan.md) and
[`orthogonal_postfix.md`](orthogonal_postfix.md). Different question, same
neighbourhood. img2emb asks "can we learn an image → ψ encoder that replaces
text conditioning"; this proposal asks "can we learn a small image-conditional
postfix that *augments* T5(tags) so DirectEdit's dry-run reconstructs without
disturbing the text-editable interface."

## Motivation

Two facts in tension:

1. `make exp-test-directedit-dry` (ψ_src = ψ_tar = T5(tags)) reconstructs *mostly* but
   not 100%. That means booru-style tags + T5 don't carry enough information for the
   frozen DiT to recover the source image through the inversion loop. The residual
   gap is real, and Anima Tagger can't close it from inside the text manifold
   alone — it's a tag-vocabulary-and-T5-bandwidth limit, not a tagger-architecture
   limit. (Mean-pool, closed vocab, threshold cliffs etc. compound it but aren't
   the structural cause.)
2. The *tag string* is DirectEdit's edit UI. "Edit `sad` → `smiling`" is a
   product; "edit a vector" is not. Replacing the tagger wholesale with img2emb
   removes the brittleness on one axis (fidelity) at the cost of the affordance
   that makes DirectEdit usable. It also re-imports the classical-inversion
   target ("ψ that generates this image from noise") that DirectEdit was
   designed to *sidestep* — see `docs/experimental/directedit_editing_v3.md` and
   the discussion that preceded this proposal.

The decomposition that resolves both:

```
ψ_src = T5(tags) + postfix(image)             # text covers what tags express,
                                              # postfix fills the visual residual
ψ_tar = T5(edited_tags) + postfix(image)      # edit acts only on the text channel;
                                              # postfix stays pinned across src→tar
```

Postfix is the right primitive because it (a) is already wired into Anima, (b)
lives natively in cross-attention conditioning space (no manifold-alignment
problem to debug), (c) is *additive* on top of T5 rather than replacing it
(text editability survives), and (d) is asked for a *small residual* rather
than a full image embedding — a much softer optimization target than
classical inversion.

## What this is not

- **Not img2emb.** img2emb (`docs/proposal/img2emb_plan.md`) trains an encoder
  to produce K tokens that *replace* T5 cross-attention via FM loss from
  noise. That's a hard target (classical inversion in amortized form).
  postfix_residual trains a smaller bank of tokens to *augment* T5(tags) and
  is supervised against DirectEdit's actual contract (dry-run reconstructs).
  The two proposals are complementary, not redundant: img2emb is "no text",
  postfix_residual is "text + residual".
- **Not IP-Adapter.** IP-Adapter routes image conditioning through a
  *parallel* KV branch (`to_k_ip`/`to_v_ip`). For DirectEdit that decouples
  the edit-acting channel (text) from the fidelity-carrying channel (KV) so
  cleanly that text edits can't touch the visual residual — which sounds good
  but also means the edit Δ never propagates into the visual stream. Postfix
  keeps both on the same conditioning sequence so the DiT processes them
  jointly and edits can shift attention away from the residual where needed.
- **Not a replacement for anima-tagger.** Anima Tagger keeps its role as the
  *edit-handle dispenser*. Postfix downgrades the tagger's fidelity burden:
  if the tagger misses a concept, postfix absorbs it; the user just can't
  *edit* that concept via text, which is the honest tradeoff.

## Architecture

Reuse the existing `cond+ortho` postfix path (the v4 config currently shipped
in `configs/methods/postfix.toml`) and swap its conditioning input from
pooled-T5 to **pooled-PE**.

```
image → frozen PE-Core-L14-336 → patch tokens → mean-pool (or attention-pool)
                                                       ↓
                          existing cond_mlp (LN + ortho-gated maxabs) → K×D postfix
                                                       ↓
                          splice into cross-attn seq via end_of_sequence / front_of_padding
```

Concretely:

- **Same `PostfixNetwork` class**, new `mode="image_cond"` (sibling of
  `mode="cond"`). Drops the caption-pooling step; reads cached
  `{stem}_anima_pe.safetensors` produced by `make preprocess-pe` and pools.
- **Same LN + ortho-gated maxabs-pool fix** from
  `project_postfix_ortho_ln_fix` (the load-bearing structural fix for slot
  collapse). The fix was about the *conditioning path*, not which modality
  fed it — should port to PE input without redesign. Re-validate on the new
  modality before trusting the K-rank assumption.
- **Image-only conditioning, NOT image + text.** Otherwise the postfix
  becomes a function of `tags_src`, and `ψ_tar = T5(tags_tgt) + postfix(image,
  tags_src)` is ambiguous (do you recompute postfix with `tags_tgt`?). Keep
  postfix text-independent so DirectEdit can treat it as a frozen offset
  across the src→tar swap. The text channel carries all editable structure.
- **K starts at 8.** Current postfix is K=48 nominal but effectively K=1
  pre-fix and ~K-rank post-fix; the image-residual carries strictly more
  information than the domain prior so K=8–16 is a reasonable starting
  budget. K too small → can't carry the residual; K too large → tag-redundant
  overgrowth (see "lane overgrowth" below).

## Training

Single stage. Loss:

```
L = L_FM + λ_zero · ‖postfix(image)‖²  + λ_dry · L_directedit_dry
```

- `L_FM`: standard flow-matching MSE through frozen DiT, with
  `crossattn_emb = T5(tags) + postfix(image)` (post-splice). The tags are
  the actual training-time captions of the image, not edited variants.
- `λ_zero · ‖postfix‖²`: zero-norm prior. Encourages postfix to do **as
  little as possible** — only encode what T5(tags) can't carry. Without this,
  postfix is incentivized to over-explain (carry tag-redundant info) because
  FM loss doesn't distinguish which channel covers a concept. λ_zero is the
  primary mechanism preventing lane overgrowth.
- `L_directedit_dry`: optional but desirable. Actual reconstruction loss
  through the DirectEdit inversion+denoise loop with ψ_src = ψ_tar =
  T5(tags) + postfix(image). This is the goal that motivated the whole
  proposal; training against FM alone might not move the dry-run metric. If
  the inversion loop is too expensive to run in-training, gate this auxiliary
  loss to every Nth step.

Drop:

- Caption-conditional contrastive term (`contrastive_weight`). It was a
  symmetry-breaking hack for pure-text cond postfix; image conditioning
  already breaks symmetry across the batch.

Keep:

- `lambda_init=0.3` style warm-start so postfix output starts at
  non-trivial magnitude — same rationale as bench `20260511-1622-cond-v2-ln-final`
  (cf. `configs/methods/postfix.toml` comments).
- ortho basis (`ortho=true`, `ortho_basis=svd_te` or random). Structural
  orthogonality is the slot-collapse fix; don't drop it for image cond.
- PE feature caching pipeline (`make preprocess-pe`) — already produces
  `{stem}_anima_pe.safetensors`, identical to IP-Adapter's input. No new
  preprocessing.

## Slot collapse and lane overgrowth — two distinct failure modes

1. **Slot collapse** (the classical postfix failure, `project_postfix_slot_collapse`):
   K slots converge to identical outputs because zero-init + symmetric splice
   has the same gradient through every slot. Fix: existing cond+ortho path
   (LN + ortho-gated maxabs + structural orthogonality from SVD basis). Risk
   for this proposal: the fix was validated on pooled-T5 input; need to
   re-verify on pooled-PE input. PE features have different statistics
   (lower entropy on Anima dataset — `project_pe_feature_diagnostics`
   reports PR=6.2), so the LN constants and λ_init may need re-tuning.
2. **Lane overgrowth** (new failure mode introduced by this design):
   postfix learns to encode tag-redundant content because FM loss doesn't
   penalize redundancy. Symptom: editing tags only partially shifts the
   image because postfix re-asserts the original concept. Diagnostic: drop
   tags entirely (ψ = postfix only) and try to reconstruct. If postfix-alone
   reconstructs well, postfix has overgrown its lane — it's no longer a
   residual, it's a parallel full encoder. Fix: increase λ_zero, decrease K,
   add an explicit tag-orthogonality loss (postfix output decorrelated from
   T5(tags)).

## Implementation steps

1. **Add `mode="image_cond"` to `networks/methods/postfix.py`.** New branch
   in `PostfixNetwork.__init__` that builds the cond_mlp with `d_in` matching
   PE feature dim (1024 for PE-Core-L14-336) instead of the T5 hidden dim.
   Forward path reads pooled PE features from the batch instead of
   pooled-T5.
2. **Wire PE feature feed.** The IP-Adapter pipeline already loads
   `{stem}_anima_pe.safetensors`; reuse the loader. Decide whether to mean-
   pool inside `PostfixNetwork.forward` or expose a pooling strategy kwarg
   (`pool="mean"|"attention"|"cls"`). Mean is the cheapest starting point.
3. **New config toggle block** in `configs/methods/postfix.toml`
   (uncomment-to-switch) for `postfix_residual` — mirrors the existing
   `cond-timestep` / `cond-func` blocks. Or, more cleanly, a fresh
   per-variant file under `configs/gui-methods/postfix_residual.toml`.
4. **DirectEdit integration.** `library/inference/directedit.py` already
   builds ψ from T5; add a path that asks an optional postfix module
   (`postfix_image_cond`) for residual tokens given the source image's PE
   features, then sums into ψ_src and ψ_tar identically (since postfix is
   image-only and the source is fixed across the edit). Should not require
   structural changes to the inversion loop.
5. **Bench harness.** `bench/postfix_residual/` (new) with the standard
   `_common.py` envelope. Two primary metrics:
   - **Dry-run reconstruction** (LPIPS / DreamSim vs source image, ψ_src = ψ_tar
     = T5(tags) + postfix(image)) — the goal metric.
   - **Edit fidelity** (does editing one tag produce a visible coherent edit
     in only the corresponding region?) — guards against lane overgrowth.
6. **Lane-overgrowth diagnostic.** Single-batch test in the bench harness:
   evaluate `ψ = postfix(image)` (zero text) and check that reconstruction
   *fails*. Run at every checkpoint.

## Validation plan

| Question | How to answer |
|----------|---------------|
| Did dry-run reconstruction actually improve? | LPIPS / DreamSim on 20 held-out images. Target: noticeably lower than `T5(tags)` alone. |
| Does postfix stay in its lane? | Postfix-only reconstruction must fail. If it succeeds, λ_zero too low. |
| Are edits still coherent? | 10 manually-curated edits (`sad→smile`, `day→night`, etc.). Compare against tag-only DirectEdit. |
| Does it compose with other adapters? | LoRA on top of postfix_residual: do edits still work? |
| Does it generalize across artists? | Test on images from artists not in the training set (`bench/dcw/dataset_diverse/` style). |

## Decisions to make

- **Pooling strategy.** Mean-pool vs attention-pool over PE patch tokens. Mean
  is the IP-Adapter resampler's input baseline. Attention-pool (with a learned
  query) is closer to what img2emb_plan.md describes. Default mean; revisit if
  postfix can't carry enough information.
- **K.** Start at 8. Compare K∈{4, 8, 16}. Larger K is more capacity but more
  lane-overgrowth risk.
- **λ_zero schedule.** Constant vs warmup-from-zero vs warmup-to-zero
  (annealed-residual). Constant first; sweep if reconstruction stalls.
- **L_directedit_dry frequency.** Every step (expensive), every 10 steps
  (cheap, may not steer enough), or off (rely on FM-loss alone, hope the dry-
  run metric improves anyway). Start with every 10 steps gated by
  `--dry_loss_interval`.
- **Conditioning input.** Pooled PE (cheap, manifold-narrow per
  `project_pe_feature_diagnostics`) vs full PE patch tokens via a small
  attention head (more expressive, more parameters). Pooled is the v1
  default; full-patch is the natural v2 upgrade and converges this proposal
  toward IP-Adapter's resampler shape.
- **Compose with other postfix variants?** Probably not — `postfix_residual`
  and the domain-prior `anima_postfix_ortho_v4` both splice into the same
  EOS region, and stacking them just duplicates the carrier. Treat as
  alternatives.
- **Inference path.** Anima Tagger → tags → T5 → ψ_src text channel; source
  image → PE → postfix → ψ_src residual channel; ψ_tar swaps text-channel
  only. Document explicitly in DirectEdit nodes
  (`custom_nodes/comfyui-anima-directedit/`) since the user-facing API needs
  a place to drop the source image *and* the source tags (currently it
  reuses the inverted image, no separate input needed).

## Relationship to other proposals

- **`img2emb_plan.md`**: a *replacement* for text conditioning, motivated by
  caption-less generation use cases. Trains against FM loss from noise.
  This proposal is *additive* to text, motivated by DirectEdit. They can
  coexist: img2emb is the "give me an image-conditioned generator" path;
  postfix_residual is the "give me an editable image-conditioned anchor"
  path.
- **`orthogonal_postfix.md`**: the structural-orthogonality move on the
  pure-text postfix. This proposal is downstream of that — it assumes the
  ortho fix carries over to image-conditional postfix (re-validation step
  required, see "slot collapse" above). If the fix doesn't port, this
  proposal blocks on the slot-collapse problem in PE-input form.

## Why no FM-from-noise objective

The same logic that argues against img2emb as DirectEdit's ψ_src applies
here: classical inversion ("ψ such that DiT generates this image from noise")
is the hard problem DirectEdit was built to avoid, and asking postfix to chase
it re-imports that brittleness. Postfix's contract is *softer*: be the
residual that, given T5(tags), survives the inversion+denoise loop. FM loss
is used as an auxiliary because it's cheap and steers in the right direction,
but the goal metric is dry-run reconstruction, not from-noise generation.
This is intentional and is the core insight separating this proposal from
img2emb.
