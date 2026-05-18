# Soft Tokens — in-artist negative sampling (idea capture)

Scratch doc for an unresolved design question on `networks/methods/soft_tokens.py`'s
InfoNCE regularizer. Captured 2026-05-08; not yet implemented or benched.

See `docs/experimental/soft_tokens.md` for the full module write-up. This doc
only covers the negative-sampling question.

## The question

The current contrastive path samples negatives by `torch.roll(text, shifts=j, dims=0)`
on whatever ended up in the batch (`networks/methods/soft_tokens.py:606`). On a
booru-style dataset that means anchor and negative differ in **both** style
(artist tag) and content (subject tags) — a confounded contrastive signal.

Should the negative-sampling policy:

- (a) stay as-is — random in-batch?
- (b) construct **hard** negatives (different artist, similar content)?
- (c) construct **weak-but-purified** negatives via **in-artist sampling**
  (same artist, different content) — the proposal this doc captures.

## Why (c) is theoretically sound

InfoNCE doesn't care which axis the model uses to win the matched-vs-mismatched
discrimination. At the velocity-field logit level (`exp(-‖v_θ - target‖² / τ)`),
**style is the easy shortcut**: artist tags fire strongly in the cached
crossattn, so most of the matched-vs-mismatched MSE gap on random negatives is
explained by style mismatch, not by content mismatch.

If soft tokens win the contrastive on style, they specialize on artist-style
features — but those features are already encoded in `crossattn_emb` (the artist
tag is right there in the prompt). That's redundant work and wastes the
~1M-param budget.

In-artist sampling matches the style axis between anchor and negative →
style-induced velocity similarity cancels → the only remaining axis InfoNCE can
win on is **content**. Contrastive gradient gets purified to point at
prompt-following, which is the load-bearing eval criterion (#3 in
`soft_tokens.md`'s "Evaluation" section).

Weak in absolute logit gap, but on the right axis.

## Why it might fall over on Anima

1. **Long-tail artist distribution.** Booru-style datasets are heavy-tailed —
   a large fraction of images are from artists with ≤ 3 samples. For those
   anchors, the in-artist negative pool is empty or near-empty. You'd silently
   fall back to random negatives for most of the batch.

   *Gating measurement:* parse `image_dataset/*.txt` for the artist tag and
   count tag frequencies. Compute the fraction of images with ≥ 2 same-artist
   siblings. If that fraction is < ~50% the proposal is structurally weak on
   this dataset.

2. **False negatives sharpen, don't soften.** Within an artist, captions are
   often highly correlated — same series, recurring characters, scene
   templates. A same-artist "mismatch" pair may genuinely produce similar
   velocity fields because the content really is similar. That's a corrupted
   signal, not a softened one. InfoNCE log-softmax then trains the soft tokens
   to discriminate things that aren't actually distinguishable in the data →
   gradient noise.

3. **Sampler complexity.** Current `torch.roll` is index-free and works with
   the existing dataloader. In-artist sampling needs:
   - Caption parsing to extract artist tag(s) per sample (booru convention is
     a tag prefix or explicit `artist:foo`; varies by dataset).
   - Per-artist index over the dataset.
   - Either a custom batch sampler (group-aware) or per-anchor lookup at step
     time.
   - Fallback policy for orphan anchors (random? skip? duplicate?).

   Meaningful engineering lift for a regularizer at `λ=0.05`.

## Diagnostic gates before implementing

The question "is the contrastive term doing anything?" isn't answered yet. The
eval criteria in `soft_tokens.md` are explicitly diagnostic:

- Run baseline at current `contrastive_weight=0.05` for ≥ 1 epoch.
- Check `‖tokens[k]‖` per layer at convergence (eval criterion #2). If layers
  converged to a shared bank, the regularizer is failing — *then* style-confound
  becomes a plausible diagnosis and (c) is worth building.
- If layers *did* differentiate across `k`, in-artist sampling is solving a
  non-problem.
- Compare against `contrastive_weight=0` (parameterization-only ablation,
  called out in the docs as the clean test). If there's no measurable
  prompt-following gap between λ=0 and λ=0.05, the contrastive term isn't
  load-bearing regardless of negative quality, and changing the negative policy
  is rearranging dead weight.

**Do not implement (c) before these gates produce a positive signal for
"the regularizer matters but is on the wrong axis."**

## Middle path — tag-overlap soft weighting

If diagnostics motivate style-confound control but full in-artist sampling is
too expensive, a cheaper variant captures most of the same effect:

- For each anchor `i` and each negative `j` already in the batch, compute
  Jaccard similarity `s_ij ∈ [0, 1]` over caption tag sets.
- In InfoNCE, down-weight the negative's logit proportionally:
  `logits[j] -= α · s_ij`.
- Effect: negatives that share many tags with the anchor (artist + content
  overlap) get a similarity penalty that **softens their contribution** to the
  contrastive denominator — they become less surprising mismatches and pull
  less gradient. Negatives that share few tags (truly different style and
  content) still pull full weight.

Properties:
- No sampler change. Works on existing `torch.roll` batches.
- Single hyperparameter `α` to tune (start at α ≈ 1, sweep 0.5–2.0).
- Operates on full tag-overlap rather than artist-only — broader axis control,
  but captures the artist correlation as a side effect (artist tags contribute
  to Jaccard).
- ~10 lines in `extra_forwards` after the per-sample MSE list:
  ```python
  # tag_jaccard: precomputed (B, B) matrix from cached caption tag sets
  for j in range(1, k_eff + 1):
      rolled_jaccard = torch.roll(tag_jaccard.diagonal(j), shifts=0)  # (B,)
      logits[j] = logits[j] - alpha * rolled_jaccard
  ```
  (`tag_jaccard` cached once per batch from the captions on disk.)

This isn't equivalent to (c) — it operates on tag-overlap rather than
artist-only — but it captures most of the style-confound control without the
sampler rewrite. If the cheap variant moves prompt-following metrics, it's
strong evidence (c) would too; if it doesn't, (c) probably won't either.

## Open sub-questions if (c) ever gets built

- **Multiple artists per image.** Some captions list collaborators (`artist:a,
  artist:b`). Define "same artist" as set intersection ≥ 1, or require full
  set match? Set intersection is more permissive (more pool coverage); full
  match is stricter (cleaner style control).
- **Anonymous / missing artist tags.** Treat as "all in same bucket" (huge
  pool, breaks the policy) or "exclude from in-artist sampling" (orphan
  policy). The latter is cleaner.
- **k > 1 with in-artist constraint.** If anchor's artist has only 1 sibling
  in the batch, only k=1 is possible. Fallback: pad with random negatives, or
  drop k for that anchor?
- **Interaction with `n_t_buckets`.** If specific t-buckets see only a few
  artists' samples (training scale ≪ dataset scale), the per-bucket signal
  could become artist-biased. Probably second-order, but worth a note.

## Status

- Theoretically sound, axis-aligned with the prompt-following objective.
- Implementation cost real but tractable.
- Diagnostic signal not yet present to justify build cost.
- Cheaper middle path (tag-overlap weighting) should be tried first if and
  when diagnostics motivate it.
