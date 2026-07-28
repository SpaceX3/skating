# Semantic staged-training handoff

## Current state

- Remote project: `/home/v100/ZYQ/skating`
- Branch: `rag`
- HEAD observed before this handoff: `af01b16 bad performance`
- Runtime: `/home/v100/anaconda3/envs/skating-action/bin/python` (Python 3.8.20)
- All attempted implementation changes from the previous agent were reverted.
- `semantic_main.py`, `semantic_rag.py`, `action_rag.py`, and `model.py` match `HEAD`.
- The baseline suite passed before any edits: 22 tests via `python -m unittest discover -v tests`.
- Backward compatibility with old semantic/RAG parameters, checkpoints, and candidate files is explicitly **not required**. Prefer a clean v2 design and fail clearly when a v1 artifact is supplied.

## User goal

Replace the current all-at-once semantic optimization with a staged pipeline:

1. Train query-side `action`, `coarse`, and exact `element` recognition first.
2. Freeze that classifier and retrieve FineFS references using its predicted distributions.
3. Learn direct GOE and reference-relative `delta GOE` without allowing those losses to damage classification.
4. Feed RAG explicit element semantics plus reference GOE / delta GOE / confidence, rather than only a mixed semantic token.

The central requirement is **soft routing**. Do not route solely by the top-1 element prediction.

## Why the current model should be split

The current `FineFSSemanticRAG` shares one query encoder across classification, retrieval, citation, direct GOE, relative GOE, and final GOE. All losses are optimized together in `semantic_main.semantic_losses`.

Observed current results for the default best retrieval checkpoint:

- Checkpoint: `semantic_seed2026_20260723_160005_best_epoch050_mrr0.2570_goemae1.3456.pth`
- Validation exact-element accuracy: `0.3784`
- Test exact-element accuracy: `0.3877`
- Validation closed-set reranked MRR: `0.2570`
- Test closed-set reranked MRR: `0.2596`
- Validation final/direct/evidence GOE MAE: about `1.3456 / 1.3439 / 1.4473`

The evidence branch is currently worse than the direct branch. The classifier is also too inaccurate for hard top-1 routing; a wrong prediction would remove every correct reference before GOE estimation.

## Data constraints

- Semantic classifier, reference bank training, validation, and test all use FineFS.
- Current corpus: 1,127 FineFS videos and 42,143 prototypes.
- Current video-disjoint split: 789 train, 169 validation, 169 test videos.
- The exact-element vocabulary currently has 242 entries, including combinations, `<background>`, and `<unknown>`.
- Only FineFS-train videos may enter the semantic reference bank.
- FineFS validation/test videos are query-only and must never enter that bank.
- FS1000 is used later as the RAG query dataset. It is not semantic pretraining data.
- Keep the existing FS1000/FineFS leakage exclusions when rebuilding the corpus or candidate artifacts.
- Never use validation or test labels for model selection beyond reporting validation metrics. Test remains final-only.

## Recommended v2 architecture

### 1. `SemanticQueryClassifier`

Create a standalone module, preferably in `semantic_rag.py` or a new focused module.

Input:

```text
query DINOv2 CLS+patch-mean feature [B, 2048]
```

Output:

```text
query_token          [B, 256]
query_retrieval      [B, 256], L2-normalized
action_logit         [B]
coarse_logits        [B, C]
element_logits       [B, E]
action_probability   [B]
coarse_probabilities [B, C]
element_probabilities[B, E]
```

The query label must not be an input. Labels are used only by the classifier losses and evaluation.

Use a clear method such as `classify_query(query_features)` so downstream stages do not need dummy references.

### 2. `ElementConditionedGOE`

Create a separate model that accepts a frozen classifier or frozen classifier outputs. It should own reference encoding, pair encoding, citation, direct GOE, delta GOE, confidence, and fusion.

Do not reuse a single opaque reference encoding for all concerns. Prefer these explicit paths:

- Retrieval visual encoder: reference DINO feature only.
- Soft semantic routing: known reference action/coarse/element metadata combined with predicted query probabilities.
- GOE pair encoder: query token, reference visual feature, raw DINO cosine, reference element/coarse embeddings, reference GOE, BV, and panel score.

This prevents GOE/BV/panel metadata from silently affecting what is named a visual similarity.

### 3. `SemanticPipeline`

Use a thin inference wrapper that combines the frozen classifier and GOE model. The final v2 semantic checkpoint should be self-contained and include both state dictionaries and all vocabulary/config metadata needed by candidate generation and RAG.

## Stage A: classifier training

Add a dedicated mode instead of a warmup hidden inside the old joint loop:

```text
semantic_main.py --mode train-classifier
```

Recommended loss:

```text
L_classifier = lambda_action * L_action
             + lambda_coarse * L_coarse
             + lambda_element * L_element
```

Implementation details:

- `L_action`: class-balanced BCE over every query.
- `L_coarse`: class-balanced cross entropy over action queries only.
- `L_element`: cross entropy over action queries with valid exact-element labels only.
- Compute class weights from unique action instances or use prototype weights so long actions with many prototypes do not dominate.
- Keep `<background>` out of the action-only element objective; action/background already has its own head.
- Preserve an explicit mapping from each element ID to its coarse class and report hierarchy consistency.
- Consider effective-number or capped inverse-frequency element weights because the 242-way distribution is highly imbalanced. Log both weighted loss and unweighted metrics.

Required validation metrics:

- Action accuracy, balanced accuracy, precision, recall, and F1.
- Coarse top-1 accuracy on action queries.
- Exact-element top-1, top-5, and top-10 accuracy on action queries.
- Exact-element metrics per coarse class.
- Top-k coverage weighted by action instance, not raw prototype count.
- Prediction entropy and confidence calibration if practical.

Select the classifier checkpoint using validation element top-k coverage followed by top-1 as a tie-breaker. Do not decide the routing width until top-5/top-10 coverage is measured.

Proposed checkpoint marker:

```text
training_stage = finefs_semantic_classifier_v2
```

## Stage B: soft retrieval and GOE training

Add a separate mode:

```text
semantic_main.py --mode train-goe \
  --classifier-checkpoint /path/to/classifier_v2.pth
```

Freeze the classifier and its query encoder initially. Do not include classifier losses in this optimizer. An optional later experiment may unfreeze the final query-encoder layer with a much smaller learning rate, but the default should remain frozen.

### Soft routing score

For query `q` and bank reference `r`, use an explicit score such as:

```text
score(q, r) = alpha * cosine(z_query, z_reference)
            + beta_element * log(P(element_r | q) + eps)
            + beta_coarse  * log(P(coarse_r  | q) + eps)
            + beta_action  * log(P(action/background status of r | q) + eps)
```

Constrain `alpha` and all `beta` values to be non-negative, for example with `exp`/`softplus` and sensible caps.

Important behavior:

- Search the full FineFS-train bank in batches.
- Take a retrieval pool such as top-64.
- Deduplicate by action instance.
- Keep top-8 for pair/citation processing.
- Do not discard candidates because their element is outside top-1.
- If efficiency later requires top-M element routing, union it with a global visual fallback pool and verify that recall does not drop.

The reference element/coarse IDs are known database metadata. The query element/coarse values must remain predictions.

### GOE paths

Expose these values separately:

```text
direct_goe
reference_goe_per_candidate
delta_goe_per_candidate
candidate_goe = reference_goe + delta_goe
evidence_reference_goe
evidence_delta_goe
evidence_goe = evidence_reference_goe + evidence_delta_goe
goe_confidence
goe_gate
predicted_goe = gate * evidence_goe + (1 - gate) * direct_goe
```

Keep values bounded to the valid GOE grade range `[-5, 5]`.

Evidence weights should include:

- Citation probability.
- Valid-score mask.
- Soft element compatibility `P(element_r | q)`.
- Optionally soft coarse/action compatibility.

When evidence is absent or confidence is low, the final prediction must fall back to `direct_goe` without NaN/Inf.

### GOE training pairs

Train the delta estimator on scored, cross-video, same-element pairs:

```text
delta_target = query_goe - reference_goe
```

Avoid a pure oracle/inference mismatch:

- Use guaranteed same-element pairs to teach the delta estimator.
- Also train/evaluate on candidates obtained from the frozen predicted soft route.
- Report oracle-route and predicted-route metrics separately so classifier routing errors are visible.
- Never feed the query ground-truth element into the predicted-route forward path.

Suggested GOE losses:

```text
L_goe = lambda_retrieval * L_multi_positive_retrieval
      + lambda_citation  * L_quality_aware_citation
      + lambda_delta     * SmoothL1(delta_goe, delta_target)
      + lambda_evidence  * SmoothL1(evidence_goe, query_goe)
      + lambda_direct    * SmoothL1(direct_goe, query_goe)
      + lambda_final     * SmoothL1(predicted_goe, query_goe)
```

Select the main GOE checkpoint by validation final GOE MAE. Save and report a separate best-retrieval checkpoint if required. Do not force one checkpoint name to represent both objectives.

Proposed checkpoint marker:

```text
training_stage = finefs_semantic_goe_v2
format_version = finefs-semantic-v2
```

## Stage C: FS1000 RAG

The RAG model should consume explicit semantic outputs instead of inferring everything from one token.

Recommended per-window RAG inputs:

```text
query_token
expected_element_embedding
expected_coarse_embedding
action_probability
element_entropy
coarse_entropy
direct_goe
evidence_reference_goe
evidence_delta_goe
evidence_goe
predicted_goe
goe_gate
goe_confidence
citation_entropy
retrieval top-1 score and margin
valid-reference ratio
dynamic context
```

Compress the 242-way element distribution with an expected embedding instead of concatenating all probabilities:

```text
expected_element_embedding = P(element | q) @ element_embedding.weight
expected_coarse_embedding  = P(coarse  | q) @ coarse_embedding.weight
```

This meets the requirement that RAG receives element information while keeping the input dimension stable and differentiable.

RAG should receive both `evidence_reference_goe` and `evidence_delta_goe`; do not pass only their sum. Their separate values let the TES residual distinguish retrieved baseline quality from the predicted correction.

The semantic classifier and GOE model should be frozen during FS1000 RAG training. Only the RAG/TES correction layers should be trainable unless an explicit ablation says otherwise.

## Artifact and CLI break

Backward compatibility is not required, so make the version break explicit:

- Reject old semantic checkpoints with a clear `format_version` error.
- Regenerate candidate `.npz` files after the v2 semantic checkpoint is trained.
- Store `format_version`, semantic checkpoint identity/hash, corpus version, routing configuration, and top-k settings in every candidate file.
- Use a new candidate directory such as `rag_artifacts/candidates_v2` so old and new files cannot be mixed accidentally.
- Do not silently load a v1 candidate artifact.

Suggested script commands:

```text
bash scripts/run_action_rag.sh train-semantic-classifier
bash scripts/run_action_rag.sh train-semantic-goe
bash scripts/run_action_rag.sh semantic-val
bash scripts/run_action_rag.sh semantic-test
bash scripts/run_action_rag.sh candidates-v2
bash scripts/run_action_rag.sh train-rag
```

Use `/home/v100/anaconda3/envs/skating-action/bin/python` for model training, evaluation, candidate generation, compilation, and unit tests.

## Files likely to change

- `semantic_rag.py`: new classifier, GOE model, soft routing, explicit GOE outputs.
- `semantic_main.py`: separate classifier/GOE modes, losses, checkpoint formats, metrics.
- `semantic_data.py`: instance-balanced classifier labels or helper mappings if needed.
- `scripts/precompute_action_candidates.py`: v2 checkpoint loading and soft-route candidates.
- `action_rag.py`: RAG consumption of element embeddings and explicit reference/delta GOE.
- `model.py`: constructor wiring for the new RAG semantic input dimensions.
- `main.py`: load v2 semantic pipeline and save its architecture metadata.
- `eval.py`: reconstruct v2 RAG exactly from checkpoint config.
- `scripts/run_action_rag.sh`: staged commands and new candidate directory.
- `tests/test_semantic_rag.py`: classifier, routing, delta, fallback, and no-label-leakage tests.
- `tests/test_action_rag.py`: explicit semantic-context RAG tests.
- `tests/test_checkpoint_compatibility.py`: replace old compatibility expectations with clear v2 rejection tests where appropriate.
- `README.md`: update the semantic and RAG workflow after code is working.

## Required tests

Add focused tests for these behaviors:

1. Classifier inference accepts only query features and never a query element label.
2. Higher predicted probability for a reference element monotonically increases its soft-route score.
3. A wrong top-1 element does not automatically eliminate the correct reference.
4. Full-bank and per-query candidate score paths return equivalent values for equivalent inputs.
5. `evidence_goe == evidence_reference_goe + evidence_delta_goe` before bounding.
6. Invalid-score references receive zero GOE evidence weight.
7. No valid evidence produces finite `direct_goe` fallback.
8. Same-video references are excluded for FineFS-train queries.
9. Validation/test videos never enter the reference bank.
10. Prototype weighting does not over-count one action instance.
11. RAG receives finite element context and explicit delta GOE tensors.
12. Old v1 checkpoints/candidates fail with an informative version error.

Verification sequence:

```bash
/home/v100/anaconda3/envs/skating-action/bin/python -m py_compile \
  semantic_rag.py semantic_data.py semantic_main.py action_rag.py \
  model.py main.py eval.py scripts/precompute_action_candidates.py

/home/v100/anaconda3/envs/skating-action/bin/python -m unittest discover -v tests
```

Then run:

- A synthetic CPU forward/backward smoke test for each training stage.
- A one-epoch, limited-batch real FineFS classifier smoke test.
- A one-epoch, limited-batch real FineFS GOE smoke test loading the classifier checkpoint.
- Candidate generation for one FS1000 video.
- One limited FS1000 RAG train and validation batch.
- Full validation only after all smoke checks pass.

## Implementation order

1. Re-run baseline tests and inspect `git status` before editing.
2. Add classifier top-1/top-5/top-10 evaluation to establish routing coverage first.
3. Implement `SemanticQueryClassifier` and `train-classifier` checkpointing.
4. Implement the frozen-classifier `ElementConditionedGOE` stage.
5. Implement full-bank soft routing and explicit GOE decomposition.
6. Add v2 checkpoint/artifact validation and regenerate candidates.
7. Wire expected element/coarse embeddings plus delta GOE into RAG.
8. Add tests, compile, run unit tests, then real-data smoke tests.
9. Only then launch full classifier and GOE training.

## Decisions that should not be revisited without user approval

- No old checkpoint/parameter compatibility layer is needed.
- Do not use top-1 element hard routing as the default.
- Query ground-truth element is supervision only, never an inference input.
- Keep direct GOE as a confidence-controlled fallback.
- Pass element semantics and reference/delta GOE explicitly to RAG.
- Keep FineFS train/validation/test video-disjoint and the reference bank train-only.
