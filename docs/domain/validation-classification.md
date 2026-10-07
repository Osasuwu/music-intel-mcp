## Validation / classification (V0)

The **validation core** is the category-agnostic deep module every derivation pipeline feeds candidates into. Python: `src/music_intel_mcp/validation.py`. Issue #62, decision `bce66b6e`. It is the *only* place the three-class verdict lives — audio/scene/temporal (#63–#65) build `Candidate`s and hand them here; they do not classify themselves.

- **`Candidate`** — a category-neutral summary of one derived pattern: `candidate_id`, `category`, `cluster_size`, `cluster_share`, `evidence_count`, `coverage`, `confidence`, optional `temporal_stability_score`, plus the pass-through `structural_descriptor` / `sample_tracks` / `actionability_hint`. The validator reads only the scalars; the descriptor rides through onto the resulting root/tendency.
- **`DatasetContext`** — `n_unique_tracks` + `history_span_days`, the dataset-wide facts the temporal gate depends on. Passed once per run (constant across candidates), not per candidate.
- **`Validator(params=ValidationParams())`** — `classify(candidate, ctx) -> Classified` (one verdict) and `validate(candidates, ctx) -> ValidationOutcome` (batch → the three RootProfile sections `roots[]` / `tendencies[]` / `quality_log[]`, input order preserved).

The four V0 tests and their order:

1. **G — calibration** (`evidence_count >= evidence_count_floor`, default 50) and **E — coverage floor** (`coverage >= coverage_floors["tendency"]`, default 0.3) are the **hard floors**: failing either → `artifact_suspect`, suppressed from the user-facing sections and logged to `quality_log[]` (G checked first — no evidence is a more fundamental failure than thin coverage; details match the schema example: `{evidence_count, floor}` resp. `{coverage, floor_artifact}`).
2. Survivors are at least a `tendency`. **E (root floor)** `coverage >= coverage_floors["root"]` (0.5) sets `coverage_pass`; **A — confidence floor** `confidence >= confidence_floor` (0.6) sets `confidence_pass`; **D — temporal stability** is evaluated **only** when the gate is open (`n_unique_tracks > N_THRESHOLD` AND `history_span_days > T_THRESHOLD_DAYS`) *and* the candidate carries a score — then it must clear `temporal_stability_floor` (new param, default 0.5, `(calibrate)`). Otherwise it is `not_evaluated` and the candidate is forced to `tendency` (we never assert stability we could not measure).
3. A candidate that fails **no** promotion test is a `root`; otherwise a `tendency` carrying `failed_tests[]` — `"coverage_floor"`, `"confidence_floor"`, `"temporal_stability"` (evaluated-but-low), or `"temporal_stability_not_evaluated"` (gate closed / no score). Failed tests accumulate.

Every threshold is read from `method_params.validation` (`ValidationParams`) — none hardcoded; they are placeholders until calibrated on the owner's real data in #66. Honest-empty holds structurally: zero passing candidates → empty `roots[]` with a populated `quality_log[]`.
