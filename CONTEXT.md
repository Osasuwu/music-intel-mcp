# CONTEXT.md — music-intel-mcp domain model

**Status: live, grown inline via `/grill`.** Authoritative source for terminology, invariants, and architectural decisions.

The brainstorm in memory (`music_intel_mcp_project_revival`, `05e849cb-2026-05-13`) is *input*. Where this file disagrees with the brainstorm, this file wins.

---

## Product framing

**One-line job:** show the user the **causal root** of their listening history — with evidence — so they can act on it (amplify / dampen / explore), instead of trusting an opaque streaming algorithm.

**Pillar order is causal, not parallel:**

1. **Understand** (primary) — derive roots from history. The product *is* this.
2. **Discover** (derived) — recommendations are downstream of root model; without a root, no recommend.
3. **Act** (byproduct) — playlist push, MCP surface. Side-effect of (1)+(2), not the goal.

This reorders the brainstorm's "3 pillars" (which presented them as parallel).

## Glossary

- **Root** — underlying driver that explains *many* surface listens (e.g. *"prefers fragile vocal timbres in evening listening"*). What this product surfaces.
- **Symptom** — surface listening pattern (top-artist counts, recent burst, genre frequency). What streaming services use directly. Symptom ≠ root; a tool that recommends from symptoms reinforces bubbles.
- **Retrospective** — analytical pass over history that derives roots with evidence, presented to the user.
- **Bubble** — local cluster of recent listening that crowds out the rest of the user's actual taste. Anti-bubble = penalty against the current bubble, computed from the *root* model not from raw recency cosine.
- **Evidence chain** — for any insight or recommendation, the ordered list of inputs/derivations that produced it. Must be inspectable by the user.
- **Root category** — a class of evidence used to derive roots. V0 ships three: `audio` (BPM/energy/valence/danceability cluster — i.e. scalar audio features that exist for any track regardless of genre), `temporal` (time-of-day / day-of-week / season pattern), `scene` (cultural/scene affinity via tags). V1 adds five more: `timbre` (deep audio-derived instrumentation signature: MFCCs, spectral envelope, instrument recognition), `career_phase` (artist-arc preference), `lyrics` (NLP themes), `structural` (song-form preference), `context_chain` (sequence/skip patterns). Naming note: `audio` was previously called `acoustic` (after the AcousticBrainz dataset) — renamed because "acoustic" is also one of Spotify's per-track scalar features (`acousticness`), and the dual use was ambiguous. AcousticBrainz the dataset keeps its proper-noun name.
- **Category weight** — per-user 0..1 scalar per root category indicating how much this category drives this user's taste. Excluded categories = 0. Lets the product fit users with very different listening modes (lyric-driven vs. pure-instrumental, scene-loyal vs. audio-driven, etc.).
- **V0 root model** — proxy approximation using only {audio, temporal, scene}. MUST be labelled as proxy in all outputs.
- **V1 root model** — full 8-category derivation, target precision. Includes audio analysis as a **deliberate killer feature** — no one else does it because it's slow/expensive; this project accepts that cost.
- **`RootProfile`** — versioned JSON artifact that *is* V0/V1 output. Top-level keys: `schema_version`, `model_maturity` (proxy/full), `snapshot_id`, `user_id`, `generated_from`, `category_weights`, `method_params`, `roots[]`, `tendencies[]`, `epochs[]`, `quality_log[]`, `analytics`. Canonical worked example: `schemas/root_profile.v0.example.json`. Source of truth for every downstream consumer (future Discover, MCP, dashboard, playlist).
  - **Per-root shape**: `id`, `category`, `classification` (`root`|`tendency`), `structural_descriptor` (category-specific), `evidence` (cluster_size/share, evidence_count, sample_tracks, coverage), `validation_scores` (confidence, temporal_stability, coverage_pass, confidence_pass), `epoch_presence`, `actionability_hint`, `curator_prose` (nullable), `caveats[]`.
  - **Root IDs are snapshot-local** in V0 — `r-audio-1` in one snapshot is not the same as `r-audio-1` in the next. Snapshot-to-snapshot matching is V1+ work (will add `linked_to_previous_id`).
  - **`tendencies[]` is structurally identical to `roots[]`** plus a `failed_tests[]` field — separated by section for cheap downstream dispatch, not by flag.
  - **`quality_log[]`** records every rejected candidate with the failed test and details. Mandatory under transparent-rejection invariant.
  - **`analytics`** is an open carve-out for future views (novelty curve, cluster_share_over_time). V0 fields are `null`; V1+ fills them.
- **RootProfile snapshot** — single `RootProfile` generated at a point in time. Per-user store keeps a *series* of snapshots so taste evolution can be analysed across years.
- **Enricher** — module that fetches/derives track metadata (ISRC, MBID, audio features, tags, scene) from Spotify / MusicBrainz dump / AcousticBrainz dump / Last.fm and writes it to the **shared metadata store**. Never per-user-specific.
- **Shared metadata store** — additive, multi-user-ready database of immutable-ish facts about tracks/artists. Refresh TTL ~90 days per entry. More users → more cache hits → faster analysis for everyone. (Mirrors the OOP-project pattern.)
- **Per-user store** — listening history (raw events) + RootProfile snapshots, scoped to one user.
- **`root` / `tendency` / `artifact_suspect`** — three-level validation classification of every candidate pattern. `root` passes all V0 tests (coverage + confidence + temporal stability); `tendency` is a visible pattern that failed at least one test (e.g. only in one chronological half, or below confidence floor) — surfaced separately, labelled, not called a root; `artifact_suspect` fails calibration / coverage floor and is suppressed from the user-facing artifact but logged in `quality_log[]` for transparency.
- **`evidence_coverage`** — per-root scalar: fraction of tracks in the cluster that have full metadata at the queried categories. Low coverage downgrades classification.
- **`temporal_stability`** — boolean+score from chronological-split test. Computed only when `n_unique_tracks > N_THRESHOLD` AND `history_span > T_THRESHOLD`; otherwise marked `not_evaluated` and the candidate is forced into `tendency`. Defaults: N=1000, T=6 months — both `(calibrate)`.
- **`quality_log[]`** — section of RootProfile listing every `artifact_suspect` plus the test it failed. We do not silently discard — we transparently show what was rejected and why.
- **Structural descriptor** — machine-readable summary of a root's evidence (for `audio`: `{bpm_band, energy_band, valence_band, danceability_band}` each in `{low,mid,high}` plus `centroid_raw` — the raw mean per clustered dim, so the band thresholds stay re-derivable; for `temporal`: `{time_bucket, conditioned_root_id, lift}`; etc.). The source of truth for what a root *is*. Every root has one; downstream consumers read this, not prose.
- **Curator prose** — optional human-readable label for a root, generated by an LLM strictly from `{structural_descriptor, sample_tracks, scene_context}`. The curator never introduces facts absent from its input; it renders, it doesn't infer. RootProfile is valid without curator prose.
- **`method_params`** — section per category capturing the actual algorithm hyperparameters used (HDBSCAN `min_cluster_size`, `min_samples`, etc.). Part of the evidence chain — two snapshots with different `method_params` are not directly comparable.
- **`time_bucket`** — coarse temporal slot used to test conditional patterns. Default partition is `day_part × weekday_kind × season`: day_part ∈ {morning, day, evening, night}, weekday_kind ∈ {weekday, weekend}, season ∈ {winter, spring, summer, autumn} → 32 buckets. The actual partition lives in `method_params.temporal_calendar` and is configurable; default 5/2 weekday split is a guess that will be overridable per-user when the V1 feedback channel exists (shift workers, non-standard schedules).
- **Conditional pattern (temporal root)** — a statistically significant `(time_bucket, conditioned_root)` association: `lift = P(conditioned_root | time_bucket) / P(conditioned_root)` above floor, plus event-count and stability tests. The temporal category does *not* cluster in its own feature space — it qualifies roots from other categories. `conditioned_root` is the id of a root from `audio`, `scene`, or (V1+) other content categories.
- **`epoch`** — a contiguous time-segment of the user's history where listening distribution is significantly different from neighbouring segments. Detected via sliding-window KS-test on cluster distributions. Epochs are *not* roots — they are historical context. They live in `RootProfile.epochs[]`, not in `roots[]`. Each root can carry an `epoch_presence` map indicating in which epochs it was active.
- **Change-point** — boundary between two epochs, with significance score from the KS test. Stored as part of the `epochs[]` timeline.
- **`scene` root** — a coherent tag-cluster derived from Last.fm track tags via NMF on the user's track×tag matrix. Each scene root carries top-K tags with weights, top-K artists in the topic, coherence score, and the chosen K (recorded in `method_params`). K is auto-selected over a small grid {3,5,8,13} by mean topic coherence above a floor; if no K passes the floor → `scene_roots = []` (honest empty).
- **Tag canonicalization** (V1 step) — optional synonym-collapse using lightweight embeddings before NMF. Each near-synonym group becomes one canonical tag; the full mapping is recorded in `method_params.tag_canonicalization` for transparency. Not in V0 pipeline.
- **Topic coherence** — average pairwise PMI of the top-K tags within a topic. Used to (a) select K, (b) gate topic acceptance. Low-coherence topics are degenerate and rejected before validation.
- **Local data layout** — plain files, no DB. `data/history.jsonl` (append-only events), `data/cache/<track_id>.json` (mutable metadata pulled from shared store), `data/profiles/<iso>.json` (RootProfile snapshots), `data/state.json` (global pointers — schema version, last enricher run, etc.). Threshold to revisit: >500 MB total or >10 s load — then move to Parquet+DuckDB.
- **Pull-and-cache** — analyser does one bulk query to shared metadata store for all tracks of interest, materialises the result to local `data/cache/`, then runs the full analysis offline. New facts derived during the run that belong in shared store get a write-back step at the end. Per-track Supabase round-trips during analysis are forbidden.
- **`ListenEvent`** (event schema) — the source-agnostic unit of the per-user history store (`data/history.jsonl`, one JSON object per line). Shape: `track` (a `TrackRef`: optional `spotify_id`/`isrc`/`mbid` filled by the identity waterfall + always-present `name`/`artist` fallback) + `played_at` (timestamp) + `source` (tag, e.g. `ifttt_csv`/`lastfm`) + nullable `context` (a `PlayContext`: `ms_played`, `skipped` — both independently nullable). Designed so a rich IFTTT extended-history row (carries play-context) and a thin Last.fm scrobble (no play-context) both land without a schema rewrite. Decision `709f0b23-f5e6-43ab-a90d-ca0748ae310e` (see §Architectural decisions). Pydantic models in `src/music_intel_mcp/models.py`.

*(More terms added as they resolve.)*

## Shared-store schema (V0)

The shared metadata store (Supabase Postgres) holds **anonymous track-level facts only** — no `user_id`, no `played_at`, no play-context in any table. Migration: `supabase/migrations/20260609000000_shared_metadata.sql`. Python client: `src/music_intel_mcp/shared_store.py`. Issue #60, decision `f7a9fcbd`.

Tables:

- **`tracks`** — canonical identity + denormalised display facts. PK is a **canonical string id** (`mbid:…` > `isrc:…` > `spotify:…` > `name:<casefold>\x1f<casefold>`), built by `canonical_track_id()` — the same waterfall #61 formalises. Columns: `id`, `spotify_id`, `isrc`, `mbid`, `name`, `artist`, `artist_id`→artists, `fetched_at` (TTL anchor). Indexed on spotify_id/isrc/mbid for the identity waterfall.
- **`artists`** — artist-level facts (`mbid`, `name`). Defined for #61 + future artist enrichment; V0's `TrackMetadataRecord` denormalises `artist` onto `tracks` and does not yet populate this table.
- **`audio_features`** — one row per track (PK = `track_id`): `bpm`, `energy`, `valence`, `danceability`, `acousticness`, `instrumentalness`, `source`. All nullable (partial enrichment is valid).
- **`tags`** — many rows per track: `(track_id, tag, weight, source)`, unique on `(track_id, tag, source)`. Scene/cultural tags from Last.fm etc.

`TrackMetadataRecord` (pydantic, `extra="forbid"`) is the **pull unit** — a track's `tracks` row + its `audio_features` + its `tags`, plus `fetched_at`. `extra="forbid"` is the structural guard that personal data can never leak into a shared-store record.

**Store access** is `SharedStore` (Protocol) with only **bulk** ops — `get_tracks(ids)` / `upsert_tracks(records)`; no singular getter exists, so the no-round-trip rule is enforced by the type. Implementations: `InMemorySharedStore` (real store in tests + offline fallback) and `SupabaseSharedStore` (network-only, lazy-imports the `supabase` optional extra, never run in CI). `pull_and_cache(ids, store, cache, now, ttl_days=90)` serves fresh entries from the local `MetadataCache` (`data/cache/`), collects uncached/stale ids, and fetches them in **one** bulk call; stale (past-TTL) store entries are surfaced for re-enrichment, missing ids for enrichment.

## Identity resolution (V0)

The **identity waterfall** turns whatever id a history event happens to carry into the one stable join key the enrichers need — the MusicBrainz **recording MBID** (the key into the AcousticBrainz/MusicBrainz dumps for #63/#64). Python: `src/music_intel_mcp/identity.py`. Issue #61.

Waterfall, deepest rung wins: **`mbid`** (already present, passthrough) → **`isrc`** → MBID via the MB dump → **`spotify_id`** → ISRC (Spotify) → MBID. The rung reached is the `ResolvedIdentity.level` ∈ {`mbid`, `isrc`, `spotify`, `name`}; only `mbid` counts as *resolved* (`ResolvedIdentity.resolved`).

- **`IsrcMbidIndex`** (Protocol) — `lookup(isrc) -> mbid|None`, the dump leg. `MusicBrainzIsrcIndex` reads a prebuilt `<isrc>\t<mbid>` TSV extract; path is **env-pointed** (`MUSICBRAINZ_ISRC_INDEX`, falling back to `$MUSICBRAINZ_DUMP_DIR/isrc_to_mbid.tsv`) and **never committed** — a missing file yields an empty index (honest low coverage, never a crash). `InMemoryIsrcMbidIndex` is the test fixture.
- **`SpotifyIsrcSource`** (Protocol) — `lookup(spotify_id) -> isrc|None`, the Spotify leg. Optional: when absent, spotify-only tracks are honestly flagged at the `spotify` level, not dropped. A live Spotify source plugs into this seam in a later slice (Spotify track metadata carries ISRC; it is *not* one of the constrained endpoints).
- **`IdentityCache`** (`data/identity/<input-key>.json`) — keyed by the *input* canonical id, so a re-run with the same history **skips the waterfall entirely** (no re-resolution). Regenerable + gitignored, percent-encoded filenames (shared `encode_cache_key` with the metadata cache).
- **`ResolutionReport`** — keyed by input canonical id (deduplicated). Derives `counts` (per-level ledger), `mbid_coverage` (fraction resolved to MBID — feeds `generated_from.coverage_per_category`), and `unresolved` (input keys that did not reach MBID — **flagged, never dropped**, per the transparent-rejection invariant). `to_metadata_records(report, now)` writes resolved identities back to the shared store, keyed by the *resolved* canonical id (so the store dedups across users by MBID).

The same `canonical_track_id()` waterfall (mbid > isrc > spotify > name/artist) is the single source of truth for track keys — the analyser's unique-track counting now calls it directly (the old `analyzer._track_key` tuple was folded into it).

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

## Audio root pipeline (V0)

The **audio** category's deep module — `history → enrich → cluster → describe → validate`. Python: `src/music_intel_mcp/audio.py`. Issue #63, decision `01ba9bb7`. Wired into `analyze()` as the first derivation stage; it is **opt-in by dependency** — it runs only when both a `SharedStore` and an `AudioFeatureSource` are supplied, otherwise the audio sections stay honest-empty.

- **Enrichment.** `enrich_audio_features(track_ids, store, source, now)` populates `audio_features` on the shared-store records by **MBID**, then writes back (bulk read + bulk write, no round-trips). `AudioFeatureSource` (Protocol, `lookup(mbid)`) has two implementations: `AcousticBrainzDump` (production; env-pointed JSONL keyed by `mbid` — `ACOUSTICBRAINZ_FEATURES_INDEX` › `$ACOUSTICBRAINZ_DUMP_DIR/acousticbrainz_features.jsonl`; **lives outside the repo**, missing file → empty index → honest low coverage, never a crash) and `InMemoryAudioFeatureSource` (tests). The `AudioEnrichmentReport` bins every considered track into exactly one of `enriched` / `already_present` / `missing_features` / `no_mbid` (transparent rejection) and reports `coverage` = fraction now carrying features. Tracks already enriched or lacking an MBID are never looked up.
- **Two feature roles** (`AudioParams`). `cluster_features` (default `[bpm, energy, valence, danceability]`) define the distance metric, the bands, and the raw centroid — a track must carry **all** of them to enter clustering. `feature_set` (those four + `acousticness`, `instrumentalness`) is the wider enrichment target: `evidence_coverage` = fraction of a cluster's tracks carrying *every* one of the six, and low coverage downgrades classification. The two extra dims are coverage signal in V0, not yet cluster axes.
- **Clustering.** `derive_audio_roots(records, track_plays, params, validation_params, dataset_ctx)` z-scores the cluster dims (per-dim, `std==0`-guarded), fits **`sklearn.cluster.HDBSCAN`** (`min_cluster_size`/`min_samples` from `method_params.audio`; `copy=True` so the input is not mutated — the z-matrix is reused for centroids/sample distances) once over the whole user population, and maps each cluster **1:1** to a candidate. HDBSCAN noise (label −1) is dropped. A population below `min_cluster_size` short-circuits to honest-empty.
- **Determinism.** Feature rows are sorted by canonical id before clustering and sklearn's HDBSCAN has no random init → identical input yields identical roots. Cluster→`r-audio-N` ids are assigned by descending cluster size, tie-broken by smallest member id, so the prominent root is `r-audio-1`.
- **Per-cluster outputs.** `structural_descriptor` = the four `*_band` (cut by `params.band_cutoffs`, `v<lo`→low / `lo≤v<hi`→mid / `v≥hi`→high) + `centroid_raw`; `evidence` = `cluster_size`/`cluster_share` (size ÷ clustered population) / `evidence_count` (= size) / `coverage` / `sample_tracks` (the `sample_track_count` members closest to the z-space centroid, each `{track_id, name, artist, distance_to_centroid}`); `confidence` = mean HDBSCAN membership probability; `temporal_stability_score` = `1 − |first_half_frac − second_half_frac|` of the cluster's plays about the global history midpoint (`None`→`not_evaluated` when the cluster has no timestamps). `curator_prose` stays `None` in V0.
- **Coverage, two senses.** The **enricher** `coverage` is a diagnostic for the enrichment step; the **category** coverage (`AudioDerivation.coverage`, fed to `generated_from.coverage_per_category["audio"]`) is the fraction of the user's records that carry usable cluster features.

Candidates are handed to the #62 `Validator` — the audio pipeline never classifies itself.

## Invariants

- **Transparency.** Every insight, score, and recommendation carries its evidence chain. No opaque numbers. If we can't explain it, we don't ship it.
- **Causality.** The system models *roots*, not just symptoms. Symptom-only recommenders are explicitly out of scope — that's Spotify's job.
- **Actionability.** A root is only useful if the user can act on it (amplify, dampen, explore). "You listen to high-valence music" is trivia; "You listen to high-valence music exclusively on weekday mornings, never weekends" is actionable. Roots must be framed for action.
- **Discover ⊂ Understand.** The recommender cannot run before the root model exists. No standalone "Discover" path that bypasses root derivation.
- **Non-commercial honesty.** The product has no churn-cost for disappointing the user, so it deliberately surfaces high-variance / potentially-disliked candidates with reasons attached. Fear-of-disappointing is Spotify's constraint, not ours.
- **Batch-first, depth over latency.** No real-time requirement. A full analysis pass may take hours-to-days; the value lives in depth, not responsiveness. Compute budget is generous; new libraries/services are allowed when they buy precision.
- **Proxy outputs are labelled.** Anything derived from V0 root model (missing 5 of 8 categories) is shipped with an explicit "proxy / preliminary" marker. We never present V0 conclusions as final root.
- **Engine-first scope.** V0 and V1 both ship *only* the analysis engine + storage + enrichers + `RootProfile` artifact. No UI, no MCP server, no Discover/recommender, no playlist push, no dashboard. Those land in V2+, on top of a mature engine. Reason: anything built on a broken root model is wasted work.
- **Data persistence is permanent.** User history and RootProfile snapshots are kept indefinitely. A user returning a year later must find their prior data intact and analysable as a time-series. No "one-shot" runs that discard intermediate state.
- **Shared metadata, per-user analysis.** Track/artist enrichment is additive to a shared store visible to all users; analysis output (RootProfile) is per-user. Enricher writes only to shared; analyser reads shared + per-user history, writes per-user snapshot. ~90-day TTL per shared entry prevents API-cost-doubling.
- **History never leaves the user's machine.** Personal data (listening events, RootProfile snapshots) is local-only. Cloud holds only track-level metadata, which is anonymous by nature. Multi-device sync of personal data is a V2+ topic with an explicit user opt-in, not a default.
- **Honest empty.** An empty `roots[]` is a valid, correct output when the data does not warrant claims. The engine never fabricates roots to fill space. Returning "we have nothing confident to tell you" is the system working as designed; it preserves transparency more strongly than a hedged invented answer.
- **Transparent rejection.** Anything the engine considered but rejected goes into `quality_log[]` with the reason. We do not silently drop candidates — the user (or a future debugger) can always see what was thrown out.
- **Category pipeline ordering.** `temporal` is a qualifier on roots from other categories, so it must run *after* the categories whose roots it conditions. V0 order: `audio` → `scene` → `temporal` → `epochs`. Engine refuses to run temporal step if upstream categories have produced no roots and no tendencies (no content to condition on).

## Architectural decisions

Decisions live in queryable memory (`record_decision` episodes), not duplicated here. Pointers:

- **`d94c44fb-03eb-4867-8440-9910f905a903`** — repo revival + Python rewrite + jarvis-style conventions.
- **`14595ec7-6049-4be9-8b16-c966be7fcb31`** — pillar order pivot: Understand primary, Discover derived, Act byproduct.
- **`cd51fa38-a741-44a1-989a-08c92bad3355`** — root taxonomy: 8 categories with per-user weights; V0 ships {audio, temporal, scene} (was `acoustic`/temporal/scene at decision time — renamed for clarity), V1 adds {timbre, career_phase, lyrics, structural, context_chain}; batch-first compute architecture, deep audio analysis explicitly in V1 scope.
- **`870623f7-95b2-42c4-a830-f6d4c5a964a9`** — V0/V1 ship `RootProfile` JSON artifact only (option B). Engine-first scope: no UI / MCP / Discover / playlist until V2+. Storage = shared metadata store (additive, multi-user-ready, 90-day TTL) + per-user store (history + RootProfile time-series snapshots). Enricher writes only to shared.
- **`f7a9fcbd-9be9-4e68-9234-801597dfacd2`** — concrete storage: Supabase Postgres (shared metadata, anonymous) + local plain JSON/JSONL files (per-user history, RootProfile snapshots, pulled metadata cache). Pull-and-cache pattern. Personal data never leaves the user's machine.
- **`bce66b6e-8a56-455b-8082-1da894c4dbe6`** — root validation: three-level classification + four V0 tests {A,D-gated,E,G} + honest-empty invariant + transparent rejection via `quality_log[]`. V1 adds user feedback (F); bootstrap (B) reserved.
- **`01ba9bb7-a190-4ce1-a984-45a416573f3b`** — `audio` root pipeline (decision text uses the older "acoustic" name): HDBSCAN clustering, 1:1 cluster-as-root, mandatory structural descriptor, optional LLM curator prose strictly bounded to descriptor+samples+scene as inputs.
- **`00f7e1eb-9848-48c2-87f3-5123aca80bb9`** — temporal root pipeline + `acoustic`→`audio` rename. Temporal as conditional patterns over `(time_bucket, conditioned_root)`. Default 32-bucket partition (day_part × weekday_kind × season), calendar configurable. Epochs in V0 via sliding-window KS-test, in their own `epochs[]` section.
- **`91229a71-70bb-4ccd-8dbe-f60ceeaa08ac`** — scene root pipeline: NMF on track×tag matrix, K auto-selected by coherence over {3,5,8,13}, tag canonicalization deferred to V1, Last.fm artist similarity excluded from derivation (saved for V2+ cross-validation/recommender), curated taxonomies (MusicBrainz/RYM/Wikipedia) deferred to V1 enrichment.
- **`7b3adb41-09cf-44a1-a64e-f449a5b2fd4f`** — V0 exit lock: 10 falsifiable acceptance criteria for V0 done, category_weights inert in V0 (recorded but unused), `RootProfile` schema canonised in `schemas/root_profile.v0.example.json`, snapshot-local root IDs (snapshot-matching is V1+).
- **`709f0b23-f5e6-43ab-a90d-ca0748ae310e`** — V0 `ListenEvent` event schema (#59): source-agnostic, nullable play-context (whole + per-field), separate optional id fields for the spotify→isrc→mbid waterfall, required name/artist fallback; JSONL per-user history store. Pydantic models in `src/music_intel_mcp/models.py`.

## Open questions

1. ~~Operational definition of "root".~~ → resolved: 8 categories, per-user weights, V0 subset = {audio, temporal, scene}.
2. ~~V0 output artifact.~~ → resolved: `RootProfile` JSON + LLM prose rendering on top. No UI / MCP / Discover until V2+.
3. ~~Storage technology.~~ → resolved: hybrid — Supabase Postgres for shared metadata; plain JSON/JSONL files locally for per-user data + pulled metadata cache; pull-and-cache pattern; scratch `.scratch/spotify_analysis/` data seeds first migration.
4. ~~Per-user weight elicitation.~~ → resolved: weights are not used in V0 (each category produces its own section; no aggregation across categories). Default V0 vector = `{audio:1, temporal:1, scene:1, others:null}`. Per-user elicitation arrives in V1 when either a recommender (needs aggregation) or a display-ordering UX lands.
5. ~~Root validation.~~ → resolved: three-level classification (`root`/`tendency`/`artifact_suspect`); V0 tests = coverage labelling (E) + confidence floor (A) + temporal stability (D, gated by dataset size) + calibration tests (G); V1 adds user feedback loop (F); bootstrap stability (B) reserved for later.
6. ~~Schema for `RootProfile`.~~ → resolved: shape canonised in `schemas/root_profile.v0.example.json`. Shared-store schema (Supabase tables for tracks/artists/tags/audio_features) is deferred to implementation phase (vertical-slice work).
7. **LLM curator role.** → partly resolved: at root-derivation time, optional, strictly renders `{descriptor, samples, scene_context}` — never introduces new claims. Recommend-time role is V2+. Open: prompt template + a verification step that flags curator output containing facts not present in input.
8. **Audio analysis infrastructure (V1).** Essentia / librosa / both? Local-files-only or can we fetch previews from Spotify (30s clips) and analyse those?
9. ~~Audio clustering method.~~ → resolved: HDBSCAN, per-user fit, 1:1 cluster-as-root, hyperparams calibrated and stored in `method_params`.
10. ~~Temporal model.~~ → resolved: temporal roots are conditional patterns `(time_bucket, conditioned_root)` with lift+confidence+stability tests; default `time_bucket` partition is `day_part × weekday_kind × season` (32 buckets), configurable; epochs (change-point detection) live in separate `epochs[]` section, not in `roots[]`.
11. **Validation thresholds.** Concrete defaults for `N_THRESHOLD` (tracks), `T_THRESHOLD` (history span), `cluster_share` floor, `evidence_count` floor, `evidence_coverage` floors for `root`/`tendency`/`artifact_suspect`. Will be calibrated on owner's real data in the first analysis run.
12. ~~Root unit / granularity.~~ → resolved across all V0 categories. `audio`: 1:1 HDBSCAN cluster. `temporal`: 1:1 `(time_bucket, conditioned_root)` pair. `scene`: 1:1 NMF topic that passed coherence floor.
13. **V1 feedback channel.** F (user feedback on tendencies) requires *some* interface — CLI prompt, simple HTTP form, MCP tool. Choice deferred to V1 grill.
