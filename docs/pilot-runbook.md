# Multi-user pilot runbook

Operational procedure for the closed pilot described in [`CONTEXT.md` §Multi-user pilot](../CONTEXT.md#multi-user-pilot--centralized-processing-node-grill-2026-09-0506-ac-locked). This is the node operator's checklist, not product documentation — it exists so onboarding, the day-0 data request, and teardown happen the same way every time, and so a participant's data is never kept around longer than the pilot requires.

Two participant types, per CONTEXT.md:

- **Type A** — has a PC, runs the full local system (`music-intel-desktop`) themselves.
- **Type B** — phone-only, sends a one-shot Extended Streaming History (ESH) export to the node operator and gets a `RootProfile` file back. Everything in this runbook that talks about "the node" or "the operator" is about running the pipeline on a type B participant's behalf.

## Participant onboarding checklist

Run through this before a participant starts (type A) or before importing their export (type B):

- **Windows ≥ 2004.** Per-process WASAPI loopback capture (the mechanism behind `capture-spike` / `capture-loop` / `music-intel-desktop`) needs this build or newer. Check with `winver` or Settings → System → About → "OS build". Below 2004, capture cannot run at all — don't onboard until the participant updates.
- **Desktop Spotify build, not the Microsoft Store package.** The collector's per-process capture has been built and exercised against the Win32 desktop installer only; if the participant has the Store version, have them install the desktop build from [spotify.com](https://www.spotify.com/download/) instead before their capture session.
- **Replay browser ≠ daily browser.** Replay (below) runs the Spotify web player in a browser tab. It must be a different browser binary from whatever the participant normally browses in, because per-process loopback captures a whole process tree — a shared browser process would mix the participant's own listening into the replay capture (and vice versa). Pick one of `chrome`, `msedge`, `firefox`, `brave`, `opera`, `vivaldi` that they don't already use daily.
- **Dedicated replay account.** Replay never runs on the participant's own Spotify account or the operator's personal account — it runs on an account from the owner's family plan, used only for replay (decision `909d28fe-109d-4d5b-9460-86659897a322`). This keeps replay plays out of the participant's own listening history (a shared account would poison their next ESH export with replay events, and would train Spotify's recommender on the wrong signal) and off any account the operator uses personally.
- **Consent file with grantor.** Before running `automated-playback` against a participant's queue, record consent:

  ```powershell
  music-intel automated-playback-consent --data-dir <root> --grant --grantor "<participant name>"
  ```

  `--grantor` is required by the CLI when granting (it exits 2 without one). The stored consent file carries `{grantor, timestamp, scope}` — `--scope` defaults to `automated-playback`. Revoke at any time with `--revoke`, which stops any running automated-playback session immediately.
- **Importer skip counts.** Every `import-*` subcommand prints a skip summary when it drops rows (e.g. `import-spotify` reports rows with no track URI; `import-account` reports podcast episodes/audiobooks it can't place). Read that line after every import — a surprisingly high skip count usually means the export directory has the wrong files in it.

## Day-0 ESH request

For every type A participant, request their Spotify Extended Streaming History **on day 0** (the day they agree to join), not later — Spotify's export typically takes on the order of weeks to arrive, and the pilot's timeline runs from **export arrival**, not from the day the request was made (decision `87277764-8173-4d87-a283-b5dc85aff76c`). Requesting late is the single easiest way to stall a participant's slice.

- Participant requests their export at <https://www.spotify.com/account/privacy/> → "Extended streaming history" (not the basic one — it lacks per-play context).
- Track the request date per participant; when the export lands, import it immediately:

  ```powershell
  music-intel import-spotify --from <export dir> --data-dir <root>
  ```
- Type B participants send the arrived export straight to the node operator instead of importing it themselves — the operator runs the same `import-spotify` command against the participant's data root on the node.

## Node procedure: derive → deliver → purge

This is the only order the node runs a type B participant's data in (decision `9fbaac14-d67a-40f7-8da8-ec7516710286`). Each step needs the previous one to have completed — there is no re-derivation after purge, because purge deletes the participant's root entirely.

1. **Derive.** Import the participant's export into a dedicated data root, then derive their `RootProfile`:

   ```powershell
   music-intel import-spotify --from <export dir> --data-dir data/participants/<id>
   music-intel resolve --data-dir data/participants/<id>
   music-intel analyze --data-dir data/participants/<id> --with-audio --with-scene
   ```

   `analyze` reads the node-level anonymous pool intersected with this participant's own history only (never the whole pool) and writes the `RootProfile` into `data/participants/<id>/`.
2. **Deliver.** Send the resulting `RootProfile` JSON (and its phone-readable rendering) back to the participant. Confirm delivery before moving on — once purge runs, there is no way to regenerate it without re-importing the export from scratch.
3. **Purge.** Delete the participant's data root:

   ```powershell
   music-intel purge --data-dir data/participants/<id>
   ```

   `purge` refuses (exit 2) if no `RootProfile` was ever written for that root — i.e. it will not let you delete a root that was never delivered — unless you pass `--force`. It deletes the participant root with `shutil.rmtree` and never touches the node-level pool or shared store; those are anonymous track-level data and are exactly what's meant to persist. Do not pass `--force` to skip the "was this delivered?" check unless the participant is withdrawing before delivery.

Because purge is the end of the line, **do not purge until delivery is confirmed.** "No re-derivation after purge" is not a warning about a bug — it is the design: the participant's personal data exists on the node only for the time it takes to produce their profile.

## Weekly checkpoint

Once participants are active, review the capture journal weekly — one pass over what each active participant's capture/replay session logged that week (successful captures, RMS-gate rejections, re-queues, skip counts from the importer). This is the point to notice a stalled participant (no captures logged, replay queue not advancing) before it costs another week, and to catch schema/import surprises early rather than at delivery time.

## Pre-pilot measurement gate: 120 s vs 30 s capture window

Before any participant starts, compare embeddings from a 120 s capture window against 30 s windows on the **owner's own data** — this is a gate, not a nice-to-have (decision `87277764-8173-4d87-a283-b5dc85aff76c`). The pilot's replay capture contract uses a 120 s window (`min(track duration, 120 s)`) to fit more tracks per hour, but the MTG models were trained on short clips, so a bias check against a 30 s window is due diligence before committing every participant's replay hours to the 120 s setting.

The measurement rides on an ordinary replay session rather than costing dedicated hours: `run_replay_capture`/`process_replay_queue` accept an `on_capture_analyzed` hook, and `make_window_probe_recorder` (`window_probe.py`) uses it to embed the *same* accepted PCM buffer a second time, truncated to its first 30 s. Both legs therefore come from one capture — content is held fixed, so the reported distance is a pure window-length effect. The flip side, which the report prints and you must not read past: it contains **no** capture-to-capture variance, so those distances are not a total noise floor.

Run the owner's replay queue with the hook wired, then read the gate back:

```powershell
music-intel window-probe-report --data-dir <owner root>
```

The report prints the per-track cosine-distance distribution (AC1), the adjusted Rand index between timbre derivations from each leg plus each leg's root/cluster and noise counts (AC2), and — below 100 tracks — a loud `under-powered sample` warning, because a thin sample must never read as a passed gate. Pairs are journaled to `<data_root>/window_probe.jsonl`; like every other per-user artifact they stay inside the gitignored data root and are never committed.

Judge the result and record it in memory (`~/.claude/projects/<project>/memory/decisions.md`): keep 120 s, switch to 30 s, or add a per-track multi-window mean. If the two windows produce meaningfully different clusters on the same tracks, document the discrepancy before proceeding — don't silently ship the 120 s setting on an unmeasured assumption.

## Pre-pilot measurement gate: whole-track vs 30 s stream-decode window

The 120 s-vs-30 s gate above only covers the loopback leg. The YouTube stream-decode leg (#170) is architecturally different: it never captures a fixed window at all, it decodes and embeds the *whole track*. #201 asks the analogous bias question for that leg: does the whole-track embedding disagree with a 30 s truncation the way the loopback windows do? This is a separate gate with its own journal and command — its numbers must never be read together with the loopback leg's (decision `d4f61147-ec4f-4759-a59f-4684b679c881`: exactly two comparison points, whole-track vs 30 s, reusing the same leg-agnostic `window_probe.py` machinery rather than adding a third 120 s point that would blur the two legs' numbers).

The measurement rides passively on an ordinary `replay-capture-youtube` run: `run_stream_decode_capture`/`process_stream_decode_queue` accept the same `on_capture_analyzed` hook as the loopback leg, and `make_window_probe_recorder` reuses the decode's own inference embedding as the whole-track leg — zero extra decode, one extra inference pass over a 30 s truncation.

Run the owner's stream-decode replay queue (the hook is wired in unconditionally), then read the gate back:

```powershell
music-intel replay-capture-youtube --data-dir <owner root>
music-intel stream-decode-window-probe-report --data-dir <owner root>
```

The report prints the same per-track cosine-distance distribution, adjusted Rand index, and `under-powered sample` warning below 100 tracks as the loopback report, but titled and labeled for this leg (`whole-track leg` / `30 s leg`, `#201`) so it can never be mistaken for the loopback leg's numbers. Pairs are journaled separately to `<data_root>/stream_decode_window_probe.jsonl`, gitignored like every other per-user artifact.

Judge the result and record it in memory (`~/.claude/projects/<project>/memory/decisions.md`): whole-track embedding is fine as-is, or the 30 s truncation should replace it for consistency with the loopback leg. If the sample reads unmeasured (fewer than 100 tracks), treat the gate as not yet passed — do not proceed to the full pilot on this leg based on an under-powered sample.

## Spotify ToS exposure: automated playback

Replay drives Spotify playback programmatically via the Web API (`automated-playback`), which sits close to Spotify's terms around automated/bot use of the service. This is a known, accepted risk for the pilot, mitigated by:

- **Dedicated accounts only** — replay never runs on a participant's or the operator's own personal account (see onboarding checklist above), so a ToS action against a replay account has no effect on anyone's real listening history or subscription.
- **Idle-only** — replay pauses whenever organic music is already playing on the node and only fills genuinely idle time; it is not a background process competing with real usage.
- **Human pace** — replay advances at roughly real-time (one capture window per track, ~30 tracks/h at the 120 s window), not accelerated or scripted to blast through a queue; it looks like a slow, ordinary listening session, not a scraper.

If a replay account gets flagged or restricted despite this, the mitigation is procedural, not technical: stop `automated-playback` for that account (`automated-playback-consent --revoke` also halts any running session immediately) and re-provision a fresh account from the family plan before resuming that participant's slice.

## Purpose limitation

Everything above operates under the pilot's data-handling invariant (decision `d0979cce-07d1-43dd-9f15-e1562f38ed18`, §Invariants → *Personal data is held only for processing*):

> Personal data (listening events, replay queues, RootProfile snapshots) is held by whoever processes it — the user's own machine or a processing node — solely to produce that user's RootProfile: never for the operator's own use, never retained past delivery of the profile, never in a shared store.

The derive → deliver → purge procedure above is how that invariant is actually enforced on the node: nothing personal survives past step 3, and the only thing that persists across participants is the anonymous track-level pool.
