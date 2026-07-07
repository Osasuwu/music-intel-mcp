"""Distil the raw MusicBrainz dump into the small ISRC -> MBID TSV (#87 AC-B).

The full MusicBrainz export is tens of millions of rows and lives OUTSIDE the
repo (env-pointed, never committed — see CLAUDE.md). :class:`~music_intel_mcp.
identity.MusicBrainzIsrcIndex` consumes a compact ``isrc_to_mbid.tsv`` extract,
not the raw dump; this module builds that extract once, offline.

Two raw tables are joined (column layout verified against the MusicBrainz
schema; the front columns are stable across dump history):

- ``isrc``      : id(0), **recording(1)**, **isrc(2)**, source, edits_pending, created
- ``recording`` : id(0), **gid(1)**, name, ...

Join path: ``isrc.recording`` -> ``recording.id`` -> ``recording.gid`` (the MBID).

Memory is bounded to the ISRC table (the set of *referenced* recording ids), not
the whole recording table: pass 1 reads ``isrc`` and remembers which recording
ids it needs; pass 2 streams ``recording`` and keeps a gid only for those. One
ISRC may reference several recordings — every joinable pair is emitted (in ISRC-
dump order), so the multi-valued index downstream sees all candidates. A missing
input table yields an empty output rather than an error, mirroring the index's
own missing-file-is-empty stance.
"""

from __future__ import annotations

from pathlib import Path

# 0-based column positions in the raw MusicBrainz TSV dump tables.
_ISRC_RECORDING_COL = 1
_ISRC_ISRC_COL = 2
_RECORDING_ID_COL = 0
_RECORDING_GID_COL = 1

_HEADER = (
    "# ISRC\tMBID — built from the MusicBrainz dump by mb_dump.build_isrc_mbid_tsv.\n"
    "# One row per (ISRC, recording-MBID); an ISRC may repeat when it maps to several.\n"
)


def _iter_rows(path: Path):
    """Yield tab-split rows from a raw dump table, skipping blank lines. A missing
    file yields nothing (honest-empty, never a crash)."""
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line:
                continue
            yield line.split("\t")


def build_isrc_mbid_tsv(
    isrc_dump: str | Path,
    recording_dump: str | Path,
    out_path: str | Path,
) -> int:
    """Build ``out_path`` (ISRC\\tMBID TSV) from the raw dump tables.

    Returns the number of ISRC->MBID pairs written. Referenced-but-unknown
    recordings (an ISRC pointing at a recording id absent from the recording
    table) are dropped rather than emitted with a blank MBID.
    """
    isrc_dump, recording_dump, out_path = Path(isrc_dump), Path(recording_dump), Path(out_path)

    # Pass 1 — collect (recording_id, isrc) pairs and the set of ids we need.
    pairs: list[tuple[str, str]] = []
    needed: set[str] = set()
    for cols in _iter_rows(isrc_dump):
        if len(cols) <= _ISRC_ISRC_COL:
            continue
        recording_id = cols[_ISRC_RECORDING_COL].strip()
        isrc = cols[_ISRC_ISRC_COL].strip()
        if not recording_id or not isrc:
            continue
        pairs.append((recording_id, isrc))
        needed.add(recording_id)

    # Pass 2 — stream the (much larger) recording table, keeping only needed gids.
    gid_by_recording: dict[str, str] = {}
    for cols in _iter_rows(recording_dump):
        if len(cols) <= _RECORDING_GID_COL:
            continue
        recording_id = cols[_RECORDING_ID_COL].strip()
        if recording_id not in needed:
            continue
        gid = cols[_RECORDING_GID_COL].strip()
        if gid:
            gid_by_recording[recording_id] = gid

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8", newline="\n") as out:
        out.write(_HEADER)
        for recording_id, isrc in pairs:
            gid = gid_by_recording.get(recording_id)
            if gid is None:
                continue
            out.write(f"{isrc}\t{gid}\n")
            written += 1
    return written
