from pathlib import Path

import pytest

from vad.labels.ami import (
    audio_duration_s,
    build_meeting_speech_intervals,
    chunk_windows,
    discover_meetings,
    meeting_channel_files,
    parse_segments_xml,
)
from vad.labels.intervals import speech_occupancy

SEGMENTS_XML_TEMPLATE = """<?xml version="1.0" encoding="ISO-8859-1" standalone="yes"?>
<nite:root nite:id="{meeting}.{channel}.segs" xmlns:nite="http://nite.sourceforge.net/">
   <segment nite:id="{meeting}.sync.1" channel="{channel_num}" transcriber_start="{s1}" transcriber_end="{e1}">
      <nite:child href="{meeting}.{channel}.words.xml#id(w0)..id(w1)"/>
   </segment>
   <segment nite:id="{meeting}.sync.2" channel="{channel_num}" transcriber_start="{s2}" transcriber_end="{e2}">
      <nite:child href="{meeting}.{channel}.words.xml#id(w2)..id(w3)"/>
   </segment>
</nite:root>
"""


def write_segments_xml(dir_path, meeting, channel, channel_num, s1, e1, s2, e2):
    content = SEGMENTS_XML_TEMPLATE.format(
        meeting=meeting, channel=channel, channel_num=channel_num, s1=s1, e1=e1, s2=s2, e2=e2
    )
    path = Path(dir_path) / f"{meeting}.{channel}.segments.xml"
    path.write_text(content)
    return path


def test_parse_segments_xml_reads_transcriber_times(tmp_path):
    path = write_segments_xml(tmp_path, "TEST01", "A", 0, 1.0, 2.5, 5.0, 6.25)
    spans = parse_segments_xml(path)
    assert spans == [(1.0, 2.5), (5.0, 6.25)]


def test_discover_meetings_finds_distinct_ids(tmp_path):
    write_segments_xml(tmp_path, "AB1001a", "A", 0, 0.0, 1.0, 2.0, 3.0)
    write_segments_xml(tmp_path, "AB1001a", "B", 1, 0.5, 1.5, 2.5, 3.5)
    write_segments_xml(tmp_path, "AB1002b", "A", 0, 0.0, 1.0, 2.0, 3.0)
    assert discover_meetings(tmp_path) == ["AB1001a", "AB1002b"]


def test_meeting_channel_files_filters_by_meeting(tmp_path):
    write_segments_xml(tmp_path, "AB1001a", "A", 0, 0.0, 1.0, 2.0, 3.0)
    write_segments_xml(tmp_path, "AB1001a", "B", 1, 0.5, 1.5, 2.5, 3.5)
    write_segments_xml(tmp_path, "AB1002b", "A", 0, 0.0, 1.0, 2.0, 3.0)
    files = meeting_channel_files(tmp_path, "AB1001a")
    assert [f.name for f in files] == ["AB1001a.A.segments.xml", "AB1001a.B.segments.xml"]


def test_build_meeting_speech_intervals_unions_channels(tmp_path):
    # Channel A speaks 0-2 and 5-6; Channel B speaks 1-3 (overlaps A's first span).
    write_segments_xml(tmp_path, "AB1001a", "A", 0, 0.0, 2.0, 5.0, 6.0)
    write_segments_xml(tmp_path, "AB1001a", "B", 1, 1.0, 3.0, 8.0, 8.5)

    covering = build_meeting_speech_intervals(tmp_path, "AB1001a", total_duration_s=10.0)

    total = sum(e - s for s, e, _ in covering)
    assert abs(total - 10.0) < 1e-9
    # union of A(0-2) and B(1-3) should merge into a single 0-3 speech span
    speech_spans = [(s, e) for s, e, label in covering if label == 1]
    assert (0.0, 3.0) in speech_spans


def test_chunk_windows_covers_full_duration_with_overlap():
    windows = chunk_windows(total_duration_s=25.0, window_s=10.0, overlap=0.5)
    assert windows[0] == (0.0, 10.0)
    assert windows[-1][0] + windows[-1][1] == pytest.approx(25.0)
    # every point in [0, 25) should be covered by at least one window
    for t in [0.0, 5.0, 12.0, 20.0, 24.9]:
        assert any(off <= t < off + dur for off, dur in windows)


def test_chunk_windows_short_meeting_returns_single_window():
    assert chunk_windows(total_duration_s=5.0, window_s=10.0, overlap=0.5) == [(0.0, 5.0)]


def test_chunk_windows_zero_duration_is_empty():
    assert chunk_windows(total_duration_s=0.0, window_s=10.0, overlap=0.5) == []


# --- Integration check against the real AMI corpus, if reachable ---

REAL_SEGMENTS_DIR = (
    "/Users/eddiegulay/Documents/Obsidiam Vault/The vault/Voice Activity Research"
    "/data/ami/annotations/segments"
)
REAL_AUDIO_DIR = (
    "/Users/eddiegulay/Documents/Obsidiam Vault/The vault/Voice Activity Research/data/ami/audio"
)
SANITY_BAND = (0.05, 0.95)  # broad regression guard; plan's tighter 0.20-0.80 band is for manual review


def test_ami_meeting_speech_occupancy_sanity():
    segments_dir = Path(REAL_SEGMENTS_DIR)
    audio_dir = Path(REAL_AUDIO_DIR)
    if not segments_dir.exists() or not audio_dir.exists():
        pytest.skip("AMI corpus not reachable in this environment")

    meetings = discover_meetings(segments_dir)
    assert len(meetings) > 0

    report: dict[str, float] = {}
    outliers: list[str] = []
    for meeting_id in meetings:
        audio_path = audio_dir / f"{meeting_id}.Mix-Headset.wav"
        if not audio_path.exists():
            continue
        duration = audio_duration_s(audio_path)
        covering = build_meeting_speech_intervals(segments_dir, meeting_id, duration)
        occ = speech_occupancy(covering)
        report[meeting_id] = occ
        if not (SANITY_BAND[0] <= occ <= SANITY_BAND[1]):
            outliers.append(f"{meeting_id}: {occ:.3f}")

    print(f"\nAMI speech occupancy across {len(report)} meetings:")
    for meeting_id, occ in sorted(report.items()):
        flag = " <-- outside plan's 0.20-0.80 review band" if not (0.20 <= occ <= 0.80) else ""
        print(f"  {meeting_id}: {occ:.3f}{flag}")

    assert not outliers, f"meetings outside sanity band {SANITY_BAND}: {outliers}"
