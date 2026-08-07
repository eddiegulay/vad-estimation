"""AMI meeting corpus label extraction.

Per meeting: parse every channel's `<meeting>.<channel>.segments.xml`
(schema confirmed by direct inspection: `<segment channel="N"
transcriber_start="S" transcriber_end="E">`), union the per-channel speech
spans (valid because Mix-Headset is a mixdown of all near-field channel
mics — speech is present at time t if any channel covers t), then fill the
gaps against the audio's total duration to get one merged "is anyone
speaking" covering interval list per meeting (plan §5).
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import soundfile as sf

from vad.labels.intervals import Interval, SpanPair, fill_gaps, merge_speech_spans


def parse_segments_xml(path: str | Path) -> list[SpanPair]:
    """Parse one channel's segments.xml into (start_s, end_s) speech spans."""
    root = ET.parse(path).getroot()
    spans: list[SpanPair] = []
    for seg in root.findall(".//segment"):
        start = seg.get("transcriber_start")
        end = seg.get("transcriber_end")
        if start is None or end is None:
            continue
        spans.append((float(start), float(end)))
    return spans


def discover_meetings(segments_dir: str | Path) -> list[str]:
    """List distinct meeting IDs present in the segments/ directory."""
    meeting_ids = {f.name.split(".")[0] for f in Path(segments_dir).glob("*.segments.xml")}
    return sorted(meeting_ids)


def meeting_channel_files(segments_dir: str | Path, meeting_id: str) -> list[Path]:
    return sorted(Path(segments_dir).glob(f"{meeting_id}.*.segments.xml"))


def audio_duration_s(audio_path: str | Path) -> float:
    info = sf.info(str(audio_path))
    return info.frames / info.samplerate


def build_meeting_speech_intervals(
    segments_dir: str | Path,
    meeting_id: str,
    total_duration_s: float,
    gap_tolerance_s: float = 0.02,
) -> list[Interval]:
    """Union all channels' speech spans for a meeting into one covering
    interval list spanning [0, total_duration_s).
    """
    all_spans: list[SpanPair] = []
    for channel_file in meeting_channel_files(segments_dir, meeting_id):
        all_spans.extend(parse_segments_xml(channel_file))
    merged = merge_speech_spans(all_spans, gap_tolerance_s=gap_tolerance_s)
    return fill_gaps(merged, total_duration_s)


def chunk_windows(total_duration_s: float, window_s: float, overlap: float) -> list[tuple[float, float]]:
    """Generate (offset_s, duration_s) windows covering [0, total_duration_s)
    with the given overlap fraction (0.0 = no overlap). Always anchors a
    final window to the end of the meeting so no tail audio is dropped.
    """
    if total_duration_s <= 0 or window_s <= 0:
        return []
    if total_duration_s <= window_s:
        return [(0.0, total_duration_s)]

    step = window_s * (1.0 - overlap)
    if step <= 0:
        raise ValueError("overlap must be < 1.0")

    windows: list[tuple[float, float]] = []
    offset = 0.0
    while offset + window_s <= total_duration_s:
        windows.append((offset, window_s))
        offset += step

    last_offset = total_duration_s - window_s
    if not windows or last_offset > windows[-1][0] + 1e-6:
        windows.append((last_offset, window_s))
    return windows
