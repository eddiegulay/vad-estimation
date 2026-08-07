"""Parser for the TEN VAD testset's `.scv` annotation format.

Format (confirmed by direct inspection, single line per file, no header):
    <filename>,<start_s>,<end_s>,<label>,<start_s>,<end_s>,<label>,...
contiguous triples covering the whole file, label 1=speech, 0=non-speech.
"""

from pathlib import Path

from vad.labels.intervals import Interval


def parse_scv(path: str | Path) -> tuple[str, list[Interval]]:
    """Parse one `.scv` file into (filename, covering interval list)."""
    text = Path(path).read_text().strip()
    fields = text.split(",")
    filename = fields[0]
    triples = fields[1:]
    if len(triples) % 3 != 0:
        raise ValueError(f"{path}: expected triples after filename, got {len(triples)} fields")

    intervals: list[Interval] = []
    for i in range(0, len(triples), 3):
        start = float(triples[i])
        end = float(triples[i + 1])
        label = int(triples[i + 2])
        intervals.append((start, end, label))
    return filename, intervals
