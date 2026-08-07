from vad.labels.ten import parse_scv


def test_parse_scv_roundtrip(tmp_path):
    content = "testset-audio-01,0.000,0.403,0,0.403,1.204,1,1.204,1.440,0,1.440,2.470,1"
    scv_path = tmp_path / "testset-audio-01.scv"
    scv_path.write_text(content)

    filename, intervals = parse_scv(scv_path)

    assert filename == "testset-audio-01"
    assert intervals == [
        (0.000, 0.403, 0),
        (0.403, 1.204, 1),
        (1.204, 1.440, 0),
        (1.440, 2.470, 1),
    ]


def test_parse_scv_real_file_if_available():
    real_path = (
        "/Users/eddiegulay/Documents/Obsidiam Vault/The vault/Voice Activity Research"
        "/data/ten-vad-testset/testset-audio-01.scv"
    )
    from pathlib import Path

    if not Path(real_path).exists():
        return  # data kit not reachable in this environment; skip silently

    filename, intervals = parse_scv(real_path)
    assert filename == "testset-audio-01"
    assert len(intervals) > 0
    # contiguous: each interval's start matches the previous one's end
    for (prev_start, prev_end, _), (start, _, _) in zip(intervals, intervals[1:]):
        assert abs(start - prev_end) < 1e-9
    # labels alternate strictly (no two adjacent same-label triples in this format)
    for (_, _, l1), (_, _, l2) in zip(intervals, intervals[1:]):
        assert l1 != l2


def test_parse_scv_malformed_raises(tmp_path):
    bad_path = tmp_path / "bad.scv"
    bad_path.write_text("testset,0.0,1.0")  # missing label field
    try:
        parse_scv(bad_path)
        raised = False
    except ValueError:
        raised = True
    assert raised
