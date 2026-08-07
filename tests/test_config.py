from vad.config import deep_merge, load_config, load_yaml


def test_deep_merge_overrides_scalar():
    base = {"a": 1, "b": 2}
    override = {"b": 3}
    assert deep_merge(base, override) == {"a": 1, "b": 3}


def test_deep_merge_recurses_into_nested_dicts():
    base = {"a": {"x": 1, "y": 2}}
    override = {"a": {"y": 3, "z": 4}}
    assert deep_merge(base, override) == {"a": {"x": 1, "y": 3, "z": 4}}


def test_deep_merge_replaces_lists_wholesale():
    base = {"a": [1, 2, 3]}
    override = {"a": [4]}
    assert deep_merge(base, override) == {"a": [4]}


def test_load_yaml_and_load_config_roundtrip(tmp_path):
    base_path = tmp_path / "base.yaml"
    override_path = tmp_path / "override.yaml"
    base_path.write_text("a: 1\nnested:\n  x: 1\n  y: 2\n")
    override_path.write_text("nested:\n  y: 20\nb: 2\n")

    assert load_yaml(base_path) == {"a": 1, "nested": {"x": 1, "y": 2}}

    merged = load_config(base_path, override_path)
    assert merged == {"a": 1, "b": 2, "nested": {"x": 1, "y": 20}}


def test_load_yaml_empty_file_returns_empty_dict(tmp_path):
    empty_path = tmp_path / "empty.yaml"
    empty_path.write_text("")
    assert load_yaml(empty_path) == {}
