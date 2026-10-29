"""Ownership graphs, captured dependencies and measured CPU trace exports."""

import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import caskade as ck
import pytest
import torch

import niagara as ng

SVG = "{http://www.w3.org/2000/svg}"


class Component(ck.Module):
    def __init__(self, name):
        super().__init__(name)
        self.amplitude = ck.Param("amplitude", 1.25)

    @ck.forward
    def __call__(self, x, amplitude):
        return (x * amplitude).sin()


class Model(ck.Module):
    def __init__(self):
        super().__init__("system")
        self.first = Component("first")
        self.second = Component("second")
        self.second.amplitude = self.first.amplitude

    @ck.forward
    def __call__(self, x):
        return self.first(x) + self.second(x).square()


@pytest.fixture
def scene():
    model = Model()
    x = torch.linspace(0.1, 0.8, 4, dtype=torch.float64)
    expected = model(x)
    with ng.assign_devices(model, default="cpu") as placement:
        compiled = ng.compile_model(model, x, backend="torchscript", record_scopes=True)
        torch.testing.assert_close(compiled(x), expected)
        yield model, placement, compiled


def _require_dot():
    pytest.importorskip("graphviz")
    if shutil.which("dot") is None:
        pytest.skip("Graphviz dot is required to render SVGs")


def _svg(graph):
    return ET.fromstring(graph.pipe(format="svg"))


def _node_titles(svg):
    return {
        element.find(SVG + "title").text
        for element in svg.iter(SVG + "g")
        if element.get("class") == "node"
    }


def _labels(svg):
    return [element.text or "" for element in svg.iter(SVG + "text")]


def test_model_views_preserve_parameter_ties_and_honor_display_flags(scene):
    _require_dot()
    model, placement, _compiled = scene
    detailed = ng.draw_model(model)
    image = _svg(detailed)
    labels = _labels(image)
    assert "system" in labels
    assert "first" in labels and "second" in labels
    assert "amplitude (tied)" in labels
    assert "cpu" in labels
    assert str(id(model.first.amplitude)) in _node_titles(image)
    assert str(id(model.second.amplitude)) in _node_titles(image)
    edges = {
        element.find(SVG + "title").text
        for element in image.iter(SVG + "g")
        if element.get("class") == "edge"
    }
    assert f"{id(model.second.amplitude)}->{id(model.first.amplitude)}" in edges

    minimal = ng.draw_model(
        model,
        placement=placement,
        parameters=False,
        devices=False,
        labels=False,
        rankdir="LR",
    )
    image = _svg(minimal)
    assert minimal.graph_attr["rankdir"] == "LR"
    assert _labels(image) == []
    assert _node_titles(image) == {str(id(node)) for node in (model, model.first, model.second)}


def test_execution_view_reports_scopes_devices_shapes_and_can_hide_them(scene):
    _require_dot()
    _model, _placement, compiled = scene
    plan = compiled.execution_plan()
    scopes = {node["scope"] for node in plan["nodes"] if node["scope"]}
    assert any("system.first" in scope for scope in scopes)
    graph = ng.draw_execution(compiled, scopes=True, devices=True, shapes=True, rankdir="TB")
    image = _svg(graph)
    labels = _labels(image)
    assert graph.graph_attr["rankdir"] == "TB"
    assert "cpu" in labels
    assert "[4]" in labels
    assert scopes.intersection(labels)

    hidden = _svg(ng.draw_execution(compiled, scopes=False, devices=False, shapes=False))
    assert _node_titles(hidden) == _node_titles(image)
    hidden_labels = _labels(hidden)
    assert "cpu" not in hidden_labels
    assert "[4]" not in hidden_labels
    assert not scopes.intersection(hidden_labels)


def test_schedule_exports_dependency_levels_on_cpu_lane(scene, tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _model, _placement, compiled = scene
    figure = ng.draw_schedule(compiled)
    try:
        axis = figure.axes[0]
        assert axis.get_xlabel() == "Dependency level"
        assert axis.get_title() == "Planned order"
        assert [label.get_text() for label in axis.get_yticklabels()] == ["cpu"]
        points = [point for group in axis.collections for point in group.get_offsets()]
        expected = sorted(
            node["level"] for node in compiled.execution_plan()["nodes"] if node["device"]
        )
        assert sorted(float(point[0]) for point in points) == expected
        assert {float(point[1]) for point in points} == {0.0}
        image = tmp_path / "dependency-levels.png"
        figure.savefig(image, dpi=100)
        assert image.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    finally:
        plt.close(figure)


@pytest.mark.parametrize("include_execution", [False, True])
def test_gallery_renders_each_requested_svg_view(scene, tmp_path, include_execution):
    _require_dot()
    model, placement, compiled = scene
    output = tmp_path / "gallery"
    paths = ng.gallery(
        model,
        output,
        placement=placement,
        compiled=compiled if include_execution else None,
    )
    assert len(paths) == (32 if include_execution else 16)
    assert len(set(paths)) == len(paths)
    models = [Path(path) for path in paths if Path(path).stem.startswith("model-")]
    execution = [Path(path) for path in paths if Path(path).stem.startswith("execution-")]
    assert len(models) == 16
    assert len(execution) == (16 if include_execution else 0)
    for path in map(Path, paths):
        assert path.parent == output
        svg = ET.fromstring(path.read_bytes())
        assert svg.tag == SVG + "svg"
        assert len(svg.get("viewBox").split()) == 4
        assert _node_titles(svg)
    assert not list(output.glob("*.gv"))


def test_cpu_profile_exports_only_measured_iterations_and_real_operations(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    calls = []

    def simulation(x):
        calls.append(x.device)
        return x.sin().square()

    x = torch.linspace(0.1, 0.8, 8, dtype=torch.float64)
    output = tmp_path / "cpu-trace.json"
    table = ng.profile(simulation, (x,), output, iterations=3, warmup=2)
    assert calls == [torch.device("cpu")] * 5
    assert "Niagara simulation" in table
    assert "aten::sin" in table
    trace = json.loads(output.read_text(encoding="utf-8"))
    events = trace["traceEvents"]
    iterations = [event for event in events if event.get("name") == "Niagara simulation"]
    assert len(iterations) == 3
    assert all(event["ph"] == "X" and event["dur"] > 0 for event in iterations)
    operations = [event for event in events if event.get("name") == "aten::sin"]
    assert len(operations) == 3
    assert all(event.get("cat") == "cpu_op" for event in operations)
    assert all("Input Dims" in event.get("args", {}) for event in operations)
    assert not any(event.get("cat") in {"kernel", "cuda_runtime"} for event in events)
