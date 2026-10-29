"""The upstream harness must report actual mixed-device execution."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import caskade as ck
import pytest
import torch


@pytest.fixture
def adapter(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/_upstream_caskade_plugin.py"
    name = "_niagara_topology_adapter_test"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_CURRENT", "topology-test")
    monkeypatch.setattr(module, "_DEFAULT", "cuda:0")
    monkeypatch.setattr(module, "_MODE", "niagara")
    monkeypatch.setattr(module, "_COMPILE_CHECKS", "eligible")
    return module


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs")
@pytest.mark.parametrize("devices,default", [
    (("cuda:0", "cuda:1"), "cuda:0"),
    (("cpu", "cuda:0"), "cuda:0"),
    (("cpu", "cuda:0"), "cpu"),
    (("cpu", "cuda:0", "cuda:1"), "cuda:0"),
    (("cpu", "cuda:0", "cuda:1"), "cpu"),
])
def test_topology_routes_real_calls_and_reports_actual_devices(adapter, monkeypatch, tmp_path, devices, default):
    monkeypatch.setattr(adapter, "_DEVICES", devices)
    monkeypatch.setattr(adapter, "_DEFAULT", default)
    monkeypatch.setattr(adapter, "_REPORT", tmp_path / "routing.json")

    class Leaf(ck.Module):
        def __init__(self):
            super().__init__("leaf")
            self.bias = ck.Param("bias", 0.5)

        @adapter._forward
        def calculate(self, x, bias):
            return x + bias

    class Middle(ck.Module):
        def __init__(self):
            super().__init__("middle")
            self.leaf = Leaf()

        @adapter._forward
        def calculate(self, x):
            return self.leaf.calculate(x).sin()

    class Root(ck.Module):
        def __init__(self):
            super().__init__("root")
            self.middle = Middle()

        @adapter._forward
        def calculate(self, x):
            return self.middle.calculate(x) * 2

    root = Root()
    x = torch.tensor([0.2, 0.8], requires_grad=True)
    out = root.calculate(x)
    torch.testing.assert_close(out, (x + 0.5).sin() * 2)
    torch.testing.assert_close(torch.autograd.grad(out.sum(), x)[0], (x.detach() + 0.5).cos() * 2)
    call = adapter._CALLS[0]
    observed = {home for _, home in call["diagnostics"]["observed_calls"]}
    assert observed == set(devices)
    assert any(count > 0 for count in call["diagnostics"]["transfers"].values())
    if "cpu" in devices:
        assert call["compiled"]["status"] == "not_eligible"
        assert "Mixed CPU/GPU" in call["compiled"]["reason"]
    adapter._TESTS["topology-test"] = {"phases": {"call": "passed"}}
    adapter.pytest_sessionfinish(SimpleNamespace(), 0)
    report = json.loads((tmp_path / "routing.json").read_text())
    assert report["contract"]["selected_devices"] == list(devices)
    assert report["tests"]["topology-test"]["observed_selected_devices"]
    assert report["tests"]["topology-test"]["true_multidevice_calls"] == 1
    assert report["tests"]["topology-test"]["all_selected_connected_calls"] == 1
    assert report["calls"][0]["all_selected_connected"]
    assert set(report["tests"]["topology-test"]["observed_devices"]) == set(devices)


@pytest.mark.parametrize("devices", [("cpu",), ("cpu", "cpu"), ("cpu", "cuda:2"), ("cpu", "cuda:1")])
def test_invalid_device_lists_are_rejected(adapter, monkeypatch, devices):
    monkeypatch.setattr(adapter, "_DEVICES", devices)
    with pytest.raises(pytest.UsageError):
        adapter.pytest_configure(SimpleNamespace())


def test_all_selected_evidence_requires_one_connected_call(adapter, monkeypatch):
    monkeypatch.setattr(adapter, "_DEVICES", ("cpu", "cuda:0", "cuda:1"))
    def call(observed, transfers):
        return {"status": "routed", "diagnostics": {"observed_calls": [("m", x) for x in observed], "transfers": transfers}}
    assert not adapter._selected_devices_connected(call(["cpu", "cuda:0"], {"cpu->cuda:0": 1}))
    assert not adapter._selected_devices_connected(call(["cpu", "cuda:0", "cuda:1"], {"cpu->cuda:0": 1}))
    assert not adapter._selected_devices_connected(call(["cpu", "cuda:0", "cuda:1"], {"cpu->cuda:0": 0, "cpu->cuda:1": 1}))
    assert adapter._selected_devices_connected(call(["cpu", "cuda:0", "cuda:1"], {"cpu->cuda:0": 1, "cuda:0->cuda:1": 1}))
    failed = call(["cpu", "cuda:0", "cuda:1"], {"cpu->cuda:0": 1, "cuda:0->cuda:1": 1})
    failed["status"] = "routing_failed"
    assert not adapter._selected_devices_connected(failed)
