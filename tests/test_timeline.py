"""Measured times, not graph order, determine timeline geometry."""

import json
import xml.etree.ElementTree as ET

import pytest
import torch

import niagara as ng

SVG = "{http://www.w3.org/2000/svg}"


def trace():
    return {"traceEvents": [
        {"ph": "X", "cat": "cpu_op", "name": "aten::<host>&", "pid": 10, "tid": 20,
         "ts": 1000, "dur": 40},
        {"ph": "X", "cat": "kernel", "name": "first", "pid": 0, "tid": 1,
         "ts": 1010, "dur": 10, "args": {"device": 0, "stream": 1}},
        {"ph": "X", "cat": "gpu_memcpy", "name": "second", "pid": 1, "tid": 2,
         "ts": 1020, "dur": 20, "args": {"device": 1, "stream": 2}},
    ]}


def rectangles(timeline):
    root = ET.fromstring(timeline.pipe())
    return {group.get("data-device"): (group, group.find(SVG + "rect"))
            for group in root.findall(SVG + "g")}


@pytest.mark.parametrize("rankdir", ["TB", "LR"])
def test_measured_start_and_duration_set_box_position_and_extent(rankdir):
    timeline = ng.draw_timeline(trace(), rankdir=rankdir, time_size=400)
    boxes = rectangles(timeline)
    group1, box1 = boxes["cuda:0"]
    group2, box2 = boxes["cuda:1"]
    position, extent = ("y", "height") if rankdir == "TB" else ("x", "width")
    assert float(group1.get("data-start-us")) == 10
    assert float(group2.get("data-duration-us")) == 20
    assert float(box2.get(position)) - float(box1.get(position)) == pytest.approx(100)
    assert float(box2.get(extent)) == pytest.approx(2 * float(box1.get(extent)))
    assert float(box1.get(extent)) == pytest.approx(100)
    assert "aten::&lt;host&gt;&amp;" in timeline.svg
    assert timeline._repr_svg_() == timeline.pipe(encoding="utf-8")
    with pytest.raises(ValueError, match="only"):
        timeline.pipe(format="png")


def test_file_input_device_selection_and_time_window(tmp_path):
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(trace()), encoding="utf-8")
    timeline = ng.draw_timeline(path, devices=("cuda:0", "cuda:1"), time_range=(5, 15), time_size=200)
    assert {event["device"] for event in timeline.events} == {"cuda:0", "cuda:1"}
    boxes = rectangles(timeline)
    assert float(boxes["cuda:0"][1].get("height")) == pytest.approx(100)
    assert float(boxes["cuda:1"][1].get("height")) == pytest.approx(100)
    assert float(boxes["cuda:1"][0].get("data-duration-us")) == 20


def test_nested_cpu_intervals_have_separate_tracks_without_changing_duration():
    data = trace()
    data["traceEvents"].append({"ph": "X", "cat": "cpu_op", "name": "nested", "pid": 10,
                                 "tid": 20, "ts": 1005, "dur": 5})
    timeline = ng.draw_timeline(data, devices=("cpu",))
    parent, child = timeline.events
    assert parent["lane"] == child["lane"]
    assert parent["track"] != child["track"]
    assert parent["duration_us"] == 40 and child["duration_us"] == 5
    assert "No GPU activity records" in timeline.svg


@pytest.mark.parametrize("options", [
    {"rankdir": "RL"}, {"time_size": 0}, {"lane_size": 1},
    {"time_range": (5, 5)}, {"time_range": (-1, 4)}, {"time_range": (100, 200)},
    {"devices": ("cuda:9",)},
])
def test_invalid_or_empty_selections_do_not_invent_timings(options):
    with pytest.raises(ValueError):
        ng.draw_timeline(trace(), **options)


def test_non_duration_records_and_invalid_timestamps_are_ignored():
    data = trace()
    data["traceEvents"] += [
        {"ph": "M", "name": "process_name", "ts": 999, "dur": 1000},
        {"ph": "X", "cat": "kernel", "name": "bad", "ts": "bad", "dur": 1},
        {"ph": "X", "cat": "kernel", "name": "bad", "ts": float("nan"), "dur": 1},
        {"ph": "X", "cat": "kernel", "name": "bad", "ts": 1000, "dur": 0},
    ]
    assert len(ng.draw_timeline(data).events) == 3
    with pytest.raises(ValueError, match="traceEvents"):
        ng.draw_timeline({})


@pytest.mark.parametrize("iterations,warmup", [(0, 1), (1, -1)])
def test_profile_rejects_invalid_counts_before_executing(tmp_path, iterations, warmup):
    calls = []
    with pytest.raises(ValueError):
        ng.profile(lambda: calls.append(True), (), tmp_path / "trace.json",
                   iterations=iterations, warmup=warmup)
    assert not calls


@pytest.mark.gpu
def test_real_two_gpu_profile_and_both_timeline_directions(tmp_path):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two physical CUDA devices")
    if torch.profiler.ProfilerActivity.CUDA not in torch.profiler.supported_activities():
        pytest.skip("CUDA profiler activity is unavailable")
    value = torch.arange(1024, device="cuda:0", dtype=torch.float32)
    path = tmp_path / "two-gpu.json"
    ng.profile(lambda x: x.sin().to("cuda:1").square(), (value,), path, iterations=1, warmup=2)
    for rankdir in ("TB", "LR"):
        timeline = ng.draw_timeline(path, rankdir=rankdir, devices=("cuda:0", "cuda:1"))
        assert {event["device"] for event in timeline.events} == {"cuda:0", "cuda:1"}
        assert all(event["duration_us"] > 0 for event in timeline.events)
        assert {event["category"] for event in timeline.events} >= {"kernel", "gpu_memcpy"}
        ET.fromstring(timeline.pipe())
