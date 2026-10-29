"""Ownership and captured numerical graphs, kept distinct from measured profiles."""

from __future__ import annotations

from collections.abc import Callable
from itertools import product
from pathlib import Path
from typing import Any

from caskade import Module, Param

from .compiler import CompiledModel
from .placement import Placement, graph_paths

COLORS = {"cpu": "#eeeeee", "cuda:0": "#b3d9f2", "cuda:1": "#ffce9e"}


def draw_model(
    root: Module,
    *,
    placement: Placement | None = None,
    parameters: bool = True,
    devices: bool = True,
    labels: bool = True,
    rankdir: str = "TB",
) -> Any:
    """Build an ownership graph using native Caskade node shapes and ties.

    Parameters
    ----------
    root : caskade.Module
        Root of the simulator DAG to visualize.
    placement : Placement or None, optional
        Applied placement to annotate; None uses the root's attached placement.
    parameters : bool, optional
        Include parameter nodes as well as module regions.
    devices : bool, optional
        Annotate declared device homes and color nodes when placement exists.
    labels : bool, optional
        Include names, module class labels and tied-parameter markers.
    rankdir : {"TB", "LR", "BT", "RL"}, optional
        Graphviz graph orientation.

    Returns
    -------
    graphviz.Digraph
        Editable graph object. Rendering requires the Graphviz ``dot`` executable
        as well as the optional Python graphviz dependency.

    Notes
    -----
    Device labels describe placement, not measured runtime or overlapping work.
    """
    import graphviz

    dot = graphviz.Digraph(
        "Niagara",
        graph_attr={"rankdir": rankdir},
        node_attr={"fontname": "Helvetica", "fontsize": "10"},
    )
    _paths, aliases = graph_paths(root)
    placement = placement or getattr(root, "_niagara_placement", None)
    shown = {n for n in aliases if parameters or isinstance(n, Module)}
    for node in shown:
        style = dict(node.graphviz_style)
        label = node.name if labels else ""
        if isinstance(node, Module) and labels:
            name = type(node).__name__.removeprefix("Niagara")
            label += f"\n{name}"
        if isinstance(node, Param) and node.pointer and labels:
            label += " (tied)"
        if devices and placement is not None:
            home = str(placement.homes[node])
            label += f"\n{home}"
            style.update(style="filled", fillcolor=COLORS.get(home, "#d5ebd0"))
        dot.node(str(id(node)), label, **style)
        for child in node.children.values():
            if child in shown:
                dot.edge(str(id(node)), str(id(child)))
    return dot


def draw_execution(
    compiled: CompiledModel,
    *,
    scopes: bool = True,
    devices: bool = True,
    shapes: bool = False,
    rankdir: str = "LR",
) -> Any:
    """Build a graph of captured tensor operations and dependencies.

    Parameters
    ----------
    compiled : CompiledModel
        Program whose captured graph should be inspected.
    scopes : bool, optional
        Include scope labels retained during capture.
    devices : bool, optional
        Include inferred output-device labels.
    shapes : bool, optional
        Include captured tensor shapes when available.
    rankdir : {"TB", "LR", "BT", "RL"}, optional
        Graphviz graph orientation.

    Returns
    -------
    graphviz.Digraph
        Editable dependency graph. This is not a measured execution trace.
    """
    import graphviz

    dot = graphviz.Digraph(
        "Dependencies",
        graph_attr={"rankdir": rankdir},
        node_attr={"shape": "box", "fontname": "Helvetica", "fontsize": "9"},
    )
    for node in compiled.execution_plan()["nodes"]:
        label = node["op"]
        if scopes and node["scope"]:
            label += f"\n{node['scope']}"
        if devices and node["device"]:
            label += f"\n{node['device']}"
        if shapes and node["shape"] is not None:
            label += f"\n{node['shape']}"
        dot.node(node["name"], label)
        for dep in node["dependencies"]:
            dot.edge(dep, node["name"])
    return dot


def draw_schedule(compiled: CompiledModel) -> Any:
    """Plot dependency levels on device lanes without inventing runtimes.

    Parameters
    ----------
    compiled : CompiledModel
        Program providing captured operation dependencies and device metadata.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with static dependency level on the x-axis and device lanes on
        the y-axis. Optional Matplotlib is imported only when called.

    Notes
    -----
    A dependency level groups potentially ready operations. It establishes
    neither elapsed time nor measured overlap; use :func:`profile` for traces.
    """
    import matplotlib.pyplot as plt

    nodes = [n for n in compiled.execution_plan()["nodes"] if n["device"]]
    lanes = sorted({n["device"] for n in nodes})
    fig, ax = plt.subplots(figsize=(8, max(2.5, len(lanes) * 0.8)), constrained_layout=True)
    for lane, home in enumerate(lanes):
        group = [n for n in nodes if n["device"] == home]
        ax.scatter(
            [n["level"] for n in group],
            [lane] * len(group),
            s=15,
            alpha=0.5,
            color=COLORS.get(home, "#447755"),
            edgecolors="#333333",
        )
    ax.set_yticks(range(len(lanes)), lanes)
    ax.set_xlabel("Dependency level")
    ax.set_title("Planned order")
    return fig


def gallery(
    root: Module,
    out: str | Path,
    *,
    placement: Placement | None = None,
    compiled: CompiledModel | None = None,
) -> list[str]:
    """Render all model-view flags and optional execution-view combinations.

    Parameters
    ----------
    root : caskade.Module
        Simulator whose native ownership graph is rendered.
    out : str or pathlib.Path
        Output directory; created if necessary.
    placement : Placement or None, optional
        Placement annotation override for the model graphs.
    compiled : CompiledModel or None, optional
        Include captured-operation graphs when supplied.

    Returns
    -------
    list of str
        Paths to 16 model SVGs, plus 16 execution SVGs when ``compiled`` is given.

    Notes
    -----
    Model flags are parameters/devices/labels; execution flags are
    scopes/devices/shapes. Each set covers both top-to-bottom and left-to-right
    orientations. The optional graphviz package and ``dot`` are required.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for parameters, devices, labels, rankdir in product(
        (False, True), (False, True), (False, True), ("TB", "LR")
    ):
        name = f"model-p{int(parameters)}-d{int(devices)}-l{int(labels)}-{rankdir}"
        dot = draw_model(
            root,
            placement=placement,
            parameters=parameters,
            devices=devices,
            labels=labels,
            rankdir=rankdir,
        )
        files.append(dot.render(str(out / name), format="svg", cleanup=True))
    if compiled is not None:
        for scopes, devices, shapes, rankdir in product(
            (False, True), (False, True), (False, True), ("TB", "LR")
        ):
            name = f"execution-c{int(scopes)}-d{int(devices)}-s{int(shapes)}-{rankdir}"
            dot = draw_execution(
                compiled, scopes=scopes, devices=devices, shapes=shapes, rankdir=rankdir
            )
            files.append(dot.render(str(out / name), format="svg", cleanup=True))
    return files


def profile(
    function: Callable[..., Any],
    args: tuple[Any, ...],
    out: str | Path,
    *,
    iterations: int = 5,
    warmup: int = 3,
) -> str:
    """Export a measured CPU/CUDA Chrome trace of repeated calls.

    Parameters
    ----------
    function : callable
        Callable to invoke as ``function(*args)``.
    args : tuple
        Positional arguments reused in warmup and measured iterations.
    out : str or pathlib.Path
        Trace JSON destination. Its parent directory must already exist.
    iterations : int, optional
        Number of profiled calls.
    warmup : int, optional
        Number of calls before profiler collection.

    Returns
    -------
    str
        Profiler summary table sorted by self CPU time. The full trace, including
        shapes and memory events, is written to ``out``.

    Notes
    -----
    CUDA activities are included when CUDA is available. A measured trace is
    necessary for overlap claims; an ownership graph or planned schedule alone
    is insufficient. Profiler overhead is separate from steady-state benchmarks.
    """
    import torch

    for _ in range(warmup):
        function(*args)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(
        activities=activities, record_shapes=True, profile_memory=True
    ) as prof:
        for _ in range(iterations):
            with torch.profiler.record_function("Niagara simulation"):
                function(*args)
            prof.step()
    prof.export_chrome_trace(str(out))
    return prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20)
