import caskade as ck
import h5py
import numpy as np
import pytest
import torch

import niagara as ng


class Region(ck.Module):
    def __init__(self):
        super().__init__("child")
        self.weight = ck.Param("weight", torch.tensor([2.0, 3.0], dtype=torch.float64), group=1)
        self.extra = torch.tensor([8.0, 9.0], dtype=torch.float64)
        self.saveattrs.add("extra")
        self.meta.note = torch.tensor([4.0, 5.0], dtype=torch.float64)
        self.saveattrs.add("meta.note")

    @ck.forward
    def __call__(self, x, weight):
        return x * weight + self.extra


class Simulator(ck.Module):
    def __init__(self):
        super().__init__("sim")
        self.child = Region()
        self.bias = ck.Param("bias", torch.tensor([0.2, 0.5], dtype=torch.float64), group=0)
        self.scalar = ck.Param("scalar", torch.tensor(0.75, dtype=torch.float64), group=1)
        self.batched = ck.Param(
            "batched",
            torch.arange(6, dtype=torch.float64).reshape(3, 2),
            shape=(2,),
            batch_shape=(3,),
            group=2,
        )
        self.empty = ck.Param("empty", None)
        self.shared = self.child.weight
        self.saved = torch.tensor([11.0, 12.0], dtype=torch.float64)
        self.saveattrs.add("saved")

    @ck.forward
    def __call__(self, x, bias, scalar):
        return self.child(x) + bias * scalar


def configure(model, two_gpu):
    if not two_gpu:
        return ng.enable(model, default="cpu", compile=False)
    return ng.enable(
        model,
        default="cuda:0",
        modules={"child": "cuda:1"},
        parameters={model.child.weight: "cuda:1"},
        compile=False,
    )


def data(path):
    with h5py.File(path, "r") as file:
        result = {}
        for name in [
            "sim/bias/value",
            "sim/scalar/value",
            "sim/batched/value",
            "sim/empty/value",
            "sim/child/weight/value",
        ]:
            result[name] = file[name][()]
        result["sim/saved"] = file["sim"].attrs["saved"]
        result["sim/child/extra"] = file["sim/child"].attrs["extra"]
        result["sim/child/meta.note"] = file["sim/child"].attrs["meta.note"]
        assert file["sim/shared"].id == file["sim/child/weight"].id
        return result


def next_values(model):
    model.bias.value = torch.tensor([1.2, 1.5], dtype=torch.float64)
    model.scalar.value = torch.tensor(1.75, dtype=torch.float64)
    model.child.weight.value = torch.tensor([4.0, 5.0], dtype=torch.float64)
    model.batched.value = torch.arange(6, dtype=torch.float64).reshape(3, 2) + 10


@pytest.mark.parametrize(
    "two_gpu",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs"),
        ),
    ],
)
@pytest.mark.parametrize("file_handle", [False, True])
def test_split_save_append_load_matches_caskade_cpu(tmp_path, two_gpu, file_handle):
    native, model = Simulator(), Simulator()
    baseline = str(tmp_path / "native.h5")
    target = str(tmp_path / "target.h5")
    native.save_state(baseline, appendable=True)
    next_values(native)
    native.append_state(baseline)
    engine = configure(model, two_gpu)
    try:
        home = torch.device("cuda:1" if two_gpu else "cpu")
        root_home = torch.device("cuda:0" if two_gpu else "cpu")
        model.child.meta.note = model.child.meta.note.to(home)
        original_extra = model.child.extra
        original_saved = model.saved
        if file_handle:
            with h5py.File(target, "w") as file:
                model.save_state(file, appendable=True)
        else:
            model.save_state(target, appendable=True)
        assert model.child.extra is original_extra and model.saved is original_saved
        assert model.child.extra.device == home and model.saved.device == root_home
        next_values(model)
        if file_handle:
            with h5py.File(target, "a") as file:
                model.append_state(file)
        else:
            model.append_state(target)
        reference, actual = data(baseline), data(target)
        assert reference.keys() == actual.keys()
        for name in reference:
            np.testing.assert_array_equal(actual[name], reference[name], err_msg=name)
        assert model.child.weight.value.device == home
        model.child.extra = torch.zeros(2, dtype=torch.float64, device=home)
        model.child.meta.note = torch.zeros(2, dtype=torch.float64, device=home)
        model.saved = torch.zeros(2, dtype=torch.float64, device=root_home)
        if file_handle:
            with h5py.File(target, "r") as file:
                model.load_state(file, index=0)
        else:
            model.load_state(target, index=0)
        native.load_state(baseline, index=0)
        if two_gpu:
            assert isinstance(model.child.extra, torch.Tensor) and model.child.extra.device == home
            assert (
                isinstance(model.child.meta.note, torch.Tensor)
                and model.child.meta.note.device == home
            )
            assert isinstance(model.saved, torch.Tensor) and model.saved.device == root_home
            np.testing.assert_array_equal(model.child.extra.cpu(), native.child.extra)
            np.testing.assert_array_equal(model.child.meta.note.cpu(), native.child.meta.note)
        else:
            assert type(model.child.extra) is type(native.child.extra)
            assert type(model.child.meta.note) is type(native.child.meta.note)
            assert type(model.saved) is type(native.saved)
            np.testing.assert_array_equal(model.child.extra, native.child.extra)
            np.testing.assert_array_equal(model.child.meta.note, native.child.meta.note)
            # Native Caskade restores HDF attrs as NumPy; convert explicitly for autograd.
            model.child.extra = torch.as_tensor(model.child.extra)
        torch.testing.assert_close(
            model.bias.value.cpu(), torch.tensor([0.2, 0.5], dtype=torch.float64)
        )
        torch.testing.assert_close(
            model.scalar.value.cpu(), torch.tensor(0.75, dtype=torch.float64)
        )
        torch.testing.assert_close(
            model.child.weight.value.cpu(), torch.tensor([2.0, 3.0], dtype=torch.float64)
        )
        torch.testing.assert_close(
            model.batched.value.cpu(), torch.arange(6, dtype=torch.float64).reshape(3, 2)
        )
        assert model.empty.value is None
        x = torch.tensor([1.0, 2.0], dtype=torch.float64, device=root_home, requires_grad=True)
        result = model.child(x)
        gradient = torch.autograd.grad(result.sum(), x)[0]
        torch.testing.assert_close(gradient.cpu(), torch.tensor([2.0, 3.0], dtype=torch.float64))
    finally:
        engine.close()


@pytest.mark.parametrize(
    "two_gpu",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs"),
        ),
    ],
)
def test_failed_append_rolls_back_all_prior_datasets_and_markers(tmp_path, monkeypatch, two_gpu):
    model = Simulator()
    target = str(tmp_path / "rollback.h5")
    engine = configure(model, two_gpu)
    try:
        model.save_state(target, appendable=True)
        before = data(target)
        next_values(model)
        original = h5py.Dataset.__setitem__
        calls = 0

        def fail_second(dataset, key, value):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("injected mid-append failure")
            return original(dataset, key, value)

        with monkeypatch.context() as m:
            m.setattr(h5py.Dataset, "__setitem__", fail_second)
            with pytest.raises(RuntimeError, match="injected mid-append failure"):
                model.append_state(target)
        assert calls == 2
        after = data(target)
        for name in before:
            np.testing.assert_array_equal(after[name], before[name], err_msg=name)
        assert not any(hasattr(p, "appended") for p in model.topological_ordering())
        model.append_state(target)
        retry = data(target)
        assert retry["sim/bias/value"].shape == (2, 2)
        assert retry["sim/scalar/value"].shape == (2,)
        assert retry["sim/child/weight/value"].shape == (2, 2)
        assert retry["sim/batched/value"].shape == (2, 3, 2)
        np.testing.assert_array_equal(retry["sim/bias/value"][-1], [1.2, 1.5])
    finally:
        engine.close()


def test_pointer_error_does_not_silently_drop_an_append(tmp_path):
    def bad_pointer(_param):
        raise RuntimeError("pointer cannot be evaluated")

    model = Simulator()
    model.broken = ck.Param("broken", bad_pointer)
    target = str(tmp_path / "pointer.h5")
    engine = configure(model, False)
    try:
        model.save_state(target, appendable=True)
        before = data(target)
        next_values(model)
        with pytest.raises(RuntimeError, match="pointer cannot be evaluated"):
            model.append_state(target)
        after = data(target)
        for name in before:
            np.testing.assert_array_equal(after[name], before[name], err_msg=name)
        assert not any(hasattr(p, "appended") for p in model.topological_ordering())
    finally:
        engine.close()


def test_custom_param_append_override_keeps_its_semantics(tmp_path):
    class CountingParam(ck.Param):
        def __init__(self):
            super().__init__("custom", 2.0)
            self.append_calls = 0

        def _append_state_hdf5(self, group):
            self.append_calls += 1
            super()._append_state_hdf5(group)

    model = Simulator()
    model.custom = CountingParam()
    target = str(tmp_path / "custom.h5")
    engine = configure(model, False)
    try:
        model.save_state(target, appendable=True)
        model.custom.value = 3.0
        model.append_state(target)
        assert model.custom.append_calls == 1
        with h5py.File(target, "r") as file:
            np.testing.assert_array_equal(file["sim/custom/value"][()], [2.0, 3.0])
    finally:
        engine.close()


def test_saveattrs_property_getters_run_once_in_native_order(tmp_path):
    class PropertySimulator(Simulator):
        def __init__(self):
            super().__init__()
            self.calls = []
            self.saveattrs.update({"first", "second", "third"})

        @property
        def first(self):
            self.calls.append("first")
            return 10

        @property
        def second(self):
            self.calls.append("second")
            return 20

        @property
        def third(self):
            self.calls.append("third")
            return 30

    native, placed = PropertySimulator(), PropertySimulator()
    native.save_state(str(tmp_path / "native-properties.h5"))
    handle = configure(placed, False)
    try:
        placed.save_state(str(tmp_path / "placed-properties.h5"))
        assert len(native.calls) == 3
        assert placed.calls == native.calls
        with (
            h5py.File(tmp_path / "native-properties.h5", "r") as expected,
            h5py.File(tmp_path / "placed-properties.h5", "r") as actual,
        ):
            for name in ("first", "second", "third"):
                assert actual["sim"].attrs[name] == expected["sim"].attrs[name]
    finally:
        handle.close()


def _side_effect_model():
    # Caskade iterates a set. Select a property name that sorts before the
    # tensor for this process so the native saved value must include the edit.
    name = next(
        f"mutate_{index}"
        for index in range(1000)
        if next(iter({"saved", f"mutate_{index}"})) == f"mutate_{index}"
    )

    class SideEffect(ck.Module):
        def __init__(self):
            super().__init__("side")
            self.saved = torch.tensor([1.0])
            self.calls = 0
            self.saveattrs.update({"saved", name})

    def mutate(self):
        self.calls += 1
        self.saved.add_(2)
        return 7

    setattr(SideEffect, name, property(mutate))
    return SideEffect()


def test_cpu_save_preserves_side_effecting_attribute_order(tmp_path):
    native = _side_effect_model()
    placed = type(native)()
    native_path = str(tmp_path / "native-side.h5")
    placed_path = str(tmp_path / "placed-side.h5")
    native.save_state(native_path)
    with ng.enable(placed, default="cpu", compile=False):
        placed.save_state(placed_path)
    with h5py.File(native_path, "r") as expected, h5py.File(placed_path, "r") as actual:
        np.testing.assert_array_equal(expected["side"].attrs["saved"], [3.0])
        np.testing.assert_array_equal(
            actual["side"].attrs["saved"], expected["side"].attrs["saved"]
        )
    assert native.calls == placed.calls == 1


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs")
def test_gpu_save_rejects_descriptor_side_effect_before_file_write(tmp_path):
    model = _side_effect_model()
    path = tmp_path / "untouched.h5"
    path.write_bytes(b"unchanged existing file")
    with ng.enable(model, default="cuda:1", compile=False):
        with pytest.raises(RuntimeError, match="descriptor-backed saveattrs"):
            model.save_state(str(path))
        assert model.calls == 0
        torch.testing.assert_close(model.saved.cpu(), torch.tensor([1.0]))
    assert path.read_bytes() == b"unchanged existing file"


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs")
def test_gpu_save_rejects_custom_attribute_lookup_without_writing(tmp_path):
    class CustomLookup(ck.Module):
        def __init__(self):
            super().__init__("lookup")
            self.saved = torch.tensor([1.0])
            self.saveattrs.add("saved")

        def __getattribute__(self, name):
            value = super().__getattribute__(name)
            return value + 10 if name == "saved" else value

    native, placed = CustomLookup(), CustomLookup()
    native_path = str(tmp_path / "native-lookup.h5")
    target = tmp_path / "unchanged-lookup.h5"
    target.write_bytes(b"do not overwrite")
    native.save_state(native_path)
    with h5py.File(native_path, "r") as expected:
        np.testing.assert_array_equal(expected["lookup"].attrs["saved"], [11.0])
    with ng.enable(placed, default="cuda:0", compile=False):
        with pytest.raises(RuntimeError, match="custom attribute lookup"):
            placed.save_state(str(target))
        torch.testing.assert_close(vars(placed)["saved"].cpu(), torch.tensor([1.0]))
    assert target.read_bytes() == b"do not overwrite"


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs")
def test_gpu_save_rejects_custom_writer_before_file_write(tmp_path):
    class CustomWriter(ck.Module):
        def __init__(self):
            super().__init__("writer")
            self.saved = torch.tensor([1.0])
            self.saveattrs.add("saved")

        def _save_state_hdf5(self, *args, **kwargs):
            self.saved.add_(2)
            return super()._save_state_hdf5(*args, **kwargs)

    native, placed = CustomWriter(), CustomWriter()
    native_path = str(tmp_path / "native-writer.h5")
    target = tmp_path / "unchanged-writer.h5"
    target.write_bytes(b"do not overwrite")
    native.save_state(native_path)
    with h5py.File(native_path, "r") as expected:
        np.testing.assert_array_equal(expected["writer"].attrs["saved"], [3.0])
    with ng.enable(placed, default="cuda:0", compile=False):
        with pytest.raises(RuntimeError, match="custom HDF5 writers"):
            placed.save_state(str(target))
        torch.testing.assert_close(placed.saved.cpu(), torch.tensor([1.0]))
    assert target.read_bytes() == b"do not overwrite"


def test_custom_public_persistence_signatures_and_calls_are_preserved(tmp_path):
    import inspect

    class CustomPublic(ck.Module):
        def __init__(self):
            super().__init__("custom_public")
            self.p = ck.Param("p", 2.0)
            self.calls = []

        def save_state(self, saveto, appendable=False, *, token):
            self.calls.append(("save", token))
            return super().save_state(saveto, appendable=appendable)

        def append_state(self, saveto, *, token):
            self.calls.append(("append", token))
            return super().append_state(saveto)

        def load_state(self, loadfrom, index=-1, *, token):
            self.calls.append(("load", token))
            return super().load_state(loadfrom, index=index)

    model = CustomPublic()
    signatures = {
        name: inspect.signature(getattr(model, name))
        for name in ("save_state", "append_state", "load_state")
    }
    target = str(tmp_path / "custom-public.h5")
    with ng.enable(model, default="cpu", compile=False):
        for name, expected in signatures.items():
            assert inspect.signature(getattr(model, name)) == expected
        model.save_state(target, appendable=True, token="a")
        model.p.value = 3.0
        model.append_state(target, token="b")
        model.load_state(target, index=0, token="c")
        assert model.calls == [("save", "a"), ("append", "b"), ("load", "c")]
        assert float(model.p.value) == 2.0


@pytest.mark.parametrize(
    "two_gpu",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs"),
        ),
    ],
)
def test_pointer_append_matches_native_two_evaluations(tmp_path, two_gpu):
    class PointerModel(ck.Module):
        def __init__(self):
            super().__init__("pointer")
            self.calls = []
            self.p = ck.Param("p", self.read_pointer)

        def read_pointer(self, _param):
            self.calls.append(1)
            return torch.tensor(float(len(self.calls)))

    native, placed = PointerModel(), PointerModel()
    native_path, placed_path = (
        str(tmp_path / "native-pointer.h5"),
        str(tmp_path / "placed-pointer.h5"),
    )
    native.save_state(native_path, appendable=True)
    native.calls.clear()
    native.append_state(native_path)
    with ng.enable(placed, default="cuda:1" if two_gpu else "cpu", compile=False):
        placed.save_state(placed_path, appendable=True)
        placed.calls.clear()
        placed.append_state(placed_path)
        assert len(placed.calls) == len(native.calls) == 2
    with h5py.File(native_path, "r") as expected, h5py.File(placed_path, "r") as actual:
        np.testing.assert_array_equal(
            actual["pointer/p/value"][()], expected["pointer/p/value"][()]
        )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs")
def test_gpu_load_rejects_custom_setter_before_any_assignment(tmp_path):
    class CustomSetter(ck.Module):
        def __init__(self):
            super().__init__("setter")
            self.events = []
            self.saved = torch.tensor([1.0])
            self.saveattrs.add("saved")

        def __setattr__(self, name, value):
            if name == "saved" and "events" in vars(self):
                self.events.append(type(value).__name__)
            super().__setattr__(name, value)

    native, placed = CustomSetter(), CustomSetter()
    target = str(tmp_path / "setter.h5")
    native.save_state(target)
    with ng.enable(placed, default="cuda:0", compile=False):
        placed.events.clear()
        old = placed.saved
        with pytest.raises(RuntimeError, match="custom attribute setter"):
            placed.load_state(target)
        assert placed.events == [] and placed.saved is old
