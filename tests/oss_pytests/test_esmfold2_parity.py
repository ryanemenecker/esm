"""Tests for the ESMFold2 parity tool's comparison logic (no model/GPU needed)."""

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

# Load the cookbook script standalone (module-level imports are torch/stdlib
# only; the esm/transformers model imports it needs are lazy).
_p = Path(__file__).resolve().parents[2] / "cookbook" / "esmfold2_parity.py"
_spec = importlib.util.spec_from_file_location("esmfold2_parity", _p)
parity = importlib.util.module_from_spec(_spec)
sys.modules["esmfold2_parity"] = parity
_spec.loader.exec_module(parity)


def _dump(plddt, ptm=0.8, iptm=None, mmcif="CIF", pae=None):
    return {
        "plddt": torch.tensor(plddt),
        "pae": None if pae is None else torch.tensor(pae),
        "ptm": ptm,
        "iptm": iptm,
        "mmcif": mmcif,
    }


def test_identical():
    ok, diffs = parity.compare_dumps(_dump([0.5, 0.9]), _dump([0.5, 0.9]))
    assert ok and diffs == []


def test_plddt_diff():
    ok, diffs = parity.compare_dumps(_dump([0.5, 0.9]), _dump([0.5, 0.8]))
    assert not ok and any("plddt" in d for d in diffs)


def test_plddt_shape_diff():
    ok, diffs = parity.compare_dumps(_dump([0.5, 0.9]), _dump([0.5]))
    assert not ok and any("plddt" in d for d in diffs)


def test_ptm_diff():
    ok, diffs = parity.compare_dumps(_dump([0.5], ptm=0.8), _dump([0.5], ptm=0.7))
    assert not ok and any("ptm" in d for d in diffs)


def test_mmcif_diff():
    ok, diffs = parity.compare_dumps(
        _dump([0.5], mmcif="l1\nl2"), _dump([0.5], mmcif="l1\nX")
    )
    assert not ok and any("mmcif" in d for d in diffs)


def test_pae_none_vs_tensor():
    ok, diffs = parity.compare_dumps(_dump([0.5], pae=None), _dump([0.5], pae=[[1.0]]))
    assert not ok and any("pae" in d for d in diffs)


def test_pae_equal():
    ok, diffs = parity.compare_dumps(
        _dump([0.5], pae=[[1.0, 2.0]]), _dump([0.5], pae=[[1.0, 2.0]])
    )
    assert ok and diffs == []
