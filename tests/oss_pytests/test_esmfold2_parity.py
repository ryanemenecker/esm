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


# --------------------------- coordinate-level comparison ---------------------------
def _mini_cif(coords):
    import io

    import numpy as np
    biotite = pytest.importorskip("biotite")
    import biotite.structure as struc
    import biotite.structure.io.pdbx as pdbx

    n = len(coords)
    arr = struc.AtomArray(n)
    arr.chain_id = np.array(["A"] * n)
    arr.res_id = np.array(list(range(1, n + 1)))
    arr.res_name = np.array(["ALA"] * n)
    arr.atom_name = np.array(["CA"] * n)  # (chain, res_id, atom_name) stays unique
    arr.element = np.array(["C"] * n)
    arr.coord = np.asarray(coords, dtype=np.float32)
    f = pdbx.CIFFile()
    pdbx.set_structure(f, arr)
    buf = io.StringIO()
    f.write(buf)
    return buf.getvalue()


def test_compare_coords_shift():
    a = _mini_cif([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    b = _mini_cif([[0.2, 0, 0], [1, 0, 0], [2, 0, 0]])  # atom 0 moved 0.2 Å
    cs = parity.compare_coords(a, b)
    assert cs is not None and cs["n_common"] == 3
    assert abs(cs["max_dev"] - 0.2) < 1e-4
    assert abs(cs["mean_dev"] - 0.2 / 3) < 1e-4
    assert cs["frac_gt_0p1"] == pytest.approx(1 / 3)


def test_compare_coords_identical():
    a = _mini_cif([[0, 0, 0], [1, 2, 3]])
    cs = parity.compare_coords(a, a)
    assert cs["max_dev"] == 0.0 and cs["rmsd"] == 0.0


def test_compare_coords_superposed_removes_translation():
    import numpy as np

    base = [[0, 0, 0], [1, 0, 0], [2, 1, 0], [0, 1, 1]]
    a = _mini_cif(base)
    b = _mini_cif((np.array(base) + [5.0, 0, 0]).tolist())  # global +5 Å shift
    cs = parity.compare_coords(a, b)
    assert cs["rmsd"] > 4.0  # direct RMSD dominated by the translation
    assert cs["rmsd_superposed"] < 1e-3  # superposition removes it


def test_compare_dumps_reports_coordinate_diff():
    a = _dump([0.5], mmcif=_mini_cif([[0, 0, 0], [1, 0, 0]]))
    b = _dump([0.5], mmcif=_mini_cif([[0.5, 0, 0], [1, 0, 0]]))
    ok, diffs = parity.compare_dumps(a, b)
    assert not ok
    joined = "\n".join(diffs)
    assert "coordinate" in joined.lower()
    assert "max=0.5000" in joined
