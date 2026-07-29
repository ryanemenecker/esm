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


# --------------------------- matrix mode ---------------------------
def test_pair_metrics_and_format():
    a = _dump([0.5, 0.9], ptm=0.80, mmcif=_mini_cif([[0, 0, 0], [1, 0, 0]]))
    b = _dump([0.5, 0.7], ptm=0.79, mmcif=_mini_cif([[0.3, 0, 0], [1, 0, 0]]))
    m = parity._pair_metrics(a, b)
    assert abs(m["plddt_max"] - 0.2) < 1e-6
    assert abs(m["ptm_delta"] - 0.01) < 1e-6
    assert abs(m["max_dev"] - 0.3) < 1e-4
    s = parity._fmt_pair(m)
    assert "RMSD(sup)=" in s and "pLDDT|Δ|=" in s and "pTM Δ=" in s


def test_matrix_end_to_end_mocked(capsys, monkeypatch):
    # Give each (which, offload) config a distinct, deterministic dump so we can
    # confirm the matrix folds all 4 + the repeat and prints every group.
    calls = []

    def fake_fold(model_cls, args, offload=None):
        calls.append((model_cls, offload))
        # coordinates depend on the config so pairs are non-trivial
        base = 0.1 * len(calls)
        return _dump([0.5], ptm=0.7 + base, mmcif=_mini_cif([[base, 0, 0], [1, 0, 0]]))

    monkeypatch.setattr(parity, "_import_fork", lambda: "FORK")
    monkeypatch.setattr(parity, "_import_original", lambda: "ORIG")
    monkeypatch.setattr(parity, "_fold", fake_fold)

    args = parity.build_parser().parse_args(["--matrix", "--seed", "0"])
    rc = parity.cmd_matrix(args)
    out = capsys.readouterr().out
    assert rc == 0
    # 4 configs + 1 noise-floor repeat = 5 folds
    assert len(calls) == 5
    # offload flags used: fork+off(T), fork+nooff(F), orig+off(T), orig+nooff(F), floor(T)
    assert [c[1] for c in calls] == [True, False, True, False, True]
    for group in ("NOISE FLOOR", "OFFLOAD IMPACT", "CODE IMPACT", "MIXED"):
        assert group in out


# --------------------------- chunk-size sweep ---------------------------
def test_parse_chunk_size():
    import argparse

    assert parity._parse_chunk_size("64") == 64
    assert parity._parse_chunk_size("none") is None
    assert parity._parse_chunk_size("0") is None
    with pytest.raises(argparse.ArgumentTypeError):
        parity._parse_chunk_size("-1")
    with pytest.raises(argparse.ArgumentTypeError):
        parity._parse_chunk_size("abc")


def test_parse_sweep_sizes():
    assert parity._parse_sweep_sizes("none,128,64,32") == [None, 128, 64, 32]
    assert parity._parse_sweep_sizes("64") == [64]


def test_lm_dropout_defaults_off_for_parity():
    # DEFAULT 0, not the model's 0.3: with dropout on, a single structure is a
    # draw from an ensemble, so contrasts measure which member was sampled rather
    # than parity (measured 4-5 A at L=780). Opt back in explicitly.
    args = parity.build_parser().parse_args(["--ab"])
    assert args.lm_dropout == 0.0
    assert parity._fold_kwargs(args)["lm_dropout"] == 0.0
    args = parity.build_parser().parse_args(["--ab", "--lm-dropout", "0.3"])
    assert parity._fold_kwargs(args)["lm_dropout"] == 0.3


def test_deterministic_defaults_on_and_can_be_disabled():
    # DEFAULT ON so the gate is bit-exactness (verified: all contrasts exactly
    # 0.0000 A at L=780), with --no-deterministic to measure kernel jitter.
    assert parity.build_parser().parse_args(["--ab"]).deterministic is True
    args = parity.build_parser().parse_args(["--ab", "--no-deterministic"])
    assert args.deterministic is False
    parity._maybe_deterministic(args)  # no-op when disabled, must not raise


def test_self_repeat_mocked(capsys, monkeypatch):
    # --self --repeat N folds N times and reports the pairwise spread.
    calls = []

    def fake_fold(model_cls, args, offload=None, chunk=parity._CHUNK_UNSET):
        i = len(calls)
        calls.append((model_cls, offload))
        # Distinct coords per fold so pairs are non-zero.
        return _dump([0.5], ptm=0.7, mmcif=_mini_cif([[0.01 * i, 0, 0], [1, 0, 0]]))

    monkeypatch.setattr(parity, "_import_fork", lambda: "FORK")
    monkeypatch.setattr(parity, "_fold", fake_fold)

    args = parity.build_parser().parse_args(["--self", "--which", "fork", "--repeat", "4"])
    rc = parity.cmd_self(args)
    out = capsys.readouterr().out
    assert rc == 0
    assert len(calls) == 4  # folded 4 times
    assert all(c[1] is True for c in calls)  # fork -> offload
    assert "4 folds, 6 pairs" in out  # C(4,2) = 6
    assert "RMSD(sup)" in out


def test_self_original_no_offload(monkeypatch):
    calls = []

    def fake_fold(model_cls, args, offload=None, chunk=parity._CHUNK_UNSET):
        calls.append((model_cls, offload))
        return _dump([0.5], ptm=0.7, mmcif=_mini_cif([[0, 0, 0]]))

    monkeypatch.setattr(parity, "_import_original", lambda: "ORIG")
    monkeypatch.setattr(parity, "_fold", fake_fold)
    args = parity.build_parser().parse_args(["--self", "--which", "original"])
    parity.cmd_self(args)
    assert all(c[0] == "ORIG" and c[1] is False for c in calls)  # original -> no offload


def test_chunk_sweep_mocked(capsys, monkeypatch):
    calls = []

    def fake_fold(model_cls, args, offload=None, chunk=parity._CHUNK_UNSET):
        calls.append(chunk)
        base = 0.0 if chunk is None else 0.001 * chunk
        return _dump([0.5], ptm=0.7, mmcif=_mini_cif([[base, 0, 0], [1, 0, 0]]))

    monkeypatch.setattr(parity, "_import_fork", lambda: "FORK")
    monkeypatch.setattr(parity, "_fold", fake_fold)

    args = parity.build_parser().parse_args(["--chunk-sweep", "--chunk-sizes", "none,64,32"])
    rc = parity.cmd_chunk_sweep(args)
    out = capsys.readouterr().out
    assert rc == 0
    # folds at none, 64, 32, then reference (none) again for the floor
    assert calls == [None, 64, 32, None]
    assert "Reference = chunk=none" in out
    assert "NOISE FLOOR" in out
    assert "chunk=64 vs chunk=none" in out and "chunk=32 vs chunk=none" in out


def test_chunk_sweep_reports_time_and_memory(capsys, monkeypatch):
    """A2: the sweep prices the memory/speed trade per chunk size and names the
    fastest one relative to the model's default 64."""
    gib = 1.0

    def fake_fold(model_cls, args, offload=None, chunk=parity._CHUNK_UNSET):
        d = _dump([0.5], ptm=0.7, mmcif=_mini_cif([[0, 0, 0], [1, 0, 0]]))
        # bigger chunk -> faster but more memory (the trade being measured)
        size = 1024 if chunk is None else chunk
        d["_time"] = 100.0 / size
        d["_mem"] = {
            "peak_alloc": 4.0 + size / 512 * gib,
            "peak_res": 5.0 + size / 512 * gib,
            "med_alloc": 3.0,
            "med_res": 4.0,
        }
        return d

    monkeypatch.setattr(parity, "_import_fork", lambda: "FORK")
    monkeypatch.setattr(parity, "_fold", fake_fold)

    args = parity.build_parser().parse_args(
        ["--chunk-sweep", "--chunk-sizes", "none,256,64"]
    )
    rc = parity.cmd_chunk_sweep(args)
    out = capsys.readouterr().out
    assert rc == 0
    assert "Cost per chunk size" in out
    # per-chunk wall clock and peak memory both surfaced
    assert "chunk=none " in out and "chunk=256 " in out and "chunk=64  " in out
    assert "GiB alloc" in out and "GiB reserved" in out
    # unchunked (size 1024) is fastest here; speedup quoted against chunk=64
    assert "Fastest: chunk=none" in out
    assert "vs 1.56s for the default chunk=64" in out
    assert "16.00x speedup" in out


def test_chunk_sweep_survives_missing_time_and_memory(capsys, monkeypatch):
    """A CPU run has no _mem and older dumps have no _time; must not crash."""

    def fake_fold(model_cls, args, offload=None, chunk=parity._CHUNK_UNSET):
        d = _dump([0.5], ptm=0.7, mmcif=_mini_cif([[0, 0, 0]]))
        d["_mem"] = None  # CPU run
        return d  # and no "_time" key at all

    monkeypatch.setattr(parity, "_import_fork", lambda: "FORK")
    monkeypatch.setattr(parity, "_fold", fake_fold)

    args = parity.build_parser().parse_args(["--chunk-sweep", "--chunk-sizes", "none,64"])
    assert parity.cmd_chunk_sweep(args) == 0
    out = capsys.readouterr().out
    assert "no GPU memory stats" in out


def test_sync_is_noop_on_cpu():
    parity._sync("cpu")  # must not raise without CUDA


def test_chunk_sweep_falls_back_when_unchunked_fails(capsys, monkeypatch):
    def fake_fold(model_cls, args, offload=None, chunk=parity._CHUNK_UNSET):
        if chunk is None:
            raise RuntimeError("CUDA out of memory")  # unchunked OOMs
        return _dump([0.5], ptm=0.7, mmcif=_mini_cif([[0.001 * chunk, 0, 0]]))

    monkeypatch.setattr(parity, "_import_fork", lambda: "FORK")
    monkeypatch.setattr(parity, "_fold", fake_fold)

    args = parity.build_parser().parse_args(["--chunk-sweep", "--chunk-sizes", "none,64,32"])
    rc = parity.cmd_chunk_sweep(args)
    out = capsys.readouterr().out
    assert rc == 0
    assert "Reference = chunk=64" in out  # fell back from failed 'none' to 64


def test_memprobe_cpu_noop():
    probe = parity._MemProbe("cpu")
    with probe:
        pass
    assert probe.stats() is None  # no CUDA -> no memory stats, no crash


def test_matrix_reports_memory(capsys, monkeypatch):
    def fake_fold(model_cls, args, offload=None, chunk=parity._CHUNK_UNSET):
        d = _dump([0.5], ptm=0.7, mmcif=_mini_cif([[0, 0, 0], [1, 0, 0]]))
        is_fork = model_cls == "FORK"
        # fork edits use less memory than the original (for the assertion)
        d["_mem"] = {
            "peak_res": 15.0 if is_fork else 20.0,
            "med_res": 8.0 if is_fork else 10.0,
            "peak_alloc": 14.0 if is_fork else 19.0,
            "med_alloc": 7.0 if is_fork else 9.0,
        }
        # Load phase peaks higher than the fold (ESMC resident before offload)
        # and is identical for both, so it must not be attributed to the edits.
        d["_mem_load"] = dict(d["_mem"], peak_alloc=22.0, med_alloc=11.0)
        d["_time"] = 10.0 if is_fork else 12.0
        return d

    monkeypatch.setattr(parity, "_import_fork", lambda: "FORK")
    monkeypatch.setattr(parity, "_import_original", lambda: "ORIG")
    monkeypatch.setattr(parity, "_fold", fake_fold)
    args = parity.build_parser().parse_args(["--matrix"])
    parity.cmd_matrix(args)
    out = capsys.readouterr().out
    assert "fold-phase alloc peak/median" in out
    # per-config: fold-phase alloc, then overall peak (max of load and fold)
    assert "fold alloc  14.00/  7.00" in out and "overall peak  22.00" in out  # fork
    assert "fold alloc  19.00/  9.00" in out  # orig
    assert "MEMORY" in out
    # headline delta is fold-phase ALLOCATED: peak 19-14=+5; median 9-7=+2
    assert "FOLD-PHASE ALLOCATED" in out
    assert "peak saved:   +5.00 GiB" in out
    assert "median saved: +2.00 GiB" in out
    # load phase is reported but explicitly not credited to the edits
    assert "load-phase peak: orig 22.00 -> fork 22.00" in out
    assert "fold time: orig 12.0s -> fork 10.0s (1.20x)" in out


def test_matrix_memory_survives_missing_load_probe(capsys, monkeypatch):
    """Dumps from an older run have no _mem_load/_time; must still report."""

    def fake_fold(model_cls, args, offload=None, chunk=parity._CHUNK_UNSET):
        d = _dump([0.5], ptm=0.7, mmcif=_mini_cif([[0, 0, 0], [1, 0, 0]]))
        d["_mem"] = {
            "peak_res": 15.0, "med_res": 8.0, "peak_alloc": 14.0, "med_alloc": 7.0,
        }
        return d  # no _mem_load, no _time

    monkeypatch.setattr(parity, "_import_fork", lambda: "FORK")
    monkeypatch.setattr(parity, "_import_original", lambda: "ORIG")
    monkeypatch.setattr(parity, "_fold", fake_fold)
    parity.cmd_matrix(parity.build_parser().parse_args(["--matrix"]))
    out = capsys.readouterr().out
    assert "overall peak  14.00" in out  # falls back to the fold peak
    assert "load-phase peak" not in out


def test_matrix_handles_failed_config(capsys, monkeypatch):
    # If a config raises (e.g. OOM), the matrix continues and marks pairs n/a.
    def fake_fold(model_cls, args, offload=None):
        if model_cls == "ORIG" and offload is False:
            raise RuntimeError("CUDA out of memory")
        return _dump([0.5], ptm=0.7, mmcif=_mini_cif([[0, 0, 0]]))

    monkeypatch.setattr(parity, "_import_fork", lambda: "FORK")
    monkeypatch.setattr(parity, "_import_original", lambda: "ORIG")
    monkeypatch.setattr(parity, "_fold", fake_fold)

    args = parity.build_parser().parse_args(["--matrix"])
    rc = parity.cmd_matrix(args)
    out = capsys.readouterr().out
    assert rc == 0
    assert "orig+no-offload   : FAILED" in out or "orig+no-offload : FAILED" in out
    assert "n/a" in out  # contrasts involving the failed config
