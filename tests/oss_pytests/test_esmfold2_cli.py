"""Tests for the ESMFold2 fold CLI's non-model logic.

Covers argument parsing, the --chunk-size constraints, FASTA-based job
resolution for all three modes, validation errors, and output-name dedup.
The model-loading / folding path (``run``) needs weights + GPU and is exercised
by the opt-in real-model test in ``test_esmfold2_memory_opts.py``.
"""

import argparse
from pathlib import Path

import pytest

# Import the CLI. In a full env this is a normal package import; in a lean env
# (missing the heavy esmfold2 __init__ deps) fall back to loading the file
# directly — its module-level code is stdlib-only, and the esm imports it needs
# are lazy (inside functions).
try:
    from esm.models.esmfold2 import fold_cli
except Exception:
    import importlib.util
    import sys

    _p = Path(__file__).resolve().parents[2] / "esm" / "models" / "esmfold2" / "fold_cli.py"
    _spec = importlib.util.spec_from_file_location("esmfold2_fold_cli", _p)
    fold_cli = importlib.util.module_from_spec(_spec)
    # Register before exec so @dataclass can resolve annotations (the module
    # uses `from __future__ import annotations`).
    sys.modules["esmfold2_fold_cli"] = fold_cli
    _spec.loader.exec_module(fold_cli)


# --------------------------- --chunk-size type ---------------------------
@pytest.mark.parametrize("val,expected", [("64", 64), ("1", 1), ("32", 32), ("4096", 4096)])
def test_chunk_size_positive_int(val, expected):
    assert fold_cli.parse_chunk_size(val) == expected


@pytest.mark.parametrize("val", ["none", "None", "OFF", "disable", "0"])
def test_chunk_size_disable(val):
    assert fold_cli.parse_chunk_size(val) is None


@pytest.mark.parametrize("val", ["-1", "-64", "abc", "1.5", "", "  "])
def test_chunk_size_invalid(val):
    with pytest.raises(argparse.ArgumentTypeError):
        fold_cli.parse_chunk_size(val)


# --------------------------- device resolution ---------------------------
def test_resolve_device_by_gpu_index():
    assert fold_cli.resolve_device(2, None, cuda_available=True, device_count=4) == "cuda:2"
    assert fold_cli.resolve_device(0, None, cuda_available=True, device_count=4) == "cuda:0"
    assert fold_cli.resolve_device(3, None, cuda_available=True, device_count=4) == "cuda:3"


def test_resolve_device_explicit_and_defaults():
    assert fold_cli.resolve_device(None, "cuda:1", cuda_available=True, device_count=4) == "cuda:1"
    assert fold_cli.resolve_device(None, "cpu", cuda_available=False, device_count=0) == "cpu"
    assert fold_cli.resolve_device(None, None, cuda_available=True, device_count=4) == "cuda"
    assert fold_cli.resolve_device(None, None, cuda_available=False, device_count=0) == "cpu"


@pytest.mark.parametrize(
    "gpu,device,cuda,count",
    [
        (5, None, True, 4),   # out of range
        (-1, None, True, 4),  # negative
        (0, None, False, 0),  # no CUDA
        (1, "cpu", True, 4),  # conflict
    ],
)
def test_resolve_device_errors(gpu, device, cuda, count):
    with pytest.raises(ValueError):
        fold_cli.resolve_device(gpu, device, cuda_available=cuda, device_count=count)


# --------------------------- helpers ---------------------------
def test_sanitize():
    assert fold_cli.sanitize("sp|P12345|NAME_HUMAN some description") == "sp_P12345_NAME_HUMAN"
    assert fold_cli.sanitize("") == "seq"
    assert fold_cli.sanitize("   ") == "seq"
    assert len(fold_cli.sanitize("x" * 300)) == 120


def test_clean_sequence():
    assert fold_cli.clean_sequence("  ac d\ne ") == "ACDE"


# --------------------------- job resolution ---------------------------
def _fasta(tmp_path: Path, name: str, entries: list[tuple[str, str]]) -> Path:
    p = tmp_path / name
    p.write_text("".join(f">{h}\n{s}\n" for h, s in entries))
    return p


def test_resolve_single_sequence():
    jobs = fold_cli.resolve_jobs(sequence="mkl", targets=None, fasta=None)
    assert len(jobs) == 1
    assert jobs[0].name == "query"
    assert jobs[0].chains == [("A", "MKL")]


def test_resolve_query_vs_targets(tmp_path):
    tf = _fasta(tmp_path, "t.fasta", [("t1", "AAAA"), ("t2", "CCCC")])
    jobs = fold_cli.resolve_jobs(sequence="MKL", targets=tf, fasta=None)
    assert [j.name for j in jobs] == ["query__t1", "query__t2"]
    assert jobs[0].chains == [("A", "MKL"), ("B", "AAAA")]
    assert jobs[1].chains == [("A", "MKL"), ("B", "CCCC")]


def test_resolve_fasta_individually(tmp_path):
    ff = _fasta(tmp_path, "f.fasta", [("a", "AA"), ("b", "CC"), ("c", "DD")])
    jobs = fold_cli.resolve_jobs(sequence=None, targets=None, fasta=ff)
    assert [j.name for j in jobs] == ["a", "b", "c"]
    assert all(len(j.chains) == 1 and j.chains[0][0] == "A" for j in jobs)


def test_resolve_dedupes_names(tmp_path):
    ff = _fasta(tmp_path, "dup.fasta", [("dup", "AA"), ("dup", "CC"), ("dup", "DD")])
    jobs = fold_cli.resolve_jobs(sequence=None, targets=None, fasta=ff)
    assert [j.name for j in jobs] == ["dup", "dup_1", "dup_2"]


def test_resolve_custom_query_name(tmp_path):
    tf = _fasta(tmp_path, "t.fasta", [("t1", "AAAA")])
    jobs = fold_cli.resolve_jobs(sequence="MKL", targets=tf, fasta=None, query_name="binderX")
    assert jobs[0].name == "binderX__t1"


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(sequence=None, targets=None, fasta=None),  # nothing
        dict(sequence=None, targets="t.fasta", fasta=None),  # targets w/o sequence
        dict(sequence="MK", targets=None, fasta="f.fasta"),  # fasta + sequence
        dict(sequence=None, targets="t.fasta", fasta="f.fasta"),  # fasta + targets
    ],
)
def test_resolve_invalid_combinations(kwargs):
    with pytest.raises(ValueError):
        fold_cli.resolve_jobs(**kwargs)


def test_empty_fasta_raises(tmp_path):
    empty = tmp_path / "empty.fasta"
    empty.write_text("")
    with pytest.raises(ValueError):
        fold_cli.resolve_jobs(sequence=None, targets=None, fasta=empty)


# --------------------------- parser ---------------------------
def test_parser_defaults_and_chunk_size():
    parser = fold_cli.build_parser()
    args = parser.parse_args(["--sequence", "MKL"])
    assert args.sequence == "MKL"
    assert args.chunk_size is fold_cli._CHUNK_UNSET  # untouched by default
    assert args.num_loops == 20 and args.num_diffusion_samples == 1

    args = parser.parse_args(["--fasta", "x.fasta", "--chunk-size", "32"])
    assert args.chunk_size == 32
    args = parser.parse_args(["--fasta", "x.fasta", "--chunk-size", "none"])
    assert args.chunk_size is None


def test_parser_rejects_bad_chunk_size():
    parser = fold_cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--sequence", "MK", "--chunk-size", "-4"])


def test_parser_stage_loading():
    parser = fold_cli.build_parser()
    assert parser.parse_args(["--sequence", "MK"]).stage_loading is False
    assert parser.parse_args(["--sequence", "MK", "--stage-loading"]).stage_loading is True


def test_parser_gpu_and_device():
    parser = fold_cli.build_parser()
    args = parser.parse_args(["--sequence", "MK", "--gpu", "2"])
    assert args.gpu == 2 and args.device is None
    args = parser.parse_args(["--sequence", "MK", "--device", "cuda:1"])
    assert args.device == "cuda:1" and args.gpu is None
    # --gpu and --device are mutually exclusive.
    with pytest.raises(SystemExit):
        parser.parse_args(["--sequence", "MK", "--gpu", "0", "--device", "cuda:1"])


# --------------------------- output writing ---------------------------
class _FakePlddt:
    def __init__(self, m):
        self._m = m

    def mean(self):
        return self._m


class _FakeComplex:
    def __init__(self, s):
        self._s = s

    def to_mmcif(self):
        return self._s


class _FakeResult:
    def __init__(self, cif, mean_plddt, ptm=None, iptm=None):
        self.complex = _FakeComplex(cif)
        self.plddt = _FakePlddt(mean_plddt)
        self.ptm = ptm
        self.iptm = iptm


def test_format_peak_memory():
    gib = 1024**3
    line = fold_cli.format_peak_memory("cuda:2", 3 * gib, 5 * gib)
    assert "cuda:2" in line
    assert "3.00 GiB allocated" in line
    assert "5.00 GiB reserved" in line


def test_write_result_single(tmp_path):
    r = _FakeResult("MMCIF-A", 87.5, ptm=0.812, iptm=0.5)
    paths = fold_cli._write_result(r, "job1", tmp_path)
    assert paths == [tmp_path / "job1.cif"]
    assert (tmp_path / "job1.cif").read_text() == "MMCIF-A"


def test_write_result_multisample(tmp_path):
    r = [_FakeResult("C0", 50.0), _FakeResult("C1", 60.0)]
    paths = fold_cli._write_result(r, "j", tmp_path)
    assert [p.name for p in paths] == ["j_sample0.cif", "j_sample1.cif"]
    assert (tmp_path / "j_sample0.cif").read_text() == "C0"
    assert (tmp_path / "j_sample1.cif").read_text() == "C1"
