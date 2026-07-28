"""Validate that this fork's ESMFold2 output is identical to the original.

Two levels of check:

* ``--ab`` (in-process, most convenient): loads the *original* model code
  (``transformers.models.esmfold2`` — the unmodified upstream) and this fork's
  copy (``esm.models.esmfold2.esmfold2_remote_code``), folds the same input with
  the same seed through the *same* input builder and weights, and compares the
  outputs bit-for-bit. This isolates exactly this fork's MODEL edits (the
  in-place ops, ResIdx scatter, deferred distogram, dels) with the hardware and
  libraries held constant.

* ``--dump`` / ``--compare`` (cross-install, most thorough): run ``--dump`` in a
  clean checkout of the *original* ``esm`` repo and again in this fork, then
  ``--compare`` the two artifacts. This validates the FULL pipeline (input
  builder + model). Produce both artifacts on the same machine / conda env so
  that only the code differs — bf16 GEMM results depend on the GPU and library
  versions, so a cross-hardware comparison is not meaningful.

Examples::

    python cookbook/esmfold2_parity.py --ab --gpu 0

    # non-determinism floor of each model (fold it twice, same seed):
    python cookbook/esmfold2_parity.py --self --which fork     --gpu 0
    python cookbook/esmfold2_parity.py --self --which original --gpu 0  # native fold(), no offload

    # original checkout:
    python cookbook/esmfold2_parity.py --dump orig.pt --gpu 0
    # this fork:
    python cookbook/esmfold2_parity.py --dump fork.pt --gpu 0
    python cookbook/esmfold2_parity.py --compare orig.pt fork.pt

Compares pLDDT, PAE, pTM, ipTM (bit-exact) and the mmCIF text (exact). Exits 0
on identical, 1 on any difference.

By default folding uses the ESMC-offloading path (same low peak memory as normal
CLI runs); pass ``--no-offload`` to use plain ``fold()`` (ESMC resident the whole
forward — higher peak, can OOM on a memory-tight GPU). Both are bit-identical, so
parity is unaffected either way.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import threading
import time

import torch

# A short, fast default target (ubiquitin, 76 aa).
_DEFAULT_SEQ = (
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
)

# Sentinel: --chunk-size not supplied -> leave the model's own default.
_CHUNK_UNSET = object()


def _parse_chunk_size(value: str):
    """argparse type: positive int, or none/off/0 -> None (disable chunking)."""
    s = value.strip().lower()
    if s in ("none", "off", "0"):
        return None
    try:
        n = int(s)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"chunk size must be a positive int or none/0, got {value!r}"
        )
    if n < 1:
        raise argparse.ArgumentTypeError(f"chunk size must be >= 1 or none/0, got {n}")
    return n


def _parse_sweep_sizes(value: str):
    """Comma-separated chunk sizes -> list of (int|None)."""
    return [_parse_chunk_size(tok) for tok in value.split(",") if tok.strip()]


def _sync(device) -> None:
    """Block until queued CUDA work finishes, so timings aren't just launch time."""
    if torch.device(device).type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


class _MemProbe:
    """Track peak + median GPU memory (GiB) over a fold. No-op off CUDA."""

    def __init__(self, device, interval_s: float = 0.02) -> None:
        self.device = torch.device(device)
        self.on = self.device.type == "cuda" and torch.cuda.is_available()
        self.interval = interval_s
        self._alloc: list[int] = []
        self._res: list[int] = []
        self._stop = threading.Event()
        self._thread = None

    def _sample(self) -> None:
        self._alloc.append(torch.cuda.memory_allocated(self.device))
        self._res.append(torch.cuda.memory_reserved(self.device))

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._sample()
            except Exception:
                break
            self._stop.wait(self.interval)

    def __enter__(self):
        if self.on:
            torch.cuda.reset_peak_memory_stats(self.device)
            self._sample()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        if self.on and self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=1.0)
        return False

    def stats(self):
        if not self.on:
            return None
        gib = 1024**3
        return {
            "peak_alloc": torch.cuda.max_memory_allocated(self.device) / gib,
            "peak_res": torch.cuda.max_memory_reserved(self.device) / gib,
            "med_alloc": (statistics.median(self._alloc) / gib) if self._alloc else 0.0,
            "med_res": (statistics.median(self._res) / gib) if self._res else 0.0,
        }


def result_to_dump(result) -> dict:
    """Extract the comparable, serializable pieces of a fold result."""
    return {
        "plddt": None if result.plddt is None else result.plddt.detach().cpu(),
        "pae": None if result.pae is None else result.pae.detach().cpu(),
        "ptm": None if result.ptm is None else float(result.ptm),
        "iptm": None if result.iptm is None else float(result.iptm),
        "mmcif": result.complex.to_mmcif(),
    }


def _cmp_tensor(name: str, a, b, diffs: list[str]) -> None:
    if (a is None) != (b is None):
        diffs.append(f"{name}: one side is None (a={a is None}, b={b is None})")
    elif a is not None:
        if a.shape != b.shape:
            diffs.append(f"{name}: shape {tuple(a.shape)} vs {tuple(b.shape)}")
        elif not torch.equal(a, b):
            d = (a.float() - b.float()).abs().max().item()
            diffs.append(f"{name}: differs (max|Δ|={d:.3e})")


def _superposed_rmsd(P, Q) -> float:
    """RMSD after optimal rigid superposition (Kabsch). Isolates real
    structural differences from a global translation/rotation of the frame."""
    import numpy as np

    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    U, _, Vt = np.linalg.svd(Qc.T @ Pc)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    Qr = Qc @ R.T
    return float(np.sqrt(((Pc - Qr) ** 2).sum(1).mean()))


def compare_coords(mmcif_a: str, mmcif_b: str):
    """Per-atom coordinate comparison (Å) between two mmCIF strings.

    Atoms are matched by (chain, residue id, atom name). Returns a stats dict,
    or None if biotite/parsing is unavailable.
    """
    try:
        import io

        import numpy as np
        import biotite.structure.io.pdbx as pdbx
    except Exception as e:
        return {"error": f"biotite unavailable ({e}); `pip install biotite`"}

    def _load(s):
        return pdbx.get_structure(pdbx.CIFFile.read(io.StringIO(s)), model=1)

    def _keymap(arr):
        cid, rid, anm, coord = arr.chain_id, arr.res_id, arr.atom_name, arr.coord
        return {
            (str(cid[i]), int(rid[i]), str(anm[i])): coord[i]
            for i in range(arr.array_length())
        }

    try:
        ma, mb = _keymap(_load(mmcif_a)), _keymap(_load(mmcif_b))
    except Exception as e:
        return {"error": f"mmCIF parse failed: {type(e).__name__}: {e}"}

    common = [k for k in ma if k in mb]
    stats = {
        "n_a": len(ma),
        "n_b": len(mb),
        "n_common": len(common),
        "n_only_a": len(ma) - len(common),
        "n_only_b": len(mb) - len(common),
    }
    if not common:
        return stats

    import numpy as np

    P = np.stack([ma[k] for k in common]).astype(np.float64)
    Q = np.stack([mb[k] for k in common]).astype(np.float64)
    dev = np.linalg.norm(P - Q, axis=1)  # per-atom Euclidean deviation (Å)
    stats.update(
        max_dev=float(dev.max()),
        mean_dev=float(dev.mean()),
        median_dev=float(np.median(dev)),
        rmsd=float(np.sqrt((dev**2).mean())),
        frac_gt_0p1=float((dev > 0.1).mean()),
        frac_gt_1p0=float((dev > 1.0).mean()),
    )
    try:
        stats["rmsd_superposed"] = _superposed_rmsd(P, Q)
    except Exception:
        pass
    return stats


def _format_coord_stats(cs) -> str:
    if cs is None:
        return "      (no coordinate diff available)"
    if "error" in cs:
        return f"      (coordinate diff unavailable — {cs['error']})"
    if cs.get("n_common", 0) == 0:
        return f"      no matching atoms (a={cs['n_a']}, b={cs['n_b']}) — atom sets differ"
    s = (
        f"      coordinate Δ over {cs['n_common']} atoms (Å): "
        f"max={cs['max_dev']:.4f}  mean={cs['mean_dev']:.4f}  "
        f"median={cs['median_dev']:.4f}  RMSD={cs['rmsd']:.4f}"
    )
    if "rmsd_superposed" in cs:
        s += f"  RMSD(superposed)={cs['rmsd_superposed']:.4f}"
    s += (
        f"\n      atoms >0.1 Å: {cs['frac_gt_0p1'] * 100:.2f}%   "
        f">1.0 Å: {cs['frac_gt_1p0'] * 100:.2f}%"
    )
    if cs["n_only_a"] or cs["n_only_b"]:
        s += f"\n      (unmatched atoms: {cs['n_only_a']} only in A, {cs['n_only_b']} only in B)"
    return s


def compare_dumps(a: dict, b: dict) -> tuple[bool, list[str]]:
    """Return (identical, list-of-difference-descriptions)."""
    diffs: list[str] = []
    _cmp_tensor("plddt", a.get("plddt"), b.get("plddt"), diffs)
    _cmp_tensor("pae", a.get("pae"), b.get("pae"), diffs)
    for scalar in ("ptm", "iptm"):
        if a.get(scalar) != b.get(scalar):
            diffs.append(f"{scalar}: {a.get(scalar)} vs {b.get(scalar)}")
    if a.get("mmcif") != b.get("mmcif"):
        la, lb = (a.get("mmcif") or "").splitlines(), (b.get("mmcif") or "").splitlines()
        n_diff = sum(1 for x, y in zip(la, lb) if x != y) + abs(len(la) - len(lb))
        line = f"mmcif: differs ({n_diff} line(s) of {max(len(la), len(lb))})\n"
        line += _format_coord_stats(compare_coords(a.get("mmcif") or "", b.get("mmcif") or ""))
        diffs.append(line)
    return (len(diffs) == 0, diffs)


def _fold_kwargs(args) -> dict:
    return dict(
        num_loops=args.num_loops,
        num_sampling_steps=args.num_sampling_steps,
        num_diffusion_samples=args.num_diffusion_samples,
        seed=args.seed,
        lm_dropout=args.lm_dropout,
    )


def _maybe_deterministic(args) -> None:
    """Force deterministic CUDA kernels (diagnostic). Best set as a shell env
    var (CUBLAS_WORKSPACE_CONFIG=:4096:8) before launch; we also set it here as
    a fallback. warn_only=True so ops lacking a deterministic impl fall back
    instead of raising."""
    if not getattr(args, "deterministic", False):
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    print("Deterministic algorithms enabled (CUBLAS_WORKSPACE_CONFIG=:4096:8). "
          "For full effect, also export that env var before launching.")


def _resolve_device(args) -> str:
    if args.gpu is not None:
        return f"cuda:{args.gpu}"
    if args.device is not None:
        return args.device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _make_input(seq: str):
    from esm.models.esmfold2 import ProteinInput, StructurePredictionInput

    return StructurePredictionInput(sequences=[ProteinInput(id="A", sequence=seq)])


def _fold(model_cls, args, offload: bool | None = None, chunk=_CHUNK_UNSET) -> dict:
    from esm.models.esmfold2 import ESMFold2InputBuilder

    if offload is None:
        offload = not args.no_offload
    # Per-call chunk overrides the global --chunk-size; either may be _CHUNK_UNSET
    # (leave the model's own default).
    effective_chunk = args.chunk_size if chunk is _CHUNK_UNSET else chunk
    device = _resolve_device(args)
    # fold_batch offloads ESMC (low peak, same as normal CLI runs); plain fold()
    # keeps ESMC resident the whole forward (native path, higher peak — can OOM
    # on a memory-tight GPU). Both are bit-identical, so parity is unaffected.
    # The probe reset happens before loading so peak covers load + fold.
    # Load and fold are probed SEPARATELY. The median is a time-sampled
    # statistic, so folding them into one window makes it a measure of how long
    # `from_pretrained` took (low allocation, slow) rather than of the fold's
    # working set — the config that loads slowest reports the lowest median.
    # `_mem` therefore covers the fold only; `_mem_load` keeps the load-phase
    # peak, which matters because loading puts ESMC on the GPU before offload.
    with _MemProbe(device) as load_probe:
        model = model_cls.from_pretrained(args.model).to(device).eval()
        if effective_chunk is not _CHUNK_UNSET:
            model.set_chunk_size(effective_chunk)
        builder = ESMFold2InputBuilder()
        inp = _make_input(args.seq)
        _sync(device)
    # Normalize the reserved pool so a previous config's cache does not inflate
    # this one's reserved figures.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    with _MemProbe(device) as probe:
        _t0 = time.perf_counter()
        if offload:
            result = builder.fold_batch(
                model, [inp], offload_esmc=True, **_fold_kwargs(args)
            )[0]
        else:
            result = builder.fold(model, inp, **_fold_kwargs(args))
        _sync(device)
        fold_s = time.perf_counter() - _t0
        dump = result_to_dump(result)
    dump["_mem"] = probe.stats()
    dump["_mem_load"] = load_probe.stats()
    dump["_time"] = fold_s
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return dump


def _import_original():
    """The unmodified upstream model (Biohub transformers fork)."""
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model

    return ESMFold2Model


def _import_fork():
    """This fork's vendored, edited copy of the model."""
    from esm.models.esmfold2.esmfold2_remote_code.modeling_esmfold2 import (
        ESMFold2Model,
    )

    return ESMFold2Model


def _best_available_model():
    """For --dump: prefer this fork's copy; fall back to upstream so the same
    script runs unchanged in an original checkout that has no vendored copy."""
    try:
        return _import_fork(), "fork (esmfold2_remote_code)"
    except Exception:
        return _import_original(), "original (transformers.models.esmfold2)"


def _pair_metrics(a: dict, b: dict) -> dict:
    """Compact difference metrics between two dumps for the matrix table."""
    m: dict = {}
    pa, pb = a.get("plddt"), b.get("plddt")
    if pa is not None and pb is not None and pa.shape == pb.shape:
        m["plddt_max"] = float((pa.float() - pb.float()).abs().max())
    if a.get("ptm") is not None and b.get("ptm") is not None:
        m["ptm_delta"] = abs(a["ptm"] - b["ptm"])
    cs = compare_coords(a.get("mmcif") or "", b.get("mmcif") or "")
    if cs and "rmsd" in cs:
        m["rmsd"] = cs["rmsd"]
        m["rmsd_sup"] = cs.get("rmsd_superposed")
        m["max_dev"] = cs["max_dev"]
    return m


def _fmt_pair(m: dict) -> str:
    rs = f"{m['rmsd_sup']:.4f}" if m.get("rmsd_sup") is not None else "?"
    pl = f"{m['plddt_max']:.2e}" if "plddt_max" in m else "?"
    pt = f"{m['ptm_delta']:.2e}" if "ptm_delta" in m else "?"
    return f"RMSD(sup)={rs} Å | pLDDT|Δ|={pl} | pTM Δ={pt}"


# The 2x2 matrix: (label, which model, offload) + a repeat of the first for the
# non-determinism floor.
_MATRIX_SPECS = [
    ("fork+offload", "fork", True),
    ("fork+no-offload", "fork", False),
    ("orig+offload", "original", True),
    ("orig+no-offload", "original", False),
]
_MATRIX_GROUPS = [
    ("NOISE FLOOR (same config, folded twice)",
     [("fork+offload", "fork+offload(2)")]),
    ("OFFLOAD IMPACT (same model code; offload on vs off)",
     [("fork+offload", "fork+no-offload"), ("orig+offload", "orig+no-offload")]),
    ("CODE IMPACT (fork vs original; same offload setting)",
     [("fork+offload", "orig+offload"), ("fork+no-offload", "orig+no-offload")]),
    ("MIXED (both axes differ — reference only)",
     [("fork+offload", "orig+no-offload"), ("fork+no-offload", "orig+offload")]),
]


def cmd_matrix(args) -> int:
    print(f"Matrix parity (seed={args.seed}): folding 4 configs + 1 repeat "
          "(non-determinism floor). Each fold reloads the model.")
    dumps: dict = {}
    errs: dict = {}
    for label, which, off in _MATRIX_SPECS:
        print(f"  folding {label} ...", flush=True)
        model_cls = _import_original() if which == "original" else _import_fork()
        try:
            dumps[label] = _fold(model_cls, args, offload=off)
            errs[label] = None
            print("    OK")
        except Exception as e:
            dumps[label] = None
            errs[label] = f"{type(e).__name__}: {str(e).splitlines()[0][:140]}"
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"    FAILED — {errs[label]}")

    print("  folding fork+offload again (noise floor) ...", flush=True)
    try:
        dumps["fork+offload(2)"] = _fold(_import_fork(), args, offload=True)
    except Exception as e:
        dumps["fork+offload(2)"] = None
        print(f"    FAILED — {type(e).__name__}")

    def _mem_str(d):
        m = d.get("_mem") if d else None
        if not m:
            return ""
        ml = d.get("_mem_load") or {}
        overall = max(m["peak_alloc"], ml.get("peak_alloc", 0.0))
        t = d.get("_time")
        t_str = f"  {t:6.1f}s" if t is not None else ""
        return (f"  fold alloc {m['peak_alloc']:6.2f}/{m['med_alloc']:6.2f}"
                f"  overall peak {overall:6.2f} GiB{t_str}")

    print("\nConfigs (fold-phase alloc peak/median, overall peak incl. load, fold time):")
    for label, _which, _off in _MATRIX_SPECS:
        d = dumps.get(label)
        status = "OK" if d is not None else f"FAILED — {errs[label]}"
        print(f"  {label:18s}: {status}{_mem_str(d)}")

    # Does the fork's model code save memory beyond what offloading already gives?
    fo, oo = dumps.get("fork+offload"), dumps.get("orig+offload")
    if fo and oo and fo.get("_mem") and oo.get("_mem"):
        mf, mo = fo["_mem"], oo["_mem"]
        print("\nMEMORY — fork's model edits ON TOP of offloading "
              "(orig+offload - fork+offload; positive = fork uses less):")
        print("  FOLD-PHASE ALLOCATED — the metric the edits act on. Measured over the")
        print("  fold only; the model load is probed separately, because a time-sampled")
        print("  median over load+fold reports load duration, not working set.")
        print(f"    peak saved:   {mo['peak_alloc'] - mf['peak_alloc']:+.2f} GiB"
              f"   (orig {mo['peak_alloc']:.2f} -> fork {mf['peak_alloc']:.2f})")
        print(f"    median saved: {mo['med_alloc'] - mf['med_alloc']:+.2f} GiB"
              f"   (orig {mo['med_alloc']:.2f} -> fork {mf['med_alloc']:.2f})")
        lf, lo = fo.get("_mem_load") or {}, oo.get("_mem_load") or {}
        if lf and lo:
            print(f"    load-phase peak: orig {lo['peak_alloc']:.2f} -> "
                  f"fork {lf['peak_alloc']:.2f} GiB (weights + ESMC before offload; "
                  "the edits do not touch this)")
        tf, to = fo.get("_time"), oo.get("_time")
        if tf and to:
            print(f"    fold time: orig {to:.1f}s -> fork {tf:.1f}s ({to / tf:.2f}x)")
        print("  NOTE: at short L the OVERALL peak is set by the ESMC encode phase, which")
        print("  is identical for both, so the edits cannot move it. They shrink the L^2")
        print("  fold working set — rerun with a long --seq for the regime where that")
        print("  dominates and the peak actually drops.")

    print("\nPairwise differences:")
    for gname, pairs in _MATRIX_GROUPS:
        print(f"  {gname}")
        for a_label, b_label in pairs:
            a, b = dumps.get(a_label), dumps.get(b_label)
            if a is None or b is None:
                missing = [lbl for lbl, d in ((a_label, a), (b_label, b)) if d is None]
                print(f"    {a_label:16s} vs {b_label:18s}: n/a ({', '.join(missing)} unavailable)")
            else:
                print(f"    {a_label:16s} vs {b_label:18s}: {_fmt_pair(_pair_metrics(a, b))}")

    print(
        "\nInterpretation: judge every contrast against the NOISE FLOOR. If OFFLOAD "
        "IMPACT and CODE IMPACT are ~the floor, then offloading ESMC and this fork's "
        "edits are both output-preserving — the differences are the model's own "
        "run-to-run non-determinism, not systematic effects."
    )
    return 0


def cmd_chunk_sweep(args) -> int:
    sizes = _parse_sweep_sizes(args.chunk_sizes)
    which = args.which
    model_cls = _import_original() if which == "original" else _import_fork()
    offload = not args.no_offload
    names = ["none" if s is None else str(s) for s in sizes]
    print(f"Chunk-size sweep on the {which} model (offload={offload}, seed={args.seed}): "
          f"{names}")

    dumps: dict = {}
    for cs, name in zip(sizes, names):
        print(f"  folding chunk={name} ...", flush=True)
        try:
            dumps[name] = _fold(model_cls, args, offload=offload, chunk=cs)
            print("    OK")
        except Exception as e:
            dumps[name] = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"    FAILED — {type(e).__name__}: {str(e).splitlines()[0][:120]}")

    # Reference = unchunked ("none") — the exact ground truth for the L² ops;
    # fall back to the first size that succeeded.
    ref, ref_cs = None, None
    if dumps.get("none") is not None:
        ref, ref_cs = "none", None
    else:
        for cs, name in zip(sizes, names):
            if dumps.get(name) is not None:
                ref, ref_cs = name, cs
                break
    if ref is None:
        print("All chunk sizes failed; nothing to compare.")
        return 1

    print(f"  folding chunk={ref} again (noise floor) ...", flush=True)
    try:
        floor = _fold(model_cls, args, offload=offload, chunk=ref_cs)
    except Exception:
        floor = None

    # --- speed / memory table: the point of retuning chunk size upward -------
    print("\nCost per chunk size (fold wall-clock, peak GPU memory):")
    best_name, best_s = None, None
    for name in names:
        d = dumps.get(name)
        if d is None:
            print(f"  chunk={name:5s}: FAILED")
            continue
        t = d.get("_time")
        m = d.get("_mem")
        t_str = f"{t:7.2f}s" if t is not None else "      ?"
        if m:
            mem_str = (f"  peak {m['peak_alloc']:6.2f} GiB alloc"
                       f" / {m['peak_res']:6.2f} GiB reserved")
        else:
            mem_str = "  (no GPU memory stats — CPU run)"
        print(f"  chunk={name:5s}: {t_str}{mem_str}")
        if t is not None and (best_s is None or t < best_s):
            best_name, best_s = name, t

    base = dumps.get("64", {}).get("_time") if dumps.get("64") else None
    if best_name is not None and base and best_name != "64":
        print(f"\n  Fastest: chunk={best_name} at {best_s:.2f}s vs {base:.2f}s for the "
              f"default chunk=64 — {base / best_s:.2f}x speedup "
              f"({base - best_s:+.2f}s).")
    elif best_name is not None:
        print(f"\n  Fastest: chunk={best_name} at {best_s:.2f}s.")

    print(f"\nReference = chunk={ref} (unchunked is the exact ground truth for the "
          "chunked L² ops).")
    print("Difference from reference  (coord RMSD superposed Å | pLDDT max|Δ| | pTM Δ):")
    if floor is not None:
        print(f"  NOISE FLOOR (chunk={ref} vs itself): "
              f"{_fmt_pair(_pair_metrics(dumps[ref], floor))}")
    for name in names:
        if name == ref or dumps.get(name) is None:
            continue
        print(f"  chunk={name} vs chunk={ref}: {_fmt_pair(_pair_metrics(dumps[name], dumps[ref]))}")

    print("\nInterpretation: if each chunk size differs from unchunked by ~the noise "
          "floor, chunking is output-preserving at that size; a systematically "
          "larger difference means that chunk size perturbs the result (expected "
          "to be tiny/ULP-scale — this quantifies it).")
    print("Pick the fastest chunk size whose difference is ~the noise floor AND whose "
          "peak memory still fits your GPU. Larger chunks re-read the right-hand "
          "stream fewer times, so they trade memory for speed — this table prices "
          "that trade with your memory savings already banked.")
    return 0


def cmd_ab(args) -> int:
    print("Folding with the ORIGINAL model code (transformers.models.esmfold2) ...")
    d_orig = _fold(_import_original(), args)
    print("Folding with THIS FORK's model code (esmfold2_remote_code) ...")
    d_fork = _fold(_import_fork(), args)
    identical, diffs = compare_dumps(d_orig, d_fork)
    if identical:
        print("PARITY: IDENTICAL — this fork's model output matches the original bit-for-bit.")
        return 0
    print("PARITY: DIFFERENCES FOUND:")
    for d in diffs:
        print(f"  - {d}")
    return 1


def cmd_self(args) -> int:
    # Fold the chosen model twice (same seed, fresh load each time) to measure
    # the GPU run-to-run non-determinism floor. Model kernels like scatter_add
    # use atomics whose accumulation order is not fixed by the seed, so a small
    # nonzero difference here is expected and is the baseline to judge --ab by.
    if args.which == "original":
        model_cls, label = _import_original(), "ORIGINAL (transformers.models.esmfold2)"
        # Original-vs-original uses the model's NATIVE path: no ESMC offload, so
        # no fork orchestration is involved in the measurement.
        offload = False
    else:
        model_cls, label = _import_fork(), "THIS FORK (esmfold2_remote_code)"
        offload = not args.no_offload
    import itertools
    import statistics as _stats

    n = max(2, args.repeat)
    path = "fold_batch, ESMC offloaded" if offload else "fold(), ESMC resident"
    print(f"Folding the {label} model {n}x (same seed; {path}) to measure the "
          "run-to-run non-determinism floor ...")
    dumps = []
    for i in range(n):
        print(f"  fold {i + 1}/{n} ...", flush=True)
        dumps.append(_fold(model_cls, args, offload=offload))

    rmsds, plddts = [], []
    for i, j in itertools.combinations(range(n), 2):
        m = _pair_metrics(dumps[i], dumps[j])
        if m.get("rmsd_sup") is not None:
            rmsds.append(m["rmsd_sup"])
        if "plddt_max" in m:
            plddts.append(m["plddt_max"])

    if not rmsds:
        print(f"SELF-CONSISTENCY ({args.which}): no comparable coordinates.")
        return 0
    if max(rmsds) == 0.0 and (not plddts or max(plddts) == 0.0):
        print(f"SELF-CONSISTENCY ({args.which}, {n} folds): bit-identical — "
              "deterministic here.")
        return 0
    print(f"\nSELF-CONSISTENCY ({args.which}, {n} folds, {len(rmsds)} pairs):")
    print(f"  RMSD(sup) Å: min={min(rmsds):.4f}  median={_stats.median(rmsds):.4f}  "
          f"max={max(rmsds):.4f}")
    if plddts:
        print(f"  pLDDT|Δ| max across pairs: {max(plddts):.2e}")
    print(
        "\nInterpretation: this is the config's own run-to-run spread. Compare it to "
        "the same config's distance from the reference in --matrix:\n"
        "  * spread ~= distance-to-reference  -> plain non-determinism (no real offset)\n"
        "  * spread <<  distance-to-reference -> a SYSTEMATIC offset (a real difference)"
    )
    return 0


def cmd_dump(args) -> int:
    if args.which == "original":
        model_cls, which = _import_original(), "original (transformers.models.esmfold2)"
    else:
        model_cls, which = _best_available_model()
    print(f"Folding with {which} and writing artifact to {args.dump} ...")
    dump = _fold(model_cls, args)
    dump["_meta"] = {
        "which": which,
        "seq": args.seq,
        "seed": args.seed,
        "num_loops": args.num_loops,
        "num_sampling_steps": args.num_sampling_steps,
        "num_diffusion_samples": args.num_diffusion_samples,
        "torch": torch.__version__,
    }
    torch.save(dump, args.dump)
    print(f"Wrote {args.dump} ({which}).")
    return 0


def cmd_compare(args) -> int:
    a = torch.load(args.compare[0], map_location="cpu", weights_only=False)
    b = torch.load(args.compare[1], map_location="cpu", weights_only=False)
    for tag, d in ((args.compare[0], a), (args.compare[1], b)):
        meta = d.get("_meta", {})
        if meta:
            print(f"{tag}: {meta.get('which')} | seq len={len(meta.get('seq',''))} "
                  f"seed={meta.get('seed')} torch={meta.get('torch')}")
    # Warn if the runs are not comparable (different input/params).
    ma, mb = a.get("_meta", {}), b.get("_meta", {})
    for key in ("seq", "seed", "num_loops", "num_sampling_steps", "num_diffusion_samples"):
        if ma.get(key) != mb.get(key):
            print(f"  WARNING: artifacts differ in '{key}' — comparison is not apples-to-apples.")
    identical, diffs = compare_dumps(a, b)
    if identical:
        print("PARITY: IDENTICAL")
        return 0
    print("PARITY: DIFFERENCES FOUND:")
    for d in diffs:
        print(f"  - {d}")
    return 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="esmfold2-parity",
        description="Validate ESMFold2 output parity between this fork and the original.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--ab", action="store_true", help="In-process original-vs-fork model comparison.")
    mode.add_argument(
        "--self",
        dest="self_check",
        action="store_true",
        help="Fold a model twice (same seed) to measure the run-to-run "
        "non-determinism floor (the baseline for judging --ab). Use --which to "
        "pick fork (default) or original.",
    )
    mode.add_argument(
        "--matrix",
        action="store_true",
        help="Fold all 4 configs (fork/original × offload/no-offload) plus a "
        "repeat, and print pairwise differences grouped by offload vs code impact.",
    )
    mode.add_argument(
        "--chunk-sweep",
        dest="chunk_sweep",
        action="store_true",
        help="Fold one model (--which) at several --chunk-sizes and compare each "
        "to the unchunked reference (isolates the chunk-size axis).",
    )
    mode.add_argument("--dump", metavar="PATH", help="Fold once and save a comparable artifact.")
    mode.add_argument("--compare", nargs=2, metavar=("A", "B"), help="Compare two saved artifacts.")

    p.add_argument(
        "--which",
        choices=["fork", "original"],
        default="fork",
        help="For --self/--dump, which model code to use (default: fork). "
        "--self --which original folds the original model twice via its native "
        "fold() path (no ESMC offload).",
    )
    p.add_argument(
        "--chunk-size",
        type=_parse_chunk_size,
        default=_CHUNK_UNSET,
        metavar="N|none",
        help="Set the L² chunk size for ALL folds (ab/self/matrix). "
        "Default: leave the model's own (64).",
    )
    p.add_argument(
        "--chunk-sizes",
        default="none,512,256,128,64",
        metavar="LIST",
        help="Comma-separated chunk sizes for --chunk-sweep "
        "(default: none,512,256,128,64 — sweeps upward from the model's 64 to "
        "trade banked memory for speed).",
    )
    p.add_argument("--seq", default=_DEFAULT_SEQ, help="Sequence to fold (default: ubiquitin).")
    p.add_argument("--model", default="biohub/ESMFold2", help="Model repo id / path.")
    p.add_argument("--gpu", type=int, default=None, help="CUDA GPU index (0-indexed).")
    p.add_argument("--device", default=None, help="Explicit torch device (e.g. cpu).")
    p.add_argument("--seed", type=int, default=0, help="Seed (default: 0 — fixed for reproducibility).")
    p.add_argument("--num-loops", type=int, default=20)
    p.add_argument("--num-sampling-steps", type=int, default=100)
    p.add_argument("--num-diffusion-samples", type=int, default=1)
    p.add_argument(
        "--lm-dropout",
        type=float,
        default=0.3,
        help="Per-loop LM dropout (model default 0.3 — an intentional stochastic "
        "ensemble knob). Set 0 to disable, e.g. to test whether it drives the "
        "fork+offload variance.",
    )
    p.add_argument(
        "--deterministic",
        action="store_true",
        help="Force deterministic CUDA algorithms (diagnostic): if this collapses "
        "the fork+offload variance, the cause is non-deterministic GPU kernels.",
    )
    p.add_argument(
        "--repeat",
        type=int,
        default=2,
        help="For --self, how many times to fold (default 2). Use e.g. 5 to see "
        "the spread and tell a systematic offset apart from noise.",
    )
    p.add_argument(
        "--no-offload",
        action="store_true",
        help="Fold with plain fold() (ESMC resident the whole forward — higher "
        "peak memory). Default offloads ESMC (fold_batch) to match normal runs.",
    )
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    _maybe_deterministic(args)
    if args.ab:
        return cmd_ab(args)
    if args.matrix:
        return cmd_matrix(args)
    if args.chunk_sweep:
        return cmd_chunk_sweep(args)
    if args.self_check:
        return cmd_self(args)
    if args.dump:
        return cmd_dump(args)
    return cmd_compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
