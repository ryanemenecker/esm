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
import sys

import torch

# A short, fast default target (ubiquitin, 76 aa).
_DEFAULT_SEQ = (
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
)


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
    )


def _resolve_device(args) -> str:
    if args.gpu is not None:
        return f"cuda:{args.gpu}"
    if args.device is not None:
        return args.device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _make_input(seq: str):
    from esm.models.esmfold2 import ProteinInput, StructurePredictionInput

    return StructurePredictionInput(sequences=[ProteinInput(id="A", sequence=seq)])


def _fold(model_cls, args) -> dict:
    from esm.models.esmfold2 import ESMFold2InputBuilder

    device = _resolve_device(args)
    model = model_cls.from_pretrained(args.model).to(device).eval()
    builder = ESMFold2InputBuilder()
    inp = _make_input(args.seq)
    # Default to the ESMC-offloading path (fold_batch) so the parity run has the
    # SAME (low) peak memory as normal CLI runs. Plain fold() keeps ESMC resident
    # the whole forward — a much higher peak that can OOM on a memory-tight GPU.
    # fold_batch is bit-identical to fold(), so parity is unaffected.
    if args.no_offload:
        result = builder.fold(model, inp, **_fold_kwargs(args))
    else:
        result = builder.fold_batch(
            model, [inp], offload_esmc=True, **_fold_kwargs(args)
        )[0]
    dump = result_to_dump(result)
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
    # Fold this fork's model twice (same seed, fresh load each time) to measure
    # the GPU run-to-run non-determinism floor. Model kernels like scatter_add
    # use atomics whose accumulation order is not fixed by the seed, so a small
    # nonzero difference here is expected and is the baseline to judge --ab by.
    print("Folding THIS FORK's model twice (same seed) to measure the "
          "run-to-run non-determinism floor ...")
    d1 = _fold(_import_fork(), args)
    d2 = _fold(_import_fork(), args)
    identical, diffs = compare_dumps(d1, d2)
    if identical:
        print("SELF-CONSISTENCY: bit-identical — the model is deterministic here.")
        print("So any --ab difference would be a real code difference, not noise.")
        return 0
    print("SELF-CONSISTENCY: the SAME model differs run-to-run "
          "(this is the GPU non-determinism floor):")
    for d in diffs:
        print(f"  - {d}")
    print(
        "\nInterpretation: compare these magnitudes with the --ab differences. If "
        "they are similar, the fork matches the original up to this inherent "
        "non-determinism — i.e. our edits are output-preserving."
    )
    return 0


def cmd_dump(args) -> int:
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
        help="Fold this fork's model twice (same seed) to measure the "
        "run-to-run non-determinism floor (the baseline for judging --ab).",
    )
    mode.add_argument("--dump", metavar="PATH", help="Fold once and save a comparable artifact.")
    mode.add_argument("--compare", nargs=2, metavar=("A", "B"), help="Compare two saved artifacts.")

    p.add_argument("--seq", default=_DEFAULT_SEQ, help="Sequence to fold (default: ubiquitin).")
    p.add_argument("--model", default="biohub/ESMFold2", help="Model repo id / path.")
    p.add_argument("--gpu", type=int, default=None, help="CUDA GPU index (0-indexed).")
    p.add_argument("--device", default=None, help="Explicit torch device (e.g. cpu).")
    p.add_argument("--seed", type=int, default=0, help="Seed (default: 0 — fixed for reproducibility).")
    p.add_argument("--num-loops", type=int, default=20)
    p.add_argument("--num-sampling-steps", type=int, default=100)
    p.add_argument("--num-diffusion-samples", type=int, default=1)
    p.add_argument(
        "--no-offload",
        action="store_true",
        help="Fold with plain fold() (ESMC resident the whole forward — higher "
        "peak memory). Default offloads ESMC (fold_batch) to match normal runs.",
    )
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.ab:
        return cmd_ab(args)
    if args.self_check:
        return cmd_self(args)
    if args.dump:
        return cmd_dump(args)
    return cmd_compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
