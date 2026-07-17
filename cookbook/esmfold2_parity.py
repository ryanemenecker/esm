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
        diffs.append(f"mmcif: differs ({n_diff} line(s) of {max(len(la), len(lb))})")
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
    result = builder.fold(model, _make_input(args.seq), **_fold_kwargs(args))
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
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.ab:
        return cmd_ab(args)
    if args.dump:
        return cmd_dump(args)
    return cmd_compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
