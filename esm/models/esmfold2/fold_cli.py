"""Command-line interface for ESMFold2 structure prediction.

Three input modes (mutually exclusive):

  1. Query vs targets  ``--sequence SEQ --targets targets.fasta``
     Folds a two-chain complex (SEQ as chain A + each target as chain B) for
     every sequence in ``targets``, one prediction per target.

  2. Single sequence   ``--sequence SEQ``
     Folds SEQ on its own.

  3. FASTA             ``--fasta seqs.fasta``
     Folds every sequence in the FASTA individually, one prediction each.

All predictions are batched through ``ESMFold2InputBuilder.fold_batch``, which
encodes every sequence with the ESMC backbone, offloads ESMC from the compute
device, then folds each structure from its cached embedding — lowering peak GPU
memory with bit-identical output.

Examples::

    # query vs a library of targets
    esmfold2-fold --sequence MKTAYIAKQR... --targets targets.fasta -o out

    # a single sequence
    esmfold2-fold --sequence MKTAYIAKQR...

    # cap memory on a very long input
    esmfold2-fold --fasta proteins.fasta --chunk-size 64 -o out

    # bit-reproducible: for checking that a code change did not alter output
    esmfold2-fold --fasta proteins.fasta --reproducible -o out

Equivalent to ``python -m esm.models.esmfold2.fold_cli ...``.
"""

from __future__ import annotations

import argparse
import os
import re
import statistics
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

# Sentinel meaning "--chunk-size was not supplied" — resolve it from
# --num-diffusion-samples via ``resolve_chunk_size``. Distinct from the value
# None, which means the user explicitly asked to DISABLE chunking.
_CHUNK_UNSET = object()

# Soft advisory bounds for --chunk-size (not hard limits).
_CHUNK_SMALL_WARN = 8

# Chunking the L^2 trunk ops partitions only the triangle-multiply einsum's
# OUTPUT rows while handing the full right-hand stream to every chunk, so the
# live byte set during the einsum is unchanged and the chunkable tensor is ~3%
# of peak. Measured at L=780, samples=1: chunk=none is 1.77x FASTER than 64 for
# 0.03 GiB more peak, and is the reference (unchunked) path the bit-exactness
# tests compare against. So disable chunking by default at samples=1.
#
# It does matter at multiple diffusion samples: the confidence-head trunk then
# runs at batch=samples, where the pair-transition SwiGLU x12 ([B,L,L,2048]) is
# ~30 GiB unchunked vs ~2 GiB chunked at L=1000. Keep 64 there.
_CHUNK_DEFAULT_SINGLE_SAMPLE = None
_CHUNK_DEFAULT_MULTI_SAMPLE = 64


def resolve_chunk_size(chunk_size, num_diffusion_samples: int):
    """Resolve ``--chunk-size`` when it was not supplied.

    Returns ``(value, explanation)``. An explicit ``--chunk-size`` is always
    honoured verbatim; otherwise the default depends on how many diffusion
    samples are drawn, since that is what decides whether the un-chunked
    confidence-head trunk fits (see ``_CHUNK_DEFAULT_*``).
    """
    if chunk_size is not _CHUNK_UNSET:
        return chunk_size, "explicit --chunk-size"
    if num_diffusion_samples <= 1:
        return (
            _CHUNK_DEFAULT_SINGLE_SAMPLE,
            "default for --num-diffusion-samples 1 (1.77x faster, +0.03 GiB peak)",
        )
    return (
        _CHUNK_DEFAULT_MULTI_SAMPLE,
        f"default for --num-diffusion-samples {num_diffusion_samples} "
        "(un-chunked needs ~30 GiB for the confidence-head trunk at batch>1)",
    )


@dataclass
class Job:
    """One prediction: an output basename and its chains."""

    name: str
    chains: list[tuple[str, str]]  # [(chain_id, sequence), ...]


def parse_chunk_size(value: str):
    """argparse type for ``--chunk-size``.

    Accepts a positive integer (the token-axis chunk width for the L^2 trunk
    ops: triangle multiply / outer-product-mean / pair transition), or one of
    ``none``/``off``/``disable``/``0`` to disable chunking entirely. Disabling
    is measurably faster and costs almost nothing in peak memory at one
    diffusion sample (L=780: 1.77x faster, +0.03 GiB), which is why it is the
    default there — see ``resolve_chunk_size``.

    Returns ``int > 0`` or ``None`` (disable).
    """
    s = value.strip().lower()
    if s in ("none", "off", "disable", "0"):
        return None
    try:
        n = int(s)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--chunk-size must be a positive integer or one of "
            f"none/off/disable/0, got {value!r}"
        )
    if n < 1:
        raise argparse.ArgumentTypeError(
            f"--chunk-size must be >= 1 (or none/0 to disable), got {n}"
        )
    return n


def parse_lm_dropout(value: str) -> float:
    """argparse type for ``--lm-dropout``: a probability in [0, 1)."""
    try:
        p = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--lm-dropout must be a number in [0, 1), got {value!r}"
        )
    if not (0.0 <= p < 1.0):
        raise argparse.ArgumentTypeError(
            f"--lm-dropout must be >= 0 and < 1, got {p}"
        )
    return p


def resolve_reproducibility(reproducible, lm_dropout, seed, deterministic):
    """Resolve the three knobs that together make a fold bit-reproducible.

    ESMFold2 is a *sampler* by default: ``lm_dropout`` (esm's API default 0.3)
    draws a fresh mask every recycling loop, and the diffusion head is
    stochastic, so repeated folds of one sequence differ by several Angstroms at
    long L. Three things must line up for bit-identical output (verified at
    L=780, where all parity contrasts came out exactly 0.0000 A):

    1. ``lm_dropout=0``   — stop drawing ensemble members
    2. a fixed ``seed``   — pin the diffusion sampling
    3. ``deterministic``  — deterministic CUDA kernels

    ``--reproducible`` sets all three, because setting only some of them is the
    trap that makes a "regression" look real when it is just a different draw.

    Returns ``(lm_dropout, seed, deterministic, notes)``. ``lm_dropout`` stays
    ``None`` when unset, which means "do not pass the kwarg" so esm's own
    default applies — passing ``None`` explicitly is NOT the same thing, since
    that would fall back to the checkpoint's config value instead of 0.3.
    """
    notes: list[str] = []
    if not reproducible:
        return lm_dropout, seed, deterministic, notes

    if lm_dropout is not None and lm_dropout != 0.0:
        raise ValueError(
            f"--reproducible requires lm_dropout=0 but --lm-dropout {lm_dropout} "
            "was given. With dropout on, every fold is a different draw from an "
            "ensemble, so the run cannot be reproducible. Drop one of the flags."
        )

    if lm_dropout is None:
        notes.append("--lm-dropout 0 (was esm's default 0.3)")
    lm_dropout = 0.0
    if seed is None:
        seed = 0
        notes.append("--seed 0")
    if not deterministic:
        notes.append("--deterministic")
    deterministic = True
    return lm_dropout, seed, deterministic, notes


def apply_deterministic(enabled: bool, torch_mod) -> bool:
    """Enable deterministic CUDA kernels. Returns whether anything was applied.

    ``CUBLAS_WORKSPACE_CONFIG`` is read by cuBLAS when it initializes, so
    exporting it before launch is strictest; setting it here covers the common
    case. ``warn_only=True`` lets ops with no deterministic implementation fall
    back rather than raising mid-fold.
    """
    if not enabled:
        return False
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch_mod.use_deterministic_algorithms(True, warn_only=True)
    return True


def resolve_device(gpu, device, *, cuda_available: bool, device_count: int) -> str:
    """Resolve the torch device from the --gpu / --device flags.

    ``gpu`` is a 0-indexed CUDA device index (or None); ``device`` is an
    explicit torch device string (or None). At most one may be set. Raises
    ValueError on a conflict, a negative or out-of-range index, or when --gpu
    is requested without CUDA. Defaults to ``cuda`` when available, else ``cpu``.
    """
    if gpu is not None and device is not None:
        raise ValueError("Pass only one of --gpu and --device.")
    if gpu is not None:
        if gpu < 0:
            raise ValueError(f"--gpu must be >= 0 (0-indexed), got {gpu}.")
        if not cuda_available:
            raise ValueError("--gpu was given but no CUDA device is available.")
        if gpu >= device_count:
            raise ValueError(
                f"--gpu {gpu} is out of range: {device_count} CUDA device(s) "
                f"visible (valid indices 0..{device_count - 1})."
            )
        return f"cuda:{gpu}"
    if device is not None:
        return device
    return "cuda" if cuda_available else "cpu"


def sanitize(header: str) -> str:
    """Turn a FASTA header into a filesystem-safe basename component."""
    token = header.strip().split()[0] if header.strip() else ""
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", token).strip("_")
    return token[:120] or "seq"


def clean_sequence(seq: str) -> str:
    """Uppercase and strip whitespace from a sequence."""
    return "".join(seq.split()).upper()


def _dedupe_names(jobs: list[Job]) -> list[Job]:
    """Ensure output basenames are unique by suffixing collisions with _1, _2, ..."""
    seen: dict[str, int] = {}
    for job in jobs:
        if job.name in seen:
            seen[job.name] += 1
            job.name = f"{job.name}_{seen[job.name]}"
        else:
            seen[job.name] = 0
    return jobs


def read_fasta(path: str | Path) -> list[tuple[str, str]]:
    """Read (header, sequence) pairs from a FASTA file (lazy import of parser)."""
    from esm.utils.parsing import read_sequences

    entries = [(e.header, clean_sequence(e.sequence)) for e in read_sequences(str(path))]
    entries = [(h, s) for h, s in entries if s]
    if not entries:
        raise ValueError(f"No sequences found in FASTA: {path}")
    return entries


def resolve_jobs(
    sequence: str | None,
    targets: str | Path | None,
    fasta: str | Path | None,
    query_name: str = "query",
) -> list[Job]:
    """Validate the input combination and expand it into a list of Jobs.

    Raises ValueError on an invalid combination of arguments.
    """
    if fasta is not None and (sequence is not None or targets is not None):
        raise ValueError("--fasta cannot be combined with --sequence/--targets.")
    if targets is not None and sequence is None:
        raise ValueError("--targets requires --sequence (mode 1: query vs targets).")
    if sequence is None and fasta is None:
        raise ValueError("Provide one of: --sequence, --sequence + --targets, or --fasta.")

    qname = sanitize(query_name)

    if sequence is not None and targets is not None:  # mode 1
        seq = clean_sequence(sequence)
        jobs = [
            Job(name=f"{qname}__{sanitize(header)}", chains=[("A", seq), ("B", tseq)])
            for header, tseq in read_fasta(targets)
        ]
    elif sequence is not None:  # mode 2
        jobs = [Job(name=qname, chains=[("A", clean_sequence(sequence))])]
    else:  # mode 3
        assert fasta is not None
        jobs = [
            Job(name=sanitize(header), chains=[("A", seq)])
            for header, seq in read_fasta(fasta)
        ]

    return _dedupe_names(jobs)


def _build_spi(job: Job):
    """Construct a StructurePredictionInput for a job (lazy import of model types)."""
    from esm.models.esmfold2 import ProteinInput, StructurePredictionInput

    return StructurePredictionInput(
        sequences=[ProteinInput(id=cid, sequence=seq) for cid, seq in job.chains]
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="esmfold2-fold",
        description="Fold protein sequences / complexes with ESMFold2.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Modes:\n"
            "  --sequence SEQ --targets t.fasta   fold SEQ+target complex per target\n"
            "  --sequence SEQ                     fold SEQ alone\n"
            "  --fasta seqs.fasta                 fold each sequence individually"
        ),
    )
    inp = p.add_argument_group("inputs")
    inp.add_argument("-s", "--sequence", help="Query amino-acid sequence.")
    inp.add_argument(
        "-t",
        "--targets",
        help="FASTA of target sequences; each is folded as a complex with --sequence.",
    )
    inp.add_argument(
        "-f",
        "--fasta",
        help="FASTA of sequences; each is folded individually.",
    )
    inp.add_argument(
        "--query-name",
        default="query",
        help="Basename for the query sequence in the output (default: query).",
    )

    out = p.add_argument_group("output")
    out.add_argument(
        "-o",
        "--output-dir",
        default="esmfold2_out",
        help="Directory for the predicted .cif files (default: esmfold2_out).",
    )

    model = p.add_argument_group("model / compute")
    model.add_argument(
        "--model",
        default="biohub/ESMFold2",
        help="HF repo id or local path of the ESMFold2 model (default: biohub/ESMFold2).",
    )
    target = model.add_mutually_exclusive_group()
    target.add_argument(
        "--gpu",
        type=int,
        default=None,
        metavar="N",
        help="CUDA GPU index to target (0-indexed), e.g. --gpu 2 uses cuda:2. "
        "Mutually exclusive with --device.",
    )
    target.add_argument(
        "--device",
        default=None,
        help="Explicit torch device, e.g. cuda:1 or cpu (default: cuda if "
        "available, else cpu). Mutually exclusive with --gpu.",
    )
    model.add_argument(
        "--chunk-size",
        type=parse_chunk_size,
        default=_CHUNK_UNSET,
        metavar="N|none",
        help=(
            "Token-axis chunk for the L^2 trunk ops. Lower = slightly less peak "
            "memory, notably more overhead; 'none' (or 0) disables chunking. "
            "Default: none at --num-diffusion-samples 1 (measured 1.77x faster "
            "for +0.03 GiB peak at L=780), else 64."
        ),
    )

    fold = p.add_argument_group("folding parameters")
    fold.add_argument("--num-loops", type=int, default=20, help="Trunk refinement loops (default: 20).")
    fold.add_argument(
        "--num-sampling-steps", type=int, default=200, help="Diffusion steps (default: 200)."
    )
    fold.add_argument(
        "--num-diffusion-samples",
        type=int,
        default=1,
        help="Parallel structure samples per input (default: 1).",
    )
    fold.add_argument("--seed", type=int, default=None, help="Random seed (default: unset).")

    repro = p.add_argument_group(
        "reproducibility",
        "ESMFold2 is a sampler by default: LM dropout draws a fresh mask every "
        "recycling loop and the diffusion head is stochastic, so folding one "
        "sequence twice gives different structures (several Angstroms apart at "
        "long L). Use --reproducible to pin all of it.",
    )
    repro.add_argument(
        "--reproducible",
        action="store_true",
        help="Make the run bit-reproducible: implies --lm-dropout 0, "
        "--deterministic, and --seed 0 if no seed was given. Use this when "
        "verifying that a code change did not alter output — at these settings "
        "repeated folds are bit-identical (verified at L=780).",
    )
    repro.add_argument(
        "--lm-dropout",
        type=parse_lm_dropout,
        default=None,
        metavar="P",
        help="Per-loop LM dropout probability in [0, 1). This is an intentional "
        "stochastic ensemble knob: >0 means each fold is a different draw. "
        "Default: unset, i.e. esm's own default (0.3). Pass 0 to disable.",
    )
    repro.add_argument(
        "--deterministic",
        action="store_true",
        help="Force deterministic CUDA kernels (sets CUBLAS_WORKSPACE_CONFIG=:4096:8). "
        "Slightly slower; needed for bit-identical output.",
    )

    fold.add_argument(
        "--no-offload-esmc",
        action="store_true",
        help="Keep ESMC resident during folding (disables the memory optimization). "
        "Ignored under --stage-loading.",
    )
    fold.add_argument(
        "--stage-loading",
        action="store_true",
        help="Lowest peak memory: load ESMC alone and encode, free it, then load "
        "the folding stack — the two are never co-resident on the device. Removes "
        "the load/encode co-residency spike (output is identical).",
    )
    return p


def format_memory_report(
    device,
    peak_allocated: int,
    peak_reserved: int,
    median_allocated: int,
    median_reserved: int,
    n_samples: int,
) -> str:
    """Two-line peak + median GPU memory summary. ``reserved`` is what the
    caching allocator held from the driver (closest to nvidia-smi); ``allocated``
    is live tensor memory. The median (sampled over the fold) reflects the
    steady-state folding usage; the peak captures the worst transient spike."""
    gib = 1024**3
    plural = "s" if n_samples != 1 else ""
    return (
        f"Peak GPU memory on {device}: "
        f"{peak_allocated / gib:.2f} GiB allocated, "
        f"{peak_reserved / gib:.2f} GiB reserved.\n"
        f"Median GPU memory on {device}: "
        f"{median_allocated / gib:.2f} GiB allocated, "
        f"{median_reserved / gib:.2f} GiB reserved "
        f"(over {n_samples} sample{plural})."
    )


class _GpuMemorySampler:
    """Background thread that samples current GPU memory (allocated + reserved)
    at a fixed interval, for reporting the median usage across a run.

    Reading the allocator's byte counters from another thread is safe: it
    queries bookkeeping, launches no kernels, and needs no per-thread context.
    """

    def __init__(self, device, torch_module, interval_s: float = 0.05) -> None:
        self._device = device
        self._torch = torch_module
        self._interval = interval_s
        self.allocated: list[int] = []
        self.reserved: list[int] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample_once(self) -> None:
        self.allocated.append(self._torch.cuda.memory_allocated(self._device))
        self.reserved.append(self._torch.cuda.memory_reserved(self._device))

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._sample_once()
            except Exception:
                break  # e.g. a CUDA error mid-run; stop sampling quietly
            self._stop.wait(self._interval)

    def __enter__(self) -> "_GpuMemorySampler":
        self._sample_once()  # guarantee at least one sample for very fast runs
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        return False

    def medians(self) -> tuple[int, int]:
        med_a = int(statistics.median(self.allocated)) if self.allocated else 0
        med_r = int(statistics.median(self.reserved)) if self.reserved else 0
        return med_a, med_r


def _write_result(result, name: str, out_dir: Path) -> list[Path]:
    """Write one job's result(s) to .cif and return the written paths."""
    results = result if isinstance(result, list) else [result]
    written: list[Path] = []
    for i, res in enumerate(results):
        suffix = "" if len(results) == 1 else f"_sample{i}"
        path = out_dir / f"{name}{suffix}.cif"
        path.write_text(res.complex.to_mmcif())
        written.append(path)
        mean_plddt = float(res.plddt.mean()) if res.plddt is not None else float("nan")
        extra = ""
        if res.ptm is not None:
            extra += f" pTM={res.ptm:.3f}"
        if res.iptm is not None:
            extra += f" ipTM={res.iptm:.3f}"
        print(f"  wrote {path}  (mean pLDDT={mean_plddt:.1f}{extra})")
    return written


def run(args: argparse.Namespace) -> int:
    import torch

    from esm.models.esmfold2 import ESMFold2InputBuilder
    from esm.models.esmfold2.esmfold2_remote_code.modeling_esmfold2 import (
        ESMFold2Model,
    )

    for knob in ("num_loops", "num_sampling_steps", "num_diffusion_samples"):
        if getattr(args, knob) < 1:
            print(f"error: --{knob.replace('_', '-')} must be >= 1", file=sys.stderr)
            return 2

    # Resolve reproducibility before anything touches CUDA, so the deterministic
    # cuBLAS workspace setting is in place before the context initializes.
    try:
        lm_dropout, seed, deterministic, repro_notes = resolve_reproducibility(
            args.reproducible, args.lm_dropout, args.seed, args.deterministic
        )
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if repro_notes:
        print(f"Reproducible mode: set {', '.join(repro_notes)}.")
    if apply_deterministic(deterministic, torch):
        print(
            "Deterministic CUDA kernels enabled (CUBLAS_WORKSPACE_CONFIG=:4096:8). "
            "Export that variable before launching for strictest effect."
        )
    if lm_dropout == 0.0:
        print("LM dropout disabled — output is deterministic given a fixed seed.")
    elif lm_dropout is None:
        print(
            "note: LM dropout is at esm's default (0.3), so this fold is one draw "
            "from a stochastic ensemble; repeated runs will differ. Use "
            "--reproducible to pin it."
        )

    jobs = resolve_jobs(args.sequence, args.targets, args.fasta, args.query_name)
    print(f"Prepared {len(jobs)} prediction(s).")

    device = resolve_device(
        args.gpu,
        args.device,
        cuda_available=torch.cuda.is_available(),
        device_count=torch.cuda.device_count() if torch.cuda.is_available() else 0,
    )
    # Track peak GPU memory across the whole run (load + encode + fold). Reset
    # now so the reported peak reflects everything this process does on `device`.
    dev = torch.device(device)
    track_mem = dev.type == "cuda" and torch.cuda.is_available()
    if track_mem:
        torch.cuda.reset_peak_memory_stats(dev)

    if args.stage_loading:
        if args.no_offload_esmc:
            print("note: --no-offload-esmc is ignored under --stage-loading.")
        print(
            f"Staged loading (experimental): ESMC and the folding stack will not "
            f"be co-resident on {device}. If this errors with a TransformerEngine "
            f"/ cuBLAS failure, drop --stage-loading — the default path still "
            f"offloads ESMC during folding."
        )
        print(f"Loading folding model {args.model!r} (ESMC loaded on demand) ...")
        model = ESMFold2Model.from_pretrained(args.model, load_esmc=False).eval()
    else:
        print(f"Loading model {args.model!r} on {device} ...")
        model = ESMFold2Model.from_pretrained(args.model).to(device).eval()

    chunk_size, chunk_why = resolve_chunk_size(
        args.chunk_size, args.num_diffusion_samples
    )
    model.set_chunk_size(chunk_size)
    if chunk_size is None:
        print(f"Chunking disabled ({chunk_why}).")
    else:
        if chunk_size < _CHUNK_SMALL_WARN:
            print(
                f"note: --chunk-size {chunk_size} is small; it minimizes "
                "memory but adds per-chunk overhead."
            )
        print(f"L^2 chunk size {chunk_size} ({chunk_why}).")

    builder = ESMFold2InputBuilder()
    inputs = [_build_spi(job) for job in jobs]
    complex_ids = [job.name for job in jobs]

    fold_kwargs = dict(
        num_loops=args.num_loops,
        num_sampling_steps=args.num_sampling_steps,
        num_diffusion_samples=args.num_diffusion_samples,
        seed=seed,
        complex_id=complex_ids,
    )
    # Only pass lm_dropout when the user set it. Passing None is NOT equivalent
    # to omitting it: esm's fold/fold_batch default is 0.3, whereas None makes
    # _lm_dropout_context a no-op and falls back to the checkpoint's config value.
    if lm_dropout is not None:
        fold_kwargs["lm_dropout"] = lm_dropout

    def _do_fold():
        if args.stage_loading:
            return builder.fold_batch_staged(model, inputs, device=device, **fold_kwargs)
        return builder.fold_batch(
            model, inputs, offload_esmc=not args.no_offload_esmc, **fold_kwargs
        )

    # Sample GPU memory across folding to report the median (steady-state)
    # alongside the peak.
    if track_mem:
        with _GpuMemorySampler(dev, torch) as sampler:
            results = _do_fold()
    else:
        sampler = None
        results = _do_fold()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for job, result in zip(jobs, results):
        print(f"{job.name}:")
        _write_result(result, job.name, out_dir)

    if track_mem and sampler is not None:
        med_alloc, med_reserved = sampler.medians()
        print(
            format_memory_report(
                dev,
                torch.cuda.max_memory_allocated(dev),
                torch.cuda.max_memory_reserved(dev),
                med_alloc,
                med_reserved,
                len(sampler.allocated),
            )
        )

    print(f"Done. {len(jobs)} prediction(s) written to {out_dir}/")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except ValueError as e:
        parser.error(str(e))  # exits 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
