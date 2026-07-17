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

    # a single sequence, no L^2 chunking (fastest for short sequences)
    esmfold2-fold --sequence MKTAYIAKQR... --chunk-size none

    # fold every sequence in a FASTA with a smaller chunk for long inputs
    esmfold2-fold --fasta proteins.fasta --chunk-size 32 -o out

Equivalent to ``python -m esm.models.esmfold2.fold_cli ...``.
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

# Sentinel meaning "--chunk-size was not supplied" — leave the model's own
# default (64) untouched. Distinct from the value None, which means the user
# explicitly asked to DISABLE chunking.
_CHUNK_UNSET = object()

# Soft advisory bounds for --chunk-size (not hard limits).
_CHUNK_SMALL_WARN = 8


@dataclass
class Job:
    """One prediction: an output basename and its chains."""

    name: str
    chains: list[tuple[str, str]]  # [(chain_id, sequence), ...]


def parse_chunk_size(value: str):
    """argparse type for ``--chunk-size``.

    Accepts a positive integer (the token-axis chunk width for the L^2 trunk
    ops: triangle multiply / outer-product-mean / pair transition), or one of
    ``none``/``off``/``disable``/``0`` to disable chunking entirely (faster for
    short sequences, but higher peak memory and OOM-prone past L~600).

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
            "Token-axis chunk for the L^2 trunk ops. Lower = less peak memory, "
            "more overhead; 'none' (or 0) disables chunking (fastest for short "
            "sequences, OOM-prone past L~600). Default: model's own value (64)."
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

    if args.chunk_size is not _CHUNK_UNSET:
        model.set_chunk_size(args.chunk_size)
        if args.chunk_size is None:
            print("Chunking disabled (--chunk-size none).")
        else:
            if args.chunk_size < _CHUNK_SMALL_WARN:
                print(
                    f"note: --chunk-size {args.chunk_size} is small; it minimizes "
                    "memory but adds per-chunk overhead."
                )
            print(f"Set L^2 chunk size to {args.chunk_size}.")

    builder = ESMFold2InputBuilder()
    inputs = [_build_spi(job) for job in jobs]
    complex_ids = [job.name for job in jobs]

    fold_kwargs = dict(
        num_loops=args.num_loops,
        num_sampling_steps=args.num_sampling_steps,
        num_diffusion_samples=args.num_diffusion_samples,
        seed=args.seed,
        complex_id=complex_ids,
    )

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
