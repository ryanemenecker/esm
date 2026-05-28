"""Gradient-guided ESMFold2 binder design CLI.

This module implements the binder-design protocol described in Appendix A.3.1.1
of the ESMFold2 preprint. The released ESMFold2 API is optimized for inference:
its public ``forward`` is wrapped in ``torch.inference_mode`` and the protein
input pipeline atomizes each residue from a discrete amino-acid identity. To
keep the design loop differentiable without patching upstream model code, this
implementation rebuilds a discrete binder scaffold from the current argmax
sequence at each step and injects the continuous binder distribution through the
model's differentiable ``res_type`` path.

The result matches the paper's optimization objectives while staying inside the
released model surface:

* ESMFold2 supplies the distogram-based intra-contact, inter-contact, and
  globularity losses.
* ESMC supplies the masked pseudo-perplexity sequence prior.
* Low-temperature candidate selection uses the full ESMFold2 confidence head to
  keep the best ipTM sequence from the trajectory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
import warnings
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence, TextIO

import torch
import torch.nn.functional as F


STANDARD_AA_ORDER = tuple("ARNDCQEGHILKMFPSTWYV")
STANDARD_AA_SET = set(STANDARD_AA_ORDER)
AA_TO_STANDARD_INDEX = {aa: idx for idx, aa in enumerate(STANDARD_AA_ORDER)}
PROTEIN_RESIDUE_TO_RES_TYPE = {
    "ALA": 2,
    "ARG": 3,
    "ASN": 4,
    "ASP": 5,
    "CYS": 6,
    "GLN": 7,
    "GLU": 8,
    "GLY": 9,
    "HIS": 10,
    "ILE": 11,
    "LEU": 12,
    "LYS": 13,
    "MET": 14,
    "PHE": 15,
    "PRO": 16,
    "SER": 17,
    "THR": 18,
    "TRP": 19,
    "TYR": 20,
    "VAL": 21,
}
ESM_PROTEIN_VOCAB = {
    "L": 4,
    "A": 5,
    "G": 6,
    "V": 7,
    "S": 8,
    "E": 9,
    "R": 10,
    "T": 11,
    "I": 12,
    "D": 13,
    "P": 14,
    "K": 15,
    "Q": 16,
    "N": 17,
    "F": 18,
    "Y": 19,
    "M": 20,
    "H": 21,
    "W": 22,
    "C": 23,
    "X": 3,
}
AA_TO_RES_TYPE = {
    "A": PROTEIN_RESIDUE_TO_RES_TYPE["ALA"],
    "R": PROTEIN_RESIDUE_TO_RES_TYPE["ARG"],
    "N": PROTEIN_RESIDUE_TO_RES_TYPE["ASN"],
    "D": PROTEIN_RESIDUE_TO_RES_TYPE["ASP"],
    "C": PROTEIN_RESIDUE_TO_RES_TYPE["CYS"],
    "Q": PROTEIN_RESIDUE_TO_RES_TYPE["GLN"],
    "E": PROTEIN_RESIDUE_TO_RES_TYPE["GLU"],
    "G": PROTEIN_RESIDUE_TO_RES_TYPE["GLY"],
    "H": PROTEIN_RESIDUE_TO_RES_TYPE["HIS"],
    "I": PROTEIN_RESIDUE_TO_RES_TYPE["ILE"],
    "L": PROTEIN_RESIDUE_TO_RES_TYPE["LEU"],
    "K": PROTEIN_RESIDUE_TO_RES_TYPE["LYS"],
    "M": PROTEIN_RESIDUE_TO_RES_TYPE["MET"],
    "F": PROTEIN_RESIDUE_TO_RES_TYPE["PHE"],
    "P": PROTEIN_RESIDUE_TO_RES_TYPE["PRO"],
    "S": PROTEIN_RESIDUE_TO_RES_TYPE["SER"],
    "T": PROTEIN_RESIDUE_TO_RES_TYPE["THR"],
    "W": PROTEIN_RESIDUE_TO_RES_TYPE["TRP"],
    "Y": PROTEIN_RESIDUE_TO_RES_TYPE["TYR"],
    "V": PROTEIN_RESIDUE_TO_RES_TYPE["VAL"],
}
STANDARD_AA_RES_TYPES = [AA_TO_RES_TYPE[aa] for aa in STANDARD_AA_ORDER]
STANDARD_AA_TOKEN_IDS = [ESM_PROTEIN_VOCAB[aa] for aa in STANDARD_AA_ORDER]
NUM_RES_TYPES = max(PROTEIN_RESIDUE_TO_RES_TYPE.values()) + 12
DEFAULT_CYS_LOGIT = -1.0e6
CONTACT_MASK_LOGIT = -1.0e7
MINI_BINDER_LM_WEIGHT = 0.15
ANTIBODY_LM_WEIGHT = 0.05
MINI_BINDER_PI_THRESHOLD = 6.0


@dataclass
class SearchConfig:
    steps: int = 150
    alpha_max: float = 0.1
    temperature_min: float = 0.01
    lm_passes: int = 4
    mask_fraction: float = 0.15
    lambda_lm: float = MINI_BINDER_LM_WEIGHT
    lambda_intra: float = 0.5
    lambda_inter: float = 0.5
    lambda_glob: float = 0.2
    confidence_temperature_threshold: float = 0.05
    confidence_sampling_steps: int = 50
    num_loops: int = 1
    use_search_lm_context: bool = True


@dataclass
class RankingConfig:
    num_loops: int = 1
    num_sampling_steps: int = 50
    selection_score: str = "mean"
    write_top_structures: int = 0


@dataclass
class PreparedComplex:
    features: dict[str, torch.Tensor]
    target_slice: slice
    binder_slice: slice
    target_length: int
    binder_length: int


@dataclass
class TrajectoryResult:
    trajectory_index: int
    sequence: str
    best_iptm: float | None
    best_step: int | None
    final_sequence: str
    final_temperature: float
    losses: dict[str, float]


@dataclass
class RankedCandidate:
    sequence: str
    source_trajectories: list[int]
    search_best_iptm: float | None
    search_best_step: int | None
    pI: float | None
    passed_pi_filter: bool
    mean_iptm: float | None
    mean_proxy: float | None
    mean_ptm: float | None
    mean_plddt: float | None
    pair_chain_iptm: float | None
    selection_score: float | None
    model_scores: list[dict[str, Any]] = field(default_factory=list)


class ProgressBar:
    def __init__(
        self,
        *,
        enabled: bool,
        stream: TextIO | None = None,
        width: int = 28,
    ) -> None:
        self.enabled = enabled
        self.stream = sys.stderr if stream is None else stream
        self.width = width
        self.label = ""
        self.total = 1
        self.completed = 0
        self.started_at = 0.0
        self.active = False
        self._last_line_length = 0
        self._last_logged_percent = -1
        self._interactive = bool(self.enabled and getattr(self.stream, "isatty", lambda: False)())

    def start(self, label: str, total: int) -> None:
        if not self.enabled:
            return
        self.label = label
        self.total = max(1, total)
        self.completed = 0
        self.started_at = time.monotonic()
        self.active = True
        self._last_line_length = 0
        self._last_logged_percent = -1
        self._render(detail="starting", force=True)

    def update(self, advance: int = 1, *, detail: str | None = None) -> None:
        if not self.enabled or not self.active:
            return
        self.completed = min(self.total, self.completed + advance)
        self._render(detail=detail)

    def finish(self, *, detail: str | None = None) -> None:
        if not self.enabled or not self.active:
            return
        self.completed = self.total
        self._render(detail=detail, force=True)
        if self._interactive:
            self.stream.write("\n")
            self.stream.flush()
        self.active = False
        self._last_line_length = 0

    def _render(self, *, detail: str | None = None, force: bool = False) -> None:
        elapsed = max(0.0, time.monotonic() - self.started_at)
        fraction = self.completed / self.total
        percent = int(100 * fraction)
        if self._interactive:
            filled = min(self.width, int(round(self.width * fraction)))
            bar = "#" * filled + "-" * (self.width - filled)
            rate = self.completed / elapsed if elapsed > 0 else 0.0
            eta = (self.total - self.completed) / rate if rate > 0 and self.completed < self.total else 0.0
            parts = [
                f"{self.label} [{bar}] {self.completed}/{self.total} {percent:3d}%",
                f"elapsed {format_duration(elapsed)}",
            ]
            if self.completed < self.total and rate > 0:
                parts.append(f"eta {format_duration(eta)}")
            if detail:
                parts.append(detail)
            line = " | ".join(parts)
            padding = " " * max(0, self._last_line_length - len(line))
            self.stream.write("\r" + line + padding)
            self.stream.flush()
            self._last_line_length = len(line)
            return

        milestone = min(100, (percent // 10) * 10)
        if not force and milestone <= self._last_logged_percent:
            return
        self._last_logged_percent = milestone
        line = f"{self.label}: {self.completed}/{self.total} ({percent}%)"
        if detail:
            line = f"{line} | {detail}"
        self.stream.write(line + "\n")
        self.stream.flush()


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    minutes, sec = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def format_progress_metric(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.3f}"


class FoldModelHandle:
    def __init__(self, model_id: str, model: Any, builder: Any) -> None:
        self.model_id = model_id
        self.model = model
        self.builder = builder


class FoldModelPool:
    def __init__(
        self,
        model_ids: Sequence[str],
        *,
        device: torch.device,
        esmc_model_id: str,
        attn_implementation: str,
        kernel_backend: str | None,
        chunk_size: int | None,
        max_loaded: int,
    ) -> None:
        self.model_ids = list(model_ids)
        self.device = device
        self.esmc_model_id = esmc_model_id
        self.attn_implementation = attn_implementation
        self.kernel_backend = kernel_backend
        self.chunk_size = chunk_size
        self.max_loaded = max_loaded
        self._cache: OrderedDict[str, FoldModelHandle] = OrderedDict()

    def get(self, model_id: str) -> FoldModelHandle:
        from esm.models.esmfold2 import ESMFold2InputBuilder

        if model_id in self._cache:
            handle = self._cache.pop(model_id)
            self._cache[model_id] = handle
            return handle

        handle = FoldModelHandle(
            model_id=model_id,
            model=load_esmfold2_model(
                model_id=model_id,
                device=self.device,
                esmc_model_id=self.esmc_model_id,
                attn_implementation=self.attn_implementation,
                kernel_backend=self.kernel_backend,
                chunk_size=self.chunk_size,
            ),
            builder=ESMFold2InputBuilder(),
        )
        self._cache[model_id] = handle
        while len(self._cache) > self.max_loaded:
            old_model_id, old_handle = self._cache.popitem(last=False)
            del old_handle
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        return handle

    def choice(self, rng: random.Random) -> FoldModelHandle:
        model_id = rng.choice(self.model_ids)
        return self.get(model_id)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Design de novo binders by jointly optimizing ESMFold2 and ESMC objectives."
    )
    parser.add_argument("--target-sequence", help="Target amino-acid sequence, or omit in favor of --target-fasta.")
    parser.add_argument("--target-fasta", type=Path, help="FASTA file containing the target sequence.")
    parser.add_argument(
        "--binder-prompt",
        help="Binder prompt using standard amino acids for fixed positions and # for mutable positions.",
    )
    parser.add_argument("--binder-length", type=int, help="Binder length when --binder-prompt is not provided.")
    parser.add_argument(
        "--binder-type",
        choices=("minibinder", "antibody"),
        default="minibinder",
        help="Controls the paper defaults for LM weight and pI filtering.",
    )
    parser.add_argument("--num-designs", type=int, default=8, help="Number of independent optimization trajectories.")
    parser.add_argument("--top-k", type=int, default=8, help="Number of ranked binders to keep in the final outputs.")
    parser.add_argument(
        "--search-model",
        dest="search_models",
        action="append",
        default=None,
        help="Repeat to provide one or more ESMFold2 checkpoints for the search loop.",
    )
    parser.add_argument(
        "--ranking-model",
        dest="ranking_models",
        action="append",
        default=None,
        help="Repeat to provide one or more ESMFold2 checkpoints for final ranking. Defaults to --search-model.",
    )
    parser.add_argument("--esmc-model", default="Biohub/ESMC-6B", help="ESMC masked-LM checkpoint for the sequence prior.")
    parser.add_argument("--device", default="cuda", help="Torch device. This workflow is intended for CUDA GPUs.")
    parser.add_argument(
        "--attn-implementation",
        choices=("flash_attention_2", "sdpa", "eager"),
        default="flash_attention_2",
        help="Attention backend for ESMC loading. The paper configuration expects FlashAttention.",
    )
    parser.add_argument(
        "--kernel-backend",
        choices=("fused", "cuequivariance", "none"),
        default="fused",
        help="ESMFold2 structure kernel backend.",
    )
    parser.add_argument("--chunk-size", type=int, default=64, help="ESMFold2 chunk size for L^2 pair operations.")
    parser.add_argument("--loaded-fold-models", type=int, default=1, help="Maximum number of ESMFold2 checkpoints kept resident at once.")
    parser.add_argument("--steps", type=int, default=150, help="Optimization steps per trajectory.")
    parser.add_argument("--alpha-max", type=float, default=0.1, help="Base SGD learning rate from Algorithm 11.")
    parser.add_argument("--temperature-min", type=float, default=0.01, help="Final temperature floor for cosine annealing.")
    parser.add_argument("--lm-passes", type=int, default=4, help="Masked pseudo-perplexity passes per optimization step.")
    parser.add_argument("--mask-fraction", type=float, default=0.15, help="Fraction of mutable positions masked in each LM pass.")
    parser.add_argument("--lm-weight", type=float, help="Override the paper default LM weight for the selected binder type.")
    parser.add_argument("--lambda-intra", type=float, default=0.5, help="Weight on the intra-binder contact loss.")
    parser.add_argument("--lambda-inter", type=float, default=0.5, help="Weight on the binder-target contact loss.")
    parser.add_argument("--lambda-glob", type=float, default=0.2, help="Weight on the globularity loss.")
    parser.add_argument(
        "--confidence-temperature-threshold",
        type=float,
        default=0.05,
        help="Start running full confidence passes once the search temperature falls below this value.",
    )
    parser.add_argument("--confidence-steps", type=int, default=50, help="Diffusion steps for low-temperature confidence passes.")
    parser.add_argument("--ranking-steps", type=int, default=50, help="Diffusion steps for final candidate ranking.")
    parser.add_argument(
        "--selection-score",
        choices=("iptm", "proxy", "mean"),
        default="mean",
        help="Final selection score computed from the per-model ipTM and Algorithm 15 proxy.",
    )
    parser.add_argument(
        "--proxy-row-indices",
        help="Optional binder row subset for the Algorithm 15 proxy, e.g. 25-32,50-56 for antibody CDRs.",
    )
    parser.add_argument(
        "--pi-threshold",
        type=float,
        default=None,
        help="Override the minibinder pI filter threshold. Use a negative value to disable filtering.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument("--no-progress", action="store_true", help="Disable live progress reporting.")
    parser.add_argument("--output-dir", type=Path, default=Path("binder_design_outputs"), help="Directory for JSONL, CSV, and optional structure outputs.")
    parser.add_argument("--write-top-structures", type=int, default=0, help="Write mmCIF files for the top ranked binders from the first ranking checkpoint.")
    parser.add_argument(
        "--disable-search-lm-context",
        action="store_true",
        help="Skip the discrete LM context inside the ESMFold2 search pass for speed. The explicit ESMC LM loss still runs.",
    )
    return parser.parse_args(argv)


def sanitize_sequence(sequence: str) -> str:
    cleaned = "".join(line.strip() for line in sequence.splitlines() if not line.startswith(">"))
    cleaned = cleaned.replace(" ", "").replace("\t", "").upper()
    if not cleaned:
        raise ValueError("Sequence is empty after stripping whitespace and FASTA headers.")
    invalid = sorted(set(cleaned) - STANDARD_AA_SET)
    if invalid:
        raise ValueError(f"Sequence contains unsupported residues: {''.join(invalid)}")
    return cleaned


def load_sequence(args: argparse.Namespace) -> str:
    if bool(args.target_sequence) == bool(args.target_fasta):
        raise ValueError("Provide exactly one of --target-sequence or --target-fasta.")
    if args.target_sequence:
        return sanitize_sequence(args.target_sequence)
    assert args.target_fasta is not None
    return sanitize_sequence(args.target_fasta.read_text())


def parse_prompt(prompt: str | None, binder_length: int | None) -> str:
    if prompt is None:
        if binder_length is None:
            raise ValueError("Provide --binder-prompt or --binder-length.")
        if binder_length <= 0:
            raise ValueError("--binder-length must be positive.")
        return "#" * binder_length
    cleaned = prompt.strip().upper()
    if not cleaned:
        raise ValueError("--binder-prompt cannot be empty.")
    invalid = sorted(set(cleaned) - (STANDARD_AA_SET | {"#"}))
    if invalid:
        raise ValueError(f"Binder prompt contains unsupported symbols: {''.join(invalid)}")
    if binder_length is not None and binder_length != len(cleaned):
        raise ValueError("--binder-length must match the prompt length when both are provided.")
    return cleaned


def parse_index_subset(spec: str | None) -> list[int] | None:
    if spec is None:
        return None
    indices: set[int] = set()
    for part in spec.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_str, end_str = token.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            if end < start:
                raise ValueError(f"Invalid range {token!r}: end before start.")
            indices.update(range(start, end + 1))
        else:
            indices.add(int(token))
    return sorted(indices)


def default_lm_weight(binder_type: str) -> float:
    if binder_type == "antibody":
        return ANTIBODY_LM_WEIGHT
    return MINI_BINDER_LM_WEIGHT


def default_pi_threshold(binder_type: str) -> float | None:
    if binder_type == "minibinder":
        return MINI_BINDER_PI_THRESHOLD
    return None


def require_cuda(device: torch.device) -> None:
    if device.type != "cuda":
        raise RuntimeError(
            "This binder-design workflow is intended for a CUDA GPU with bfloat16 support. "
            "Run it on the target Blackwell workstation or pass a CUDA device explicitly."
        )


def configure_torch_for_gpu(device: torch.device) -> None:
    require_cuda(device)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The selected GPU does not report bfloat16 support.")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def maybe_autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cpu", enabled=False)


def temperature_at_step(step_index: int, total_steps: int, temperature_min: float) -> float:
    return temperature_min + (1.0 - temperature_min) * 0.5 * (1.0 + math.cos(math.pi * step_index / total_steps))


def initialize_binder_logits(prompt: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    logits = torch.empty(len(prompt), len(STANDARD_AA_ORDER), device=device, dtype=torch.float32)
    gradient_mask = torch.zeros_like(logits)
    cys_index = AA_TO_STANDARD_INDEX["C"]
    for position, token in enumerate(prompt):
        if token == "#":
            logits[position].normal_(mean=0.0, std=1.0e-2)
            logits[position, cys_index] = DEFAULT_CYS_LOGIT
            gradient_mask[position] = 1.0
            gradient_mask[position, cys_index] = 0.0
        else:
            logits[position].zero_()
            logits[position, AA_TO_STANDARD_INDEX[token]] = 10.0
    return logits, gradient_mask


def binder_distribution_to_sequence(distribution: torch.Tensor) -> str:
    indices = distribution.argmax(dim=-1).tolist()
    return "".join(STANDARD_AA_ORDER[index] for index in indices)


def binder_distribution_to_res_type_soft(soft_binder: torch.Tensor) -> torch.Tensor:
    res_type = soft_binder.new_zeros((soft_binder.shape[0], NUM_RES_TYPES))
    res_type[:, STANDARD_AA_RES_TYPES] = soft_binder
    return res_type


def prepare_complex(handle: FoldModelHandle, target_sequence: str, binder_sequence: str) -> PreparedComplex:
    from esm.models.esmfold2 import ProteinInput, StructurePredictionInput

    spi = StructurePredictionInput(
        sequences=[
            ProteinInput(id="A", sequence=target_sequence),
            ProteinInput(id="B", sequence=binder_sequence),
        ]
    )
    features, chain_infos = handle.builder.prepare_input(spi, device=handle.model.device)
    if len(chain_infos) != 2:
        raise ValueError(f"Expected two chains in the prepared complex, got {len(chain_infos)}")
    target_tokens = chain_infos[0].tokens
    binder_tokens = chain_infos[1].tokens
    if not target_tokens or not binder_tokens:
        raise ValueError("Prepared complex is missing target or binder tokens.")
    target_slice = slice(target_tokens[0].token_index, target_tokens[-1].token_index + 1)
    binder_slice = slice(binder_tokens[0].token_index, binder_tokens[-1].token_index + 1)
    return PreparedComplex(
        features=features,
        target_slice=target_slice,
        binder_slice=binder_slice,
        target_length=len(target_tokens),
        binder_length=len(binder_tokens),
    )


def build_soft_res_type_tensor(prepared: PreparedComplex, soft_binder: torch.Tensor) -> torch.Tensor:
    template = F.one_hot(prepared.features["res_type"].long(), num_classes=NUM_RES_TYPES).float()
    template[:, prepared.binder_slice, :] = binder_distribution_to_res_type_soft(soft_binder).unsqueeze(0)
    return template


def esmfold2_distogram_pass(
    model: Any,
    prepared: PreparedComplex,
    soft_res_type: torch.Tensor,
    *,
    input_ids: torch.Tensor | None,
    num_loops: int,
) -> torch.Tensor:
    from transformers.models.esmfold2.modeling_esmfold2_common import (  # pyright: ignore[reportMissingImports]
        CHAR_VOCAB_SIZE,
        MAX_ATOMIC_NUMBER,
    )

    features = prepared.features
    tok_mask = features["token_attention_mask"]
    atm_mask = features["atom_attention_mask"]
    atom_to_token = features["atom_to_token"] * atm_mask.long()
    deletion_mean = torch.zeros(
        soft_res_type.shape[0],
        soft_res_type.shape[1],
        device=soft_res_type.device,
        dtype=torch.float32,
    )
    ref_element_oh = F.one_hot(features["ref_element"].long(), num_classes=MAX_ATOMIC_NUMBER).float()
    ref_atom_name_chars_oh = F.one_hot(
        features["ref_atom_name_chars"].long(), num_classes=CHAR_VOCAB_SIZE
    ).float()
    atm_mask_f = atm_mask.float()
    ref_element_oh = ref_element_oh * atm_mask_f.unsqueeze(-1)
    ref_atom_name_chars_oh = ref_atom_name_chars_oh * atm_mask_f.unsqueeze(-1).unsqueeze(-1)

    with maybe_autocast(model.device):
        x_inputs = model.inputs_embedder(
            aatype=soft_res_type,
            profile=soft_res_type.float(),
            deletion_mean=deletion_mean,
            ref_pos=features["ref_pos"],
            atom_attention_mask=atm_mask,
            ref_space_uid=features["ref_space_uid"],
            ref_charge=features["ref_charge"],
            ref_element=ref_element_oh,
            ref_atom_name_chars=ref_atom_name_chars_oh,
            atom_to_token=atom_to_token,
        )
        z_init = model.z_init_1(x_inputs).unsqueeze(2) + model.z_init_2(x_inputs).unsqueeze(1)
        relative_position_encoding = model.rel_pos(
            residue_index=features["residue_index"],
            asym_id=features["asym_id"],
            sym_id=features["sym_id"],
            entity_id=features["entity_id"],
            token_index=features["token_index"],
        )
        token_bonds_encoding = model.token_bonds(features["token_bonds"].float())
        z_init = z_init + relative_position_encoding + token_bonds_encoding

        lm_z = None
        if input_ids is not None and getattr(model, "_esmc", None) is not None:
            with torch.no_grad():
                lm_hidden_states = model._compute_lm_hidden_states(
                    input_ids,
                    features["asym_id"],
                    features["residue_index"],
                    features["mol_type"],
                    tok_mask,
                )
                lm_z = model.language_model(lm_hidden_states)

        pair_mask = tok_mask[:, :, None].float() * tok_mask[:, None, :].float()
        z = model._init_pair_state(z_init)
        a, b = model._discretized_dynamics()
        a = a.view(1, 1, 1, -1).to(device=z.device, dtype=z.dtype)
        b_mat = b.to(device=z.device, dtype=z.dtype)
        z = model._run_one_loop(
            z=z,
            z_init=z_init,
            lm_z=lm_z,
            _msa_kwargs=None,
            pair_mask=pair_mask,
            a=a,
            b_mat=b_mat,
            total_steps=max(1, num_loops + 1),
        )
        z = model.parcae_readout(z)
        z = model.parcae_coda(z, pair_attention_mask=pair_mask)
        z = z.float()
        return model.distogram_head(z + z.transpose(-2, -3))


def restricted_contact_cross_entropy(distogram_block: torch.Tensor, contact_cutoff: float) -> torch.Tensor:
    bin_centers = distogram_bin_centers(
        distogram_block.shape[-1],
        device=distogram_block.device,
        dtype=distogram_block.dtype,
    )
    restricted_logits = distogram_block - (bin_centers >= contact_cutoff).to(distogram_block.dtype) * CONTACT_MASK_LOGIT
    p_contact = torch.softmax(restricted_logits, dim=-1)
    return -(p_contact * torch.log_softmax(distogram_block, dim=-1)).sum(dim=-1)


def contact_loss(
    distogram_block: torch.Tensor,
    *,
    k_contacts: int,
    min_sequence_separation: int,
    contact_cutoff: float,
) -> torch.Tensor:
    cross_entropy = restricted_contact_cross_entropy(distogram_block, contact_cutoff)
    row_count, col_count = cross_entropy.shape
    losses: list[torch.Tensor] = []
    for row_index in range(row_count):
        eligible = torch.ones(col_count, device=cross_entropy.device, dtype=torch.bool)
        if row_count == col_count and min_sequence_separation > 0:
            col_indices = torch.arange(col_count, device=cross_entropy.device)
            eligible = (col_indices - row_index).abs() >= min_sequence_separation
        row_values = cross_entropy[row_index][eligible]
        if row_values.numel() == 0:
            continue
        k = min(k_contacts, row_values.numel())
        losses.append(row_values.topk(k=k, largest=False).values.mean())
    if not losses:
        return cross_entropy.new_tensor(0.0)
    return torch.stack(losses).mean()


def intra_contact_loss(distogram_logits: torch.Tensor, binder_slice: slice) -> torch.Tensor:
    # The binder occupies a contiguous suffix of the target+binder complex.
    # Slicing the binder-binder block isolates tertiary contacts that only
    # report whether the designed chain is predicted to fold into itself.
    distogram_block = distogram_logits[:, binder_slice, binder_slice, :].squeeze(0)
    return contact_loss(
        distogram_block,
        k_contacts=2,
        min_sequence_separation=9,
        contact_cutoff=14.0,
    )


def inter_contact_loss(distogram_logits: torch.Tensor, target_slice: slice, binder_slice: slice) -> torch.Tensor:
    # The paper defines the inter-chain loss on the target->binder block so each
    # target residue is scored by the confidence of its strongest binder contact.
    distogram_block = distogram_logits[:, target_slice, binder_slice, :].squeeze(0)
    return contact_loss(
        distogram_block,
        k_contacts=1,
        min_sequence_separation=0,
        contact_cutoff=22.0,
    )


def globularity_loss(distogram_logits: torch.Tensor, binder_slice: slice) -> torch.Tensor:
    binder_block = distogram_logits[:, binder_slice, binder_slice, :].squeeze(0)
    bin_centers = distogram_bin_centers(
        binder_block.shape[-1],
        device=binder_block.device,
        dtype=binder_block.dtype,
    )
    clamped_sq = torch.minimum(bin_centers, bin_centers.new_tensor(27.0)).pow(2)
    expected_sq_dist = (torch.softmax(binder_block, dim=-1) * clamped_sq).sum(dim=-1)
    triu = torch.triu(expected_sq_dist, diagonal=1)
    binder_length = binder_block.shape[0]
    radius_of_gyration = torch.sqrt(triu.sum() / max(binder_length * binder_length, 1))
    packing_radius = 2.38 * (binder_length ** 0.365)
    return F.elu(radius_of_gyration - packing_radius)


def normalize_masked_gradient(gradient: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked_gradient = gradient * mask
    norm = masked_gradient.norm()
    if torch.isfinite(norm) and float(norm) > 0.0:
        scale = math.sqrt(mask[..., 0].sum().item()) if mask.ndim == 2 else 1.0
        return scale * masked_gradient / norm
    return torch.zeros_like(gradient)


def straight_through_onehot(soft_probs: torch.Tensor) -> torch.Tensor:
    hard_indices = soft_probs.argmax(dim=-1)
    hard = F.one_hot(hard_indices, num_classes=soft_probs.shape[-1]).to(soft_probs.dtype)
    return hard + soft_probs - soft_probs.detach()


def soft_binder_to_vocab_distribution(
    soft_binder: torch.Tensor,
    *,
    cls_token_id: int,
    eos_token_id: int,
    vocab_size: int,
) -> torch.Tensor:
    distribution = soft_binder.new_zeros((soft_binder.shape[0] + 2, vocab_size))
    distribution[0, cls_token_id] = 1.0
    distribution[-1, eos_token_id] = 1.0
    distribution[1:-1, STANDARD_AA_TOKEN_IDS] = soft_binder
    return distribution.unsqueeze(0)


def forward_esmc_soft(masked_lm_model: Any, token_distribution: torch.Tensor) -> torch.Tensor:
    encoder = masked_lm_model.esmc
    embed_weight = encoder.get_input_embeddings().weight
    hidden = token_distribution.to(embed_weight.dtype) @ embed_weight
    batch_size, sequence_length, _ = hidden.shape
    attention_mask = torch.ones(batch_size, sequence_length, dtype=torch.bool, device=hidden.device)

    if getattr(encoder, "_use_flash_attn", False):
        from flash_attn.bert_padding import pad_input, unpad_input  # pyright: ignore[reportMissingImports]

        hidden_unpadded, indices, *_ = unpad_input(hidden, attention_mask)
        last_hidden_state, _, _, _ = encoder.transformer(
            hidden_unpadded,
            sequence_id=attention_mask,
            output_attentions=False,
        )
        last_hidden_state = pad_input(last_hidden_state, indices, batch_size, sequence_length)
    else:
        last_hidden_state, _, _, _ = encoder.transformer(
            hidden,
            sequence_id=None,
            output_attentions=False,
        )

    return masked_lm_model.lm_head(last_hidden_state)


def masked_pseudo_perplexity_loss(
    masked_lm_model: Any,
    tokenizer: Any,
    soft_binder: torch.Tensor,
    mutable_positions: torch.Tensor,
    *,
    lm_passes: int,
    mask_fraction: float,
) -> torch.Tensor:
    mutable_indices = mutable_positions.nonzero(as_tuple=False).flatten()
    if mutable_indices.numel() == 0:
        return soft_binder.new_tensor(0.0)

    binder_st = straight_through_onehot(soft_binder)
    base_distribution = soft_binder_to_vocab_distribution(
        binder_st,
        cls_token_id=tokenizer.cls_token_id,
        eos_token_id=tokenizer.eos_token_id,
        vocab_size=masked_lm_model.config.vocab_size,
    )
    losses: list[torch.Tensor] = []
    num_to_mask = max(1, math.ceil(mask_fraction * mutable_indices.numel()))
    aa_token_ids = torch.tensor(STANDARD_AA_TOKEN_IDS, device=soft_binder.device)

    for _ in range(lm_passes):
        selection = mutable_indices[torch.randperm(mutable_indices.numel(), device=soft_binder.device)[:num_to_mask]]
        masked_distribution = base_distribution.clone()
        masked_distribution[0, selection + 1, :] = 0.0
        masked_distribution[0, selection + 1, tokenizer.mask_token_id] = 1.0
        with maybe_autocast(soft_binder.device):
            logits = forward_esmc_soft(masked_lm_model, masked_distribution)
        masked_logits = logits[0, selection + 1][:, aa_token_ids]
        log_probs = torch.log_softmax(masked_logits.float(), dim=-1)
        losses.append(-(soft_binder[selection] * log_probs).sum(dim=-1).mean())

    return torch.stack(losses).mean()


def distogram_iptm_proxy(
    distogram_logits: torch.Tensor,
    *,
    target_length: int,
    binder_length: int,
    binder_row_subset: Sequence[int] | None,
) -> float:
    if distogram_logits.ndim != 3:
        raise ValueError(f"Expected distogram logits with shape [L, L, B], got {tuple(distogram_logits.shape)}")
    binder_to_target = distogram_logits[target_length : target_length + binder_length, 0:target_length, :]
    if binder_row_subset is not None:
        subset_tensor = torch.tensor(list(binder_row_subset), device=binder_to_target.device)
        binder_to_target = binder_to_target.index_select(0, subset_tensor)
        if binder_to_target.shape[0] == 0:
            return float("nan")

    bin_centers = distogram_bin_centers(
        binder_to_target.shape[-1],
        device=binder_to_target.device,
        dtype=binder_to_target.dtype,
    )
    contact_mask = (bin_centers < 22.0).to(binder_to_target.dtype)
    p_full = torch.softmax(binder_to_target, dim=-1)
    p_cut = torch.softmax(binder_to_target - (1.0 - contact_mask) * CONTACT_MASK_LOGIT, dim=-1)
    pair_scores = -(p_cut * torch.log(p_full.clamp(min=1.0e-12))).sum(dim=-1).reshape(-1)
    k = min(binder_to_target.shape[0], pair_scores.numel())
    if k == 0:
        return float("nan")
    mean_low_scores = pair_scores.topk(k=k, largest=False).values.mean()
    proxy = torch.clamp(1.0 - mean_low_scores / math.log(51.0), min=0.0, max=1.0)
    return float(proxy.item())


def sequence_pI(sequence: str) -> float | None:
    try:
        from Bio.SeqUtils.ProtParam import ProteinAnalysis

        return float(ProteinAnalysis(sequence).isoelectric_point())
    except Exception:
        return None


def distogram_bin_centers(
    num_bins: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    min_distance: float = 1.5,
    max_distance: float = 54.5,
) -> torch.Tensor:
    # The paper defines distogram midpoints over [1.5, 54.5] A. Some released
    # checkpoints emit fewer bins than the appendix's 128, so derive the centers
    # from the live tensor shape instead of hard-coding the bin count.
    if num_bins <= 0:
        raise ValueError(f"Distogram bin count must be positive, got {num_bins}.")
    return torch.linspace(min_distance, max_distance, num_bins, device=device, dtype=dtype)


def resolve_fold_esmc_attn_implementation(attn_implementation: str, *, warn: bool = True) -> str:
    # Binder design always folds a target+binder complex, so the ESMC instance
    # embedded inside ESMFold2 must support chain-aware attention masks.
    if attn_implementation == "flash_attention_2":
        if warn:
            warnings.warn(
                "ESMFold2 binder design uses multi-chain complexes; overriding the "
                "ESMFold2-internal ESMC attention backend from 'flash_attention_2' "
                "to 'sdpa'. The standalone ESMC pseudo-perplexity model keeps the "
                "requested backend.",
                stacklevel=2,
            )
        return "sdpa"
    return attn_implementation


def load_esmfold2_model(
    *,
    model_id: str,
    device: torch.device,
    esmc_model_id: str,
    attn_implementation: str,
    kernel_backend: str | None,
    chunk_size: int | None,
) -> Any:
    from transformers.models.esmc.modeling_esmc import ESMCModel  # pyright: ignore[reportMissingImports]
    from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model  # pyright: ignore[reportMissingImports]

    model = ESMFold2Model.from_pretrained(model_id, load_esmc=False).to(device=device, dtype=torch.bfloat16).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if kernel_backend is not None:
        model.set_kernel_backend(kernel_backend)
    if chunk_size is not None:
        model.set_chunk_size(chunk_size)

    fold_esmc_attn_implementation = resolve_fold_esmc_attn_implementation(attn_implementation)
    esmc = ESMCModel.from_pretrained(
        esmc_model_id,
        attn_implementation=fold_esmc_attn_implementation,
    )
    esmc = esmc.to(device=device, dtype=torch.bfloat16).eval()
    for parameter in esmc.parameters():
        parameter.requires_grad_(False)
    model._esmc = esmc
    model._esmc_fp8 = False
    return model


def load_esmc_masked_lm(model_id: str, device: torch.device, attn_implementation: str) -> tuple[Any, Any]:
    from transformers import AutoTokenizer
    from transformers.models.esmc.modeling_esmc import ESMCForMaskedLM  # pyright: ignore[reportMissingImports]

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = ESMCForMaskedLM.from_pretrained(model_id, attn_implementation=attn_implementation)
    model = model.to(device=device, dtype=torch.bfloat16).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, tokenizer


def full_fold_candidate(
    handle: FoldModelHandle,
    target_sequence: str,
    binder_sequence: str,
    *,
    seed: int,
    num_loops: int,
    num_sampling_steps: int,
) -> Any:
    from esm.models.esmfold2 import ProteinInput, StructurePredictionInput

    spi = StructurePredictionInput(
        sequences=[
            ProteinInput(id="A", sequence=target_sequence),
            ProteinInput(id="B", sequence=binder_sequence),
        ]
    )
    return handle.builder.fold(
        handle.model,
        spi,
        num_loops=num_loops,
        num_sampling_steps=num_sampling_steps,
        num_diffusion_samples=1,
        seed=seed,
    )


def combine_selection_score(mean_iptm: float | None, mean_proxy: float | None, mode: str) -> float | None:
    if mode == "iptm":
        return mean_iptm
    if mode == "proxy":
        return mean_proxy
    values = [value for value in (mean_iptm, mean_proxy) if value is not None and math.isfinite(value)]
    if not values:
        return None
    return float(sum(values) / len(values))


def nanmean(values: Iterable[float | None]) -> float | None:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    if not finite:
        return None
    return float(sum(finite) / len(finite))


def trajectory_search(
    trajectory_index: int,
    *,
    target_sequence: str,
    binder_prompt: str,
    search_config: SearchConfig,
    search_pool: FoldModelPool,
    lm_model: Any,
    lm_tokenizer: Any,
    seed: int,
    rng: random.Random,
    progress: ProgressBar | None = None,
    num_trajectories: int | None = None,
) -> TrajectoryResult:
    device = search_pool.device
    logits, gradient_mask = initialize_binder_logits(binder_prompt, device=device)
    mutable_positions = torch.tensor([token == "#" for token in binder_prompt], device=device)
    best_sequence = binder_distribution_to_sequence(torch.softmax(logits, dim=-1))
    best_iptm = float("-inf")
    best_step: int | None = None
    final_losses: dict[str, float] = {}

    for step in range(1, search_config.steps + 1):
        logits = logits.detach().requires_grad_(True)
        temperature = temperature_at_step(step, search_config.steps, search_config.temperature_min)
        alpha = search_config.alpha_max * temperature
        soft_binder = torch.softmax(logits / temperature, dim=-1)
        hard_binder = binder_distribution_to_sequence(soft_binder)

        handle = search_pool.choice(rng)
        prepared = prepare_complex(handle, target_sequence, hard_binder)
        soft_res_type = build_soft_res_type_tensor(prepared, soft_binder)
        input_ids = prepared.features["input_ids"] if search_config.use_search_lm_context else None
        distogram_logits = esmfold2_distogram_pass(
            handle.model,
            prepared,
            soft_res_type,
            input_ids=input_ids,
            num_loops=search_config.num_loops,
        )

        loss_intra = intra_contact_loss(distogram_logits, prepared.binder_slice)
        loss_inter = inter_contact_loss(distogram_logits, prepared.target_slice, prepared.binder_slice)
        loss_glob = globularity_loss(distogram_logits, prepared.binder_slice)
        loss_struct = (
            search_config.lambda_intra * loss_intra
            + search_config.lambda_inter * loss_inter
            + search_config.lambda_glob * loss_glob
        )
        loss_lm = masked_pseudo_perplexity_loss(
            lm_model,
            lm_tokenizer,
            soft_binder,
            mutable_positions,
            lm_passes=search_config.lm_passes,
            mask_fraction=search_config.mask_fraction,
        )

        grad_struct = torch.autograd.grad(loss_struct, logits, retain_graph=True)[0]
        grad_lm = torch.autograd.grad(loss_lm, logits)[0]
        total_grad = normalize_masked_gradient(grad_struct, gradient_mask)
        total_grad = total_grad + search_config.lambda_lm * normalize_masked_gradient(grad_lm, gradient_mask)

        with torch.no_grad():
            logits = logits - alpha * total_grad
            logits[:, AA_TO_STANDARD_INDEX["C"]] = DEFAULT_CYS_LOGIT

        if temperature < search_config.confidence_temperature_threshold:
            result = full_fold_candidate(
                handle,
                target_sequence,
                hard_binder,
                seed=seed + step,
                num_loops=search_config.num_loops,
                num_sampling_steps=search_config.confidence_sampling_steps,
            )
            if result.iptm is not None and float(result.iptm) > best_iptm:
                best_iptm = float(result.iptm)
                best_sequence = hard_binder
                best_step = step

        final_losses = {
            "temperature": float(temperature),
            "loss_intra": float(loss_intra.item()),
            "loss_inter": float(loss_inter.item()),
            "loss_glob": float(loss_glob.item()),
            "loss_struct": float(loss_struct.item()),
            "loss_lm": float(loss_lm.item()),
        }
        if progress is not None:
            best_iptm_value = None if best_step is None else best_iptm
            trajectory_total = num_trajectories if num_trajectories is not None else "?"
            progress.update(
                detail=(
                    f"traj {trajectory_index + 1}/{trajectory_total} "
                    f"step {step}/{search_config.steps} "
                    f"temp={temperature:.3f} "
                    f"struct={loss_struct.item():.3f} "
                    f"lm={loss_lm.item():.3f} "
                    f"best_ipTM={format_progress_metric(best_iptm_value)}"
                )
            )

    final_soft = torch.softmax(logits.detach() / search_config.temperature_min, dim=-1)
    final_sequence = binder_distribution_to_sequence(final_soft)
    if best_step is None:
        best_sequence = final_sequence

    return TrajectoryResult(
        trajectory_index=trajectory_index,
        sequence=best_sequence,
        best_iptm=None if best_step is None else float(best_iptm),
        best_step=best_step,
        final_sequence=final_sequence,
        final_temperature=search_config.temperature_min,
        losses=final_losses,
    )


def deduplicate_trajectories(trajectories: Sequence[TrajectoryResult]) -> list[RankedCandidate]:
    best_by_sequence: dict[str, RankedCandidate] = {}
    for trajectory in trajectories:
        entry = best_by_sequence.get(trajectory.sequence)
        if entry is None:
            best_by_sequence[trajectory.sequence] = RankedCandidate(
                sequence=trajectory.sequence,
                source_trajectories=[trajectory.trajectory_index],
                search_best_iptm=trajectory.best_iptm,
                search_best_step=trajectory.best_step,
                pI=None,
                passed_pi_filter=True,
                mean_iptm=None,
                mean_proxy=None,
                mean_ptm=None,
                mean_plddt=None,
                pair_chain_iptm=None,
                selection_score=None,
            )
            continue

        entry.source_trajectories.append(trajectory.trajectory_index)
        current = trajectory.best_iptm if trajectory.best_iptm is not None else float("-inf")
        existing = entry.search_best_iptm if entry.search_best_iptm is not None else float("-inf")
        if current > existing:
            entry.search_best_iptm = trajectory.best_iptm
            entry.search_best_step = trajectory.best_step

    return list(best_by_sequence.values())


def rank_candidates(
    candidates: Sequence[RankedCandidate],
    *,
    target_sequence: str,
    ranking_pool: FoldModelPool,
    ranking_config: RankingConfig,
    binder_type: str,
    binder_row_subset: Sequence[int] | None,
    pi_threshold: float | None,
    output_dir: Path,
    progress: ProgressBar | None = None,
) -> list[RankedCandidate]:
    ranked: list[RankedCandidate] = []
    structures_dir = output_dir / "structures"
    if ranking_config.write_top_structures > 0:
        structures_dir.mkdir(parents=True, exist_ok=True)

    total_candidates = len(candidates)
    total_models = len(ranking_pool.model_ids)
    for candidate_index, candidate in enumerate(candidates, start=1):
        candidate.pI = sequence_pI(candidate.sequence)
        threshold = pi_threshold
        if threshold is None:
            candidate.passed_pi_filter = True
        elif threshold < 0:
            candidate.passed_pi_filter = True
        else:
            candidate.passed_pi_filter = candidate.pI is None or candidate.pI <= threshold

        per_model_scores: list[dict[str, Any]] = []
        for model_index, model_id in enumerate(ranking_pool.model_ids):
            handle = ranking_pool.get(model_id)
            result = full_fold_candidate(
                handle,
                target_sequence,
                candidate.sequence,
                seed=17 * (model_index + 1),
                num_loops=ranking_config.num_loops,
                num_sampling_steps=ranking_config.num_sampling_steps,
            )
            proxy = None
            if result.distogram is not None:
                proxy = distogram_iptm_proxy(
                    result.distogram,
                    target_length=len(target_sequence),
                    binder_length=len(candidate.sequence),
                    binder_row_subset=binder_row_subset,
                )
            pair_chain_iptm = None
            if result.pair_chains_iptm is not None and result.pair_chains_iptm.numel() >= 4:
                pair_chain_iptm = float(result.pair_chains_iptm[1, 0].item())

            per_model_scores.append(
                {
                    "model_id": model_id,
                    "iptm": None if result.iptm is None else float(result.iptm),
                    "ptm": None if result.ptm is None else float(result.ptm),
                    "plddt_mean": float(result.plddt.mean().item()),
                    "pair_chain_iptm": pair_chain_iptm,
                    "distogram_proxy": proxy,
                }
            )

            if model_index == 0 and ranking_config.write_top_structures > 0:
                structure_path = structures_dir / f"{candidate.sequence}.cif"
                structure_path.write_text(result.complex.to_mmcif())

            if progress is not None:
                progress.update(
                    detail=(
                        f"candidate {candidate_index}/{total_candidates} "
                        f"model {model_index + 1}/{total_models} "
                        f"ipTM={format_progress_metric(None if result.iptm is None else float(result.iptm))} "
                        f"proxy={format_progress_metric(proxy)}"
                    )
                )

        candidate.model_scores = per_model_scores
        candidate.mean_iptm = nanmean(score["iptm"] for score in per_model_scores)
        candidate.mean_proxy = nanmean(score["distogram_proxy"] for score in per_model_scores)
        candidate.mean_ptm = nanmean(score["ptm"] for score in per_model_scores)
        candidate.mean_plddt = nanmean(score["plddt_mean"] for score in per_model_scores)
        candidate.pair_chain_iptm = nanmean(score["pair_chain_iptm"] for score in per_model_scores)
        candidate.selection_score = combine_selection_score(
            candidate.mean_iptm,
            candidate.mean_proxy,
            ranking_config.selection_score,
        )
        ranked.append(candidate)

    filtered = [candidate for candidate in ranked if candidate.passed_pi_filter]
    if binder_type == "minibinder" and filtered:
        ranked = filtered

    ranked.sort(
        key=lambda candidate: (
            candidate.selection_score is None,
            -(candidate.selection_score or float("-inf")),
            -(candidate.search_best_iptm or float("-inf")),
        )
    )
    return ranked


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: to_jsonable(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [to_jsonable(inner) for inner in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return None
    if dataclass_isinstance(value):
        return to_jsonable(asdict(value))
    return value


def dataclass_isinstance(value: Any) -> bool:
    return hasattr(value, "__dataclass_fields__")


def write_outputs(
    output_dir: Path,
    *,
    config: dict[str, Any],
    trajectories: Sequence[TrajectoryResult],
    ranked_candidates: Sequence[RankedCandidate],
    top_k: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(json.dumps(to_jsonable(config), indent=2))

    with (output_dir / "raw_trajectories.jsonl").open("w") as handle:
        for trajectory in trajectories:
            handle.write(json.dumps(to_jsonable(trajectory)) + "\n")

    with (output_dir / "ranked_candidates.jsonl").open("w") as handle:
        for candidate in ranked_candidates[:top_k]:
            handle.write(json.dumps(to_jsonable(candidate)) + "\n")

    csv_path = output_dir / "ranked_candidates.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sequence",
                "source_trajectories",
                "search_best_iptm",
                "search_best_step",
                "pI",
                "passed_pi_filter",
                "mean_iptm",
                "mean_proxy",
                "mean_ptm",
                "mean_plddt",
                "pair_chain_iptm",
                "selection_score",
            ],
        )
        writer.writeheader()
        for candidate in ranked_candidates[:top_k]:
            writer.writerow(
                {
                    "sequence": candidate.sequence,
                    "source_trajectories": ";".join(map(str, candidate.source_trajectories)),
                    "search_best_iptm": candidate.search_best_iptm,
                    "search_best_step": candidate.search_best_step,
                    "pI": candidate.pI,
                    "passed_pi_filter": candidate.passed_pi_filter,
                    "mean_iptm": candidate.mean_iptm,
                    "mean_proxy": candidate.mean_proxy,
                    "mean_ptm": candidate.mean_ptm,
                    "mean_plddt": candidate.mean_plddt,
                    "pair_chain_iptm": candidate.pair_chain_iptm,
                    "selection_score": candidate.selection_score,
                }
            )


def build_run_config(args: argparse.Namespace, search_config: SearchConfig, ranking_config: RankingConfig) -> dict[str, Any]:
    return {
        "target_sequence": args.target_sequence if args.target_sequence else str(args.target_fasta),
        "binder_prompt": args.binder_prompt,
        "binder_length": args.binder_length,
        "binder_type": args.binder_type,
        "search_models": args.search_models,
        "ranking_models": args.ranking_models,
        "esmc_model": args.esmc_model,
        "device": args.device,
        "attn_implementation": args.attn_implementation,
        "fold_esmc_attn_implementation": resolve_fold_esmc_attn_implementation(
            args.attn_implementation,
            warn=False,
        ),
        "kernel_backend": args.kernel_backend,
        "chunk_size": args.chunk_size,
        "loaded_fold_models": args.loaded_fold_models,
        "search_config": search_config,
        "ranking_config": ranking_config,
        "proxy_row_indices": args.proxy_row_indices,
        "pi_threshold": args.pi_threshold,
        "progress": not args.no_progress,
        "seed": args.seed,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    target_sequence = load_sequence(args)
    binder_prompt = parse_prompt(args.binder_prompt, args.binder_length)
    binder_row_subset = parse_index_subset(args.proxy_row_indices)
    search_models = args.search_models or ["biohub/ESMFold2"]
    ranking_models = args.ranking_models or search_models

    device = torch.device(args.device)
    configure_torch_for_gpu(device)

    kernel_backend = None if args.kernel_backend == "none" else args.kernel_backend
    search_config = SearchConfig(
        steps=args.steps,
        alpha_max=args.alpha_max,
        temperature_min=args.temperature_min,
        lm_passes=args.lm_passes,
        mask_fraction=args.mask_fraction,
        lambda_lm=args.lm_weight if args.lm_weight is not None else default_lm_weight(args.binder_type),
        lambda_intra=args.lambda_intra,
        lambda_inter=args.lambda_inter,
        lambda_glob=args.lambda_glob,
        confidence_temperature_threshold=args.confidence_temperature_threshold,
        confidence_sampling_steps=args.confidence_steps,
        use_search_lm_context=not args.disable_search_lm_context,
    )
    ranking_config = RankingConfig(
        num_sampling_steps=args.ranking_steps,
        selection_score=args.selection_score,
        write_top_structures=args.write_top_structures,
    )
    pi_threshold = args.pi_threshold
    if pi_threshold is None:
        pi_threshold = default_pi_threshold(args.binder_type)

    search_pool = FoldModelPool(
        search_models,
        device=device,
        esmc_model_id=args.esmc_model,
        attn_implementation=args.attn_implementation,
        kernel_backend=kernel_backend,
        chunk_size=args.chunk_size,
        max_loaded=args.loaded_fold_models,
    )
    ranking_pool = FoldModelPool(
        ranking_models,
        device=device,
        esmc_model_id=args.esmc_model,
        attn_implementation=args.attn_implementation,
        kernel_backend=kernel_backend,
        chunk_size=args.chunk_size,
        max_loaded=args.loaded_fold_models,
    )
    lm_model, lm_tokenizer = load_esmc_masked_lm(args.esmc_model, device, args.attn_implementation)
    progress = ProgressBar(enabled=not args.no_progress)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    trajectories: list[TrajectoryResult] = []
    search_total_steps = args.num_designs * search_config.steps
    if search_total_steps > 0:
        progress.start("search", search_total_steps)
    for trajectory_index in range(args.num_designs):
        trajectory_seed = args.seed + trajectory_index * 1009
        random_generator = random.Random(trajectory_seed)
        torch.manual_seed(trajectory_seed)
        trajectories.append(
            trajectory_search(
                trajectory_index,
                target_sequence=target_sequence,
                binder_prompt=binder_prompt,
                search_config=search_config,
                search_pool=search_pool,
                lm_model=lm_model,
                lm_tokenizer=lm_tokenizer,
                seed=trajectory_seed,
                rng=random_generator,
                progress=progress,
                num_trajectories=args.num_designs,
            )
        )
    if search_total_steps > 0:
        progress.finish(detail=f"completed {len(trajectories)} trajectories")

    deduplicated = deduplicate_trajectories(trajectories)
    ranking_total = len(deduplicated) * len(ranking_pool.model_ids)
    if ranking_total > 0:
        progress.start("ranking", ranking_total)
    ranked_candidates = rank_candidates(
        deduplicated,
        target_sequence=target_sequence,
        ranking_pool=ranking_pool,
        ranking_config=ranking_config,
        binder_type=args.binder_type,
        binder_row_subset=binder_row_subset,
        pi_threshold=pi_threshold,
        output_dir=args.output_dir,
        progress=progress,
    )
    if ranking_total > 0:
        progress.finish(detail=f"scored {len(deduplicated)} unique candidates")
    write_outputs(
        args.output_dir,
        config=build_run_config(args, search_config, ranking_config),
        trajectories=trajectories,
        ranked_candidates=ranked_candidates,
        top_k=args.top_k,
    )

    print(f"Generated {len(trajectories)} trajectories and ranked {min(len(ranked_candidates), args.top_k)} binders.")
    for rank, candidate in enumerate(ranked_candidates[: args.top_k], start=1):
        print(
            f"[{rank}] seq={candidate.sequence} score={candidate.selection_score} "
            f"iptm={candidate.mean_iptm} proxy={candidate.mean_proxy} pI={candidate.pI}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())