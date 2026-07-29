import inspect
import random
from collections.abc import Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from esm.models.esmfold2.conformers import load_ccd
from esm.models.esmfold2.output import build_molecular_complex_from_features
from esm.models.esmfold2.prepare_input import ChainInfo, prepare_esmfold2_input
from esm.models.esmfold2.types import (
    MSA,
    Modification,
    ProteinInput,
    StructurePredictionInput,
)
from esm.utils.structure.molecular_complex import MolecularComplexResult


@contextmanager
def _seed_context(seed: int | None):
    if seed is None:
        yield
        return
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
        if cuda_state is not None:
            try:
                torch.cuda.set_rng_state_all(cuda_state)
            except Exception:
                # If the wrapped block raised a CUDA error, the context may be
                # unusable; restoring RNG here must not mask the original error.
                pass


@contextmanager
def _lm_dropout_context(model: Any, lm_dropout: float | None):
    """Temporarily set LM-embedding dropout for the wrapped forward, restoring on exit.

    Applies dropout to ``lm_z`` at inference (``training=True``, fresh mask per
    loop) so repeated folds give a diverse ensemble. Release models read
    ``config.lm_encoder.lm_dropout``, the experimental model a top-level
    ``config.lm_dropout`` — disambiguated by ``config.type``. ``None``/``0`` is a no-op.
    """
    if not lm_dropout:
        yield
        return

    cfg = model.config
    lm_encoder_cfg = getattr(cfg, "lm_encoder", None)
    if lm_encoder_cfg is not None and getattr(cfg, "type", None) != "experimental":
        saved = (lm_encoder_cfg.lm_dropout, lm_encoder_cfg.per_loop_lm_dropout)
        lm_encoder_cfg.lm_dropout = lm_dropout
        lm_encoder_cfg.per_loop_lm_dropout = True
        try:
            yield
        finally:
            lm_encoder_cfg.lm_dropout, lm_encoder_cfg.per_loop_lm_dropout = saved
    elif hasattr(cfg, "lm_dropout"):
        saved = (
            cfg.lm_dropout,
            getattr(cfg, "force_lm_dropout_during_inference", False),
        )
        cfg.lm_dropout = lm_dropout
        cfg.force_lm_dropout_during_inference = True
        try:
            yield
        finally:
            cfg.lm_dropout, cfg.force_lm_dropout_during_inference = saved
    else:
        raise ValueError(
            "lm_dropout was requested but this model's config exposes neither "
            "`lm_encoder.lm_dropout` nor a top-level `lm_dropout`."
        )


def clean_esmfold2_input(input: StructurePredictionInput) -> StructurePredictionInput:
    """Group identical protein sequences into the same ProteinInput with multiple ids.

    Example: Passing a tetramer like [ProteinInput(id=["0"], seq="AAA|AAA|BBB|BBB")]
    gets converted into [ProteinInput(id=["0_0", "0_1"], seq="AAA"),
                         ProteinInput(id=["0_2", "0_3"], seq="BBB")]

    Preserves the original order of unique sequences. Also converts "|" chainbreak
    tokens to ":" in the sequence.
    """
    cleaned_sequences: list = []
    chain_to_ids: dict[str, list[str]] = {}
    chain_to_modifications: dict[str, list] = {}
    chain_to_msa: dict[str, MSA | None] = {}

    for item in input.sequences:
        if isinstance(item, ProteinInput):
            sequence = ":".join(item.sequence.split("|"))
            if ":" not in sequence:
                cleaned_sequences.append(item)
                continue

            if ":" in sequence and input.covalent_bonds is not None:
                raise ValueError(
                    "Covalent bonds are not supported when using chainbreaks. "
                    "Chains must be separated into multiple ProteinInput objects."
                )

            base_id = item.id[0] if isinstance(item.id, list) else item.id
            chain_to_ids = {}
            chain_to_modifications = {}
            chain_to_msa = {}
            chains = sequence.split(":")

            chain_start_positions = []
            pos = 0
            for chain in chains:
                chain_start_positions.append(pos)
                pos += len(chain) + 1

            if item.modifications is not None:
                for chain_idx, chain in enumerate(chains):
                    chain_start = chain_start_positions[chain_idx]
                    chain_end = chain_start + len(chain)
                    chain_modifications = []
                    for mod in item.modifications:
                        if chain_start <= mod.position < chain_end:
                            adjusted_mod = Modification(
                                position=mod.position - chain_start, ccd=mod.ccd
                            )
                            chain_modifications.append(adjusted_mod)
                    if chain not in chain_to_modifications:
                        chain_to_modifications[chain] = chain_modifications
                    else:
                        chain_to_modifications[chain].extend(chain_modifications)

            if item.msa is not None:
                for chain_idx, chain in enumerate(chains):
                    if chain not in chain_to_msa:
                        chain_start = chain_start_positions[chain_idx]
                        chain_end = chain_start + len(chain)
                        chain_msa = item.msa.select_positions(  # type: ignore
                            np.arange(chain_start, chain_end)
                        )
                        chain_to_msa[chain] = chain_msa

            for i, chain in enumerate(chains):
                chain_id = base_id + "_" + str(i)
                if chain in chain_to_ids:
                    chain_to_ids[chain].append(chain_id)
                else:
                    chain_to_ids[chain] = [chain_id]
                    cleaned_sequences.append((item, chain))
        else:
            cleaned_sequences.append(item)

    for i in range(len(cleaned_sequences)):
        if isinstance(cleaned_sequences[i], tuple):
            item, chain = cleaned_sequences[i]
            chain_ids = chain_to_ids[chain]
            chain_modifications = (
                chain_to_modifications.get(chain) if item.modifications else None
            )
            chain_msa = chain_to_msa.get(chain) if item.msa else None
            cleaned_sequences[i] = ProteinInput(
                id=chain_ids,
                sequence=chain,
                msa=chain_msa,
                modifications=chain_modifications,
            )

    return StructurePredictionInput(
        sequences=cleaned_sequences,
        distogram_conditioning=input.distogram_conditioning,
        covalent_bonds=input.covalent_bonds,
    )


class ESMFold2InputBuilder:
    def __init__(self, ccd_cache: Path | None = None):
        load_ccd(ccd_cache)

    def prepare_input(
        self,
        input: StructurePredictionInput,
        seed: int | None = None,
        device: torch.device | str | None = None,
    ) -> tuple[dict, list[ChainInfo]]:
        """Prepare raw input for the folding model.

        Converts user-provided StructurePredictionInput into batched tensors
        ready for model inference.

        Parameters
        ----------
        input : StructurePredictionInput
            Input specification (sequences, structures, constraints, etc.).
        seed : int, optional
            Random seed for reproducibility.
        device : torch.device or str, optional
            Target device for the returned tensors. Defaults to CPU; pass
            ``model.device`` to skip a separate ``.to(...)`` step. ``fold()``
            forwards ``model.device`` automatically.

        Returns
        -------
        tuple[dict, list[ChainInfo]]
            Batched input tensors and chain metadata for output processing.
        """
        structure_prediction_input = clean_esmfold2_input(input)
        with _seed_context(seed) if seed is not None else nullcontext():
            features, chain_infos = prepare_esmfold2_input(
                structure_prediction_input, seed=seed
            )
            features = {
                k: (v[None].to(device) if device is not None else v[None])
                if isinstance(v, torch.Tensor)
                else v
                for k, v in features.items()
            }

        return features, chain_infos

    def __call__(
        self,
        input: StructurePredictionInput,
        seed: int | None = None,
        device: torch.device | str | None = None,
    ) -> tuple[dict, list[ChainInfo]]:
        return self.prepare_input(input, seed=seed, device=device)

    def decode(
        self,
        output: dict[str, torch.Tensor],
        features: dict[str, torch.Tensor],
        chain_infos: list[ChainInfo],
        *,
        num_diffusion_samples: int = 1,
        complex_id: str = "pred",
    ) -> MolecularComplexResult | list[MolecularComplexResult]:
        """Convert raw model outputs into one MolecularComplexResult per sample.

        Parameters
        ----------
        output : dict[str, Tensor]
            Output dict returned by ESMFold2Model.forward.
        features : dict[str, Tensor]
            Feature dict from :meth:`prepare_input` (batched, on the model device).
        chain_infos : list[ChainInfo]
            Chain metadata returned alongside `features`.
        num_diffusion_samples : int
            Number of diffusion samples present in the output (Bm = B * num_diffusion_samples).
        complex_id : str
            Identifier assigned to each MolecularComplex.

        Returns
        -------
        MolecularComplexResult or list[MolecularComplexResult]
            A single result when num_diffusion_samples == 1, otherwise a list of length Bm.
        """
        atom_mask = features["atom_attention_mask"][0]
        ref_element = features["ref_element"][0]
        ref_atom_name_chars = features["ref_atom_name_chars"][0]

        sample_coords = output["sample_atom_coords"]
        plddts = output["plddt"]
        Bm = sample_coords.shape[0]

        ptm_t = output.get("ptm")
        iptm_t = output.get("iptm")
        pae_t = output.get("pae")
        distogram_t = output.get("distogram_logits")
        pair_chains_t = output.get("pair_chains_iptm")
        residue_index_t = output.get("residue_index")
        entity_id_t = output.get("entity_id")

        results: list[MolecularComplexResult] = []
        for i in range(Bm):
            mc = build_molecular_complex_from_features(
                coords=sample_coords[i],
                plddt=plddts[i],
                atom_mask=atom_mask,
                ref_element=ref_element,
                ref_atom_name_chars=ref_atom_name_chars,
                chain_infos=chain_infos,
                complex_id=complex_id,
            )
            results.append(
                MolecularComplexResult(
                    complex=mc,
                    plddt=plddts[i].detach().cpu(),
                    ptm=float(ptm_t[i].item()) if ptm_t is not None else None,
                    iptm=float(iptm_t[i].item()) if iptm_t is not None else None,
                    pae=pae_t[i].detach().cpu() if pae_t is not None else None,
                    distogram=(
                        distogram_t[0].detach().cpu()
                        if distogram_t is not None
                        else None
                    ),
                    pair_chains_iptm=(
                        pair_chains_t[i].detach().cpu()
                        if pair_chains_t is not None
                        else None
                    ),
                    residue_index=(
                        residue_index_t[0].detach().cpu()
                        if residue_index_t is not None
                        else None
                    ),
                    entity_id=(
                        entity_id_t[0].detach().cpu()
                        if entity_id_t is not None
                        else None
                    ),
                )
            )

        if num_diffusion_samples == 1 and len(results) == 1:
            return results[0]
        return results

    def fold(
        self,
        model: Any,
        input: StructurePredictionInput,
        *,
        num_loops: int = 20,
        num_sampling_steps: int = 200,
        num_diffusion_samples: int = 1,
        seed: int | None = None,
        noise_scale: float | None = None,
        step_scale: float | None = None,
        max_inference_sigma: float | None = None,
        lm_mask_pct: float | None = None,
        early_exit: bool = False,
        lm_dropout: float | None = 0.3,
        msa_max_depth: int | None = 1024,
        msa_column_mask_rate: float = 0.1,
        complex_id: str = "pred",
    ) -> MolecularComplexResult | list[MolecularComplexResult]:
        """Fold a structure end-to-end: encode → model → decode.

        Parameters
        ----------
        model : ESMFold2Model
            The folding model. Must already be on the target device and in eval mode.
        input : StructurePredictionInput
            User-facing input specification.
        num_loops, num_sampling_steps, num_diffusion_samples : int
            Inference knobs forwarded to the model.
        seed : int, optional
            Seeds both input prep (SMILES conformer generation) and diffusion sampling.
        noise_scale, step_scale, max_inference_sigma, early_exit
            Optional sampler overrides forwarded to the model when not None.
        lm_mask_pct : float, optional
            Fraction of sequence residues randomly masked before the PLM backbone.
            Overrides the checkpoint config when not None.
        lm_dropout : float, optional
            LM-embedding dropout for this fold (fresh mask per loop → diverse
            ensemble on repeated folds). Defaults to ``0.3`` (paper folding-eval
            value); ``0``/``None`` disables.
        msa_max_depth : int, optional
            Maximum number of MSA rows kept per loop (row subsampling
            is drawn fresh per loop). When ``None``, MSA row subsampling is
            disabled and the full MSA depth is used. Only affects inputs that
            carry an MSA.
        msa_column_mask_rate : float
            Fraction of MSA columns masked once before the loop
            (shared across loops). Only affects inputs that carry an MSA.
        complex_id : str
            Identifier assigned to the predicted MolecularComplex(es).

        Returns
        -------
        MolecularComplexResult or list[MolecularComplexResult]
            A single result when num_diffusion_samples == 1, otherwise a list.
        """
        features, chain_infos = self.prepare_input(
            input, seed=seed, device=model.device
        )

        output = self._run_model_forward(
            model,
            features,
            num_loops=num_loops,
            num_sampling_steps=num_sampling_steps,
            num_diffusion_samples=num_diffusion_samples,
            seed=seed,
            noise_scale=noise_scale,
            step_scale=step_scale,
            max_inference_sigma=max_inference_sigma,
            lm_mask_pct=lm_mask_pct,
            early_exit=early_exit,
            lm_dropout=lm_dropout,
            msa_max_depth=msa_max_depth,
            msa_column_mask_rate=msa_column_mask_rate,
        )

        return self.decode(
            output,
            features,
            chain_infos,
            num_diffusion_samples=num_diffusion_samples,
            complex_id=complex_id,
        )

    def _run_model_forward(
        self,
        model: Any,
        features: dict[str, Any],
        *,
        num_loops: int,
        num_sampling_steps: int,
        num_diffusion_samples: int,
        seed: int | None,
        noise_scale: float | None,
        step_scale: float | None,
        max_inference_sigma: float | None,
        lm_mask_pct: float | None,
        early_exit: bool,
        lm_dropout: float | None,
        msa_max_depth: int | None,
        msa_column_mask_rate: float,
        lm_hidden_states: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run one ESMFold2 forward pass under the standard inference contexts.

        Shared by :meth:`fold` (which lets the model run its ESMC backbone
        inline) and :meth:`fold_batch` (which passes a precomputed
        ``lm_hidden_states`` so the ESMC backbone is skipped). The RNG-seeding
        and LM-dropout contexts, and every forward kwarg, are identical across
        both callers — the only difference is whether ``lm_hidden_states`` is
        supplied. Because the model recomputes ``lm_z`` deterministically from
        ``lm_hidden_states`` and the ESMC backbone consumes no RNG, supplying a
        cached ``lm_hidden_states`` yields bit-identical outputs to the inline
        path.
        """
        sampler_kwargs: dict[str, Any] = {}
        if noise_scale is not None:
            sampler_kwargs["noise_scale"] = noise_scale
        if step_scale is not None:
            sampler_kwargs["step_scale"] = step_scale
        if max_inference_sigma is not None:
            sampler_kwargs["max_inference_sigma"] = max_inference_sigma
        if lm_mask_pct is not None:
            sampler_kwargs["lm_mask_pct"] = lm_mask_pct

        # Only pass lm_hidden_states when precomputed; omitting it entirely (vs.
        # passing None) keeps fold()'s forward call byte-for-byte unchanged.
        lm_kwargs: dict[str, Any] = {}
        if lm_hidden_states is not None:
            lm_kwargs["lm_hidden_states"] = lm_hidden_states

        with torch.no_grad():
            with _seed_context(seed) if seed is not None else nullcontext():
                with _lm_dropout_context(model, lm_dropout):
                    output = model(
                        **features,
                        num_loops=num_loops,
                        num_sampling_steps=num_sampling_steps,
                        num_diffusion_samples=num_diffusion_samples,
                        early_exit=early_exit,
                        msa_max_depth=msa_max_depth,
                        msa_column_mask_rate=msa_column_mask_rate,
                        # A null depth means "use the full MSA" => no subsampling.
                        msa_subsample_at_inference=msa_max_depth is not None,
                        **lm_kwargs,
                        **sampler_kwargs,
                    )
        return output

    def fold_batch(
        self,
        model: Any,
        inputs: StructurePredictionInput | Sequence[StructurePredictionInput],
        *,
        num_loops: int = 20,
        num_sampling_steps: int = 200,
        num_diffusion_samples: int = 1,
        seed: int | Sequence[int | None] | None = None,
        noise_scale: float | None = None,
        step_scale: float | None = None,
        max_inference_sigma: float | None = None,
        lm_mask_pct: float | None = None,
        early_exit: bool = False,
        lm_dropout: float | None = 0.3,
        msa_max_depth: int | None = 1024,
        msa_column_mask_rate: float = 0.1,
        complex_id: str | Sequence[str] = "pred",
        offload_esmc: bool = True,
        restore_esmc: bool = False,
        esmc_precision: str = "bf16",
    ) -> list[MolecularComplexResult | list[MolecularComplexResult]]:
        """Fold a batch of inputs while keeping ESMC off the compute device during folding.

        Memory-optimized equivalent of calling :meth:`fold` once per input.
        The peak GPU footprint is reduced by never holding the ESMC PLM
        backbone resident while the (memory-heavy) folding trunk and diffusion
        head run. The flow is:

        1. **Encode** — load ESMC onto the compute device (if not already
           there), and for every input run only the ESMC backbone to produce
           its ``lm_hidden_states``. Each encoding is moved to CPU RAM
           immediately; nothing but the current item's ESMC activations lives
           on the GPU during this phase.
        2. **Offload** — move ESMC off the compute device (to CPU RAM), freeing
           its parameters from the GPU for the duration of folding.
        3. **Fold** — for each input, move its features + cached
           ``lm_hidden_states`` back to the compute device and run the folding
           forward (ESMC skipped, since ``lm_hidden_states`` is supplied), then
           decode.

        By default ESMC is left on CPU when the batch finishes (no wasted
        re-upload, and the GPU stays clear for the next batch). If ESMC was not
        loaded on entry, it is unloaded entirely. See ``restore_esmc`` to move
        it back onto the compute device instead.

        The result is **bit-for-bit identical** to :meth:`fold` per input
        (given the same ``seed``): ESMC is a deterministic feature extractor
        that consumes no RNG, ``prepare_input`` touches only RDKit's RNG (not
        the torch/NumPy/Python global RNG), and the folding forward is run
        under the exact same seeding and LM-dropout contexts as :meth:`fold`.
        Encodings are cached in CPU RAM (not on disk) for speed.

        Parameters
        ----------
        model : ESMFold2Model
            The folding model, already on the target device and in eval mode.
            Must expose the ESMC hooks used here (``_esmc``,
            ``_compute_lm_hidden_states``, ``load_esmc``, ``config.esmc_id``)
            and accept a ``lm_hidden_states`` forward kwarg.
        inputs : StructurePredictionInput or sequence of them
            One or more input specifications to fold.
        seed : int, sequence, or None
            Per-fold seed. A scalar (or None) applies to every input; a
            sequence assigns one seed per input and must match ``len(inputs)``.
        complex_id : str or sequence of str
            Identifier(s) assigned to the predicted complex(es). A scalar
            applies to every input; a sequence must match ``len(inputs)``.
        offload_esmc : bool
            When True (default), ESMC is moved off the compute device during
            the folding phase. Set False to keep it resident (no memory
            benefit; useful for A/B parity checks against :meth:`fold`).
        restore_esmc : bool
            When True, ESMC is moved back onto its original device after the
            batch (restoring the model to its entry state). Default False:
            ESMC that was already loaded on entry is left on CPU, which keeps
            the GPU clear for the next batch and avoids a wasted re-upload —
            note a subsequent inline :meth:`fold` on the same model would then
            need ESMC moved back (``model._esmc.to(model.device)``), or just
            call :meth:`fold_batch` again. ESMC that this call *loaded* is
            always unloaded regardless of this flag.
        esmc_precision : str
            Precision used if ESMC has to be loaded here (``"bf16"``,
            ``"fp32"``, or ``"fp8"``). Ignored when the model already has ESMC
            loaded, or when the model's ``load_esmc`` takes no precision arg.

        All other parameters match :meth:`fold` and are applied to every input.

        Returns
        -------
        list
            One entry per input, aligned with ``inputs``. Each entry matches
            what :meth:`fold` returns for that input (a single
            ``MolecularComplexResult`` when ``num_diffusion_samples == 1``,
            otherwise a list).
        """
        if isinstance(inputs, StructurePredictionInput):
            inputs = [inputs]
        else:
            inputs = list(inputs)
        n = len(inputs)
        if n == 0:
            return []

        seeds = self._broadcast_arg(seed, n, "seed")
        complex_ids = self._broadcast_arg(complex_id, n, "complex_id")

        if not hasattr(model, "_compute_lm_hidden_states") or not hasattr(
            model, "load_esmc"
        ):
            raise TypeError(
                "fold_batch requires an ESMFold2 model exposing the ESMC hooks "
                "(`_compute_lm_hidden_states`, `load_esmc`). Use fold() per input "
                "for models without a detachable ESMC backbone."
            )

        self._reject_unappliable_lm_mask_pct(model, lm_mask_pct)

        # Capture the compute device once, before any offload can change what
        # `model.device` reports.
        compute_device = model.device

        # --- Phase 1: load ESMC + encode every input, caching to CPU RAM ---
        esmc_was_present = getattr(model, "_esmc", None) is not None
        if not esmc_was_present:
            # `precision` is only accepted by the standard model's load_esmc;
            # the experimental variant's signature omits it.
            load_kwargs: dict[str, Any] = {}
            if "precision" in inspect.signature(model.load_esmc).parameters:
                load_kwargs["precision"] = esmc_precision
            model.load_esmc(model.config.esmc_id, **load_kwargs)
        esmc_origin_device = next(model._esmc.parameters()).device
        if esmc_origin_device != compute_device:
            model._esmc.to(compute_device)

        encodings: list[_CachedEncoding] = []
        try:
            for i in range(n):
                features, chain_infos = self.prepare_input(
                    inputs[i], seed=seeds[i], device=None
                )
                lm_hidden_states = self._encode_lm_hidden_states(
                    model, features, compute_device
                )
                encodings.append(
                    _CachedEncoding(
                        features=features,
                        chain_infos=chain_infos,
                        lm_hidden_states=lm_hidden_states,
                        seed=seeds[i],
                        complex_id=complex_ids[i],
                    )
                )

            # --- Phase 2: offload ESMC off the compute device for folding ---
            # After encoding, ESMC sits on the compute device; move it to CPU to
            # free that device (e.g. the GPU) for folding. Keyed on the compute
            # device (not ESMC's origin) so it also fires when a user keeps ESMC
            # on CPU but folds on GPU. No-op when folding on CPU.
            if offload_esmc and torch.device(compute_device).type != "cpu":
                model._esmc.to("cpu")
                self._empty_cache(compute_device)

            # --- Phase 3: fold each input from its cached encoding ---
            results: list[MolecularComplexResult | list[MolecularComplexResult]] = []
            for enc in encodings:
                features = self._features_to_device(enc.features, compute_device)
                lm_hidden_states = (
                    enc.lm_hidden_states.to(compute_device)
                    if enc.lm_hidden_states is not None
                    else None
                )
                output = self._run_model_forward(
                    model,
                    features,
                    num_loops=num_loops,
                    num_sampling_steps=num_sampling_steps,
                    num_diffusion_samples=num_diffusion_samples,
                    seed=enc.seed,
                    noise_scale=noise_scale,
                    step_scale=step_scale,
                    max_inference_sigma=max_inference_sigma,
                    lm_mask_pct=lm_mask_pct,
                    early_exit=early_exit,
                    lm_dropout=lm_dropout,
                    msa_max_depth=msa_max_depth,
                    msa_column_mask_rate=msa_column_mask_rate,
                    lm_hidden_states=lm_hidden_states,
                )
                results.append(
                    self.decode(
                        output,
                        features,
                        enc.chain_infos,
                        num_diffusion_samples=num_diffusion_samples,
                        complex_id=enc.complex_id,
                    )
                )
                del features, lm_hidden_states, output
                self._empty_cache(compute_device)
        finally:
            # ESMC that we loaded here is always unloaded (leaves the model as
            # found). ESMC that was present on entry is left offloaded by
            # default — no wasted re-upload, GPU stays clear for the next batch
            # — unless restore_esmc asks to move it back. Runs even on error.
            if getattr(model, "_esmc", None) is not None:
                if not esmc_was_present:
                    model._esmc = None
                    self._empty_cache(compute_device)
                elif restore_esmc:
                    if next(model._esmc.parameters()).device != esmc_origin_device:
                        model._esmc.to(esmc_origin_device)

        return results

    def fold_batch_staged(
        self,
        model: Any,
        inputs: StructurePredictionInput | Sequence[StructurePredictionInput],
        *,
        device: torch.device | str,
        esmc_id: str | None = None,
        esmc_precision: str = "bf16",
        num_loops: int = 20,
        num_sampling_steps: int = 200,
        num_diffusion_samples: int = 1,
        seed: int | Sequence[int | None] | None = None,
        noise_scale: float | None = None,
        step_scale: float | None = None,
        max_inference_sigma: float | None = None,
        lm_mask_pct: float | None = None,
        early_exit: bool = False,
        lm_dropout: float | None = 0.3,
        msa_max_depth: int | None = 1024,
        msa_column_mask_rate: float = 0.1,
        complex_id: str | Sequence[str] = "pred",
    ) -> list[MolecularComplexResult | list[MolecularComplexResult]]:
        """Fold a batch with staged loading so ESMC and the folding stack are never co-resident on ``device``.

        :meth:`fold_batch` offloads a *resident* ESMC only during the folding
        phase, so the two parameter sets still coexist on ``device`` while the
        model loads and during encoding. This method instead:

        1. moves the folding stack to CPU and loads **only** ESMC onto
           ``device``,
        2. encodes every input to CPU RAM (folding stack stays off ``device``),
        3. **frees** ESMC from ``device``, then
        4. moves the folding stack onto ``device`` and folds each input from its
           cached embedding.

        Peak footprint is therefore
        ``max(ESMC + encode activations, folding stack + fold activations)`` —
        the two never overlap, eliminating the load/encode co-residency spike.

        Precondition: ``model`` is a folding model loaded **without** its ESMC
        backbone (ideally still on CPU), e.g.::

            model = ESMFold2Model.from_pretrained(repo, load_esmc=False)
            results = builder.fold_batch_staged(model, inputs, device="cuda")

        Any ESMC already attached is dropped and reloaded here, so the two
        parameter sets never coexist on ``device``. On return the folding stack
        is on ``device`` and ESMC is unloaded.

        The result is **bit-for-bit identical** to :meth:`fold` / :meth:`fold_batch`
        for the same ``seed``: the encoding calls the model's own
        ``_compute_lm_hidden_states`` (a deterministic, RNG-free feature
        extractor), and the folding forward runs under the exact same seeding and
        LM-dropout contexts, consuming the cached ``lm_hidden_states``.

        Parameters mirror :meth:`fold_batch`; ``device`` is required (the model
        typically starts on CPU, so it cannot be inferred), ``esmc_id`` defaults
        to ``model.config.esmc_id``.
        """
        if isinstance(inputs, StructurePredictionInput):
            inputs = [inputs]
        else:
            inputs = list(inputs)
        n = len(inputs)
        if n == 0:
            return []

        seeds = self._broadcast_arg(seed, n, "seed")
        complex_ids = self._broadcast_arg(complex_id, n, "complex_id")

        if not hasattr(model, "_compute_lm_hidden_states") or not hasattr(
            model, "load_esmc"
        ):
            raise TypeError(
                "fold_batch_staged requires an ESMFold2 model exposing the ESMC "
                "hooks (`_compute_lm_hidden_states`, `load_esmc`)."
            )

        compute_device = torch.device(device)
        esmc_id = esmc_id or model.config.esmc_id

        load_supports_precision = (
            "precision" in inspect.signature(model.load_esmc).parameters
        )
        # fp8 is unsupported here: load_esmc runs the TransformerEngine fp8
        # quantization on model.device, which staging forces to CPU while the
        # folding stack is parked there — TE requires CUDA. Use fold_batch (which
        # loads ESMC on-GPU) for fp8, or bf16/fp32 here.
        if load_supports_precision and esmc_precision == "fp8":
            raise ValueError(
                "fold_batch_staged does not support esmc_precision='fp8' (fp8 "
                "quantization must run on-GPU, but staging loads ESMC while the "
                "folding stack occupies CPU). Use fold_batch for fp8, or "
                "'bf16'/'fp32' here."
            )

        self._reject_unappliable_lm_mask_pct(model, lm_mask_pct)

        # Clean staged start: drop any attached ESMC and force the folding stack
        # to CPU, so nothing but ESMC will sit on `device` during encoding.
        model._esmc = None
        model.to("cpu")
        self._empty_cache(compute_device)

        encodings: list[_CachedEncoding] = []
        try:
            # --- Phase 1: load ESMC (only) onto the compute device + encode. ---
            # Inside the try so the finally frees ESMC even if load/move raises.
            load_kwargs: dict[str, Any] = {}
            if load_supports_precision:
                load_kwargs["precision"] = esmc_precision
            model.load_esmc(esmc_id, **load_kwargs)  # lands on model.device (CPU)
            model._esmc.to(compute_device)  # ESMC -> device; folding stack stays CPU

            for i in range(n):
                features, chain_infos = self.prepare_input(
                    inputs[i], seed=seeds[i], device=None
                )
                lm_hidden_states = self._encode_lm_hidden_states(
                    model, features, compute_device
                )
                encodings.append(
                    _CachedEncoding(
                        features=features,
                        chain_infos=chain_infos,
                        lm_hidden_states=lm_hidden_states,
                        seed=seeds[i],
                        complex_id=complex_ids[i],
                    )
                )
        finally:
            # Never strand ESMC on the device (even if encoding raised).
            model._esmc = None
            self._empty_cache(compute_device)

        # --- Phase 2: now that ESMC is gone, bring the folding stack on. ---
        model.to(compute_device)

        # --- Phase 3: fold each input from its cached embedding. ---
        results: list[MolecularComplexResult | list[MolecularComplexResult]] = []
        for enc in encodings:
            features = self._features_to_device(enc.features, compute_device)
            lm_hidden_states = (
                enc.lm_hidden_states.to(compute_device)
                if enc.lm_hidden_states is not None
                else None
            )
            output = self._run_model_forward(
                model,
                features,
                num_loops=num_loops,
                num_sampling_steps=num_sampling_steps,
                num_diffusion_samples=num_diffusion_samples,
                seed=enc.seed,
                noise_scale=noise_scale,
                step_scale=step_scale,
                max_inference_sigma=max_inference_sigma,
                lm_mask_pct=lm_mask_pct,
                early_exit=early_exit,
                lm_dropout=lm_dropout,
                msa_max_depth=msa_max_depth,
                msa_column_mask_rate=msa_column_mask_rate,
                lm_hidden_states=lm_hidden_states,
            )
            results.append(
                self.decode(
                    output,
                    features,
                    enc.chain_infos,
                    num_diffusion_samples=num_diffusion_samples,
                    complex_id=enc.complex_id,
                )
            )
            del features, lm_hidden_states, output
            self._empty_cache(compute_device)

        return results

    @staticmethod
    def _reject_unappliable_lm_mask_pct(model: Any, lm_mask_pct: float | None) -> None:
        """Fail loudly when ``lm_mask_pct`` could not be honoured on this path.

        LM token masking is applied inside the model's *inline* ESMC call. The
        precomputed-embedding paths supply ``lm_hidden_states``, which makes
        ``forward`` skip that call entirely — so a nonzero ``lm_mask_pct`` would
        be silently dropped, giving different LM inputs (and one fewer
        L-dependent RNG draw) than the inline path. That is exactly the kind of
        quiet divergence these paths exist to avoid, so raise instead.

        No-op for the common cases: falsy ``lm_mask_pct``, or a model whose
        ``_compute_lm_hidden_states`` accepts it (then encoding can apply it).
        """
        if not lm_mask_pct:
            return
        try:
            params = inspect.signature(model._compute_lm_hidden_states).parameters
        except (AttributeError, TypeError, ValueError):
            params = {}
        if "lm_mask_pct" in params:
            return
        raise ValueError(
            f"lm_mask_pct={lm_mask_pct!r} cannot be applied when ESMC embeddings "
            "are precomputed: this model's _compute_lm_hidden_states() does not "
            "accept it, and supplying lm_hidden_states makes forward() skip the "
            "inline ESMC call where masking happens. Passing it would silently "
            "diverge from the inline path. Use fold() (or offload_esmc=False) if "
            "you need LM token masking, or drop lm_mask_pct."
        )

    def _encode_lm_hidden_states(
        self,
        model: Any,
        features: dict[str, Any],
        device: torch.device | str,
    ) -> torch.Tensor | None:
        """Run only the ESMC backbone for one input, returning its LM hidden states on CPU.

        Delegates to the model's own ``_compute_lm_hidden_states`` so the
        result is identical to what the inline forward would compute. Returns
        ``None`` when the input carries no ``input_ids`` (mirrors the model's
        forward, which then produces no LM signal).
        """
        input_ids = features.get("input_ids")
        if input_ids is None:
            return None

        def _dev(key: str) -> torch.Tensor:
            return features[key].to(device)

        with torch.no_grad():
            lm_hidden_states = model._compute_lm_hidden_states(
                input_ids.to(device),
                _dev("asym_id"),
                _dev("residue_index"),
                _dev("mol_type"),
                _dev("token_attention_mask"),
            )
        return lm_hidden_states.detach().to("cpu")

    @staticmethod
    def _features_to_device(
        features: dict[str, Any], device: torch.device | str
    ) -> dict[str, Any]:
        return {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in features.items()
        }

    @staticmethod
    def _empty_cache(device: torch.device | str) -> None:
        dev = torch.device(device)
        if dev.type == "cuda" and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                # A prior CUDA error can leave the context unusable; a
                # best-effort cache flush must not mask the original exception
                # (e.g. surface a TransformerEngine/cuBLAS failure, not this).
                pass

    @staticmethod
    def _broadcast_arg(value: Any, n: int, name: str) -> list[Any]:
        # Strings are scalars here (not sequences to spread over the batch).
        if isinstance(value, Sequence) and not isinstance(value, str):
            value = list(value)
            if len(value) != n:
                raise ValueError(
                    f"{name} has length {len(value)} but there are {n} inputs; "
                    "pass a single value or one per input."
                )
            return value
        return [value] * n


@dataclass
class _CachedEncoding:
    """One input's prepared features + cached ESMC hidden states, held in CPU RAM."""

    features: dict[str, Any]
    chain_infos: list[ChainInfo]
    lm_hidden_states: torch.Tensor | None
    seed: int | None
    complex_id: str = "pred"


__all__ = ["ESMFold2InputBuilder", "clean_esmfold2_input"]
