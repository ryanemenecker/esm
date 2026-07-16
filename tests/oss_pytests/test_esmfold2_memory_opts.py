"""Regression tests for the output-preserving ESMFold2 memory optimizations.

Two layers:

1. Module-level bit-exact guards (always run when the model package imports; no
   weights or GPU needed). Each edited module is instantiated with random
   weights and compared, over several random inputs, against a reference
   reimplementation of the *original* (pre-optimization) formula. These pin the
   optimized modules to their intended arithmetic — if a future change alters
   any module's output, ``torch.equal`` fails here.

2. A real-model ``fold`` vs ``fold_batch`` parity + determinism test. This needs
   the ESMFold2 weights (and ideally a GPU), so it is opt-in: set
   ``ESMFOLD2_REGRESSION=1`` (and optionally ``ESMFOLD2_MODEL=<repo-or-path>``).
   It validates that the dynamic ESMC-offload batched path reproduces the inline
   fold bit-for-bit, and that a fixed seed is deterministic.
"""

import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("transformers")

# Import the optimized modules directly from the vendored esmfold2_remote_code
# package. We add the esmfold2 directory to sys.path and import it as a
# top-level package so we do NOT trigger esm.models.esmfold2.__init__, which
# eagerly pulls in heavy deps (Bio/rdkit/zstd) that these pure-torch module
# tests don't need. Only torch + transformers are required here.
_ESMFOLD2_DIR = Path(__file__).resolve().parents[2] / "esm" / "models" / "esmfold2"
if str(_ESMFOLD2_DIR) not in sys.path:
    sys.path.insert(0, str(_ESMFOLD2_DIR))

try:
    from esmfold2_remote_code.modeling_esmfold2_common import (
        AttentionPairBias,
        ConditionedTransitionBlock,
        ResIdxAsymIdSymIdEntityIdEncoding,
        SwiGLU,
        SwiGLUFFN,
        TriangleMultiplicativeBlock,
    )

    _HAVE_MODULES = True
    _IMPORT_ERR = ""
except Exception as e:  # pragma: no cover - environment dependent
    _HAVE_MODULES = False
    _IMPORT_ERR = repr(e)

module_test = pytest.mark.skipif(
    not _HAVE_MODULES,
    reason=f"esmfold2_remote_code not importable: {_IMPORT_ERR}",
)


# ===========================================================================
# 1. Module-level bit-exact guards
# ===========================================================================


@module_test
def test_residx_scatter_matches_onehot():
    """#5: one-hot->scatter fp32 buffer == cat of int64 one-hots (.float())."""

    def reference(mod, residue_index, asym_id, sym_id, entity_id, token_index):
        R, C = mod.n_relative_residx_bins, mod.n_relative_chain_bins
        same_chain = asym_id.unsqueeze(2) == asym_id.unsqueeze(1)
        same_res = residue_index.unsqueeze(2) == residue_index.unsqueeze(1)
        same_ent = entity_id.unsqueeze(2) == entity_id.unsqueeze(1)
        dr = torch.clip(residue_index.unsqueeze(2) - residue_index.unsqueeze(1) + R, 0, 2 * R)
        dr = torch.where(same_chain, dr, 2 * R + 1)
        dt = torch.clip(token_index.unsqueeze(2) - token_index.unsqueeze(1) + R, 0, 2 * R)
        dt = torch.where(same_chain & same_res, dt, 2 * R + 1)
        dc = torch.clip(sym_id.unsqueeze(2) - sym_id.unsqueeze(1) + C, 0, 2 * C)
        dc = torch.where(same_chain, 2 * C + 1, dc)
        feats = torch.cat(
            [
                F.one_hot(dr, 2 * R + 2).float(),
                F.one_hot(dt, 2 * R + 2).float(),
                same_ent.float().unsqueeze(-1),
                F.one_hot(dc, 2 * C + 2).float(),
            ],
            dim=-1,
        )
        return mod.embed(feats)

    for (R, C, d) in [(32, 2, 16), (3, 2, 8), (5, 1, 12)]:
        torch.manual_seed(R * 100 + C * 10 + d)
        mod = ResIdxAsymIdSymIdEntityIdEncoding(R, C, d).eval()
        for t in range(5):
            B, L, nch = 1 + t % 2, [1, 2, 5, 17][t % 4], 1 + t % 3
            args = (
                torch.randint(-40, 40, (B, L)),  # residue_index
                torch.randint(0, nch, (B, L)),  # asym_id
                torch.randint(0, nch, (B, L)),  # sym_id
                torch.randint(0, nch, (B, L)),  # entity_id
                torch.randint(-40, 40, (B, L)),  # token_index
            )
            with torch.no_grad():
                assert torch.equal(mod(*args), reference(mod, *args)), (R, C, d, B, L)


@module_test
def test_trimul_free_is_bit_identical():
    """#3: freeing dead intermediates does not change TriangleMultiplicativeBlock output."""

    def reference(mod, pair, vis=None):
        if vis is None:
            vis = pair.new_ones(pair.shape[:-1])
        ng = mod.norm_start(pair)
        bundled = mod.proj_bundle(ng)
        sig, gate = bundled.split(2 * mod.latent_channels, dim=-1)
        routed = (sig * torch.sigmoid(gate)) * vis.unsqueeze(-1)
        left, right = routed.float().chunk(2, dim=-1)
        if mod._chunk_size is not None:
            contracted = mod._triangular_contract_chunked(left, right, mod._chunk_size)
        else:
            contracted = mod._triangular_contract(left, right)
        return mod.proj_emit(mod.norm_mix(contracted)) * torch.sigmoid(mod.proj_gate(ng))

    for flow in ("outgoing", "incoming"):
        torch.manual_seed(hash(flow) % 10007)
        mod = TriangleMultiplicativeBlock(input_channels=8, latent_channels=8, flow=flow).eval()
        for chunk in (None, 3):
            mod.set_chunk_size(chunk)
            for t in range(3):
                B, L = 1 + t % 2, [2, 5, 9][t % 3]
                pair = torch.randn(B, L, L, 8)
                vis = (torch.rand(B, L, L) > 0.3).float() if t % 2 else None
                with torch.no_grad():
                    assert torch.equal(mod(pair, vis), reference(mod, pair, vis))


@module_test
def test_swiglu_inplace_gating():
    """minor#8: in-place SwiGLU / SwiGLUFFN gating is bit-identical."""
    torch.manual_seed(1)
    m1 = SwiGLU(in_features=12, hidden_features=16, out_features=12).eval()
    m2 = SwiGLUFFN(d_model=12, expansion_ratio=2).eval()
    for t in range(4):
        x = torch.randn(2, [1, 5, 9][t % 3], 12)
        with torch.no_grad():
            a1, a2 = m1.w12(x).split(m1.hidden_features, dim=-1)
            assert torch.equal(m1(x), m1.w3(F.silu(a1) * a2))
            xx = x.to(m2.w_up.weight.dtype)
            b1, b2 = m2.w_up(xx).chunk(2, dim=-1)
            assert torch.equal(m2(x), m2.w_down(F.silu(b1) * b2))


@module_test
def test_conditioned_transition_inplace_gating():
    """minor#8: in-place ConditionedTransitionBlock gating is bit-identical."""

    def reference(mod, a, s):
        x = mod.adaln(a, s) if s is not None else mod.pre_norm(a)
        sa, sb = mod.lin_swish(x).chunk(2, dim=-1)
        out = mod.lin_out(F.silu(sa) * sb)
        if s is not None:
            out = torch.sigmoid(mod.output_gate(s)) * out
        return out

    for use_cond in (True, False):
        torch.manual_seed(int(use_cond))
        mod = ConditionedTransitionBlock(d_model=12, d_cond=12, use_conditioning=use_cond).eval()
        for t in range(3):
            a = torch.randn(2, [4, 7][t % 2], 12)
            s = torch.randn(2, a.shape[1], 12) if use_cond else None
            with torch.no_grad():
                assert torch.equal(mod(a, s), reference(mod, a, s))


@module_test
def test_attention_pair_bias_inplace_logits():
    """minor#9: in-place logit bias adds in the reference attention path are bit-identical."""

    def reference(mod, a, s, z, mask=None):
        bsz, nq, dm = a.shape
        x = mod.adaln(a, s) if s is not None else mod.pre_norm(a)
        q = mod.q_proj(x).view(bsz, nq, mod.num_heads, mod.head_dim)
        k, v = mod.kv_proj(x).chunk(2, dim=-1)
        k = k.view(bsz, x.shape[1], mod.num_heads, mod.head_dim)
        v = v.view(bsz, x.shape[1], mod.num_heads, mod.head_dim)
        g = torch.sigmoid(mod.g_proj(x)).view(bsz, nq, mod.num_heads, mod.head_dim)
        logits = torch.einsum("... i h d, ... j h d -> ... i j h", q, k) * mod.scale
        pair_bias = mod.pair_bias_proj(mod.pair_norm(z)) if z.dim() == 4 else z.unsqueeze(-1)
        logits = logits + pair_bias.to(dtype=logits.dtype)
        if mask is not None:
            mn = torch.finfo(logits.dtype).min
            logits = logits + torch.where(mask.bool()[:, None, :, None], 0.0, mn).to(logits.dtype)
        attn = torch.softmax(logits, dim=-2).to(dtype=v.dtype)
        ctx = g * torch.einsum("... i j h, ... j h d -> ... i h d", attn, v)
        out = mod.out_proj(ctx.reshape(bsz, nq, dm))
        if s is not None:
            out = torch.sigmoid(mod.out_gate(s)) * out
        return out

    for use_cond in (True, False):
        torch.manual_seed(50 + int(use_cond))
        # default backend (None) -> reference path, which contains the edited adds
        mod = AttentionPairBias(
            d_model=16, d_pair=8, num_heads=4, d_cond=16, use_conditioning=use_cond
        ).eval()
        for t in range(4):
            L = [3, 6, 9][t % 3]
            a = torch.randn(2, L, 16)
            s = torch.randn(2, L, 16) if use_cond else None
            z = torch.randn(2, L, L, 8)
            mask = (torch.rand(2, L) > 0.3) if t % 2 else None
            with torch.no_grad():
                assert torch.equal(
                    mod(a, s, z, attention_mask=mask), reference(mod, a, s, z, mask)
                )


# ===========================================================================
# 2. Real-model fold vs fold_batch parity + determinism (opt-in)
# ===========================================================================

_UBQ = (
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
)


@pytest.mark.skipif(
    not os.environ.get("ESMFOLD2_REGRESSION"),
    reason="opt-in: set ESMFOLD2_REGRESSION=1 (needs ESMFold2 weights; GPU recommended). "
    "Optionally ESMFOLD2_MODEL=<hf-repo-or-local-path>.",
)
def test_fold_batch_matches_fold_and_is_deterministic():
    from esm.models.esmfold2 import (
        ESMFold2InputBuilder,
        ProteinInput,
        StructurePredictionInput,
    )
    from esm.models.esmfold2.esmfold2_remote_code.modeling_esmfold2 import (
        ESMFold2Model,
    )

    repo = os.environ.get("ESMFOLD2_MODEL", "biohub/ESMFold2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ESMFold2Model.from_pretrained(repo).to(device).eval()
    builder = ESMFold2InputBuilder()

    inp = StructurePredictionInput(sequences=[ProteinInput(id="A", sequence=_UBQ)])
    kw = dict(num_loops=4, num_sampling_steps=20, num_diffusion_samples=1)

    def outputs(res):
        return res.plddt.cpu(), res.complex.to_mmcif(), res.ptm

    # Inline fold (ESMC runs inside the model) vs batched fold (ESMC encoded,
    # offloaded, then folded from the cached lm_hidden_states).
    inline_plddt, inline_cif, inline_ptm = outputs(builder.fold(model, inp, seed=7, **kw))
    batched_plddt, batched_cif, batched_ptm = outputs(
        builder.fold_batch(model, [inp], seed=7, **kw)[0]
    )
    assert torch.equal(inline_plddt, batched_plddt), "fold_batch pLDDT != fold"
    assert inline_cif == batched_cif, "fold_batch coordinates != fold"
    assert inline_ptm == batched_ptm, "fold_batch pTM != fold"

    # Determinism: same seed twice -> identical.
    again_plddt, again_cif, _ = outputs(builder.fold(model, inp, seed=7, **kw))
    assert torch.equal(inline_plddt, again_plddt)
    assert inline_cif == again_cif
