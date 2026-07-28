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
        _autocast_downcasts,
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

    # Staged loading (ESMC and the folding stack never co-resident): a
    # folding-only model that reloads ESMC internally must match the inline fold.
    staged_model = ESMFold2Model.from_pretrained(repo, load_esmc=False).eval()
    staged_plddt, staged_cif, staged_ptm = outputs(
        builder.fold_batch_staged(staged_model, [inp], device=device, seed=7, **kw)[0]
    )
    assert torch.equal(inline_plddt, staged_plddt), "fold_batch_staged pLDDT != fold"
    assert inline_cif == staged_cif, "fold_batch_staged coordinates != fold"
    assert inline_ptm == staged_ptm, "fold_batch_staged pTM != fold"

    # Determinism: same seed twice -> identical.
    again_plddt, again_cif, _ = outputs(builder.fold(model, inp, seed=7, **kw))
    assert torch.equal(inline_plddt, again_plddt)
    assert inline_cif == again_cif



# ===========================================================================
# A1: triangle-multiply fp32 mask excursion — bit-exactness guards
# ===========================================================================
#
# A1 removes a wasted fp32 round-trip in TriangleMultiplicativeBlock.forward.
# The 0/1 `visibility` mask arrives in fp32 (trunk/confidence paths) or bool
# (MSA-encoder path); an fp32 mask type-promoted the block's largest surviving
# intermediate (`routed`) from bf16 to fp32, after which `.float()` was a no-op
# and `einsum` cast it straight back down. A1 casts the mask down instead, and
# skips `.float()` only when `_autocast_downcasts` proves the round-trip is dead.
#
# IMPORTANT — why the interesting cases are CUDA-gated: `aten::einsum` is an
# autocast FALLTHROUGH on CPU (it is only registered for
# AutocastCUDA/XPU/MPS/MTIA/MAIA). Under CPU autocast the downcast happens only
# incidentally inside einsum's `sumproduct_pair` decomposition at the internal
# `bmm`, and an extent-1 contraction (L == 1) short-circuits to `mul` and never
# downcasts at all. So CPU autocast is NOT a faithful proxy for CUDA autocast
# here, and `_autocast_downcasts` deliberately returns False on CPU.
#
# What the CPU tests therefore pin: (a) the mask downcast alone is bit-exact,
# (b) the guard refuses to fire on CPU, (c) the fp32 contract is preserved
# wherever the guard is False. The end-to-end fast-path proof is CUDA-only.

_needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fast path only engages on CUDA autocast"
)


def _trimul_forward_pre_a1(block, pair_grid, visibility):
    """Byte-for-byte replica of TriangleMultiplicativeBlock.forward before A1."""
    if visibility is None:
        visibility = pair_grid.new_ones(pair_grid.shape[:-1])
    normalized_grid = block.norm_start(pair_grid)
    bundled = block.proj_bundle(normalized_grid)
    signal, gate_logits = bundled.split(2 * block.latent_channels, dim=-1)
    routed = signal * torch.sigmoid(gate_logits)
    routed = routed * visibility.unsqueeze(-1)  # fp32 mask -> promotes routed
    left_stream, right_stream = routed.float().chunk(2, dim=-1)  # dead round-trip
    if block._chunk_size is not None:
        contracted = block._triangular_contract_chunked(
            left_stream, right_stream, block._chunk_size
        )
    else:
        contracted = block._triangular_contract(left_stream, right_stream)
    mixed = block.proj_emit(block.norm_mix(contracted))
    output_gate = torch.sigmoid(block.proj_gate(normalized_grid))
    return mixed * output_gate


def _trimul_case(flow="outgoing", chunk=None, L=6, seed=0, device="cpu", mask_dtype=torch.float32):
    torch.manual_seed(seed)
    block = (
        TriangleMultiplicativeBlock(input_channels=16, latent_channels=8, flow=flow)
        .eval()
        .to(device)
    )
    block.set_chunk_size(chunk)
    pair_grid = torch.randn(1, L, L, 16, device=device)
    # Mask must stay NON-DEGENERATE at small L: zeroing a fixed-size tail would
    # blank the whole mask at L<=2 and hide real divergence (this is exactly how
    # the first version of these tests missed the L==1 counterexample).
    tok = torch.ones(1, L, device=device)
    if L > 2:
        tok[:, -2:] = 0.0
    visibility = (tok[:, :, None] * tok[:, None, :]).to(mask_dtype)
    return block, pair_grid, visibility


# --------------------------- the guard ---------------------------
@module_test
def test_autocast_downcasts_false_without_autocast():
    assert _autocast_downcasts(torch.randn(2, 2, dtype=torch.bfloat16)) is False


@module_test
def test_autocast_downcasts_false_for_fp32():
    # fp32 tensor: .float() is already a no-op, nothing to skip.
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert _autocast_downcasts(torch.randn(2, 2)) is False


@module_test
def test_autocast_downcasts_false_on_cpu_even_when_dtype_matches():
    """Pins the fix for the refuted premise: einsum is an autocast fallthrough on
    CPU, so the fp32 round-trip is NOT dead there and the guard must not fire."""
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert _autocast_downcasts(torch.randn(2, 2, dtype=torch.bfloat16)) is False


@module_test
def test_cpu_einsum_does_not_downcast_at_extent_one():
    """Documents WHY cpu is excluded: under CPU bf16 autocast einsum returns
    bf16 at L>=2 (via its internal bmm) but fp32 at L==1, where the
    sumproduct_pair decomposition short-circuits to `mul`. If this ever changes,
    revisit _EINSUM_AUTOCAST_DEVICES rather than deleting this test."""
    eq = "bikd,bjkd->bijd"
    with torch.autocast("cpu", dtype=torch.bfloat16):
        big = torch.einsum(eq, torch.randn(1, 3, 3, 4), torch.randn(1, 3, 3, 4))
        one = torch.einsum(eq, torch.randn(1, 1, 1, 4), torch.randn(1, 1, 1, 4))
    assert big.dtype == torch.bfloat16
    assert one.dtype == torch.float32, "extent-1 einsum downcast on CPU after all"


@module_test
@_needs_cuda
def test_autocast_downcasts_true_on_cuda_when_dtype_matches():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        t = torch.randn(2, 2, dtype=torch.bfloat16, device="cuda")
        assert _autocast_downcasts(t) is True


@module_test
@_needs_cuda
def test_autocast_downcasts_false_on_cuda_dtype_mismatch():
    # fp16 tensor under a bf16 autocast: einsum would not leave it alone.
    with torch.autocast("cuda", dtype=torch.bfloat16):
        t = torch.randn(2, 2, dtype=torch.float16, device="cuda")
        assert _autocast_downcasts(t) is False


@module_test
def test_autocast_downcasts_false_for_integer_tensor():
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert _autocast_downcasts(torch.ones(2, 2, dtype=torch.int64)) is False


@module_test
def test_autocast_downcasts_does_not_raise_on_meta():
    """Unknown/unqueryable device types must fail closed, not raise. A meta
    forward used to work pre-A1 and must keep working."""
    assert _autocast_downcasts(torch.randn(2, 2, dtype=torch.bfloat16, device="meta")) is False


# --------------------------- bit-exactness (CPU: mask cast only) ---------------------------
@module_test
@pytest.mark.parametrize("flow", ["outgoing", "incoming"])
@pytest.mark.parametrize("chunk", [None, 1, 2, 64])
@pytest.mark.parametrize("L", [1, 2, 3, 7])
def test_trimul_a1_bitexact_cpu_autocast(flow, chunk, L):
    """On CPU the guard is False, so this pins that the mask downcast on its own
    is bit-exact — including at L==1, the case that refuted the first version."""
    block, pair_grid, visibility = _trimul_case(flow=flow, chunk=chunk, L=L)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        new = block(pair_grid, visibility)
        old = _trimul_forward_pre_a1(block, pair_grid, visibility)
    assert new.dtype == old.dtype
    assert torch.equal(new, old), (
        f"A1 changed output (flow={flow}, chunk={chunk}, L={L}); "
        f"max|delta|={(new.float() - old.float()).abs().max().item():.3e}"
    )


@module_test
@pytest.mark.parametrize("mask_dtype", [torch.float32, torch.bool, torch.float64])
def test_trimul_a1_bitexact_across_mask_dtypes(mask_dtype):
    """fp32 (trunk), bool (MSA encoder) and fp64 masks all hold exactly 0/1, so
    casting any of them to routed.dtype must be lossless."""
    block, pair_grid, visibility = _trimul_case(chunk=None, mask_dtype=mask_dtype)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        new = block(pair_grid, visibility)
        old = _trimul_forward_pre_a1(block, pair_grid, visibility)
    assert torch.equal(new, old)


@module_test
@pytest.mark.parametrize("chunk", [None, 2])
def test_trimul_a1_bitexact_without_autocast(chunk):
    """Off autocast the guard falls back to .float(); output must be unchanged."""
    block, pair_grid, visibility = _trimul_case(chunk=chunk)
    new = block(pair_grid, visibility)
    old = _trimul_forward_pre_a1(block, pair_grid, visibility)
    assert torch.equal(new, old)


@module_test
def test_trimul_a1_bitexact_visibility_none():
    """visibility=None takes pair_grid's dtype; the .to() must be a no-op there."""
    block, pair_grid, _ = _trimul_case(chunk=None)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        new = block(pair_grid, None)
        old = _trimul_forward_pre_a1(block, pair_grid, None)
    assert torch.equal(new, old)


@module_test
def test_trimul_a1_fractional_mask_would_not_be_exact():
    """Pins the load-bearing invariant: the exactness argument holds ONLY for a
    0/1 mask. If someone introduces a soft mask, A1's premise breaks — this test
    documents that rather than letting it pass silently."""
    block, pair_grid, visibility = _trimul_case(chunk=None)
    soft = visibility * 0.1
    # 0.1 is not representable in bf16, so casting the mask down loses bits.
    assert soft.to(torch.bfloat16).to(torch.float32).ne(soft).any(), (
        "0.1 unexpectedly exact in bf16 — the invariant note needs revisiting"
    )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        new = block(pair_grid, soft)
        old = _trimul_forward_pre_a1(block, pair_grid, soft)
    # The mask cast applies regardless of the guard, so a soft mask diverges even
    # on CPU. This is the invariant failing loudly, exactly as intended: A1 is
    # output-preserving ONLY for a 0/1 mask. Every mask in ESMFold2 today is 0/1
    # (verified across all call sites); if that ever changes, revisit A1.
    assert not torch.equal(new, old), (
        "a fractional mask unexpectedly round-tripped exactly — if masks are now "
        "guaranteed 0/1 by construction this test can go, but do not just delete it"
    )


# --------------------------- CPU: fp32 contract preserved ---------------------------
def _einsum_dtype_spy(monkeypatch):
    seen = []
    real_einsum = torch.einsum

    def spy(eq, *operands, **kw):
        seen.append(tuple(o.dtype for o in operands))
        return real_einsum(eq, *operands, **kw)

    monkeypatch.setattr(torch, "einsum", spy)
    return seen


@module_test
def test_trimul_a1_keeps_fp32_operands_on_cpu(monkeypatch):
    """Guard is False on CPU, so einsum must still receive fp32 — no silent
    precision drop on the device where the downcast is not guaranteed."""
    block, pair_grid, visibility = _trimul_case(chunk=None)
    seen = _einsum_dtype_spy(monkeypatch)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        block(pair_grid, visibility)
    assert seen and all(d == torch.float32 for dts in seen for d in dts), seen


@module_test
def test_trimul_a1_promotes_to_fp32_without_autocast(monkeypatch):
    block, pair_grid, visibility = _trimul_case(chunk=None)
    seen = _einsum_dtype_spy(monkeypatch)
    block(pair_grid, visibility)
    assert seen and all(d == torch.float32 for dts in seen for d in dts)


@module_test
def test_trimul_a1_mask_still_zeroes_padding():
    """Sanity: casting the mask down must not break masking semantics."""
    block, pair_grid, visibility = _trimul_case(chunk=None, L=6)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        masked = block(pair_grid, visibility)
        unmasked = block(pair_grid, torch.ones_like(visibility))
    assert not torch.equal(masked, unmasked)


# --------------------------- CUDA: the actual fast path ---------------------------
@module_test
@_needs_cuda
@pytest.mark.parametrize("flow", ["outgoing", "incoming"])
@pytest.mark.parametrize("chunk", [None, 1, 64, 128])
@pytest.mark.parametrize("L", [1, 2, 7, 65, 130])
def test_trimul_a1_bitexact_cuda_autocast(flow, chunk, L):
    """The real claim: with the guard firing on CUDA, output is bit-identical to
    the pre-A1 code. Covers L==1 (the CPU counterexample) and strided views into
    `routed` at chunked production-ish shapes."""
    block, pair_grid, visibility = _trimul_case(
        flow=flow, chunk=chunk, L=L, device="cuda"
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        new = block(pair_grid, visibility)
        old = _trimul_forward_pre_a1(block, pair_grid, visibility)
    assert torch.equal(new, old), (
        f"A1 changed output on CUDA (flow={flow}, chunk={chunk}, L={L}); "
        f"max|delta|={(new.float() - old.float()).abs().max().item():.3e}"
    )


@module_test
@_needs_cuda
def test_trimul_a1_keeps_einsum_operands_in_bf16_on_cuda(monkeypatch):
    """Behavioural proof A1 actually removes the fp32 materialization."""
    block, pair_grid, visibility = _trimul_case(chunk=None, device="cuda")
    seen = _einsum_dtype_spy(monkeypatch)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        block(pair_grid, visibility)
    assert seen and all(d == torch.bfloat16 for dts in seen for d in dts), seen


@module_test
@_needs_cuda
def test_trimul_a1_bitexact_bf16_input_no_autocast_cuda():
    """bf16 input WITHOUT autocast: einsum would not downcast, so the fp32
    contract must be preserved rather than silently dropped to bf16."""
    block, pair_grid, visibility = _trimul_case(chunk=None, device="cuda")
    block = block.to(torch.bfloat16)
    pair_grid = pair_grid.to(torch.bfloat16)
    new = block(pair_grid, visibility)
    old = _trimul_forward_pre_a1(block, pair_grid, visibility)
    assert torch.equal(new, old)
