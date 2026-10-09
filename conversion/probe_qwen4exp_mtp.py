#!/usr/bin/env python3
"""Self-check for the MTP export of Qwen3.8-Flash-Next.

Checks the PATH a tensor takes: filter_tensors (inherited from _QwenMtpMixin) rewrites
the `mtp.` namespace, modify_tensors handles whatever the parent remapper does not know.

    python3 conversion/probe_qwen4exp_mtp.py

The first version tested `_mtp_tensor` directly and passed with 29 families, while
the real run failed immediately. It now checks the path instead of the function.

What the real run still has to answer: the concatenation order of eh_proj
(the wrong way round gives valid numbers and nonsense) and whether llama.cpp loads the result.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "gguf-py"))

import torch
import gguf
import conversion.qwen4exp as q

N_LAYER = 48


def make_class(mtp_only=False):
    k = type("Probe", (q.Qwen4ExpTextModel,), {"model_arch": gguf.MODEL_ARCH.QWEN4EXP})
    k.mtp_only, k.no_mtp = mtp_only, False
    k._original_block_count = N_LAYER      # otherwise set by index_tensors
    k.opt_num_mtp_layers = 0
    return k


def main() -> int:
    errors = 0

    print("  ── 1. filter_tensors: where does the parent class rewrite to? ──")
    K = make_class()
    PATHS = [
        ("mtp.layers.0.self_attn.q_proj.weight", "model.layers.48.self_attn.q_proj.weight"),
        ("mtp.layers.0.mlp.experts.7.gate_proj.weight", "model.layers.48.mlp.experts.7.gate_proj.weight"),
        ("mtp.pre_fc_norm_embedding.weight", "model.layers.48.enorm.weight"),
        ("mtp.pre_fc_norm_hidden.weight", "model.layers.48.hnorm.weight"),
        ("mtp.fc_embedding.weight", "mtp.fc_embedding.weight"),          # stays: handled by own code
        ("mtp.hyper_connection_mixer.hc_norm.weight", "mtp.hyper_connection_mixer.hc_norm.weight"),
    ]
    for src, expected in PATHS:
        out = K.filter_tensors((src, lambda: None))
        actual = out[0] if out else "DISCARDED"
        flag = "OK   " if actual == expected else "ERROR"
        if actual != expected:
            errors += 1
        print("    %-6s %-46s -> %s" % (flag, src[:46], actual))
    print("    The last two keep the mtp. prefix — the parent remapper does not know them")
    print("    (the model has TWO fc projections, and the mixer has no predecessor).")

    print("\n  ── 2. modify_tensors: what the own code makes of them ──")
    m = object.__new__(make_class())   # ModelBase.__init__ needs real paths
    m._mtp_bid = N_LAYER
    m._mtp_fc = {}
    m.hparams = {"num_hidden_layers": N_LAYER, "indexer_n_heads": 8, "indexer_head_dim": 64}
    m.block_count = N_LAYER + 1
    m.tensor_map = gguf.get_tensor_name_map(gguf.MODEL_ARCH.QWEN4EXP, N_LAYER + 1)
    m.fuse_gate_up_exps = False
    m._ple_pending, m._ple_shard_rows = {}, {}
    m._ple_row_dim = m._ple_rows_per_shard = m._ple_map = m._ple_path = None

    for n in ("mtp.fc_embedding.weight", "mtp.fc_hidden.weight",
              "mtp.hyper_connection_mixer.hc_norm.weight",
              "mtp.hyper_connection_mixer.input_mix_weight_down.weight",
              "mtp.hyper_connection_mixer.input_mix_weight_up.weight"):
        try:
            out = list(m.modify_tensors(torch.zeros(2560, 2560), n, N_LAYER))
            if not out:
                print("    %-56s (held back until fc_hidden arrives)" % n[:56])
            for gn, t in out:
                print("    %-56s -> %-30s %s" % (n[:56], gn, tuple(t.shape)))
        except Exception as e:
            print("    %-56s ERROR %s: %s" % (n[:56], type(e).__name__, e)); errors += 1

    print("\n  ── 3. The concatenation: eh_proj must be twice as wide ──")
    m2 = object.__new__(make_class())
    m2._mtp_bid, m2._mtp_fc = N_LAYER, {}
    m2.tensor_map = gguf.get_tensor_name_map(gguf.MODEL_ARCH.QWEN4EXP, N_LAYER + 1)
    list(m2._mtp_tensor(torch.zeros(2560, 2560), "mtp.fc_embedding.weight"))
    out = list(m2._mtp_tensor(torch.zeros(2560, 2560), "mtp.fc_hidden.weight"))
    width = out[0][1].shape[-1] if out else 0
    flag = "OK   " if width == 5120 else "ERROR"
    if width != 5120:
        errors += 1
    print("    %-6s 2560 + 2560 -> %d  (expected 5120)" % (flag, width))

    print("\n  ── 4. filter_tensors under --mtp-only ──")
    KM = make_class(mtp_only=True)
    PASS_THROUGH = [
        ("mtp.fc_embedding.weight", True, "the MTP block itself"),
        ("model.language_model.layers.1.ple.ple_embedding.layer_multipliers", True,
         "PLE constant — set_gguf_parameters ALWAYS reads it"),
        ("model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets", True, "ditto"),
        ("model.language_model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes", True, "ditto"),
        ("model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight",
         False, "the 51-billion-entry table MUST stay out"),
        ("model.language_model.layers.7.self_attn.q_proj.weight", False,
         "ordinary layer, belongs in the main model"),
    ]
    for name, expected, why in PASS_THROUGH:
        actual = KM.filter_tensors((name, lambda: None)) is not None
        flag = "OK   " if actual == expected else "ERROR"
        if actual != expected:
            errors += 1
        print("    %-6s %-5s %-62s %s" % (flag, "pass" if expected else "drop", name[:62], why))

    print("\n  ── 5. The probe must be able to fail ──")
    try:
        m._mtp_tensor(torch.zeros(2, 2), "mtp.doesnotexist.weight")
        print("    FAILED — invented tensor accepted."); errors += 1
    except ValueError as e:
        print("    OK — rejected: %s" % e)

    print("\n  %d errors" % errors)
    print("  -> %s" % ("passed" if not errors else "FAILED"))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
