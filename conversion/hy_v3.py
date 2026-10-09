from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf, logger
from .qwen import QwenModel


@ModelBase.register("HYV3ForCausalLM")
class Hy3Model(TextModel):
    # Tencent Hunyuan-V3 (Hy3): DeepSeek-V3-style sigmoid router on a Hunyuan-MoE
    # skeleton (GQA + per-head qk-norm) plus one MTP/NextN layer.
    model_arch = gguf.MODEL_ARCH.HY_V3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The checkpoint carries num_nextn_predict_layers extra (MTP) layers at the
        # tail (index == num_hidden_layers). Keep the KV block_count at num_hidden_layers
        # (mirrors minimax-m3), but size the tensor map so blk.<n_layer>.* resolve.
        self._n_layer_nextn = int(self.hparams.get("num_nextn_predict_layers", 0) or 0)
        self._n_blk = int(self.hparams["num_hidden_layers"]) + self._n_layer_nextn
        self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self._n_blk)

    # --- FP8 dequant --------------------------------------------------------
    # Hy3-FP8 stores a per-tensor scale as "<w>.weight_scale" (scalar), which the
    # generic base fp8 path (which only matches "_scale_inv") does not handle.
    def dequant_model(self):
        quant_config = self.hparams.get("quantization_config") or {}
        if quant_config.get("quant_method") != "fp8":
            return super().dequant_model()

        fp8_dtypes = tuple(
            d for d in (getattr(torch, "float8_e4m3fn", None),
                        getattr(torch, "float8_e5m2", None)) if d is not None
        )

        def dequant_fp8(weight: Tensor, scale: Tensor) -> Tensor:
            # per-tensor (scalar) scale -> plain broadcast multiply
            return weight.float() * scale.float()

        to_remove: list[str] = []
        n_dequant = 0
        for name in list(self.model_tensors.keys()):
            if name.endswith(".weight_scale"):
                weight_name = name.removesuffix("_scale")  # -> "<...>.weight"
                if weight_name in self.model_tensors:
                    w = self.model_tensors[weight_name]
                    s = self.model_tensors[name]
                    if not fp8_dtypes or w().dtype in fp8_dtypes:
                        self.model_tensors[weight_name] = (
                            lambda w=w, s=s: dequant_fp8(w(), s())
                        )
                        n_dequant += 1
                to_remove.append(name)
            elif name.endswith((".input_scale", ".activation_scale",
                                ".weight_scale_2", ".kv_scale",
                                ".k_scale", ".v_scale")):
                # static-activation / kv-cache scales: unused by the decoder
                to_remove.append(name)

        for name in to_remove:
            self.model_tensors.pop(name, None)
        logger.info("Hy3: dequantized %d FP8 weight tensor(s)", n_dequant)

    # --- vocab (Hunyuan tiktoken / mergeable_ranks) -------------------------
    def set_vocab(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.dir_model, trust_remote_code=True)

        tokpre = self.get_vocab_base_pre(tokenizer)

        merges = []
        vocab = {}
        mergeable_ranks = tokenizer.mergeable_ranks
        for token, rank in mergeable_ranks.items():
            vocab[QwenModel.token_bytes_to_string(token)] = rank
            if len(token) == 1:
                continue
            merged = QwenModel.bpe(mergeable_ranks, token, max_rank=rank)
            if len(merged) == 2:
                merges.append(" ".join(map(QwenModel.token_bytes_to_string, merged)))

        vocab_size = self.hparams["vocab_size"]
        assert tokenizer.vocab_size == vocab_size
        special_tokens = tokenizer.special_tokens
        reverse_vocab = {id_: encoded_tok for encoded_tok, id_ in {**vocab, **special_tokens}.items()}
        tokens: list[str] = []
        toktypes: list[int] = []
        for i in range(vocab_size):
            if i not in reverse_vocab:
                tokens.append(f"[PAD{i}]")
                toktypes.append(gguf.TokenType.UNUSED)
            else:
                token = reverse_vocab[i]
                tokens.append(token)
                if i in special_tokens.values():
                    toktypes.append(gguf.TokenType.CONTROL)
                else:
                    toktypes.append(gguf.TokenType.NORMAL)

        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)
        self.gguf_writer.add_token_merges(merges)

        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=False)
        special_vocab.add_to_gguf(self.gguf_writer)

        # Special-IDs from the Hy3 config (do NOT hardcode Hunyuan-A13B ids).
        if (bos := self.hparams.get("bos_token_id")) is not None:
            self.gguf_writer.add_bos_token_id(bos)
        if (eos := self.hparams.get("eos_token_id")) is not None:
            self.gguf_writer.add_eos_token_id(eos)
        if (pad := self.hparams.get("pad_token_id")) is not None:
            self.gguf_writer.add_pad_token_id(pad)
        if (sep := self.hparams.get("sep_token_id")) is not None:
            self.gguf_writer.add_sep_token_id(sep)
        if (eod := self.hparams.get("eod_token_id")) is not None:
            self.gguf_writer.add_eot_token_id(eod)

    # --- hparams / gguf KV --------------------------------------------------
    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams

        self.gguf_writer.add_vocab_size(hparams["vocab_size"])

        # MoE geometry
        moe_ff = hparams["moe_intermediate_size"]
        self.gguf_writer.add_expert_feed_forward_length(moe_ff)
        n_shared = int(hparams.get("num_shared_experts", 1))
        self.gguf_writer.add_expert_shared_feed_forward_length(moe_ff * n_shared)
        self.gguf_writer.add_expert_shared_count(n_shared)
        self.gguf_writer.add_leading_dense_block_count(hparams["first_k_dense_replace"])

        # DeepSeek-V3-style sigmoid router with expert bias + renorm + scale.
        self.gguf_writer.add_expert_weights_scale(hparams["router_scaling_factor"])
        self.gguf_writer.add_expert_weights_norm(bool(hparams.get("route_norm", True)))
        self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)

        # MTP / NextN
        if self._n_layer_nextn > 0:
            self.gguf_writer.add_nextn_predict_layers(self._n_layer_nextn)

    # --- tensor transforms --------------------------------------------------
    _experts: list[dict[str, Tensor]] | None = None

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name == "lm_head.weight" and self.hparams.get("tie_word_embeddings", False):
            logger.info("Skipping tied output layer 'lm_head.weight'")
            return

        # fuse the 192 per-expert projections into stacked 3d tensors
        if name.find("mlp.experts") != -1:
            n_experts = self.hparams["num_experts"]
            assert bid is not None
            if self._experts is None:
                self._experts = [{} for _ in range(self._n_blk)]
            self._experts[bid][name] = data_torch
            if len(self._experts[bid]) >= n_experts * 3:
                for w_name in ["down_proj", "gate_proj", "up_proj"]:
                    datas: list[Tensor] = []
                    for xid in range(n_experts):
                        ename = f"model.layers.{bid}.mlp.experts.{xid}.{w_name}.weight"
                        datas.append(self._experts[bid][ename])
                        del self._experts[bid][ename]
                    data_torch = torch.stack(datas, dim=0)
                    merged_name = f"model.layers.{bid}.mlp.experts.{w_name}.weight"
                    yield from super().modify_tensors(data_torch, merged_name, bid)
                return
            else:
                return

        yield from super().modify_tensors(data_torch, name, bid)

    def prepare_tensors(self):
        super().prepare_tensors()
        if self._experts is not None:
            experts = [k for d in self._experts for k in d.keys()]
            if len(experts) > 0:
                raise ValueError(f"Unprocessed experts: {experts}")
