# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Mt2TransparentLlm: TransparentLlm adapter for MT2 audio ViT via TransformerLens HookedMT2."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torchaudio
from jaxtyping import Float, Int

from llm_transparency_tool.models.transparent_llm import ModelInfo, TransparentLlm

# Ensure TransformerLens and mt2 are on path when running from llm-transparency-tool
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (os.path.join(_REPO_ROOT, "TransformerLens"), os.path.join(_REPO_ROOT, "mt2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from transformer_lens import HookedMT2
except ImportError:
    HookedMT2 = None  # type: ignore


@dataclass
class _Mt2RunInfo:
    tokens: Int[torch.Tensor, "batch pos"]
    logits: Float[torch.Tensor, "batch pos d_vocab"]
    cache_dict: Dict[str, torch.Tensor]
    seq_len: int


class Mt2TransparentLlm(TransparentLlm):
    """
    TransparentLlm implementation wrapping HookedMT2.
    run(sentences) interprets each string as an audio file path.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        **kwargs: Any,
    ):
        if HookedMT2 is None:
            raise ImportError("transformer_lens.HookedMT2 not available; add TransformerLens and mt2 to path.")
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"MT2 checkpoint not found: {checkpoint_path}")
        self._device = "cuda" if device == "gpu" else device
        self._dtype = dtype
        self._model = HookedMT2.from_pretrained(
            "mt2",
            checkpoint_path=checkpoint_path,
            device=self._device,
            dtype=dtype,
        )
        self._last_run: Optional[_Mt2RunInfo] = None
        self._run_exception = RuntimeError("Call run() before using model output.")

    def copy(self) -> "Mt2TransparentLlm":
        import copy
        return copy.copy(self)

    def model_info(self) -> ModelInfo:
        cfg = self._model.cfg
        return ModelInfo(
            name="mt2",
            n_params_estimate=cfg.n_params_estimate,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            d_model=cfg.d_model,
            d_vocab=cfg.d_vocab,
        )

    def _load_audio_to_tensor(self, path: str) -> torch.Tensor:
        """Load one audio file to (1, samples, 3) for MT2 (same segment three times)."""
        wav, sr = torchaudio.load(path)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        wav = wav.mean(dim=0, keepdim=True)
        target_sr = self._model.mt2.get_sr()
        duration = self._model.mt2.get_duration()
        target_samples = target_sr * duration
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        wav = wav.squeeze(0)
        if wav.numel() >= target_samples:
            wav = wav[:target_samples]
        else:
            wav = torch.nn.functional.pad(wav, (0, target_samples - wav.numel()))
        wav = wav.clamp(-1.0, 1.0)
        wav = (wav + 1.0) / 2.0
        wav = wav.unsqueeze(0).unsqueeze(-1).expand(1, -1, 3)
        return wav

    @torch.no_grad()
    def run(self, sentences: List[str]) -> None:
        tensors = []
        for path in sentences:
            t = self._load_audio_to_tensor(path)
            tensors.append(t)
        audio = torch.cat(tensors, dim=0).to(self._device)
        logits, cache = self._model.run_with_cache(audio)
        batch_size = audio.shape[0]
        seq_len = logits.shape[1]
        tokens = self._model.tokens(batch_size, seq_len)
        self._last_run = _Mt2RunInfo(
            tokens=tokens,
            logits=logits,
            cache_dict=self._model._last_cache_dict,
            seq_len=seq_len,
        )

    def batch_size(self) -> int:
        if self._last_run is None:
            raise self._run_exception
        return self._last_run.tokens.shape[0]

    def tokens(self) -> Int[torch.Tensor, "batch pos"]:
        if self._last_run is None:
            raise self._run_exception
        return self._last_run.tokens

    def tokens_to_strings(self, tokens: Int[torch.Tensor, "pos"]) -> List[str]:
        pos_list = tokens.tolist()
        names = []
        for i in pos_list:
            if i == 0:
                names.append("CLS_STONE")
            elif i == 1:
                names.append("CLS_CONTR")
            else:
                names.append("t=%d" % (i - 2))
        return names

    def logits(self) -> Float[torch.Tensor, "batch pos d_vocab"]:
        if self._last_run is None:
            raise self._run_exception
        return self._last_run.logits

    def _get_block(self, layer: int, name: str) -> torch.Tensor:
        if self._last_run is None:
            raise self._run_exception
        key = f"blocks.{layer}.{name}"
        return self._last_run.cache_dict[key]

    @torch.no_grad()
    def unembed(
        self,
        t: Float[torch.Tensor, "d_model"],
        normalize: bool,
    ) -> Float[torch.Tensor, "vocab"]:
        t = t.to(self._device, dtype=self._dtype)
        if normalize:
            t = torch.nn.functional.layer_norm(t.unsqueeze(0), (t.shape[-1],))
            t = t.squeeze(0)
        out = self._model.mt2.linear_out_stone(t.unsqueeze(0))
        out = self._model.mt2.octave_pool(out)
        return out.squeeze(0)

    def residual_in(self, layer: int) -> Float[torch.Tensor, "batch pos d_model"]:
        return self._get_block(layer, "hook_resid_pre")

    def residual_after_attn(self, layer: int) -> Float[torch.Tensor, "batch pos d_model"]:
        return self._get_block(layer, "hook_resid_mid")

    def residual_out(self, layer: int) -> Float[torch.Tensor, "batch pos d_model"]:
        return self._get_block(layer, "hook_resid_post")

    def ffn_out(self, layer: int) -> Float[torch.Tensor, "batch pos d_model"]:
        return self._get_block(layer, "hook_mlp_out")

    def decomposed_ffn_out(
        self,
        batch_i: int,
        layer: int,
        pos: int,
    ) -> Float[torch.Tensor, "hidden d_model"]:
        post = self._get_block(layer, "mlp.hook_post")[batch_i][pos]
        fc2 = self._model.mt2.backbone.blocks[layer].mlp.fc2
        return torch.mul(post.unsqueeze(-1), fc2.weight.T)

    def neuron_activations(
        self,
        batch_i: int,
        layer: int,
        pos: int,
    ) -> Float[torch.Tensor, "d_ffn"]:
        return self._get_block(layer, "mlp.hook_pre")[batch_i][pos]

    def neuron_output(
        self,
        layer: int,
        neuron: int,
    ) -> Float[torch.Tensor, "d_model"]:
        return self._model.mt2.backbone.blocks[layer].mlp.fc2.weight.T[neuron]

    def attention_matrix(
        self,
        batch_i: int,
        layer: int,
        head: int,
    ) -> Float[torch.Tensor, "query_pos key_pos"]:
        return self._get_block(layer, "attn.hook_pattern")[batch_i][head]

    def attention_output_per_head(
        self,
        batch_i: int,
        layer: int,
        pos: int,
        head: int,
    ) -> Float[torch.Tensor, "d_model"]:
        return self._get_block(layer, "attn.hook_result")[batch_i][pos][head]

    def attention_output(
        self,
        batch_i: int,
        layer: int,
        pos: int,
        head: Optional[int] = None,
    ) -> Float[torch.Tensor, "d_model"]:
        if head is not None:
            return self._get_block(layer, "attn.hook_result")[batch_i][pos][head]
        return self._get_block(layer, "hook_attn_out")[batch_i][pos]

    @torch.no_grad()
    def decomposed_attn(
        self,
        batch_i: int,
        layer: int,
    ) -> Float[torch.Tensor, "pos key_pos head d_model"]:
        return self._model.decomposed_attn(batch_i, layer)
