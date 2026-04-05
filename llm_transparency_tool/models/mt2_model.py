# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Mt2TransparentLlm: TransparentLlm adapter for the deezer/mt2 audio ViT model.

Uses the actual deezer/mt2 model (https://github.com/deezer/mt2) with
PyTorch forward hooks to capture internal activations — no HookedMT2 needed.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torchaudio
from jaxtyping import Float, Int

from llm_transparency_tool.models.transparent_llm import ModelInfo, TransparentLlm

# ── Add deezer/mt2 repo to sys.path ──────────────────────────────────────────
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_MT2_REPO = os.path.join(_REPO_ROOT, "..", "mt2")  # sibling directory
for _p in [_MT2_REPO]:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

try:
    # gin.enter_interactive_mode() allows @gin.configurable classes to use
    # their Python default args without needing a parsed config file.
    # We do NOT parse mt2.gin here because it contains training-only bindings
    # (e.g. dataloader) that fail unless the full src package is pre-imported.
    # The model architecture defaults (dim_embed=192, depth=12, n_heads=3)
    # already match the paper config so no explicit binding is needed.
    import gin
    gin.enter_interactive_mode()

    from src.model import MT2  # type: ignore
    from einops import rearrange, repeat  # type: ignore
    _MT2_AVAILABLE = True
    _MT2_IMPORT_ERROR = ""
except Exception as _e:
    import traceback as _tb
    MT2 = None  # type: ignore
    _MT2_AVAILABLE = False
    _MT2_IMPORT_ERROR = _tb.format_exc()


@dataclass
class _Mt2RunInfo:
    """Stores everything from a single MT2 forward pass with hooks."""
    tokens: Int[torch.Tensor, "batch pos"]
    logits: Float[torch.Tensor, "batch pos d_vocab"]
    # Per-layer activations captured by hooks
    resid_pre: Dict[int, torch.Tensor]    # blocks.{i}.hook_resid_pre
    resid_mid: Dict[int, torch.Tensor]    # blocks.{i}.hook_resid_mid  (after attn)
    resid_post: Dict[int, torch.Tensor]   # blocks.{i}.hook_resid_post (after mlp)
    attn_pattern: Dict[int, torch.Tensor] # blocks.{i}.attn.hook_pattern [b h t t]
    attn_v: Dict[int, torch.Tensor]       # blocks.{i}.attn.hook_v [b h t e]
    attn_result: Dict[int, torch.Tensor]  # blocks.{i}.attn.hook_result [b t h e]
    mlp_pre: Dict[int, torch.Tensor]      # blocks.{i}.mlp.hook_pre [b t hidden]
    mlp_post: Dict[int, torch.Tensor]     # blocks.{i}.mlp.hook_post [b t hidden]
    mlp_out: Dict[int, torch.Tensor]      # blocks.{i}.hook_mlp_out [b t d]
    seq_len: int


class Mt2TransparentLlm(TransparentLlm):
    """
    TransparentLlm implementation wrapping the deezer/mt2 audio ViT.

    run(sentences) interprets each string as an audio file path (wav/mp3/etc).
    The model checkpoint is loaded from 'mt2/model_state_dict.pt' relative to
    the workspace root, or from the path given in config/mt2.json.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        **kwargs: Any,
    ):
        if not _MT2_AVAILABLE:
            raise ImportError(
                f"deezer/mt2 not available. Error: {_MT2_IMPORT_ERROR}\n"
                f"Make sure the mt2 repo is cloned at {_MT2_REPO} and "
                f"its dependencies (nnAudio, einops, gin-config) are installed."
            )
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"MT2 checkpoint not found: {checkpoint_path}\n"
                f"Download it with: git clone https://github.com/deezer/mt2.git"
            )

        self._device = "cuda" if device == "gpu" else device
        self._dtype = dtype

        # Load model
        self._model = MT2(encoder_type="vit", device=self._device)
        self._model.load_state_dict(torch.load(checkpoint_path, map_location=self._device))
        self._model.eval()
        self._model = self._model.to(self._device)

        self._last_run: Optional[_Mt2RunInfo] = None
        self._run_exception = RuntimeError("Call run() before accessing model outputs.")

    def copy(self) -> "Mt2TransparentLlm":
        import copy
        return copy.copy(self)

    def model_info(self) -> ModelInfo:
        cfg = self._model.backbone
        n_layers = cfg.get_num_layers()
        n_heads = cfg.blocks[0].attn.n_heads
        d_model = cfg.dim_embed
        # The UI presents octave-pooled pitch classes.
        d_vocab = 12
        return ModelInfo(
            name="mt2",
            n_params_estimate=sum(p.numel() for p in self._model.parameters()),
            n_layers=n_layers,
            n_heads=n_heads,
            d_model=d_model,
            d_vocab=d_vocab,
        )

    # ── Audio Loading ─────────────────────────────────────────────────────────

    def _load_audio(self, path: str) -> torch.Tensor:
        """Load audio file and return tensor of shape (sr*duration,) normalized 0..1."""
        # torchaudio.load requires torchcodec in newer builds; use scipy/soundfile instead.
        try:
            import soundfile as sf
            data, sr = sf.read(path, always_2d=True)  # (samples, channels)
            wav = torch.from_numpy(data.mean(axis=1).astype("float32"))  # mono
        except Exception:
            import scipy.io.wavfile as wv
            sr, data = wv.read(path)
            if data.ndim > 1:
                data = data.mean(axis=1)
            if data.dtype == "int16":
                data = data.astype("float32") / 32768.0
            elif data.dtype == "int32":
                data = data.astype("float32") / 2147483648.0
            wav = torch.from_numpy(data.astype("float32"))

        target_sr: int = self._model.get_sr()
        duration: int = self._model.get_duration()
        target_samples = target_sr * duration

        # Resample if needed (use torchaudio.functional which doesn't need torchcodec)
        if sr != target_sr:
            wav = torch.nn.functional.interpolate(
                wav.unsqueeze(0).unsqueeze(0),
                size=int(len(wav) * target_sr / sr),
                mode="linear",
                align_corners=False,
            ).squeeze()

        # Trim / pad
        if wav.numel() >= target_samples:
            wav = wav[:target_samples]
        else:
            wav = torch.nn.functional.pad(wav, (0, target_samples - wav.numel()))

        # Normalize to [0, 1]
        wav = wav.clamp(-1.0, 1.0)
        wav = (wav + 1.0) / 2.0
        return wav  # (target_samples,)

    # ── Hook-based forward pass ───────────────────────────────────────────────

    def _run_with_hooks(self, audio_1d: torch.Tensor) -> _Mt2RunInfo:
        """
        Run the MT2 backbone with PyTorch forward hooks to capture all
        intermediate activations that LLM-TT needs.

        audio_1d: (batch, sr*duration) — pre-processed audio
        """
        backbone = self._model.backbone
        n_layers = backbone.get_num_layers()

        # Storage for hook captures
        resid_pre: Dict[int, torch.Tensor] = {}
        resid_mid: Dict[int, torch.Tensor] = {}
        resid_post: Dict[int, torch.Tensor] = {}
        attn_pattern: Dict[int, torch.Tensor] = {}
        attn_v: Dict[int, torch.Tensor] = {}
        attn_result: Dict[int, torch.Tensor] = {}
        mlp_pre: Dict[int, torch.Tensor] = {}
        mlp_post: Dict[int, torch.Tensor] = {}
        mlp_out: Dict[int, torch.Tensor] = {}

        handles = []

        for layer_i, block in enumerate(backbone.blocks):
            i = layer_i  # capture in closure

            # resid_pre: input to this block
            def make_pre_hook(li):
                def hook(module, inp, out):
                    # inp[0] is the residual stream entering the block
                    resid_pre[li] = inp[0].detach().clone()
                return hook
            handles.append(block.register_forward_hook(make_pre_hook(i)))

            # Hook attention module to get pattern, v, result
            def make_attn_hook(li):
                def hook(module, inp, out):
                    x_in = inp[0]  # [b, t, d]
                    # Recompute QKV to extract v and pattern
                    from einops import rearrange as _r
                    qkv = _r(
                        module.qkv(x_in),
                        "b t (e h qkv) -> qkv b h t e",
                        qkv=3,
                        e=module.head_dim,
                        h=module.n_heads,
                    )
                    q, k, v = qkv[0], qkv[1], qkv[2]
                    scale = module.scale
                    pattern = (q @ k.transpose(-2, -1)) * scale
                    pattern = pattern.softmax(dim=-1)  # [b, h, t, t]
                    attn_pattern[li] = pattern.detach().clone()
                    attn_v[li] = v.detach().clone()
                    # result: per-head output before proj [b, t, h, e]
                    result = (pattern @ v)  # [b, h, t, e]
                    result = _r(result, "b h t e -> b t h e")
                    attn_result[li] = result.detach().clone()
                return hook
            handles.append(block.attn.register_forward_hook(make_attn_hook(i)))

            # Hook MLP to get pre/post activations
            def make_mlp_hook(li):
                def hook(module, inp, out):
                    # pre: after fc1+act, post: after fc2
                    x_in = inp[0]
                    pre = module.act(module.fc1(x_in))
                    post = module.fc2(pre)
                    mlp_pre[li] = pre.detach().clone()
                    mlp_post[li] = post.detach().clone()
                    mlp_out[li] = out.detach().clone()
                return hook
            handles.append(block.mlp.register_forward_hook(make_mlp_hook(i)))

            # resid_mid: after attn, resid_post: after mlp — hook the block output for mid
            def make_block_out_hook(li, blocks_ref):
                block_li = blocks_ref[li]
                def hook(module, inp, out):
                    # out = resid_post for this layer
                    resid_post[li] = out.detach().clone()
                    # resid_mid = out - mlp_out (subtract MLP contribution)
                    if li in mlp_out:
                        resid_mid[li] = (out - mlp_out[li]).detach().clone()
                    else:
                        resid_mid[li] = out.detach().clone()
                return hook
            handles.append(block.register_forward_hook(make_block_out_hook(i, backbone.blocks)))

        with torch.no_grad():
            # --- Replicate the extract_feature forward pass ---
            audio = audio_1d.to(self._device)

            # Equivariant path (CQT / "stone")
            hcqt = self._model.hcqt(audio)
            hcqt = rearrange(hcqt, "b 1 f t -> b t f")
            x_stone = self._model.norm_in_stone(hcqt)
            x_stone = self._model.act_in(self._model.linear_in_stone(x_stone))
            cls_stone = repeat(self._model.cls_stone, "e -> b 1 e", b=audio.shape[0])
            cls_stone = cls_stone + torch.mean(x_stone, dim=1, keepdim=True)

            # Contrastive path (mel)
            mel = rearrange(self._model.spec(audio.unsqueeze(-1)), "b 1 e t -> b t e")
            x_contr = self._model.norm_in_contrastive(mel)
            x_contr = self._model.act_in(self._model.linear_in_contrastive(x_contr))
            cls_contr = repeat(self._model.cls_contrastive, "e -> b 1 e", b=audio.shape[0])
            cls_contr = cls_contr + torch.mean(x_contr, dim=1, keepdim=True)

            # Combine + run backbone (hooks fire here)
            x = x_contr + x_stone
            x = torch.cat((cls_stone, cls_contr, x), dim=1)
            x = self._model.pos_emb(x)
            x = self._model.backbone(x)  # ← hooks fire

            # Logits: use linear_out_stone on each position
            # shape: [batch, pos, d_vocab]
            logits = self._model.linear_out_stone(x)              # [b, t, n_bins]
            logits = self._model.octave_pool(logits)              # [b, t, 12]

            seq_len = x.shape[1]
            batch_size = x.shape[0]

            # "tokens" = position indices (0=CLS_STONE, 1=CLS_CONTR, 2..=seq positions)
            tokens = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        # Remove hooks
        for h in handles:
            h.remove()

        return _Mt2RunInfo(
            tokens=tokens,
            logits=logits,
            resid_pre=resid_pre,
            resid_mid=resid_mid,
            resid_post=resid_post,
            attn_pattern=attn_pattern,
            attn_v=attn_v,
            attn_result=attn_result,
            mlp_pre=mlp_pre,
            mlp_post=mlp_post,
            mlp_out=mlp_out,
            seq_len=seq_len,
        )

    # ── TransparentLlm API ────────────────────────────────────────────────────

    @torch.no_grad()
    def run(self, sentences: List[str]) -> None:
        """Load audio from file paths and run the model with hook capture."""
        tensors = []
        for path in sentences:
            t = self._load_audio(path.strip())
            tensors.append(t)
        audio = torch.stack(tensors, dim=0).to(self._device)  # (batch, samples)
        self._last_run = self._run_with_hooks(audio)

    def batch_size(self) -> int:
        if self._last_run is None:
            raise self._run_exception
        return self._last_run.tokens.shape[0]

    def tokens(self) -> Int[torch.Tensor, "batch pos"]:
        if self._last_run is None:
            raise self._run_exception
        return self._last_run.tokens

    def tokens_to_strings(self, tokens: Int[torch.Tensor, "pos"]) -> List[str]:
        """Map position indices to human-readable names."""
        result = []
        audio_token_count = max(len(tokens.tolist()) - 2, 1)
        duration = float(self._model.get_duration())
        denom = max(audio_token_count - 1, 1)

        for i in tokens.tolist():
            if i == 0:
                result.append("CLS_STONE")
            elif i == 1:
                result.append("CLS_CONTR")
            else:
                position = i - 2
                time_s = duration * position / denom
                result.append(f"{time_s:.1f}s")
        return result

    def logits(self) -> Float[torch.Tensor, "batch pos d_vocab"]:
        if self._last_run is None:
            raise self._run_exception
        return self._last_run.logits

    # ── Residual stream ───────────────────────────────────────────────────────

    def residual_in(self, layer: int) -> Float[torch.Tensor, "batch pos d_model"]:
        if self._last_run is None:
            raise self._run_exception
        return self._last_run.resid_pre[layer]

    def residual_after_attn(self, layer: int) -> Float[torch.Tensor, "batch pos d_model"]:
        if self._last_run is None:
            raise self._run_exception
        return self._last_run.resid_mid[layer]

    def residual_out(self, layer: int) -> Float[torch.Tensor, "batch pos d_model"]:
        if self._last_run is None:
            raise self._run_exception
        return self._last_run.resid_post[layer]

    # ── Attention ─────────────────────────────────────────────────────────────

    def ffn_out(self, layer: int) -> Float[torch.Tensor, "batch pos d_model"]:
        if self._last_run is None:
            raise self._run_exception
        return self._last_run.mlp_out[layer]

    def decomposed_ffn_out(
        self, batch_i: int, layer: int, pos: int
    ) -> Float[torch.Tensor, "hidden d_model"]:
        """Return per-neuron contributions: outer product of post-activations × W_out rows."""
        if self._last_run is None:
            raise self._run_exception
        post = self._last_run.mlp_post[layer][batch_i][pos]  # (hidden,)
        fc2 = self._model.backbone.blocks[layer].mlp.fc2
        # Each neuron i contributes post[i] * W_out[i, :]
        return torch.mul(post.unsqueeze(-1), fc2.weight.T)  # (hidden, d_model)

    def neuron_activations(
        self, batch_i: int, layer: int, pos: int
    ) -> Float[torch.Tensor, "d_ffn"]:
        if self._last_run is None:
            raise self._run_exception
        return self._last_run.mlp_pre[layer][batch_i][pos]

    def neuron_output(self, layer: int, neuron: int) -> Float[torch.Tensor, "d_model"]:
        return self._model.backbone.blocks[layer].mlp.fc2.weight.T[neuron]

    def attention_matrix(
        self, batch_i: int, layer: int, head: int
    ) -> Float[torch.Tensor, "query_pos key_pos"]:
        if self._last_run is None:
            raise self._run_exception
        return self._last_run.attn_pattern[layer][batch_i][head]  # [t, t]

    def attention_output_per_head(
        self, batch_i: int, layer: int, pos: int, head: int
    ) -> Float[torch.Tensor, "d_model"]:
        if self._last_run is None:
            raise self._run_exception
        # result: [b, t, h, e] → project back to d_model via W_O
        v_h = self._last_run.attn_result[layer][batch_i][pos][head]  # (head_dim,)
        W_O = self._model.backbone.blocks[layer].attn.proj.weight  # (d_model, d_model)
        d_model = W_O.shape[0]
        n_heads = self._model.backbone.blocks[layer].attn.n_heads
        head_dim = d_model // n_heads
        # Extract the slice of W_O corresponding to this head
        W_O_head = W_O[:, head * head_dim: (head + 1) * head_dim]  # (d_model, head_dim)
        return W_O_head @ v_h  # (d_model,)

    def attention_output(
        self, batch_i: int, layer: int, pos: int, head: Optional[int] = None
    ) -> Float[torch.Tensor, "d_model"]:
        if head is not None:
            return self.attention_output_per_head(batch_i, layer, pos, head)
        if self._last_run is None:
            raise self._run_exception
        return (
            self._last_run.resid_mid[layer][batch_i][pos]
            - self._last_run.resid_pre[layer][batch_i][pos]
        )

    @torch.no_grad()
    def decomposed_attn(
        self, batch_i: int, layer: int
    ) -> Float[torch.Tensor, "pos key_pos head d_model"]:
        """
        Return per-(query, key, head) attention contributions in d_model space.
        Shape: [pos, key_pos, head, d_model]
        """
        if self._last_run is None:
            raise self._run_exception
        pattern = self._last_run.attn_pattern[layer][batch_i]  # [h, t, t]
        v = self._last_run.attn_v[layer][batch_i]               # [h, t, head_dim]
        W_O = self._model.backbone.blocks[layer].attn.proj.weight  # [d_model, d_model]
        n_heads = pattern.shape[0]
        t = pattern.shape[1]
        d_model = W_O.shape[0]
        head_dim = d_model // n_heads

        # result[h, q, k, :] = pattern[h, q, k] * (W_O_h @ v[h, k])
        # W_O_h: [d_model, head_dim]
        result = torch.zeros(t, t, n_heads, d_model, device=pattern.device)
        for h in range(n_heads):
            W_O_h = W_O[:, h * head_dim: (h + 1) * head_dim]  # [d_model, head_dim]
            # v[h]: [t, head_dim] → projected: [t, d_model]
            v_proj = v[h] @ W_O_h.T  # [t, d_model]
            # pattern[h]: [t, t] (query × key)
            # result[:, :, h, :] = pattern[h] outer-broadcast v_proj
            result[:, :, h, :] = (pattern[h].unsqueeze(-1) * v_proj.unsqueeze(0))
        return result

    @torch.no_grad()
    def unembed(
        self,
        t: Float[torch.Tensor, "d_model"],
        normalize: bool,
    ) -> Float[torch.Tensor, "vocab"]:
        t = t.to(self._device, dtype=self._dtype)
        if normalize:
            t = torch.nn.functional.layer_norm(t.unsqueeze(0), (t.shape[-1],)).squeeze(0)
        out = self._model.linear_out_stone(t.unsqueeze(0))   # [1, n_bins]
        out = self._model.octave_pool(out)                    # [1, 12]
        return out.squeeze(0)                                  # [12]
