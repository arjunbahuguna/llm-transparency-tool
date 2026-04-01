# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for Mt2TransparentLlm and contribution graph. Skip if MT2 checkpoint not found."""

import os
import tempfile
import unittest

import torch
import torchaudio

from llm_transparency_tool.models.mt2_model import Mt2TransparentLlm

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_MT2_CKPT = os.path.join(_REPO_ROOT, "mt2", "model_state_dict.pt")


def _mt2_available():
    return os.path.isfile(_MT2_CKPT)


@unittest.skipIf(not _mt2_available(), "MT2 checkpoint not found")
class TestMt2TransparentLlm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = Mt2TransparentLlm(checkpoint_path=_MT2_CKPT, device="cpu")

    def test_model_info(self):
        info = self.model.model_info()
        self.assertEqual(info.name, "mt2")
        self.assertGreater(info.n_layers, 0)
        self.assertGreater(info.n_heads, 0)
        self.assertGreater(info.d_model, 0)
        self.assertEqual(info.d_vocab, 12)

    def test_run_and_contribution_graph(self):
        sr = 16000
        duration = 4
        samples = sr * duration
        wav = torch.randn(1, samples)
        wav = (wav - wav.min()) / (wav.max() - wav.min() + 1e-8)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            torchaudio.save(path, wav, sr)
            self.model.run([path])
            self.assertEqual(self.model.batch_size(), 1)
            tokens = self.model.tokens()
            logits = self.model.logits()
            self.assertEqual(tokens.shape[0], 1)
            self.assertEqual(logits.shape[0], 1)
            self.assertEqual(logits.shape[2], 12)
            from llm_transparency_tool.routes.graph import build_full_graph
            graph = build_full_graph(self.model, batch_i=0, renormalizing_threshold=0.04)
            self.assertIsNotNone(graph)
            self.assertGreater(graph.number_of_nodes(), 0)
            self.assertGreater(graph.number_of_edges(), 0)
        finally:
            if os.path.isfile(path):
                os.unlink(path)
