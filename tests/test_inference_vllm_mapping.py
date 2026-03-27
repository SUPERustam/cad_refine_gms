"""Tests for ``infer.py`` vLLM path: HF kwargs → vLLM ``SamplingParams``."""

import pytest

pytest.importorskip("vllm")

from cad_rl.pipelines.inference_vllm import hf_generate_kwargs_to_vllm_sampling_params


def test_hf_to_vllm_sampling_params_greedy_and_stops():
    class Tok:
        def decode(self, ids, skip_special_tokens=False):
            return "<x>"

    sp = hf_generate_kwargs_to_vllm_sampling_params(
        Tok(),
        {
            "max_new_tokens": 512,
            "do_sample": False,
            "temperature": 0.9,
            "top_p": 0.95,
            "top_k": 40,
            "repetition_penalty": 1.05,
            "eos_token_id": 151645,
            "pad_token_id": 0,
        },
    )
    assert sp.max_tokens == 512
    assert sp.temperature == 0.0
    assert sp.top_k == 40
    assert sp.repetition_penalty == 1.05
    assert sp.stop_token_ids == [151645]


def test_hf_to_vllm_sampling_params_respects_top_p_when_sampling():
    class Tok:
        def decode(self, ids, skip_special_tokens=False):
            return "<x>"

    sp = hf_generate_kwargs_to_vllm_sampling_params(
        Tok(),
        {
            "max_new_tokens": 128,
            "do_sample": True,
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 20,
            "repetition_penalty": 1.0,
        },
    )
    assert sp.temperature == 0.8
    assert sp.top_p == 0.95
    assert sp.top_k == 20


def test_hf_to_vllm_sampling_params_bad_words():
    class Tok:
        def decode(self, ids, skip_special_tokens=False):
            return {7: "a", 8: "bc"}.get(ids[0], "?")

    sp = hf_generate_kwargs_to_vllm_sampling_params(
        Tok(),
        {
            "max_new_tokens": 64,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 1.0,
            "top_k": -1,
            "repetition_penalty": 1.0,
            "bad_words_ids": [[7], [8]],
        },
    )
    assert sp.bad_words == ["a", "bc"]
