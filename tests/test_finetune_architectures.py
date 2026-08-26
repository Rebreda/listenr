"""Tests for architecture detection and the Whisper/Moonshine differences.

The end-to-end case builds a randomly-initialised Moonshine from a tiny config
so it runs offline: it proves the collator's output actually feeds the model,
which is the part a mock cannot check.
"""

import numpy as np
import pytest

from listenr.finetune.architectures import (
    MOONSHINE,
    SUPPORTED,
    WHISPER,
    UnsupportedArchitecture,
    detect,
)


class TestArchitectureTable:
    def test_every_supported_family_is_registered(self):
        assert set(SUPPORTED) == {"whisper", "moonshine", "moonshine_streaming"}

    def test_whisper_takes_prepadded_log_mel(self):
        assert WHISPER.feature_key == "input_features"
        assert WHISPER.pad_features is False

    def test_moonshine_takes_variable_length_waveform(self):
        assert MOONSHINE.feature_key == "input_values"
        assert MOONSHINE.pad_features is True

    def test_only_whisper_carries_language_and_task_tokens(self):
        assert WHISPER.supports_language_and_task is True
        assert MOONSHINE.supports_language_and_task is False

    def test_architecture_is_immutable(self):
        with pytest.raises(Exception):
            WHISPER.feature_key = "nope"  # type: ignore[misc]


class TestDetect:
    """Detection falls back to the model id when no config can be fetched."""

    @staticmethod
    def _model_type(monkeypatch, value):
        """Stub the config lookup. transformers is an optional dependency, so
        the seam is patched rather than transformers itself."""
        monkeypatch.setattr(
            "listenr.finetune.architectures.model_type_of", lambda model_id: value
        )

    @pytest.mark.parametrize(
        "model_id, expected",
        [
            ("openai/whisper-small", WHISPER),
            ("openai/whisper-large-v3-turbo", WHISPER),
            ("UsefulSensors/moonshine-base", MOONSHINE),
            ("UsefulSensors/moonshine-tiny", MOONSHINE),
        ],
    )
    def test_detects_from_name_without_network(self, model_id, expected, monkeypatch):
        self._model_type(monkeypatch, None)
        assert detect(model_id) is expected

    def test_config_model_type_wins_over_the_name(self, monkeypatch):
        self._model_type(monkeypatch, "moonshine")
        # A local directory name that says nothing about the family.
        assert detect("/tmp/my-merged-model") is MOONSHINE

    def test_unsupported_model_type_is_a_clear_error(self, monkeypatch):
        self._model_type(monkeypatch, "wav2vec2")
        with pytest.raises(UnsupportedArchitecture, match="wav2vec2"):
            detect("facebook/wav2vec2-base")

    def test_unrecognisable_name_is_a_clear_error(self, monkeypatch):
        self._model_type(monkeypatch, None)
        with pytest.raises(UnsupportedArchitecture) as exc:
            detect("some-org/mystery-asr")
        # The error must list what is supported, whatever that set becomes.
        for family in SUPPORTED:
            assert family in str(exc.value)


class TestMoonshineBatchFeedsTheModel:
    """A collated Moonshine batch must be accepted by a real Moonshine forward."""

    @staticmethod
    def _tiny_model():
        transformers = pytest.importorskip("transformers")
        pytest.importorskip("torch")
        cfg = transformers.MoonshineConfig(
            hidden_size=32,
            intermediate_size=64,
            encoder_num_hidden_layers=1,
            decoder_num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=64,
            # Real checkpoints define these; a bare config does not, and the
            # decoder needs both to build its shifted inputs.
            pad_token_id=0,
            decoder_start_token_id=1,
        )
        return transformers.MoonshineForConditionalGeneration(cfg)

    @staticmethod
    def _processor():
        transformers = pytest.importorskip("transformers")
        return transformers.Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=16_000,
            padding_value=0.0,
            return_attention_mask=True,
        )

    def test_variable_length_clips_pad_to_the_longest(self):
        from listenr.finetune.data import SpeechDataCollator

        torch = pytest.importorskip("torch")
        feature_extractor = self._processor()

        class Processor:
            pass

        processor = Processor()
        processor.feature_extractor = feature_extractor
        processor.tokenizer = _StubTokenizer(torch)

        collator = SpeechDataCollator(
            processor=processor,
            decoder_start_token_id=1,
            feature_key="input_values",
            pad_features=True,
        )
        batch = collator(
            [
                {"input_values": np.zeros(8_000, dtype="float32"), "labels": [5, 6]},
                {"input_values": np.zeros(4_000, dtype="float32"), "labels": [7]},
            ]
        )

        assert batch["input_values"].shape == (2, 8_000)
        assert "attention_mask" in batch
        # The shorter clip is masked out past its real length.
        assert batch["attention_mask"][1, 4_000:].sum().item() == 0

    def test_forward_pass_accepts_the_collated_batch(self):
        torch = pytest.importorskip("torch")
        model = self._tiny_model()
        feature_extractor = self._processor()

        from listenr.finetune.data import SpeechDataCollator

        class Processor:
            pass

        processor = Processor()
        processor.feature_extractor = feature_extractor
        processor.tokenizer = _StubTokenizer(torch)

        collator = SpeechDataCollator(
            processor=processor,
            decoder_start_token_id=model.config.decoder_start_token_id,
            feature_key="input_values",
            pad_features=True,
        )
        batch = collator(
            [
                {"input_values": np.zeros(8_000, dtype="float32"), "labels": [5, 6]},
                {"input_values": np.zeros(6_000, dtype="float32"), "labels": [7, 8]},
            ]
        )

        with torch.no_grad():
            out = model(**batch)

        assert out.loss is not None
        assert torch.isfinite(out.loss)


class _StubTokenizer:
    """Minimal stand-in for the label half of a processor."""

    def __init__(self, torch):
        self._torch = torch

    def pad(self, encoded, return_tensors=None):
        rows = [item["input_ids"] for item in encoded]
        width = max(len(r) for r in rows)
        ids = self._torch.tensor([r + [0] * (width - len(r)) for r in rows])
        mask = self._torch.tensor([[1] * len(r) + [0] * (width - len(r)) for r in rows])

        class Padded:
            pass

        padded = Padded()
        padded.input_ids = ids
        padded.attention_mask = mask
        return padded


class TestPadToken:
    """Label padding needs a pad token; not every checkpoint ships one."""

    @staticmethod
    def _processor(pad_token=None, eos_token=None):
        class Tokenizer:
            def __init__(self):
                self.pad_token = pad_token
                self.eos_token = eos_token

            def convert_ids_to_tokens(self, token_id):
                return f"<tok-{token_id}>"

        class Processor:
            pass

        processor = Processor()
        processor.tokenizer = Tokenizer()
        return processor

    def test_existing_pad_token_is_left_alone(self):
        from listenr.finetune.data import _ensure_pad_token

        processor = self._processor(pad_token="<pad>", eos_token="<eos>")
        _ensure_pad_token(processor, "openai/whisper-small")
        assert processor.tokenizer.pad_token == "<pad>"

    def test_falls_back_to_eos(self):
        from listenr.finetune.data import _ensure_pad_token

        processor = self._processor(eos_token="<eos>")
        _ensure_pad_token(processor, "openai/whisper-small")
        assert processor.tokenizer.pad_token == "<eos>"

    def test_falls_back_to_the_config_pad_token_id(self, monkeypatch):
        """Moonshine's tokenizer exposes no special tokens; its config does."""
        from listenr.finetune.data import _ensure_pad_token

        class FakeConfig:
            pad_token_id = 2
            eos_token_id = 2

        monkeypatch.setattr(
            "listenr.finetune.data._load_config", lambda model_id: FakeConfig()
        )
        processor = self._processor()
        _ensure_pad_token(processor, "UsefulSensors/moonshine-tiny")
        assert processor.tokenizer.pad_token == "<tok-2>"

    def test_no_pad_token_anywhere_is_not_fatal(self, monkeypatch):
        from listenr.finetune.data import _ensure_pad_token

        monkeypatch.setattr("listenr.finetune.data._load_config", lambda model_id: None)
        processor = self._processor()
        _ensure_pad_token(processor, "some-org/mystery-asr")
        assert processor.tokenizer.pad_token is None


class TestMoonshineStreaming:
    """The streaming variant is a separate model_type, not a flag on the offline one."""

    def test_registered(self):
        from listenr.finetune.architectures import MOONSHINE_STREAMING, SUPPORTED

        assert SUPPORTED["moonshine_streaming"] is MOONSHINE_STREAMING

    def test_takes_the_raw_waveform_like_offline_moonshine(self):
        from listenr.finetune.architectures import MOONSHINE, MOONSHINE_STREAMING

        assert MOONSHINE_STREAMING.feature_key == MOONSHINE.feature_key == "input_values"
        assert MOONSHINE_STREAMING.pad_features is True

    def test_pads_to_the_frame_size(self):
        """The encoder reshapes to [batch, -1, 80] and raises on anything else."""
        from listenr.finetune.architectures import MOONSHINE_STREAMING

        assert MOONSHINE_STREAMING.pad_to_multiple == 80

    def test_only_streaming_needs_frame_alignment(self):
        from listenr.finetune.architectures import MOONSHINE, WHISPER

        assert WHISPER.pad_to_multiple is None
        assert MOONSHINE.pad_to_multiple is None

    def test_name_detection_prefers_the_more_specific_family(self, monkeypatch):
        """"moonshine-streaming-small" contains "moonshine"; longest must win."""
        from listenr.finetune.architectures import MOONSHINE_STREAMING, detect

        monkeypatch.setattr(
            "listenr.finetune.architectures.model_type_of", lambda model_id: None
        )
        assert detect("moonshine-ai/moonshine-streaming-small") is MOONSHINE_STREAMING

    def test_offline_moonshine_still_resolves(self, monkeypatch):
        from listenr.finetune.architectures import MOONSHINE, detect

        monkeypatch.setattr(
            "listenr.finetune.architectures.model_type_of", lambda model_id: None
        )
        assert detect("UsefulSensors/moonshine-base") is MOONSHINE


class TestFrameAlignedCollation:
    """A batch padded to the longest clip lands on a frame boundary only by luck."""

    @staticmethod
    def _collator(pad_to_multiple):
        # Needs the finetune extras; CI installs only the dev extra.
        torch = pytest.importorskip("torch")
        transformers = pytest.importorskip("transformers")
        Wav2Vec2FeatureExtractor = transformers.Wav2Vec2FeatureExtractor

        from listenr.finetune.data import SpeechDataCollator

        class Processor:
            pass

        processor = Processor()
        processor.feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1, sampling_rate=16_000, padding_value=0.0, return_attention_mask=True
        )

        class Tokenizer:
            def pad(self, encoded, return_tensors=None):
                rows = [i["input_ids"] for i in encoded]
                width = max(len(r) for r in rows)
                out = type("P", (), {})()
                out.input_ids = torch.tensor([r + [0] * (width - len(r)) for r in rows])
                out.attention_mask = torch.tensor(
                    [[1] * len(r) + [0] * (width - len(r)) for r in rows]
                )
                return out

        processor.tokenizer = Tokenizer()
        return SpeechDataCollator(
            processor=processor,
            decoder_start_token_id=1,
            feature_key="input_values",
            pad_features=True,
            pad_to_multiple=pad_to_multiple,
        )

    def test_batch_is_padded_up_to_the_frame_size(self):
        batch = self._collator(80)(
            [
                {"input_values": np.zeros(16_037, dtype="float32"), "labels": [5]},
                {"input_values": np.zeros(9_000, dtype="float32"), "labels": [6]},
            ]
        )
        length = batch["input_values"].shape[1]
        assert length % 80 == 0
        assert length >= 16_037

    def test_without_it_the_batch_keeps_a_ragged_length(self):
        batch = self._collator(None)(
            [
                {"input_values": np.zeros(16_037, dtype="float32"), "labels": [5]},
                {"input_values": np.zeros(9_000, dtype="float32"), "labels": [6]},
            ]
        )
        assert batch["input_values"].shape[1] == 16_037
