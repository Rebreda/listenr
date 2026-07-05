"""Unit tests for listenr.settings.

Covers: defaults, TOML file loading (via LISTENR_CONFIG), environment-variable
overrides, precedence, validation errors, and smoke-imports of the consumers.
"""

import pytest
from pydantic import ValidationError

from listenr.settings import ASR_RATE, DEFAULT_CORRECTIONS, Settings


@pytest.fixture
def isolated_env(monkeypatch, tmp_path):
    """Point LISTENR_CONFIG at an (initially missing) TOML in tmp_path."""
    config_file = tmp_path / "config.toml"
    monkeypatch.setenv("LISTENR_CONFIG", str(config_file))
    return config_file


class TestDefaults:
    def test_sane_defaults_without_config_file(self, isolated_env):
        s = Settings()
        assert s.whisper.model == "Whisper-Base"
        assert s.audio.sample_rate == 48_000
        assert s.vad.silence_duration_ms == 800
        assert s.llm.enabled is True
        assert s.dataset.split == "80/10/10"
        assert s.finetune.lora_target_modules == ["q_proj", "v_proj"]
        assert s.corrections == DEFAULT_CORRECTIONS

    def test_asr_rate_is_fixed(self):
        assert ASR_RATE == 16_000

    def test_path_defaults_are_absolute(self, isolated_env):
        s = Settings()
        assert s.storage.audio_clips_path.is_absolute()
        assert s.dataset.output_path.is_absolute()
        assert s.finetune.output_dir.is_absolute()


class TestTomlSource:
    def test_toml_values_override_defaults(self, isolated_env):
        isolated_env.write_text(
            '[finetune]\nbase_model = "openai/whisper-tiny"\nmax_steps = 42\n'
            '[vad]\nthreshold = 0.11\n'
        )
        s = Settings()
        assert s.finetune.base_model == "openai/whisper-tiny"
        assert s.finetune.max_steps == 42
        assert s.vad.threshold == 0.11
        # Untouched keys keep their defaults.
        assert s.finetune.lora_r == 8

    def test_toml_paths_are_expanded(self, isolated_env):
        isolated_env.write_text('[dataset]\noutput_path = "~/somewhere"\n')
        s = Settings()
        assert "~" not in str(s.dataset.output_path)
        assert s.dataset.output_path.is_absolute()

    def test_corrections_can_be_replaced(self, isolated_env):
        isolated_env.write_text('[corrections]\nkubernetes = "Kubernetes"\n')
        s = Settings()
        assert s.corrections == {"kubernetes": "Kubernetes"}

    def test_unknown_sections_are_ignored(self, isolated_env):
        isolated_env.write_text('[not_a_real_section]\nfoo = "bar"\n')
        s = Settings()
        assert s.whisper.model == "Whisper-Base"


class TestEnvSource:
    def test_env_overrides_default(self, isolated_env, monkeypatch):
        monkeypatch.setenv("LISTENR_FINETUNE__MAX_STEPS", "123")
        assert Settings().finetune.max_steps == 123

    def test_env_overrides_toml(self, isolated_env, monkeypatch):
        isolated_env.write_text("[finetune]\nmax_steps = 42\n")
        monkeypatch.setenv("LISTENR_FINETUNE__MAX_STEPS", "123")
        assert Settings().finetune.max_steps == 123

    def test_lora_target_modules_accepts_csv(self, isolated_env, monkeypatch):
        monkeypatch.setenv("LISTENR_FINETUNE__LORA_TARGET_MODULES", "q_proj,k_proj,v_proj")
        assert Settings().finetune.lora_target_modules == ["q_proj", "k_proj", "v_proj"]


class TestValidation:
    def test_bad_int_is_a_loud_error(self, isolated_env):
        isolated_env.write_text('[finetune]\nmax_steps = "twohundred"\n')
        with pytest.raises(ValidationError):
            Settings()

    def test_bad_dataset_format_rejected(self, isolated_env):
        isolated_env.write_text('[dataset]\nformat = "parquet"\n')
        with pytest.raises(ValidationError):
            Settings()

    def test_input_device_default_maps_to_none(self, isolated_env):
        isolated_env.write_text('[audio]\ninput_device = "default"\n')
        assert Settings().audio.input_device is None


class TestConsumersImport:
    """The migrated modules must import cleanly against the settings object."""

    def test_smoke_imports(self):
        import listenr.build_dataset  # noqa: F401
        import listenr.categorize  # noqa: F401
        import listenr.importers.manifest  # noqa: F401
        import listenr.llm_processor  # noqa: F401
        import listenr.storage  # noqa: F401
        import listenr.transcript_utils  # noqa: F401
