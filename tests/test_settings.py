"""Unit tests for listenr.settings.

Covers: defaults, TOML file loading (via LISTENR_CONFIG), environment-variable
overrides, precedence, validation errors, and smoke-imports of the consumers.
"""

from pathlib import Path

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

    def test_section_named_var_holding_a_path_is_ignored(self, isolated_env, monkeypatch):
        """A stale LISTENR_FINETUNE=<host path> must not crash the import."""
        monkeypatch.setenv("LISTENR_FINETUNE", "/home/you/listenr_finetune")
        with pytest.warns(RuntimeWarning, match="LISTENR_FINETUNE"):
            s = Settings()
        assert s.finetune.max_steps == Settings.model_fields["finetune"].default.max_steps

    def test_section_named_var_holding_json_still_applies(self, isolated_env, monkeypatch):
        monkeypatch.setenv("LISTENR_FINETUNE", '{"max_steps": 77}')
        assert Settings().finetune.max_steps == 77

    def test_stale_dotenv_is_ignored(self, isolated_env, monkeypatch, tmp_path):
        (tmp_path / ".env").write_text("LISTENR_DATASET=/home/you/listenr_dataset\n")
        monkeypatch.chdir(tmp_path)
        with pytest.warns(RuntimeWarning, match="LISTENR_DATASET"):
            assert Settings().dataset.split == "80/10/10"


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


class TestExampleConfig:
    """examples/config.toml must stay in lockstep with the code defaults."""

    EXAMPLE = Path(__file__).parent.parent / "examples" / "config.toml"

    def test_example_equals_defaults(self, isolated_env, monkeypatch):
        defaults = Settings()
        monkeypatch.setenv("LISTENR_CONFIG", str(self.EXAMPLE))
        assert Settings() == defaults

    def test_example_covers_every_setting(self):
        import tomllib

        with open(self.EXAMPLE, "rb") as f:
            data = tomllib.load(f)

        for section, field in Settings.model_fields.items():
            if section == "corrections":
                continue  # free-form table, covered by the equality test
            assert section in data, f"section [{section}] missing from example"
            model_keys = set(field.default.__class__.model_fields)
            file_keys = set(data[section])
            assert file_keys == model_keys, (
                f"[{section}] drift — missing {model_keys - file_keys}, "
                f"stale {file_keys - model_keys}"
            )


class TestConsumersImport:
    """The migrated modules must import cleanly against the settings object."""

    def test_smoke_imports(self):
        import listenr.build_dataset  # noqa: F401
        import listenr.categorize  # noqa: F401
        import listenr.importers.manifest  # noqa: F401
        import listenr.llm_processor  # noqa: F401
        import listenr.storage  # noqa: F401
        import listenr.transcript_utils  # noqa: F401


class TestStaleIniWarning:
    """An inert config.ini is worse than no config at all: it looks configured."""

    def test_warns_when_ini_exists_and_toml_does_not(self, monkeypatch, tmp_path):
        (tmp_path / "config.ini").write_text("[llm]\napi_base = http://localhost:8080/api/v1\n")
        monkeypatch.setenv("LISTENR_CONFIG", str(tmp_path / "config.toml"))
        with pytest.warns(RuntimeWarning, match="TOML only"):
            Settings()

    def test_silent_when_the_toml_exists(self, monkeypatch, tmp_path):
        (tmp_path / "config.ini").write_text("[llm]\n")
        toml = tmp_path / "config.toml"
        toml.write_text("[whisper]\nmodel = \"Whisper-Base\"\n")
        monkeypatch.setenv("LISTENR_CONFIG", str(toml))
        import warnings as w

        with w.catch_warnings():
            w.simplefilter("error")
            assert Settings().whisper.model == "Whisper-Base"

    def test_silent_when_there_is_no_ini(self, monkeypatch, tmp_path):
        monkeypatch.setenv("LISTENR_CONFIG", str(tmp_path / "config.toml"))
        import warnings as w

        with w.catch_warnings():
            w.simplefilter("error")
            Settings()
