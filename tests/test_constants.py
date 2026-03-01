"""
Unit tests for listenr.constants.

Verifies:
  - Every public constant exists and has the expected Python type.
  - Path constants are absolute (already expanded).
  - Numeric constants are within sane ranges.
  - reload() correctly picks up config changes made at runtime.
  - Constants are consumed by the modules that use them (smoke-import checks).
"""

import importlib
from pathlib import Path

import pytest

import listenr.constants as C


# ---------------------------------------------------------------------------
# Type checks
# ---------------------------------------------------------------------------

class TestConstantTypes:
    def test_lemonade_api_base_is_str(self):
        assert isinstance(C.LEMONADE_API_BASE, str)

    def test_whisper_model_is_str(self):
        assert isinstance(C.WHISPER_MODEL, str)

    def test_capture_rate_is_int(self):
        assert isinstance(C.CAPTURE_RATE, int)

    def test_asr_rate_is_int(self):
        assert isinstance(C.ASR_RATE, int)

    def test_channels_is_int(self):
        assert isinstance(C.CHANNELS, int)

    def test_chunk_size_is_int(self):
        assert isinstance(C.CHUNK_SIZE, int)

    def test_input_device_is_str_or_none(self):
        assert C.INPUT_DEVICE is None or isinstance(C.INPUT_DEVICE, str)

    def test_storage_base_is_path(self):
        assert isinstance(C.STORAGE_BASE, Path)

    def test_storage_clips_enabled_is_bool(self):
        assert isinstance(C.STORAGE_CLIPS_ENABLED, bool)

    def test_storage_retention_days_is_int(self):
        assert isinstance(C.STORAGE_RETENTION_DAYS, int)

    def test_storage_max_gb_is_float(self):
        assert isinstance(C.STORAGE_MAX_GB, float)

    def test_vad_threshold_is_float(self):
        assert isinstance(C.VAD_THRESHOLD, float)

    def test_vad_silence_ms_is_int(self):
        assert isinstance(C.VAD_SILENCE_MS, int)

    def test_vad_prefix_padding_ms_is_int(self):
        assert isinstance(C.VAD_PREFIX_PADDING_MS, int)

    def test_llm_enabled_is_bool(self):
        assert isinstance(C.LLM_ENABLED, bool)

    def test_llm_model_is_str(self):
        assert isinstance(C.LLM_MODEL, str)

    def test_llm_api_base_is_str(self):
        assert isinstance(C.LLM_API_BASE, str)

    def test_llm_temperature_is_float(self):
        assert isinstance(C.LLM_TEMPERATURE, float)

    def test_llm_max_tokens_is_int(self):
        assert isinstance(C.LLM_MAX_TOKENS, int)

    def test_llm_timeout_is_int(self):
        assert isinstance(C.LLM_TIMEOUT, int)

    def test_llm_context_window_is_int(self):
        assert isinstance(C.LLM_CONTEXT_WINDOW, int)

    def test_dataset_output_is_path(self):
        assert isinstance(C.DATASET_OUTPUT, Path)

    def test_dataset_split_is_str(self):
        assert isinstance(C.DATASET_SPLIT, str)

    def test_dataset_min_duration_is_float(self):
        assert isinstance(C.DATASET_MIN_DURATION, float)

    def test_dataset_min_chars_is_int(self):
        assert isinstance(C.DATASET_MIN_CHARS, int)

    def test_dataset_seed_is_int(self):
        assert isinstance(C.DATASET_SEED, int)

    def test_dataset_format_is_str(self):
        assert isinstance(C.DATASET_FORMAT, str)

    def test_output_file_is_path_or_none(self):
        assert C.OUTPUT_FILE is None or isinstance(C.OUTPUT_FILE, Path)

    def test_output_llm_file_is_path_or_none(self):
        assert C.OUTPUT_LLM_FILE is None or isinstance(C.OUTPUT_LLM_FILE, Path)

    def test_output_line_format_is_str(self):
        assert isinstance(C.OUTPUT_LINE_FORMAT, str)

    def test_output_timestamp_format_is_str(self):
        assert isinstance(C.OUTPUT_TIMESTAMP_FORMAT, str)

    def test_output_show_raw_is_bool(self):
        assert isinstance(C.OUTPUT_SHOW_RAW, bool)

    def test_log_level_is_str(self):
        assert isinstance(C.LOG_LEVEL, str)

    def test_log_file_is_path_or_none(self):
        assert C.LOG_FILE is None or isinstance(C.LOG_FILE, Path)


# ---------------------------------------------------------------------------
# Value sanity checks
# ---------------------------------------------------------------------------

class TestConstantValues:
    def test_asr_rate_always_16000(self):
        """ASR_RATE is fixed — Lemonade /realtime always requires 16 kHz."""
        assert C.ASR_RATE == 16000

    def test_capture_rate_positive(self):
        assert C.CAPTURE_RATE > 0

    def test_chunk_size_positive(self):
        assert C.CHUNK_SIZE > 0

    def test_channels_at_least_one(self):
        assert C.CHANNELS >= 1

    def test_vad_threshold_between_0_and_1(self):
        assert 0.0 < C.VAD_THRESHOLD < 1.0

    def test_vad_silence_ms_positive(self):
        assert C.VAD_SILENCE_MS > 0

    def test_vad_prefix_padding_ms_non_negative(self):
        assert C.VAD_PREFIX_PADDING_MS >= 0

    def test_llm_temperature_in_range(self):
        assert 0.0 <= C.LLM_TEMPERATURE <= 2.0

    def test_llm_max_tokens_positive(self):
        assert C.LLM_MAX_TOKENS > 0

    def test_llm_timeout_positive(self):
        assert C.LLM_TIMEOUT > 0

    def test_llm_context_window_positive(self):
        assert C.LLM_CONTEXT_WINDOW > 0

    def test_dataset_split_parses_as_three_ints(self):
        parts = C.DATASET_SPLIT.split('/')
        assert len(parts) == 3, f"Expected 3 parts, got: {C.DATASET_SPLIT!r}"
        assert all(p.strip().isdigit() for p in parts)

    def test_dataset_split_sums_to_100(self):
        parts = [int(p) for p in C.DATASET_SPLIT.split('/')]
        assert sum(parts) == 100, f"Split must sum to 100, got {sum(parts)}"

    def test_dataset_min_duration_non_negative(self):
        assert C.DATASET_MIN_DURATION >= 0.0

    def test_dataset_min_chars_non_negative(self):
        assert C.DATASET_MIN_CHARS >= 0

    def test_dataset_format_valid(self):
        assert C.DATASET_FORMAT in {'csv', 'hf', 'parquet'}

    def test_lemonade_api_base_starts_with_http(self):
        assert C.LEMONADE_API_BASE.startswith('http')

    def test_llm_api_base_starts_with_http(self):
        assert C.LLM_API_BASE.startswith('http')

    def test_storage_base_is_absolute(self):
        assert C.STORAGE_BASE.is_absolute()

    def test_dataset_output_is_absolute(self):
        assert C.DATASET_OUTPUT.is_absolute()

    def test_whisper_model_non_empty(self):
        assert C.WHISPER_MODEL.strip() != ''

    def test_llm_model_non_empty(self):
        assert C.LLM_MODEL.strip() != ''

    def test_log_level_is_valid(self):
        import logging
        assert C.LOG_LEVEL.upper() in {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}

    def test_output_line_format_contains_text_placeholder(self):
        assert '{text}' in C.OUTPUT_LINE_FORMAT

    def test_retention_days_positive(self):
        assert C.STORAGE_RETENTION_DAYS > 0

    def test_storage_max_gb_positive(self):
        assert C.STORAGE_MAX_GB > 0


# ---------------------------------------------------------------------------
# reload() smoke test — checks the mechanism works without real config change
# ---------------------------------------------------------------------------

class TestReload:
    def test_reload_returns_same_types(self):
        """reload() should succeed and constants should still have correct types."""
        C.reload()
        # Re-import to get updated module state
        importlib.invalidate_caches()
        import listenr.constants as C2
        assert isinstance(C2.CAPTURE_RATE, int)
        assert isinstance(C2.LLM_MODEL, str)
        assert isinstance(C2.STORAGE_BASE, Path)
        assert isinstance(C2.DATASET_MIN_DURATION, float)

    def test_reload_preserves_asr_rate(self):
        """ASR_RATE must remain 16000 regardless of any reload."""
        C.reload()
        import listenr.constants as C2
        assert C2.ASR_RATE == 16000

    def test_reload_with_patched_config(self, monkeypatch):
        """Constants module attribute reflects patched config after reload()."""
        import listenr.config_manager as cfg_mod
        monkeypatch.setattr(
            cfg_mod, 'get_int_setting',
            lambda section, key, fallback=0: 9999 if (section, key) == ('LLM', 'max_tokens') else fallback,
        )
        C.reload()
        import listenr.constants as C2
        assert C2.LLM_MAX_TOKENS == 9999

    def test_reload_restores_after_patch(self, monkeypatch):
        """After monkeypatch teardown + reload, value returns to real config."""
        import listenr.config_manager as cfg_mod
        original_fn = cfg_mod.get_int_setting
        monkeypatch.setattr(
            cfg_mod, 'get_int_setting',
            lambda section, key, fallback=0: 1 if (section, key) == ('LLM', 'context_window') else original_fn(section, key, fallback),
        )
        C.reload()
        import listenr.constants as C2
        assert C2.LLM_CONTEXT_WINDOW == 1

        # monkeypatch teardown restores original_fn automatically;
        # call reload once more here to re-read with real config
        monkeypatch.undo()
        C.reload()
        import listenr.constants as C3
        assert isinstance(C3.LLM_CONTEXT_WINDOW, int)


# ---------------------------------------------------------------------------
# Smoke imports — ensure migrated modules still import cleanly
# ---------------------------------------------------------------------------

class TestModuleImports:
    def test_cli_imports_constants(self):
        import listenr.cli  # noqa: F401 — just verify no ImportError

    def test_llm_processor_imports_constants(self):
        import listenr.llm_processor  # noqa: F401

    def test_unified_asr_imports_constants(self):
        import listenr.unified_asr  # noqa: F401

    def test_build_dataset_imports_constants(self):
        import listenr.build_dataset  # noqa: F401

    def test_build_dataset_defaults_match_constants(self):
        import listenr.build_dataset as bd
        assert bd.DEFAULT_OUTPUT == C.DATASET_OUTPUT
        assert bd.DEFAULT_SPLIT == C.DATASET_SPLIT
        assert bd.DEFAULT_MIN_DURATION == C.DATASET_MIN_DURATION
        assert bd.DEFAULT_MIN_CHARS == C.DATASET_MIN_CHARS
        assert bd.DEFAULT_SEED == C.DATASET_SEED
        assert bd.DEFAULT_FORMAT == C.DATASET_FORMAT


# ---------------------------------------------------------------------------
# No stale cfg.get_* calls in migrated modules (grep-based AST-free check)
# ---------------------------------------------------------------------------

class TestNoCfgCallsInMigratedModules:
    """
    Ensure migrated modules do not contain inline cfg.get_*_setting() calls
    that duplicate what constants.py already exposes.  We allow cfg.get_setting
    ONLY inside _api_base() in llm_processor (URL may be overridden at runtime).
    """

    def _source(self, module_name: str) -> str:
        mod = importlib.import_module(module_name)
        import inspect
        return inspect.getsource(mod)

    def test_cli_has_no_inline_cfg_get_calls(self):
        src = self._source('listenr.cli')
        # cli.py should contain no cfg.get_*_setting() calls at all
        import re
        calls = re.findall(r'cfg\.get_\w+_setting\(', src)
        assert calls == [], f"cli.py still has inline cfg calls: {calls}"

    def test_build_dataset_has_no_inline_cfg_get_calls(self):
        src = self._source('listenr.build_dataset')
        import re
        calls = re.findall(r'cfg\.get_\w+_setting\(', src)
        assert calls == [], f"build_dataset.py still has inline cfg calls: {calls}"

    def test_unified_asr_has_no_inline_cfg_get_calls(self):
        src = self._source('listenr.unified_asr')
        import re
        calls = re.findall(r'cfg\.get_\w+_setting\(', src)
        assert calls == [], f"unified_asr.py still has inline cfg calls: {calls}"

    def test_llm_processor_cfg_calls_only_in_api_base(self):
        src = self._source('listenr.llm_processor')
        import re
        # Find all lines with cfg.get_*_setting
        lines_with_cfg = [
            (i + 1, line.strip())
            for i, line in enumerate(src.splitlines())
            if re.search(r'cfg\.get_\w+_setting\(', line)
        ]
        for lineno, line in lines_with_cfg:
            assert '_api_base' in src.splitlines()[lineno - 2] or 'def _api_base' in line or '_api_base' in line, (
                f"llm_processor.py has unexpected cfg call outside _api_base at line ~{lineno}: {line!r}"
            )
