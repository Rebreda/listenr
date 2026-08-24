"""Tests for the pre-flight checks.

The point of these checks is that they fire before anything expensive loads, so
the tests construct Accelerator values directly rather than depending on
whatever hardware the test runner happens to have.
"""

import pytest

from listenr.finetune.preflight import (
    Accelerator,
    check_precision,
    check_step_schedule,
    check_torch_build,
    describe_accelerator,
    describe_accelerator_line,
    format_problems,
)


def _acc(build=None, available=False, amd_present=False, device_name=None):
    return Accelerator(
        torch_version="2.12.0+cu130" if build == "cuda" else "2.12.0",
        build=build,
        available=available,
        device_name=device_name,
        amd_present=amd_present,
    )


class TestTorchBuild:
    def test_cuda_wheel_on_an_amd_box_is_an_error(self):
        """The headline AMD bug: pip resolves a CUDA wheel and it imports fine."""
        problems = check_torch_build(_acc(build="cuda", available=False, amd_present=True))
        assert len(problems) == 1
        assert problems[0].severity == "error"
        assert "AMD GPU" in problems[0].message
        assert "rocm" in problems[0].message.lower()

    def test_rocm_wheel_that_sees_nothing_suggests_the_isa_override(self):
        problems = check_torch_build(_acc(build="rocm", available=False, amd_present=True))
        assert len(problems) == 1
        assert problems[0].severity == "error"
        # 11.0.0 masquerades as gfx1100. Setting gfx1151 to its own value does nothing.
        assert "11.0.0" in problems[0].message

    def test_working_gpu_is_silent(self):
        assert check_torch_build(_acc(build="rocm", available=True, amd_present=True)) == []

    def test_no_gpu_on_a_non_amd_box_is_only_a_warning(self):
        problems = check_torch_build(_acc(build=None, available=False, amd_present=False))
        assert [p.severity for p in problems] == ["warning"]

    def test_missing_torch_is_an_error(self):
        acc = Accelerator("not installed", None, False, None, False)
        problems = check_torch_build(acc)
        assert problems[0].severity == "error"
        assert "listenr[finetune]" in problems[0].message


class TestPrecision:
    def test_bf16_without_a_gpu_is_an_error(self):
        """transformers raises 'Your setup doesn't support bf16/gpu' at build time."""
        problems = check_precision(fp16=False, bf16=True, acc=_acc(available=False))
        assert any(p.severity == "error" and "--bf16" in p.message for p in problems)

    def test_error_names_the_flag_that_turns_it_off(self):
        problems = check_precision(fp16=False, bf16=True, acc=_acc(available=False))
        assert any("--no-bf16" in p.message for p in problems)

    def test_both_precisions_at_once_is_an_error(self):
        problems = check_precision(fp16=True, bf16=True, acc=_acc(available=True, build="rocm"))
        assert any("mutually exclusive" in p.message for p in problems)

    def test_bf16_with_a_gpu_is_fine(self):
        assert check_precision(False, True, _acc(build="rocm", available=True)) == []

    def test_fp16_on_rocm_warns_but_does_not_block(self):
        problems = check_precision(True, False, _acc(build="rocm", available=True))
        assert [p.severity for p in problems] == ["warning"]


class TestStepSchedule:
    def test_defaults_are_valid(self):
        assert check_step_schedule(eval_steps=200, save_steps=400) == []

    def test_non_multiple_is_an_error(self):
        """transformers rejects this, but only after the model has loaded."""
        problems = check_step_schedule(eval_steps=150, save_steps=400)
        assert len(problems) == 1
        assert "round multiple" not in problems[0].message  # ours is plainer
        assert "whole multiple" in problems[0].message

    def test_error_suggests_a_workable_value(self):
        problems = check_step_schedule(eval_steps=150, save_steps=400)
        assert "450" in problems[0].message

    def test_skipped_when_not_loading_the_best_model(self):
        assert check_step_schedule(150, 400, load_best_model_at_end=False) == []

    @pytest.mark.parametrize("eval_steps, save_steps", [(0, 400), (200, 0), (-1, 400)])
    def test_non_positive_steps_are_an_error(self, eval_steps, save_steps):
        problems = check_step_schedule(eval_steps, save_steps)
        assert problems and problems[0].severity == "error"


class TestDescribe:
    def test_describe_accelerator_never_raises(self):
        acc = describe_accelerator()
        assert isinstance(acc.amd_present, bool)
        assert isinstance(acc.torch_version, str)

    def test_line_names_the_device_when_there_is_one(self):
        line = describe_accelerator_line(_acc(build="rocm", available=True, device_name="Radeon 8060S"))
        assert "Radeon 8060S" in line

    def test_line_says_cpu_when_there_is_not(self):
        assert "CPU" in describe_accelerator_line(_acc(build="cuda"))

    def test_format_problems_labels_severity(self):
        text = format_problems(check_step_schedule(150, 400))
        assert text.strip().startswith("ERROR:")
