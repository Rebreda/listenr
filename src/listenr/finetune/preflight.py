"""
preflight.py — Checks that fail fast, before a run wastes time or crashes deep.

Every check here answers a question the user would otherwise have answered by a
traceback several minutes into a run, after the dataset and the model have both
loaded:

* Can torch see this machine's GPU at all? On AMD the usual failure is silent.
  ``pip`` resolves a CUDA wheel from default PyPI, it imports without
  complaint, and it reports no devices. Training then runs on CPU at a
  thousandth of the speed, or dies on a precision flag, and nothing says why.
* Do the precision flags suit the device that will actually run? ``bf16=True``
  raises inside ``Seq2SeqTrainingArguments`` when there is no GPU.
* Are ``eval_steps`` and ``save_steps`` compatible? ``load_best_model_at_end``
  requires save to be a round multiple of eval, and that check also fires late.

Checks are pure and return problems as data, so ``--dry-run`` can report all of
them at once rather than stopping at the first.

Public API
----------
Problem                       -> dataclass, .severity is "error" or "warning"
Accelerator                   -> dataclass describing what torch can actually use
describe_accelerator()        -> Accelerator
check_all(...)                -> list[Problem]
format_problems(problems)     -> str
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Present whenever the amdkfd kernel driver is loaded, regardless of whether any
# ROCm userspace is installed. This is how we tell "AMD GPU in this machine"
# apart from "torch can use it".
AMD_KFD_DEVICE = Path("/dev/kfd")

ROCM_INSTALL_HINT = (
    "Install a ROCm build of torch into this environment, for example:\n"
    "    uv pip install --torch-backend=rocm torch\n"
    "  or with pip:\n"
    "    pip install --index-url https://download.pytorch.org/whl/rocm6.4 torch\n"
    "  The ROCm wheels bundle their own runtime, so a host ROCm install is not\n"
    "  required. Only the kernel driver is."
)


@dataclass(frozen=True)
class Problem:
    severity: str  # "error" or "warning"
    message: str


@dataclass(frozen=True)
class Accelerator:
    """What torch can actually use, as opposed to what the machine contains."""

    torch_version: str
    #: "rocm", "cuda" or None. Read from the build, not from device availability.
    build: str | None
    #: True when torch reports at least one usable device.
    available: bool
    device_name: str | None
    #: True when this machine has an AMD GPU, whatever torch thinks.
    amd_present: bool

    @property
    def will_train_on_cpu(self) -> bool:
        return not self.available


def describe_accelerator() -> Accelerator:
    """Report what torch can use here. Never raises."""
    amd_present = AMD_KFD_DEVICE.exists()

    try:
        import torch
    except ImportError:
        return Accelerator(
            torch_version="not installed",
            build=None,
            available=False,
            device_name=None,
            amd_present=amd_present,
        )

    # ROCm torch aliases the cuda namespace, so version.hip is the only
    # reliable way to tell the two builds apart.
    build = "rocm" if getattr(torch.version, "hip", None) else None
    if build is None and getattr(torch.version, "cuda", None):
        build = "cuda"

    try:
        available = torch.cuda.is_available()
    except Exception:
        available = False

    device_name = None
    if available:
        try:
            device_name = torch.cuda.get_device_name(0)
        except Exception:
            device_name = None

    return Accelerator(
        torch_version=torch.__version__,
        build=build,
        available=available,
        device_name=device_name,
        amd_present=amd_present,
    )


def check_torch_build(acc: Accelerator) -> list[Problem]:
    """Catch the AMD-user-gets-a-CUDA-wheel trap, which is otherwise silent."""
    if acc.torch_version == "not installed":
        return [Problem("error", "torch is not installed. Install with: uv pip install 'listenr[finetune]'")]

    if acc.available:
        return []

    if acc.amd_present and acc.build == "cuda":
        return [
            Problem(
                "error",
                f"This machine has an AMD GPU (/dev/kfd exists) but torch "
                f"{acc.torch_version} is a CUDA build and sees no devices, so "
                f"training would fall back to CPU.\n  " + ROCM_INSTALL_HINT,
            )
        ]

    if acc.amd_present and acc.build == "rocm":
        return [
            Problem(
                "error",
                f"torch {acc.torch_version} is a ROCm build but reports no "
                "usable device. If your GPU is not on AMD's support matrix, "
                "retry once with HSA_OVERRIDE_GFX_VERSION set to a supported "
                "ISA. For gfx1151 (Strix Halo) that is 11.0.0, which "
                "masquerades as gfx1100. Setting it to your own ISA is a no-op.",
            )
        ]

    return [
        Problem(
            "warning",
            f"torch {acc.torch_version} reports no usable GPU. Training will "
            "run on CPU, which is workable for a smoke test and impractical "
            "for a real run.",
        )
    ]


def check_precision(fp16: bool, bf16: bool, acc: Accelerator) -> list[Problem]:
    """Precision flags that transformers would reject, reported before the load."""
    problems: list[Problem] = []

    if fp16 and bf16:
        problems.append(
            Problem("error", "--fp16 and --bf16 are mutually exclusive. Pick one.")
        )

    if (fp16 or bf16) and acc.will_train_on_cpu:
        flag = "--bf16" if bf16 else "--fp16"
        problems.append(
            Problem(
                "error",
                f"{flag} needs a GPU and none is usable, so training would "
                f"raise as soon as it starts. Pass --no-bf16 --no-fp16 to run "
                f"on CPU, or fix the GPU first.",
            )
        )

    if fp16 and acc.build == "rocm":
        problems.append(
            Problem(
                "warning",
                "--fp16 on ROCm is not recommended. Prefer --bf16 on RDNA2 and newer.",
            )
        )

    return problems


def check_step_schedule(
    eval_steps: int, save_steps: int, load_best_model_at_end: bool = True
) -> list[Problem]:
    """load_best_model_at_end requires save_steps to be a multiple of eval_steps.

    transformers enforces this, but only once training arguments are built,
    which is after the dataset and the model have loaded.
    """
    if not load_best_model_at_end:
        return []
    if eval_steps <= 0 or save_steps <= 0:
        return [Problem("error", "--eval-steps and --save-steps must both be positive.")]
    if save_steps % eval_steps != 0:
        return [
            Problem(
                "error",
                f"--save-steps ({save_steps}) must be a whole multiple of "
                f"--eval-steps ({eval_steps}), because the best checkpoint is "
                f"loaded at the end. Try --save-steps "
                f"{eval_steps * max(1, round(save_steps / eval_steps))}.",
            )
        ]
    return []


def check_all(
    *,
    fp16: bool,
    bf16: bool,
    eval_steps: int,
    save_steps: int,
    accelerator: Accelerator | None = None,
) -> list[Problem]:
    acc = accelerator or describe_accelerator()
    return [
        *check_torch_build(acc),
        *check_precision(fp16, bf16, acc),
        *check_step_schedule(eval_steps, save_steps),
    ]


def format_problems(problems: list[Problem]) -> str:
    lines = []
    for p in problems:
        prefix = "ERROR" if p.severity == "error" else "WARNING"
        lines.append(f"  {prefix}: {p.message}")
    return "\n".join(lines)


def describe_accelerator_line(acc: Accelerator) -> str:
    """One line for the log, so a run records what it actually trained on."""
    if acc.available:
        return (
            f"torch {acc.torch_version} ({acc.build or 'unknown'} build), "
            f"device: {acc.device_name}"
        )
    return f"torch {acc.torch_version} ({acc.build or 'cpu'} build), no usable GPU, will use CPU"
