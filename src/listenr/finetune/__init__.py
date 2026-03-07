"""
listenr.finetune — LoRA fine-tuning pipeline for Whisper ASR.

Public API (all heavy imports are deferred until actually called so that the
package can be imported without the optional fine-tuning dependencies installed):

    from listenr.finetune.data   import make_processor, make_dataset, WhisperDataCollator
    from listenr.finetune.model  import load_base_model, make_lora_config, apply_lora, freeze_encoder
    from listenr.finetune.metrics import make_compute_metrics

Entry point::

    listenr-finetune [options]
"""
