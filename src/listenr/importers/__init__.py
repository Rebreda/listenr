"""Optional data-source importers for Listenr.

Each importer loads an external ASR dataset and writes a Listenr-compatible
``manifest.jsonl`` so the existing ``listenr build-dataset`` and fine-tuning
flow keeps working unchanged. Importers are optional: the SDKs they depend on
(``datacollective`` for MDC, ``datasets`` for Hugging Face) are only imported
when the corresponding importer actually runs.
"""
