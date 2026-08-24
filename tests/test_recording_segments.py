"""Tests for pairing a transcript with the audio it was actually made from.

The bug these cover wrote real, plausible-looking training rows: 0.085 s of
audio labelled with a 25 word sentence. Both halves pass every individual
check, so nothing downstream caught it, and the row teaches a model to produce
a full sentence from near silence.

The recording loop itself is a long async function over a websocket, so these
tests model the event sequence rather than driving the loop. The sequence is
taken verbatim from a debug log of a session that produced corrupt rows.
"""

from collections import deque

import pytest

from listenr.transcript_utils import (
    MAX_WORDS_PER_SECOND,
    implausible_speech_rate,
)


class SegmentPairing:
    """The buffer discipline from cli.py, isolated so it can be driven directly."""

    def __init__(self):
        self.pcm_buffer: list[str] = []
        self.pending: deque[list[str]] = deque(maxlen=16)

    def audio(self, chunk: str) -> None:
        self.pcm_buffer.append(chunk)

    def committed(self) -> None:
        self.pending.append(list(self.pcm_buffer))
        self.pcm_buffer.clear()

    def speech_started(self) -> None:
        self.pcm_buffer.clear()

    def completed(self) -> list[str]:
        if self.pending:
            return self.pending.popleft()
        segment = list(self.pcm_buffer)
        self.pcm_buffer.clear()
        return segment


class TestForcedCommitOrdering:
    def test_transcript_gets_its_own_audio_when_speech_continues(self):
        """The exact sequence that corrupted rows.

        max_segment_s forces a commit while the speaker is still talking, so
        speech_started for the next utterance arrives before the transcript for
        the committed one.
        """
        s = SegmentPairing()
        for chunk in ("a1", "a2", "a3"):
            s.audio(chunk)

        s.committed()          # forced by max_segment_s
        s.speech_started()     # next utterance begins
        s.audio("b1")          # a few chunks of the NEW utterance
        segment = s.completed()  # transcript for the FIRST utterance arrives

        assert segment == ["a1", "a2", "a3"]
        # and the new utterance's audio is untouched
        assert s.pcm_buffer == ["b1"]

    def test_normal_ordering_still_works(self):
        s = SegmentPairing()
        s.audio("a1")
        s.audio("a2")
        s.committed()
        assert s.completed() == ["a1", "a2"]

    def test_two_segments_stay_in_order(self):
        s = SegmentPairing()
        s.audio("a1")
        s.committed()
        s.audio("b1")
        s.committed()
        assert s.completed() == ["a1"]
        assert s.completed() == ["b1"]

    def test_dropped_transcript_does_not_misalign_the_queue(self):
        """A hallucination is discarded, and must take its own audio with it."""
        s = SegmentPairing()
        s.audio("a1")
        s.committed()
        s.audio("b1")
        s.committed()

        s.completed()  # first transcript, dropped as a hallucination
        assert s.completed() == ["b1"]

    def test_falls_back_to_the_live_buffer_without_a_commit(self):
        """Older servers may not emit committed; behaviour must not regress."""
        s = SegmentPairing()
        s.audio("a1")
        assert s.completed() == ["a1"]
        assert s.pcm_buffer == []

    def test_speech_started_never_discards_committed_audio(self):
        s = SegmentPairing()
        s.audio("a1")
        s.committed()
        s.speech_started()
        assert s.completed() == ["a1"]

    def test_queue_is_bounded(self):
        s = SegmentPairing()
        for i in range(40):
            s.audio(f"c{i}")
            s.committed()
        assert len(s.pending) == 16


class TestImplausibleSpeechRate:
    def test_the_real_corrupt_row_is_caught(self):
        """25 words against 0.085s, straight from the manifest."""
        text = " ".join(["word"] * 25)
        rate = implausible_speech_rate(text, 0.085)
        assert rate is not None
        assert round(rate) == 294

    def test_the_other_real_corrupt_row_is_caught(self):
        """21 words against 0.342s passed both min_duration and min_chars."""
        assert implausible_speech_rate(" ".join(["word"] * 21), 0.342) is not None

    def test_ordinary_speech_passes(self):
        # 12 words in 4 seconds is 3 w/s, comfortably normal.
        assert implausible_speech_rate(" ".join(["word"] * 12), 4.0) is None

    def test_fast_but_possible_speech_passes(self):
        assert implausible_speech_rate(" ".join(["word"] * 25), 5.0) is None

    def test_boundary_is_not_flagged(self):
        assert implausible_speech_rate(" ".join(["word"] * 8), 1.0) is None

    @pytest.mark.parametrize("duration", [0.0, -1.0])
    def test_unknown_duration_is_not_judged(self, duration):
        assert implausible_speech_rate("some words here", duration) is None

    def test_empty_text_is_not_judged(self):
        assert implausible_speech_rate("", 0.01) is None

    def test_threshold_is_above_real_human_speech(self):
        """Fast conversational speech is around 5 w/s; the limit must clear it."""
        assert MAX_WORDS_PER_SECOND > 5.0


class TestBuildDatasetRejectsMismatches:
    def test_validate_entry_drops_a_mismatched_clip(self, tmp_path):
        from collections import Counter

        from listenr.build_dataset import validate_entry

        wav = tmp_path / "clip.wav"
        wav.write_bytes(b"RIFF")
        reasons: Counter[str] = Counter()
        entry = validate_entry(
            {
                "uuid": "corrupt",
                "raw_transcription": " ".join(["word"] * 21),
                "audio_path": str(wav),
                "duration_s": 0.342,
            },
            min_duration=0.3,   # the real default: this clip passes it
            min_chars=10,       # and passes this too
            reasons=reasons,
        )
        assert entry is None
        assert any("do not match" in r for r in reasons)

    def test_a_normal_clip_survives(self, tmp_path):
        from listenr.build_dataset import validate_entry

        wav = tmp_path / "clip.wav"
        wav.write_bytes(b"RIFF")
        entry = validate_entry(
            {
                "uuid": "fine",
                "raw_transcription": "this is a normal sentence of speech",
                "audio_path": str(wav),
                "duration_s": 3.0,
            },
            min_duration=0.3,
            min_chars=10,
        )
        assert entry is not None


class TestRealRecordingLoop:
    """Drives the actual _run loop, not a model of it.

    Only the microphone and the websocket are replaced. The event sequence is
    the one from the debug log of the session that produced corrupt rows, and
    everything in between, including save_recording, is the real code.
    """

    @staticmethod
    def _drive(monkeypatch, tmp_path, script):
        """Run _run against *script*, a list of (feed_chunks, event) pairs."""
        import asyncio

        import numpy as np

        from listenr import cli

        storage = tmp_path / "clips"
        monkeypatch.setattr(cli, "STORAGE_BASE", storage)
        monkeypatch.setattr(cli, "USE_LLM", False)
        monkeypatch.setattr(cli, "ensure_models_loaded", lambda debug=False: None)
        monkeypatch.setattr(cli, "get_lemonade_ws_url", lambda: "ws://test/v1/realtime")

        captured: dict = {}

        # Not an async def: calling it must capture the buffer immediately,
        # because _run builds the generator before iteration begins.
        def fake_mic_stream(pcm_buffer, debug=False):
            captured["buffer"] = pcm_buffer

            async def _gen():
                return
                yield  # pragma: no cover

            return _gen()

        monkeypatch.setattr(cli, "mic_stream", fake_mic_stream)

        def chunk(seed: int) -> bytes:
            t = np.linspace(0, 1, 1600, endpoint=False)
            return (np.sin(2 * np.pi * (100 + seed * 50) * t) * 8000).astype("<i2").tobytes()

        async def fake_stream_transcribe(self, audio_stream, **kwargs):
            buf = captured["buffer"]
            for feed, event in script:
                for i in range(feed):
                    buf.append(chunk(i))
                yield event

        monkeypatch.setattr(
            cli.LemonadeUnifiedASR, "stream_transcribe", fake_stream_transcribe
        )
        asyncio.run(cli._run(save=True, show_raw=False, debug=False))
        return storage

    @staticmethod
    def _saved(storage):
        import json

        manifest = storage / "manifest.jsonl"
        if not manifest.exists():
            return []
        return [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]

    @staticmethod
    def _frames(path):
        import wave

        with wave.open(str(path), "rb") as w:
            return w.getnframes()

    def test_forced_commit_saves_the_audio_that_was_committed(self, monkeypatch, tmp_path):
        """The regression. 10 chunks committed, then the next utterance starts.

        Before the fix the saved clip held only the 1 chunk of the new
        utterance, which is how 0.085s got labelled with a full sentence.
        """
        script = [
            (10, {"type": "input_audio_buffer.committed"}),
            (0, {"type": "input_audio_buffer.speech_started"}),
            (1, {"type": "session.updated"}),  # a chunk of the NEXT utterance
            (
                0,
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "transcript": "this is the first utterance and it is long enough",
                },
            ),
        ]
        storage = self._drive(monkeypatch, tmp_path, script)
        records = self._saved(storage)
        assert len(records) == 1
        assert self._frames(records[0]["audio_path"]) == 10 * 1600

    def test_two_forced_commits_keep_their_own_audio(self, monkeypatch, tmp_path):
        script = [
            (10, {"type": "input_audio_buffer.committed"}),
            (0, {"type": "input_audio_buffer.speech_started"}),
            (4, {"type": "input_audio_buffer.committed"}),
            (
                0,
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "transcript": "the first utterance which is nice and long",
                },
            ),
            (
                0,
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "transcript": "the second utterance also long enough to keep",
                },
            ),
        ]
        storage = self._drive(monkeypatch, tmp_path, script)
        records = self._saved(storage)
        assert len(records) == 2
        assert self._frames(records[0]["audio_path"]) == 10 * 1600
        assert self._frames(records[1]["audio_path"]) == 4 * 1600

    def test_still_works_when_the_server_sends_no_commit(self, monkeypatch, tmp_path):
        script = [
            (6, {"type": "session.updated"}),
            (
                0,
                {
                    "type": "conversation.item.input_audio_transcription.completed",
                    "transcript": "an utterance with no commit event at all",
                },
            ),
        ]
        storage = self._drive(monkeypatch, tmp_path, script)
        records = self._saved(storage)
        assert len(records) == 1
        assert self._frames(records[0]["audio_path"]) == 6 * 1600
