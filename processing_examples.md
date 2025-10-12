# Processing Examples

The following examples illustrate the expected behaviour for the transcription post-processing pipeline. Each example pairs the raw Whisper output with the cleaned result and highlights the key fixes that should be applied consistently.

| Raw Transcription | Corrected Output | Notable Corrections |
| --- | --- | --- |
| `i went to the store to by 2 bags of milk` | `I went to the store to buy two bags of milk.` | Capitalization, homophone fix (`by`→`buy`), numeric normalization, closing punctuation |
| `can you hear me now im testing the mic` | `Can you hear me now? I'm testing the mic.` | Leading capitalization, inserted question mark, apostrophe restoration, sentence split |
| `weather looks great today maybe we could eat outside` | `Weather looks great today. Maybe we could eat outside.` | Capitalization, period insertion, sentence boundary detection |
| `its 5 pm already we shouldve left` | `It's 5 p.m. already; we should've left.` | Apostrophes, time formatting, semicolon insertion for clarity |
| `um um i think thats all for now` | `Um, I think that's all for now.` | Duplicate filler reduction, comma insertion, apostrophes, terminal period |
| `dont forget the meeting is on friday at 10 am sharp` | `Don't forget: the meeting is on Friday at 10 a.m. sharp.` | Apostrophe restoration, colon insertion, proper-noun capitalization, time formatting |

These examples are deliberately varied to demonstrate punctuation, capitalization, spacing, and word-choice adjustments. The `LLMProcessor` now generates lightweight metadata summarising the corrections so downstream components (for example, JSON export or UI surfaces) can display what changed alongside the final transcript.
