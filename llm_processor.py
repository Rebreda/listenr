#!/usr/bin/env python3
"""
LLM Processor for ASR Post-Processing
Handles transcription improvement using local LLMs via Ollama
"""

import json
import logging
import re
from collections import deque
from difflib import SequenceMatcher
from typing import Optional, List, Tuple, Dict

import requests
import threading

class LLMProcessor:
    """Handles LLM post-processing of transcriptions"""
    
    def __init__(self, 
                 model: str = "gemma2:2b",
                 ollama_host: str = "http://localhost:11434",
                 context_window: int = 3,
                 temperature: float = 0.1,
                 max_tokens: int = 100,
                 timeout: int = 10,
                 correction_types: Optional[List[str]] = None,
                 correction_threshold: float = 0.7,
                 enable_fallback_processing: bool = True):
        """
        Initialize LLM Processor
        
        Args:
            model: Ollama model name
            ollama_host: Ollama API URL
            context_window: Number of previous transcriptions to use as context
            temperature: LLM temperature (0.0-1.0, lower = more deterministic)
            max_tokens: Maximum tokens to generate
            timeout: API timeout in seconds
        """
        self.model = model
        self.ollama_host = ollama_host
        self.context_buffer = deque(maxlen=context_window)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        if isinstance(correction_types, str):
            correction_types = [item.strip() for item in correction_types.split(",") if item.strip()]
        self.correction_types = correction_types or [
            "punctuation",
            "capitalization",
            "grammar",
            "homophone",
            "numeric",
            "spacing",
        ]
        self.correction_threshold = correction_threshold
        self.enable_fallback_processing = enable_fallback_processing
        self.logger = logging.getLogger(__name__)
        self.available = False
        
        # System prompt for transcription correction
        self.system_prompt = """You are an expert transcription editor. Your ONLY job is to fix speech-to-text output.

MANDATORY RULES:
1. Reply with the corrected transcription only. No explanations or markup.
2. Preserve the speaker's meaning, tense, and tone.
3. Apply professional punctuation, sentence casing, and paragraphing.
4. Repair grammar issues, subject/verb agreement, and obvious word choice mistakes.
5. Resolve homophone substitutions using the surrounding context.
6. Normalize numeric expressions (e.g., '2' -> 'two') when they are spoken as words.
7. Preserve contractions as spoken (e.g., "don't") and keep emphasis words in place.
8. Remove filler words only if they are duplicated (e.g., 'um um'), otherwise keep them.
9. Never invent new ideas or add commentary.
10. If the text is already correct, return it unchanged."""
        
        # Check availability on init
        self.check_availability()
    
    def check_availability(self) -> bool:
        """Check if Ollama is running and model is available"""
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=2)
            if response.status_code != 200:
                self.logger.warning(f"Ollama not responding: Status {response.status_code}")
                self.available = False
                return False
            
            # Check if model is available
            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]
            
            if not any(self.model in name for name in model_names):
                self.logger.warning(f"Model {self.model} not found. Available: {model_names}")
                self.logger.info(f"Pull the model with: ollama pull {self.model}")
                self.available = False
                return False
            
            self.logger.info(f"LLM processor ready with model: {self.model}")
            self.available = True
            return True
            
        except requests.exceptions.RequestException as e:
            self.logger.warning(f"Cannot connect to Ollama at {self.ollama_host}: {e}")
            self.logger.info("Start Ollama with: ollama serve")
            self.available = False
            return False
        except Exception as e:
            self.logger.error(f"Error checking Ollama availability: {e}")
            self.available = False
            return False
    
    def process(self, text: str, use_context: bool = True) -> str:
        """Compatibility wrapper that returns only the corrected text."""
        corrected, _ = self.process_with_metadata(text, use_context)
        return corrected

    def process_with_metadata(self, text: str, use_context: bool = True) -> Tuple[str, List[Dict[str, str]]]:
        """
        Process transcription with LLM to fix errors
        
        Args:
            text: Raw transcription text
            use_context: Whether to use previous transcriptions as context
            
        Returns:
            Tuple of (corrected text, list of correction metadata dictionaries)
        """
        if not text or not text.strip():
            return text, []
        
        if not self.available:
            # Try to reconnect
            if not self.check_availability():
                fallback_text = self._apply_heuristic_repairs(text, text)
                corrections = self._summarize_corrections(text, fallback_text)
                return fallback_text, corrections
        
        try:
            # Build the prompt
            prompt = self._build_prompt(text, use_context)
            
            # Make API request to Ollama
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "system": self.system_prompt,
                    "temperature": self.temperature,
                    "stream": False,
                    "options": {
                        "num_predict": self.max_tokens,
                        "stop": ["\n\n", "---", "Note:", "Explanation:", "Context:", "Previous:"],
                    }
                },
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                self.logger.warning(f"LLM API error: {response.status_code}")
                return self._handle_fallback(text)
            
            # Extract corrected text
            result = response.json()
            corrected = result.get('response', '').strip()
            
            # Validate the response
            if not corrected or len(corrected) < 3:
                self.logger.debug("LLM returned empty/short response, using original")
                return self._handle_fallback(text)
            
            # Check if response is too different (might be an error)
            if len(corrected) > len(text) * 2.5 or len(corrected) < len(text) * 0.3:
                self.logger.debug("LLM response length suspicious, using original")
                return self._handle_fallback(text)
            
            # Remove any accidental meta-text the LLM might have added
            corrected = self._clean_response(corrected)
            corrected = self._apply_heuristic_repairs(text, corrected)
            corrections = self._summarize_corrections(text, corrected)
            
            # Update context buffer with the corrected text
            self.context_buffer.append(corrected)
            
            self.logger.debug(f"LLM corrected: '{text}' -> '{corrected}'")
            return corrected, corrections
            
        except requests.exceptions.Timeout:
            self.logger.warning("LLM request timed out")
            return self._handle_fallback(text)
        except requests.exceptions.RequestException as e:
            self.logger.error(f"LLM request failed: {e}")
            self.available = False  # Mark as unavailable
            return self._handle_fallback(text)
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON from LLM: {e}")
            return self._handle_fallback(text)
        except Exception as e:
            self.logger.error(f"Unexpected error in LLM processing: {e}")
            return self._handle_fallback(text)
    
    def _build_prompt(self, text: str, use_context: bool) -> str:
        """Build the prompt for the LLM"""
        if use_context and len(self.context_buffer) > 0:
            # Include context from previous transcriptions
            context = " ".join(self.context_buffer)
            prompt = f"""Previous context: {context}

Current transcription to correct: {text}

Corrected transcription:"""
        else:
            # Simple correction without context
            prompt = f"""Transcription to correct: {text}

Corrected transcription:"""
        
        return prompt
    
    def _clean_response(self, text: str) -> str:
        """Remove any meta-text the LLM might have accidentally included"""
        # Remove common prefixes the LLM might add
        prefixes_to_remove = [
            "Corrected transcription:",
            "Corrected:",
            "Fixed:",
            "Here's the corrected version:",
            "The corrected text is:",
        ]
        
        cleaned = text
        for prefix in prefixes_to_remove:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):].strip()
        
        # Remove quotes if the LLM wrapped the response
        if cleaned.startswith('"') and cleaned.endswith('"'):
            cleaned = cleaned[1:-1]
        if cleaned.startswith("'") and cleaned.endswith("'"):
            cleaned = cleaned[1:-1]
        
        return cleaned.strip()
    
    def _apply_heuristic_repairs(self, original: str, candidate: str) -> str:
        """
        Apply lightweight deterministic fixes to stabilize punctuation and spacing.
        Does not introduce meaning changes beyond surface-level cleanup.
        """
        if not candidate:
            return candidate
        
        text = candidate.strip()
        
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        text = re.sub(r"([,.;:!?])([^\s])", r"\1 \2", text)
        text = text.replace(" ,", ",").replace(" .", ".")
        
        # Capitalize standalone first-person pronouns
        text = re.sub(r"\bi\b", "I", text)
        text = re.sub(r"\bi'm\b", "I'm", text, flags=re.IGNORECASE)
        text = re.sub(r"\bi'd\b", "I'd", text, flags=re.IGNORECASE)
        text = re.sub(r"\bi've\b", "I've", text, flags=re.IGNORECASE)
        text = re.sub(r"\bi'll\b", "I'll", text, flags=re.IGNORECASE)
        
        # Ensure leading capitalization for sentences
        if text and text[0].isalpha():
            text = text[0].upper() + text[1:]
        
        # Ensure ending punctuation if the utterance sounds complete
        ending_punct = {".", "!", "?"}
        if text and text[-1] not in ending_punct:
            if original.strip().endswith("?") or any(word in original.lower().split()[:3] for word in ["why", "what", "how", "when", "where", "who", "will", "did", "do", "can", "could", "should"]):
                text = text.rstrip(".") + "?"
            else:
                text = text.rstrip(".") + "."
        
        return text.strip()
    
    def _summarize_corrections(self, original: str, corrected: str) -> List[Dict[str, str]]:
        """Generate a simple diff summary between original and corrected text."""
        if original == corrected:
            return []
        
        summary: List[Dict[str, str]] = []
        matcher = SequenceMatcher(None, original, corrected)
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            original_segment = original[i1:i2]
            corrected_segment = corrected[j1:j2]
            correction_type = self._infer_correction_type(original_segment, corrected_segment)
            summary.append({
                "type": correction_type,
                "original": original_segment,
                "corrected": corrected_segment,
                "position": str(j1),
            })
        return summary
    
    def _infer_correction_type(self, original_fragment: str, corrected_fragment: str) -> str:
        """Classify the correction type in a coarse way for downstream reporting."""
        original_fragment = original_fragment.strip()
        corrected_fragment = corrected_fragment.strip()
        
        if not original_fragment and corrected_fragment:
            if re.fullmatch(r"[,.!?;:]+", corrected_fragment):
                return "punctuation"
            return "insertion"
        if original_fragment and not corrected_fragment:
            return "deletion"
        
        if re.sub(r"[^\w]", "", original_fragment, flags=re.UNICODE).lower() == \
           re.sub(r"[^\w]", "", corrected_fragment, flags=re.UNICODE).lower():
            return "spacing"
        
        if original_fragment.islower() and corrected_fragment[:1].isupper():
            return "capitalization"
        
        if re.fullmatch(r"[0-9]+", original_fragment) and corrected_fragment.isalpha():
            return "numeric"
        
        if any(h in self.correction_types for h in ["punctuation", "spacing"]) and \
           re.fullmatch(r"[,.!?;:]+", corrected_fragment):
            return "punctuation"
        
        return "grammar"
    
    def _handle_fallback(self, text: str) -> Tuple[str, List[Dict[str, str]]]:
        """Fallback pathway when LLM processing fails or is unavailable."""
        if not self.enable_fallback_processing:
            return text, []
        
        repaired = self._apply_heuristic_repairs(text, text)
        corrections = self._summarize_corrections(text, repaired)
        return repaired, corrections
    
    def clear_context(self):
        """Clear the context buffer"""
        self.context_buffer.clear()
        self.logger.debug("Context buffer cleared")
    
    def add_to_context(self, text: str):
        """Manually add text to context without processing"""
        self.context_buffer.append(text)


class AsyncLLMProcessor:
    """Async wrapper for non-blocking LLM processing"""
    
    def __init__(self, processor: LLMProcessor):
        """
        Initialize async wrapper
        
        Args:
            processor: LLMProcessor instance to wrap
        """
        import queue
        
        self.processor = processor
        self.input_queue = queue.Queue()
        self.output_callbacks = {}
        self.worker_thread = None
        self.running = False
        self.logger = logging.getLogger(__name__)
    
    def start(self):
        """Start the async processing thread"""
        if self.running:
            return
        
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        self.logger.debug("Async LLM processor started")
    
    def stop(self):
        """Stop the async processing thread"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=2)
        self.logger.debug("Async LLM processor stopped")
    
    def process_async(self, text: str, callback=None, use_context=True):
        """
        Queue text for async processing
        
        Args:
            text: Text to process
            callback: Optional callback function(original, corrected)
            use_context: Whether to use context
        """
        import uuid
        task_id = str(uuid.uuid4())
        
        if callback:
            self.output_callbacks[task_id] = callback
        
        self.input_queue.put((task_id, text, use_context))
        return task_id
    
    def _worker(self):
        """Worker thread for processing queue"""
        import queue as q
        
        while self.running:
            try:
                task_id, text, use_context = self.input_queue.get(timeout=0.5)
                
                # Process with LLM
                corrected = self.processor.process(text, use_context)
                
                # Call callback if provided
                if task_id in self.output_callbacks:
                    callback = self.output_callbacks.pop(task_id)
                    try:
                        callback(text, corrected)
                    except Exception as e:
                        self.logger.error(f"Error in LLM callback: {e}")
                
            except q.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error in LLM worker: {e}")


def test_llm_processor():
    """Test function to verify LLM processor is working"""
    logging.basicConfig(level=logging.DEBUG)
    
    processor = LLMProcessor(model="gemma2:2b")
    
    if not processor.available:
        print("LLM processor not available. Check Ollama setup.")
        return
    
    test_cases = [
        "hello world this is a test",
        "im going to the store two by some milk",
        "the whether is nice today i think ill go for a walk",
        "can you here me now",
    ]
    
    print("Testing LLM Processor:\n")
    for test in test_cases:
        corrected = processor.process(test)
        print(f"Original:  {test}")
        print(f"Corrected: {corrected}\n")


if __name__ == "__main__":
    test_llm_processor()
