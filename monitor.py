#!/usr/bin/env python3
"""
ASR Monitor - Real-time monitoring and assessment for listnr transcription system
Tracks transcription quality, performance metrics, and system resources
"""

import os
import sys
import time
import json
import threading
import queue
from datetime import datetime, timedelta
from collections import deque, defaultdict
from pathlib import Path
import argparse
import curses
import signal

# For file monitoring
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# For system monitoring
import psutil

# For text analysis
import difflib
from typing import Dict, List, Tuple, Optional

class TranscriptionMetrics:
    """Tracks metrics for transcription quality and performance"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        
        # Performance metrics
        self.transcription_times = deque(maxlen=window_size)
        self.audio_durations = deque(maxlen=window_size)
        self.processing_delays = deque(maxlen=window_size)
        
        # Quality metrics
        self.text_lengths = deque(maxlen=window_size)
        self.correction_ratios = deque(maxlen=window_size)
        self.word_counts = deque(maxlen=window_size)
        
        # Session statistics
        self.total_transcriptions = 0
        self.total_corrections = 0
        self.session_start = datetime.now()
        self.last_transcription_time = None
        
        # Error tracking
        self.errors = deque(maxlen=20)
        self.vad_triggers = deque(maxlen=window_size)
        
        # LLM metrics
        self.llm_response_times = deque(maxlen=window_size)
        self.correction_types = defaultdict(int)
        
    def add_transcription(self, raw_text: str, corrected_text: Optional[str] = None, 
                         duration_ms: float = 0, processing_time_ms: float = 0):
        """Add a new transcription and calculate metrics"""
        self.total_transcriptions += 1
        self.last_transcription_time = datetime.now()
        
        # Text metrics
        self.text_lengths.append(len(raw_text))
        self.word_counts.append(len(raw_text.split()))
        
        # Performance metrics
        if duration_ms > 0:
            self.audio_durations.append(duration_ms)
        if processing_time_ms > 0:
            self.transcription_times.append(processing_time_ms)
            if duration_ms > 0:
                self.processing_delays.append(processing_time_ms / duration_ms)
        
        # Correction metrics
        if corrected_text and corrected_text != raw_text:
            self.total_corrections += 1
            similarity = difflib.SequenceMatcher(None, raw_text, corrected_text).ratio()
            self.correction_ratios.append(1 - similarity)
            self._analyze_corrections(raw_text, corrected_text)
    
    def _analyze_corrections(self, raw: str, corrected: str):
        """Analyze types of corrections made"""
        raw_words = raw.lower().split()
        corrected_words = corrected.lower().split()
        
        # Common correction patterns
        for i, (r, c) in enumerate(zip(raw_words, corrected_words)):
            if r != c:
                # Homophone detection
                homophones = [('by', 'buy'), ('there', 'their'), ('your', "you're"), 
                             ('its', "it's"), ('two', 'to'), ('two', 'too')]
                for h1, h2 in homophones:
                    if (r == h1 and c == h2) or (r == h2 and c == h1):
                        self.correction_types['homophone'] += 1
                        break
                
                # Number formatting
                if r.isdigit() and c.isalpha():
                    self.correction_types['number_format'] += 1
                elif r != c:
                    self.correction_types['contextual'] += 1
    
    def get_stats(self) -> Dict:
        """Get current statistics"""
        stats = {
            'total_transcriptions': self.total_transcriptions,
            'total_corrections': self.total_corrections,
            'session_duration': str(datetime.now() - self.session_start).split('.')[0],
            'last_activity': self.last_transcription_time.strftime('%H:%M:%S') if self.last_transcription_time else 'N/A',
        }
        
        # Performance averages
        if self.transcription_times:
            stats['avg_transcription_time'] = f"{sum(self.transcription_times) / len(self.transcription_times):.1f}ms"
        if self.audio_durations:
            stats['avg_audio_duration'] = f"{sum(self.audio_durations) / len(self.audio_durations):.1f}ms"
        if self.processing_delays:
            stats['rtf'] = f"{sum(self.processing_delays) / len(self.processing_delays):.2f}x"  # Real-time factor
        
        # Quality metrics
        if self.word_counts:
            stats['avg_words_per_segment'] = f"{sum(self.word_counts) / len(self.word_counts):.1f}"
        if self.correction_ratios:
            stats['avg_correction_ratio'] = f"{sum(self.correction_ratios) / len(self.correction_ratios) * 100:.1f}%"
        
        # Correction breakdown
        if self.correction_types:
            stats['corrections'] = dict(self.correction_types)
        
        # Activity rate
        if self.total_transcriptions > 0:
            elapsed = (datetime.now() - self.session_start).total_seconds()
            stats['transcriptions_per_minute'] = f"{(self.total_transcriptions / elapsed) * 60:.1f}"
        
        return stats


class FileMonitor(FileSystemEventHandler):
    """Monitor output files for changes"""
    
    def __init__(self, metrics: TranscriptionMetrics, update_queue: queue.Queue):
        self.metrics = metrics
        self.update_queue = update_queue
        self.last_raw_line = None
        self.last_corrected_line = None
        self.line_cache = {}
        
    def on_modified(self, event):
        """Handle file modification events"""
        if event.is_directory:
            return
            
        try:
            filepath = event.src_path
            filename = os.path.basename(filepath)
            
            # Read new lines from file
            if filepath not in self.line_cache:
                self.line_cache[filepath] = 0
            
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                new_lines = lines[self.line_cache[filepath]:]
                self.line_cache[filepath] = len(lines)
                
                for line in new_lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Parse timestamp and text
                    if '] ' in line:
                        timestamp_str, text = line.split('] ', 1)
                        timestamp_str = timestamp_str.strip('[')
                        
                        # Track based on file type
                        if 'raw' in filename.lower():
                            self.last_raw_line = text
                            if text.startswith('[RAW]'):
                                text = text[5:].strip()
                            self.metrics.add_transcription(text)
                        elif 'clean' in filename.lower() or 'llm' in filename.lower():
                            self.last_corrected_line = text
                            # Try to match with recent raw
                            if self.last_raw_line:
                                self.metrics.add_transcription(
                                    self.last_raw_line, 
                                    text,
                                    duration_ms=5000,  # Estimate
                                    processing_time_ms=500  # Estimate
                                )
                        
                        # Queue update for UI
                        self.update_queue.put({
                            'type': 'transcription',
                            'timestamp': timestamp_str,
                            'text': text,
                            'source': filename
                        })
                        
        except Exception as e:
            self.metrics.errors.append(f"File monitor error: {str(e)}")


class SystemMonitor:
    """Monitor system resources"""
    
    def __init__(self):
        self.process_name = 'python'  # Or specific process
        self.asr_processes = []
        
    def find_asr_processes(self) -> List[psutil.Process]:
        """Find ASR-related processes"""
        asr_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = ' '.join(proc.info['cmdline'] or [])
                if 'asr.py' in cmdline or 'listnr' in cmdline:
                    asr_processes.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return asr_processes
    
    def get_stats(self) -> Dict:
        """Get system statistics"""
        stats = {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
        }
        
        # ASR process specific stats
        asr_procs = self.find_asr_processes()
        if asr_procs:
            total_cpu = sum(p.cpu_percent() for p in asr_procs)
            total_mem = sum(p.memory_percent() for p in asr_procs)
            stats['asr_cpu'] = f"{total_cpu:.1f}%"
            stats['asr_memory'] = f"{total_mem:.1f}%"
            stats['asr_processes'] = len(asr_procs)
        
        # Check if Ollama is running
        for proc in psutil.process_iter(['name']):
            try:
                if 'ollama' in proc.info['name'].lower():
                    stats['ollama_running'] = True
                    break
            except:
                pass
        else:
            stats['ollama_running'] = False
        
        return stats


class MonitorUI:
    """Terminal UI for monitoring"""
    
    def __init__(self, stdscr, metrics: TranscriptionMetrics, sys_monitor: SystemMonitor):
        self.stdscr = stdscr
        self.metrics = metrics
        self.sys_monitor = sys_monitor
        self.update_queue = queue.Queue()
        self.recent_transcriptions = deque(maxlen=10)
        self.running = True
        
        # Setup curses
        curses.curs_set(0)  # Hide cursor
        self.stdscr.nodelay(True)  # Non-blocking input
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
        curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)
        curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)
        curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)
        
    def draw_header(self, y: int) -> int:
        """Draw header section"""
        self.stdscr.addstr(y, 0, "╔" + "═" * 78 + "╗")
        y += 1
        self.stdscr.addstr(y, 0, "║", curses.color_pair(1))
        self.stdscr.addstr(y, 2, "LISTNR ASR MONITOR", curses.color_pair(1) | curses.A_BOLD)
        self.stdscr.addstr(y, 35, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), curses.color_pair(4))
        self.stdscr.addstr(y, 79, "║", curses.color_pair(1))
        y += 1
        self.stdscr.addstr(y, 0, "╚" + "═" * 78 + "╝")
        return y + 1
    
    def draw_metrics(self, y: int) -> int:
        """Draw metrics section"""
        stats = self.metrics.get_stats()
        
        # Session info
        self.stdscr.addstr(y, 0, "Session Info:", curses.A_BOLD)
        y += 1
        self.stdscr.addstr(y, 2, f"Duration: {stats.get('session_duration', 'N/A')}")
        self.stdscr.addstr(y, 30, f"Total: {stats.get('total_transcriptions', 0)} segments")
        self.stdscr.addstr(y, 55, f"Last: {stats.get('last_activity', 'N/A')}")
        y += 1
        
        # Performance metrics
        self.stdscr.addstr(y, 0, "Performance:", curses.A_BOLD)
        y += 1
        self.stdscr.addstr(y, 2, f"Avg Time: {stats.get('avg_transcription_time', 'N/A')}")
        self.stdscr.addstr(y, 25, f"RTF: {stats.get('rtf', 'N/A')}")
        self.stdscr.addstr(y, 40, f"Rate: {stats.get('transcriptions_per_minute', '0')} /min")
        y += 1
        
        # Quality metrics
        self.stdscr.addstr(y, 0, "Quality:", curses.A_BOLD)
        y += 1
        self.stdscr.addstr(y, 2, f"Corrections: {stats.get('total_corrections', 0)}")
        self.stdscr.addstr(y, 25, f"Correction Rate: {stats.get('avg_correction_ratio', 'N/A')}")
        self.stdscr.addstr(y, 50, f"Avg Words: {stats.get('avg_words_per_segment', 'N/A')}")
        y += 1
        
        # Correction types
        if 'corrections' in stats:
            self.stdscr.addstr(y, 0, "Correction Types:", curses.A_BOLD)
            y += 1
            corr_str = ", ".join(f"{k}: {v}" for k, v in stats['corrections'].items())
            self.stdscr.addstr(y, 2, corr_str[:76])
            y += 1
        
        return y + 1
    
    def draw_system(self, y: int) -> int:
        """Draw system metrics"""
        sys_stats = self.sys_monitor.get_stats()
        
        self.stdscr.addstr(y, 0, "System:", curses.A_BOLD)
        y += 1
        
        # CPU and Memory
        cpu = sys_stats.get('cpu_percent', 0)
        mem = sys_stats.get('memory_percent', 0)
        
        cpu_color = curses.color_pair(1) if cpu < 50 else curses.color_pair(2) if cpu < 80 else curses.color_pair(3)
        mem_color = curses.color_pair(1) if mem < 50 else curses.color_pair(2) if mem < 80 else curses.color_pair(3)
        
        self.stdscr.addstr(y, 2, f"CPU: {cpu:.1f}%", cpu_color)
        self.stdscr.addstr(y, 20, f"Memory: {mem:.1f}%", mem_color)
        self.stdscr.addstr(y, 40, f"Disk: {sys_stats.get('disk_usage', 0):.1f}%")
        y += 1
        
        # ASR process info
        if 'asr_processes' in sys_stats:
            self.stdscr.addstr(y, 2, f"ASR Processes: {sys_stats['asr_processes']}")
            self.stdscr.addstr(y, 25, f"ASR CPU: {sys_stats.get('asr_cpu', 'N/A')}")
            self.stdscr.addstr(y, 45, f"ASR Mem: {sys_stats.get('asr_memory', 'N/A')}")
            y += 1
        
        # Ollama status
        ollama_status = "Running" if sys_stats.get('ollama_running', False) else "Not Running"
        ollama_color = curses.color_pair(1) if sys_stats.get('ollama_running', False) else curses.color_pair(3)
        self.stdscr.addstr(y, 2, f"Ollama: {ollama_status}", ollama_color)
        
        return y + 2
    
    def draw_recent(self, y: int) -> int:
        """Draw recent transcriptions"""
        self.stdscr.addstr(y, 0, "Recent Transcriptions:", curses.A_BOLD)
        y += 1
        
        max_y, max_x = self.stdscr.getmaxyx()
        available_lines = max_y - y - 3
        
        for i, trans in enumerate(list(self.recent_transcriptions)[-available_lines:]):
            if y >= max_y - 2:
                break
            
            timestamp = trans.get('timestamp', '')[:8]  # Time only
            text = trans.get('text', '')[:65]  # Truncate long text
            source = 'R' if 'raw' in trans.get('source', '').lower() else 'C'
            
            color = curses.color_pair(4) if source == 'R' else curses.color_pair(1)
            self.stdscr.addstr(y, 0, f"[{timestamp}]", curses.color_pair(2))
            self.stdscr.addstr(y, 11, f"[{source}]", color)
            self.stdscr.addstr(y, 15, text)
            y += 1
        
        return y
    
    def draw(self):
        """Main draw function"""
        self.stdscr.clear()
        
        y = 0
        y = self.draw_header(y)
        y = self.draw_metrics(y)
        y = self.draw_system(y)
        y = self.draw_recent(y)
        
        # Footer
        max_y, _ = self.stdscr.getmaxyx()
        self.stdscr.addstr(max_y - 1, 0, "[Q] Quit | [C] Clear | [R] Reset Stats", curses.color_pair(2))
        
        self.stdscr.refresh()
    
    def run(self):
        """Main UI loop"""
        while self.running:
            # Check for updates
            try:
                while True:
                    update = self.update_queue.get_nowait()
                    if update['type'] == 'transcription':
                        self.recent_transcriptions.append(update)
            except queue.Empty:
                pass
            
            # Draw UI
            self.draw()
            
            # Handle input
            try:
                key = self.stdscr.getch()
                if key == ord('q') or key == ord('Q'):
                    self.running = False
                elif key == ord('c') or key == ord('C'):
                    self.recent_transcriptions.clear()
                elif key == ord('r') or key == ord('R'):
                    self.metrics.__init__(self.metrics.window_size)
            except:
                pass
            
            time.sleep(0.5)


def run_monitor(config_path: str = None, watch_paths: List[str] = None):
    """Main monitor function"""
    
    # Load config if provided
    if config_path:
        sys.path.insert(0, os.path.dirname(config_path))
        import config_manager as cfg
        cfg.CONFIG_FILE = config_path
        cfg.load_config()
    
    # Default watch paths
    if not watch_paths:
        watch_paths = [
            os.path.expanduser('~/transcripts_raw.txt'),
            os.path.expanduser('~/transcripts_clean.txt'),
        ]
    
    # Filter existing paths
    watch_paths = [p for p in watch_paths if os.path.exists(p)]
    if not watch_paths:
        print("No transcript files found to monitor. Please run ASR first.")
        return
    
    # Initialize components
    metrics = TranscriptionMetrics()
    sys_monitor = SystemMonitor()
    
    # Setup file monitoring
    update_queue = queue.Queue()
    event_handler = FileMonitor(metrics, update_queue)
    observer = Observer()
    
    for path in watch_paths:
        observer.schedule(event_handler, path, recursive=False)
    
    observer.start()
    
    # Run UI
    def ui_wrapper(stdscr):
        ui = MonitorUI(stdscr, metrics, sys_monitor)
        ui.update_queue = update_queue
        
        # Handle graceful shutdown
        def signal_handler(sig, frame):
            ui.running = False
        signal.signal(signal.SIGINT, signal_handler)
        
        ui.run()
    
    try:
        curses.wrapper(ui_wrapper)
    finally:
        observer.stop()
        observer.join()


def main():
    parser = argparse.ArgumentParser(
        description='Monitor listnr ASR transcription system',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Monitor default transcript files
  %(prog)s -w ~/my_transcripts.txt  # Monitor specific file
  %(prog)s --json output.json       # Monitor JSON output (if implemented)
  %(prog)s --config ~/.config/listnr/config.ini  # Use specific config
        """
    )
    
    parser.add_argument('-w', '--watch', 
                       action='append',
                       dest='watch_paths',
                       help='File path to monitor (can be used multiple times)')
    
    parser.add_argument('-c', '--config',
                       help='Path to listnr config file')
    
    parser.add_argument('--json',
                       help='Monitor JSON output file')
    
    parser.add_argument('--no-ui',
                       action='store_true',
                       help='Run without UI, output to console')
    
    args = parser.parse_args()
    
    # Build watch paths
    watch_paths = args.watch_paths or []
    if args.json and os.path.exists(args.json):
        watch_paths.append(args.json)
    
    # Run monitor
    try:
        run_monitor(config_path=args.config, watch_paths=watch_paths)
    except KeyboardInterrupt:
        print("\nMonitor stopped.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()