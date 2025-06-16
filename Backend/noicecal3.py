#!/usr/bin/env python3
import os
import sys
import argparse
import logging
import time
import tempfile
import threading
import json
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict, Tuple, Any
from threading import Lock
import asyncio
import pkg_resources

# Suppress TensorFlow informational logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import required packages
from faster_whisper import WhisperModel
import pyaudio
import wave
from colorama import Fore, Style, init
import torch
import numpy as np
import librosa
from scipy.signal import butter, filtfilt
from transformers import MarianMTModel, MarianTokenizer, M2M100ForConditionalGeneration, M2M100Tokenizer
import soundfile as sf  # For writing audio

# For FastAPI integration
try:
    from fastapi import FastAPI, File, UploadFile, Form, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
except ImportError:
    pass

# For Flask integration
try:
    from flask import Flask, request, jsonify
except ImportError:
    pass

# Initialize colorama for colored terminal output
init(autoreset=True)

# ---------------------- Logging Configuration ---------------------- #
logger = logging.getLogger()
debug_mode = os.environ.get("DEBUG_MODE", "0") == "1"
logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)
console_handler = logging.StreamHandler()
console_formatter = logging.Formatter(
    f'{Fore.CYAN}%(asctime)s.%(msecs)03d{Style.RESET_ALL} - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)
file_handler = logging.FileHandler("app.log")
file_formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# ---------------------- Constants and Language Settings ---------------------- #
AUDIO_FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
VALID_MODELS = ["tiny", "base", "small", "medium", "large"]

SUPPORTED_INPUT_LANGUAGES: Dict[str, str] = {
    "auto": "Auto-detect",
    "en": "English",
    "ta": "Tamil",
    "hi": "Hindi",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "ru": "Russian"
}

SUPPORTED_OUTPUT_LANGUAGES: Dict[str, str] = {
    "en": "English",
    "ta": "Tamil",
    "hi": "Hindi"
}

TRANSLATION_MODELS: Dict[str, str] = {
    "en-hi": "Helsinki-NLP/opus-mt-en-hi",
    "ta-en": "Helsinki-NLP/opus-mt-tc-big-ta-en",
    "hi-en": "Helsinki-NLP/opus-mt-hi-en",
    "ta-hi": "Helsinki-NLP/opus-mt-ta-hi",
    "hi-ta": "Helsinki-NLP/opus-mt-hi-ta"
}

# ---------------------- Audio Preprocessing ---------------------- #
class AudioPreprocessor:
    @staticmethod
    def apply_noise_reduction(audio: np.ndarray, sr: int = SAMPLE_RATE, scaling_factor: float = 2**15) -> np.ndarray:
        try:
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
            _ = librosa.pcen(mel_spec * scaling_factor, sr=sr)
            D = librosa.stft(audio.astype(np.float32))
            mag = np.abs(D)
            threshold = np.median(mag) * 0.5
            mask = mag > threshold
            clean_D = D * mask
            cleaned_audio = librosa.istft(clean_D)
            return cleaned_audio
        except Exception as e:
            logging.error(f"Noise reduction failed: {str(e)}")
            return audio

    @staticmethod
    def apply_high_pass_filter(audio: np.ndarray, sr: int = SAMPLE_RATE, cutoff: float = 80.0, order: int = 4) -> np.ndarray:
        nyq = 0.5 * sr
        normal_cutoff = cutoff / nyq
        if normal_cutoff >= 1.0:
            logging.warning("High-pass filter cutoff frequency is too high; skipping.")
            return audio
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        filtered_audio = filtfilt(b, a, audio)
        return filtered_audio

    @staticmethod
    def normalize_audio(audio: np.ndarray) -> np.ndarray:
        peak = np.max(np.abs(audio))
        return audio / peak if peak > 0 else audio

    @staticmethod
    def detect_voice_activity(audio: np.ndarray, threshold: float = 0.5) -> bool:
        """Detect if audio chunk contains voice activity"""
        energy = np.mean(np.abs(audio))
        return energy > threshold

    @staticmethod
    def remove_silence(audio: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Remove silent parts from audio"""
        energy = np.abs(audio)
        mask = energy > (threshold * np.mean(energy))
        return audio[mask]

# ---------------------- Translation ---------------------- #
class Translator:
    def __init__(self) -> None:
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        logging.info("Initializing translator...")

    def get_translation_pair(self, source_lang: str, target_lang: str) -> Tuple[Optional[Any], Optional[Any]]:
        model_key = f"{source_lang}-{target_lang}"
        if source_lang == "en" and target_lang == "ta":
            if model_key not in self.models:
                try:
                    logging.info("Loading M2M100 model for en->ta translation...")
                    self.tokenizers[model_key] = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
                    self.models[model_key] = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
                except Exception as e:
                    logging.error(f"Failed to load M2M100 model for en-ta: {str(e)}")
                    return None, None
            return self.models[model_key], self.tokenizers[model_key]
        if model_key in TRANSLATION_MODELS:
            model_name = TRANSLATION_MODELS[model_key]
        elif source_lang != "en" and target_lang != "en":
            logging.info(f"No direct translation model for {model_key}, using English as pivot")
            return None, None
        else:
            logging.error(f"Unsupported translation pair: {model_key}")
            return None, None
        if model_key not in self.models:
            try:
                logging.info(f"Loading translation model for {model_key}...")
                self.tokenizers[model_key] = MarianTokenizer.from_pretrained(model_name)
                self.models[model_key] = MarianMTModel.from_pretrained(model_name)
                logging.info(f"Translation model for {model_key} loaded successfully")
            except Exception as e:
                logging.error(f"Failed to load translation model for {model_key}: {str(e)}")
                return None, None
        return self.models.get(model_key), self.tokenizers.get(model_key)

    def translate(self, text: str, source_lang: str, target_lang: str, num_beams: int = 5) -> str:
        if source_lang == target_lang:
            return text
        if source_lang == "en" and target_lang == "ta":
            model, tokenizer = self.get_translation_pair(source_lang, target_lang)
            if model and tokenizer:
                try:
                    tokenizer.src_lang = "en"
                    encoded = tokenizer(text, return_tensors="pt")
                    generated_tokens = model.generate(
                        **encoded, forced_bos_token_id=tokenizer.get_lang_id("ta"),
                        num_beams=num_beams, early_stopping=True
                    )
                    return tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
                except Exception as e:
                    logging.error(f"M2M100 translation error: {str(e)}")
                    return text
            return text
        model, tokenizer = self.get_translation_pair(source_lang, target_lang)
        if model and tokenizer:
            try:
                inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                outputs = model.generate(**inputs, num_beams=num_beams, early_stopping=True)
                return tokenizer.decode(outputs[0], skip_special_tokens=True)
            except Exception as e:
                logging.error(f"Translation error: {str(e)}")
                return text
        if source_lang != "en" and target_lang != "en":
            en_text = self.translate(text, source_lang, "en", num_beams=num_beams)
            return self.translate(en_text, "en", target_lang, num_beams=num_beams)
        logging.warning(f"Could not translate: {source_lang} -> {target_lang}")
        return text

# ---------------------- Speech Recognition ---------------------- #
class SpeechRecognizer:
    """
    Enhanced speech recognition system supporting:
      - File transcription, batch processing, microphone recording (CLI and REST),
      - Live captioning via WebSocket streaming, and
      - Synchronous server-side recording sessions.
    """
    def __init__(self) -> None:
        self.stop_event = threading.Event()
        self.audio_buffer = deque(maxlen=int(SAMPLE_RATE / CHUNK_SIZE * 5))
        self.buffer_lock = threading.Lock()
        self.transcript_history: List[str] = []
        self.current_output_file: Optional[str] = None
        self.output_lock = threading.Lock()
        self.batch_executor = ThreadPoolExecutor(max_workers=4)
        self.translator = Translator()
        self.whisper_model: Optional[WhisperModel] = None
        self.whisper_model_size: Optional[str] = None

        self.advanced_processing = False
        self.advanced_hp_cutoff = 80.0
        self.advanced_hp_order = 4

        # For server-side recording sessions
        self.recording_sessions: Dict[str, Dict[str, Any]] = {}
        self.session_lock = Lock()

    def shutdown(self) -> None:
        logging.info("Shutting down ThreadPoolExecutor...")
        self.batch_executor.shutdown(wait=True)

    # -------------- MODEL LOADING -------------- #
    def _get_model(self, model_size: str = "medium", device: str = "auto") -> WhisperModel:
        if self.whisper_model is not None and self.whisper_model_size == model_size:
            logging.info("Reusing existing Whisper model instance.")
            return self.whisper_model
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        logging.info(f"Initializing {model_size} model on {device.upper()} with {compute_type} precision")
        self.whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type, download_root="./models")
        self.whisper_model_size = model_size
        return self.whisper_model

    # -------------- AUDIO PREPROCESSING -------------- #
    def _preprocess_audio(self, input_path: str, output_path: str, apply_hp: bool = True,
                          hp_cutoff: float = 80.0, hp_order: int = 4) -> None:
        try:
            audio, sr = librosa.load(input_path, sr=SAMPLE_RATE)
            processed = AudioPreprocessor.apply_noise_reduction(audio, sr)
            if apply_hp:
                processed = AudioPreprocessor.apply_high_pass_filter(processed, sr, cutoff=hp_cutoff, order=hp_order)
            processed = AudioPreprocessor.normalize_audio(processed)
            sf.write(output_path, processed, SAMPLE_RATE)
        except Exception as e:
            logging.error(f"Audio preprocessing failed: {str(e)}")
            raise

    # -------------- TRANSCRIBE SINGLE FILE -------------- #
    def transcribe_file(self, audio_path: str, model_size: str = "medium",
                        input_language: Optional[str] = None, output_language: str = "en",
                        beam_size: int = 15, best_of: Optional[int] = None, vad_filter: bool = True,
                        word_timestamps: bool = True, output_format: str = "txt", preprocess: bool = True,
                        temperature: float = 0.0, apply_highpass: bool = True, hp_cutoff: float = 80.0,
                        hp_order: int = 4) -> Optional[Dict]:
        if not os.path.exists(audio_path):
            logging.error(f"File not found: {audio_path}")
            return None
        try:
            model = self._get_model(model_size)
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            temp_output = f"{base_name}_transcript_{time.time()}.tmp"
            final_output = temp_output.replace(".tmp", f".{output_format}")
            processed_path = audio_path
            if preprocess:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as processed_file:
                    self._preprocess_audio(audio_path, processed_file.name, apply_hp=apply_highpass,
                                           hp_cutoff=hp_cutoff, hp_order=hp_order)
                    processed_path = processed_file.name
            logging.info(f"Processing {os.path.basename(audio_path)}...")
            start_time = time.time()
            if best_of is None:
                best_of = beam_size
            segments, info = model.transcribe(
                processed_path,
                language=input_language,
                beam_size=beam_size,
                best_of=best_of,
                vad_filter=vad_filter,
                word_timestamps=word_timestamps,
                temperature=temperature
            )
            segment_list = self._process_segments_to_list(segments)
            detected_language = info.language
            if output_language != detected_language and output_language in SUPPORTED_OUTPUT_LANGUAGES:
                segment_list = self._translate_segments(segment_list, detected_language, output_language)
            with open(temp_output, "w", encoding="utf-8") as f:
                if output_format == "json":
                    json.dump(segment_list, f, ensure_ascii=False, indent=2)
                else:
                    for idx, seg in enumerate(segment_list, 1):
                        if output_format == "txt":
                            f.write(seg["text"] + "\n")
                        elif output_format == "srt":
                            f.write(self._format_srt_segment(seg, idx) + "\n")
            os.rename(temp_output, final_output)
            duration = time.time() - start_time
            logging.info(f"Transcription completed in {duration:.1f}s. Output saved to {final_output}")
            result = {
                'original_language': detected_language,
                'output_language': output_language,
                'language_probability': info.language_probability,
                'segments': segment_list,
                'output_file': final_output
            }
            return result
        except Exception as e:
            logging.error(f"Transcription failed: {str(e)}")
            return None

    # -------------- BATCH TRANSCRIPTION -------------- #
    def transcribe_batch(self, file_paths: List[str], model_size: str = "medium",
                         input_language: Optional[str] = None, output_language: str = "en",
                         **kwargs) -> Dict[str, Optional[Dict]]:
        futures = {
            self.batch_executor.submit(
                self.transcribe_file,
                path,
                model_size=model_size,
                input_language=input_language,
                output_language=output_language,
                **kwargs
            ): path for path in file_paths
        }
        results = {}
        for future in futures:
            try:
                results[futures[future]] = future.result()
            except Exception as e:
                results[futures[future]] = {'error': str(e)}
        return results

    # -------------- LIVE CAPTIONING WITH CALLBACK -------------- #
    def _processor_loop_with_callback(self, model: WhisperModel, input_language: Optional[str],
                                      output_language: str, beam_size: int, best_of: int,
                                      vad_filter: bool, callback: callable) -> None:
        last_update = time.time()
        detected_language = input_language
        while not self.stop_event.is_set():
            if time.time() - last_update < 0.5:
                time.sleep(0.1)
                continue
            with self.buffer_lock:
                frames = list(self.audio_buffer)
                self.audio_buffer.clear()
            if frames:
                tmp_file = None
                try:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        tmp_file = tmp.name
                        self._save_audio_chunk(frames, tmp_file)
                    tmp_to_use = tmp_file
                    if self.advanced_processing:
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as proc_tmp:
                            proc_tmp_path = proc_tmp.name
                        try:
                            self._preprocess_audio(tmp_file, proc_tmp_path, apply_hp=True,
                                                   hp_cutoff=self.advanced_hp_cutoff,
                                                   hp_order=self.advanced_hp_order)
                            tmp_to_use = proc_tmp_path
                        except Exception as e:
                            logging.error(f"Advanced processing failed: {e}")
                    segments, info = model.transcribe(
                        tmp_to_use,
                        language=input_language,
                        beam_size=beam_size,
                        best_of=best_of,
                        vad_filter=vad_filter
                    )
                    if not input_language:
                        detected_language = info.language
                    for segment in segments:
                        clean_text = segment.text.strip()
                        if clean_text:
                            if output_language != detected_language and output_language in SUPPORTED_OUTPUT_LANGUAGES:
                                translated_text = self.translator.translate(clean_text, detected_language, output_language)
                            else:
                                translated_text = clean_text
                            callback(translated_text)
                except Exception as proc_err:
                    logging.error(f"Live transcription processing error: {proc_err}")
                finally:
                    if tmp_file and os.path.exists(tmp_file):
                        try:
                            os.remove(tmp_file)
                        except Exception as cleanup_error:
                            logging.warning(f"Failed to remove temporary file: {cleanup_error}")
                last_update = time.time()

    def live_caption_with_callback(self, options: dict, callback: callable) -> None:
        """
        Start live captioning and stream transcript chunks via the callback.
        Runs for a fixed duration (60 seconds for demonstration).
        """
        model_size = options.get("modelSize", "medium")
        input_lang = options.get("inputLanguage", "auto")
        output_lang = options.get("outputLanguage", "en")
        beam_size = options.get("beamSize", 15)
        best_of = options.get("bestOf", beam_size)
        vad_filter = options.get("vadFilter", True)

        model = self._get_model(model_size)
        self.stop_event.clear()
        recorder_thread = threading.Thread(target=self._recorder_loop, daemon=True)
        processor_thread = threading.Thread(
            target=self._processor_loop_with_callback,
            args=(model, input_lang, output_lang, beam_size, best_of, vad_filter, callback),
            daemon=True
        )
        recorder_thread.start()
        processor_thread.start()
        # Run live captioning for 60 seconds (adjust as needed)
        time.sleep(60)
        self.stop_event.set()
        recorder_thread.join(timeout=2)
        processor_thread.join(timeout=2)

    # -------------- CLI: RECORD FROM MICROPHONE -------------- #
    def transcribe_from_microphone(self, duration: float, model_size: str = "medium",
                                   input_language: Optional[str] = None, output_language: str = "en",
                                   beam_size: int = 15, best_of: Optional[int] = None, vad_filter: bool = True,
                                   word_timestamps: bool = True, output_format: str = "txt", temperature: float = 0.0,
                                   apply_highpass: bool = True, hp_cutoff: float = 80.0, hp_order: int = 4) -> Optional[Dict]:
        if best_of is None:
            best_of = beam_size
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            self._record_audio(duration, tmpfile.name)
            return self.transcribe_file(
                tmpfile.name,
                model_size=model_size,
                input_language=None if input_language == "auto" else input_language,
                output_language=output_language,
                beam_size=beam_size,
                best_of=best_of,
                vad_filter=vad_filter,
                word_timestamps=word_timestamps,
                output_format=output_format,
                preprocess=True,
                temperature=temperature,
                apply_highpass=apply_highpass,
                hp_cutoff=hp_cutoff,
                hp_order=hp_order
            )

    def _record_audio(self, duration: float, output_path: str) -> None:
        try:
            p = pyaudio.PyAudio()
            try:
                stream = p.open(format=AUDIO_FORMAT, channels=CHANNELS, rate=SAMPLE_RATE,
                                input=True, frames_per_buffer=CHUNK_SIZE)
            except Exception as e:
                logging.error(f"Failed to open microphone stream: {str(e)}")
                p.terminate()
                raise Exception("Microphone not accessible")
            logging.info(f"{Fore.YELLOW}Recording for {duration} seconds...")
            frames = []
            try:
                for _ in range(0, int(SAMPLE_RATE / CHUNK_SIZE * duration)):
                    data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                    frames.append(data)
            except Exception as e:
                logging.error(f"Recording error: {e}")
            finally:
                stream.stop_stream()
                stream.close()
                p.terminate()
            self._save_audio_chunk(frames, output_path)
            logging.info(f"{Fore.GREEN}Recording saved to {output_path}")
        except Exception as e:
            logging.error(f"Error during microphone recording: {str(e)}")
            raise

    def _save_audio_chunk(self, frames: List[bytes], filename: str) -> None:
        try:
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(pyaudio.get_sample_size(AUDIO_FORMAT))
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(b''.join(frames))
        except Exception as e:
            logging.error(f"Error saving audio chunk: {e}")

    def _format_srt_segment(self, segment: Dict, idx: int) -> str:
        return (f"{idx}\n"
                f"{self._sec_to_srt(segment['start'])} --> {self._sec_to_srt(segment['end'])}\n"
                f"{segment['text']}\n")

    def _sec_to_srt(self, seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        sec = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{sec:06.3f}".replace(".", ",")

    def _process_segments_to_list(self, segments) -> List[Dict]:
        results = []
        for segment in segments:
            segment_data = {
                'start': segment.start,
                'end': segment.end,
                'text': segment.text.strip(),
                'confidence': segment.avg_logprob,
                'words': [{
                    'word': word.word,
                    'start': word.start,
                    'end': word.end,
                    'confidence': word.probability
                } for word in getattr(segment, 'words', [])]
            }
            results.append(segment_data)
        return results

    def _translate_segments(self, segments: List[Dict], source_lang: str, target_lang: str) -> List[Dict]:
        if source_lang == target_lang:
            return segments
        translated_segments = []
        for segment in segments:
            translated_segment = segment.copy()
            translated_segment['text'] = self.translator.translate(segment['text'], source_lang, target_lang)
            if 'words' in segment and segment['words']:
                translated_words = []
                for word in segment['words']:
                    translated_word = word.copy()
                    translated_word['word'] = self.translator.translate(word['word'], source_lang, target_lang)
                    translated_words.append(translated_word)
                translated_segment['words'] = translated_words
            translated_segments.append(translated_segment)
        return translated_segments

    # ---------------------- SYNCHRONOUS SERVER-SIDE RECORDING ---------------------- #
    def start_recording_sync(self, session_id: str) -> None:
        try:
            p = pyaudio.PyAudio()
            try:
                stream = p.open(format=AUDIO_FORMAT, channels=CHANNELS, rate=SAMPLE_RATE,
                                input=True, frames_per_buffer=CHUNK_SIZE)
            except Exception as e:
                logging.error(f"Failed to open microphone for recording: {str(e)}")
                p.terminate()
                raise Exception("Microphone not accessible")
            frames = []
            logging.info(f"{Fore.YELLOW}Recording (server-side) for 5 seconds...")
            try:
                for _ in range(0, int(SAMPLE_RATE / CHUNK_SIZE * 5)):
                    data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                    frames.append(data)
            except Exception as e:
                logging.error(f"Recording error during server-side capture: {e}")
            finally:
                stream.stop_stream()
                stream.close()
                p.terminate()
            output_path = f"{session_id}.wav"
            self._save_audio_chunk(frames, output_path)
            logging.info(f"{Fore.GREEN}Recording saved to {output_path}")
            with self.session_lock:
                self.recording_sessions[session_id] = {"status": "done", "file_path": output_path}
        except Exception as e:
            logging.error(f"Error in start_recording_sync: {str(e)}")
            raise

    def stop_recording(self, session_id: str, model_size="medium", input_lang="auto", output_lang="en") -> Optional[Dict]:
        with self.session_lock:
            session = self.recording_sessions.get(session_id)
            if not session:
                logging.error(f"No session found for {session_id}")
                return None
            self.recording_sessions.pop(session_id, None)
        file_path = session["file_path"]
        result = self.transcribe_file(
            file_path,
            model_size=model_size,
            input_language=None if input_lang == "auto" else input_lang,
            output_language=output_lang,
            preprocess=True
        )
        if os.path.exists(file_path):
            os.remove(file_path)
        return result

    def live_caption(self, model_size: str, input_language: Optional[str], output_language: str):
        def print_callback(text: str):
            print(f"{text}\n")
        
        self.live_caption_with_callback(
            {
                "modelSize": model_size,
                "inputLanguage": input_language,
                "outputLanguage": output_language,
                "beamSize": 15,
                "vadFilter": True
            },
            print_callback
        )

    async def process_live_audio(self, audio_data: np.ndarray, model_size: str,
                               input_language: Optional[str], output_language: str) -> Optional[str]:
        try:
            # Use temporary file for processing
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                sf.write(tmp.name, audio_data, SAMPLE_RATE)
                model = self._get_model(model_size)
                segments, info = model.transcribe(
                    tmp.name,
                    language=input_language,
                    beam_size=5,
                    vad_filter=True
                )
                
                text = " ".join(segment.text for segment in segments)
                if text and output_language != info.language:
                    text = self.translator.translate(text, info.language, output_language)
                return text

        except Exception as e:
            logging.error(f"Live audio processing error: {e}")
            return None

    def _cleanup_stale_sessions(self, max_age: int = 3600):
        """Clean up sessions older than max_age seconds"""
        current_time = time.time()
        with self.session_lock:
            for session_id in list(self.recording_sessions.keys()):
                session = self.recording_sessions[session_id]
                if current_time - session.get("start_time", 0) > max_age:
                    self.recording_sessions.pop(session_id, None)

# ---------------------- FASTAPI INTEGRATION ---------------------- #
def create_fastapi_app() -> "FastAPI":
    from fastapi import FastAPI, File, UploadFile, Form, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    recognizer = SpeechRecognizer()

    @app.post("/transcribe")
    async def transcribe(
        file: UploadFile = File(...),
        model_size: str = Form("medium"),
        input_lang: str = Form("auto"),
        output_lang: str = Form("en"),
        out_format: str = Form("txt")
    ):
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_filename = tmp.name
            result = recognizer.transcribe_file(
                tmp_filename,
                model_size=model_size,
                input_language=None if input_lang=="auto" else input_lang,
                output_language=output_lang,
                output_format=out_format,
                preprocess=True
            )
            os.remove(tmp_filename)
            if not result:
                raise HTTPException(status_code=500, detail="Transcription failed")
            return JSONResponse(content=result)
        except Exception as e:
            logging.error(f"FastAPI endpoint error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/transcribe-batch")
    async def transcribe_batch(
        files: List[UploadFile] = File(...),
        model_size: str = Form("medium"),
        input_lang: str = Form("auto"),
        output_lang: str = Form("en"),
        out_format: str = Form("txt")
    ):
        try:
            temp_files = []
            file_paths = []
            for f in files:
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                content = await f.read()
                tmp.write(content)
                tmp.flush()
                temp_files.append(tmp.name)
                file_paths.append(tmp.name)
            results = recognizer.transcribe_batch(
                file_paths,
                model_size=model_size,
                input_language=None if input_lang=="auto" else input_lang,
                output_language=output_lang,
                output_format=out_format,
                preprocess=True
            )
            for t in temp_files:
                if os.path.exists(t):
                    os.remove(t)
            return JSONResponse(content=results)
        except Exception as e:
            logging.error(f"Batch endpoint error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/start-recording")
    async def start_recording(options: dict):
        try:
            model_size = options.get("modelSize", "medium")
            input_lang = options.get("inputLanguage", "auto")
            output_lang = options.get("outputLanguage", "en")
            session_id = str(uuid.uuid4())
            recognizer.start_recording_sync(session_id)
            return JSONResponse(content={"sessionId": session_id})
        except Exception as e:
            logging.error(f"Start-recording endpoint error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/stop-recording")
    async def stop_recording(session_data: dict):
        try:
            session_id = session_data.get("sessionId")
            if not session_id:
                raise HTTPException(status_code=400, detail="No sessionId provided")
            result = recognizer.stop_recording(session_id)
            if not result:
                raise HTTPException(status_code=500, detail="Recording session not finished or not found")
            return JSONResponse(content=result)
        except Exception as e:
            logging.error(f"Stop-recording endpoint error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.websocket("/live-captioning")
    async def live_captioning(websocket: WebSocket):
        session = None
        try:
            await websocket.accept()
            
            # Receive configuration
            data = await websocket.receive_text()
            options = json.loads(data)
            
            # Set default options if not provided
            options.setdefault("modelSize", "medium")
            options.setdefault("inputLanguage", None)  # Auto-detect
            options.setdefault("outputLanguage", "en")
            options.setdefault("vadFilter", True)
            options.setdefault("beamSize", 5)
            options.setdefault("preprocess", True)
            
            session = LiveCaptioningSession(websocket, options)
            await session.start()

            # Main WebSocket loop
            while not session.stop_event.is_set():
                try:
                    # Receive audio data
                    data = await websocket.receive_bytes()
                    await session.audio_buffer.add_with_backpressure(data)
                    
                except WebSocketDisconnect:
                    logging.info("WebSocket disconnected")
                    break
                except Exception as e:
                    logging.error(f"WebSocket error: {e}")
                    break

        except Exception as e:
            logging.error(f"Live captioning error: {e}")
        finally:
            if session:
                await session.stop()

    return app

# ---------------------- FLASK INTEGRATION (OPTIONAL) ---------------------- #
def create_flask_app() -> "Flask":
    from flask import Flask, request, jsonify
    app = Flask(__name__)
    recognizer = SpeechRecognizer()
    @app.route("/transcribe", methods=["POST"])
    def transcribe():
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400
        file = request.files['file']
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                file.save(tmp)
                tmp_filename = tmp.name
            model_size = request.form.get("model_size", "medium")
            input_lang = request.form.get("input_lang", "auto")
            output_lang = request.form.get("output_lang", "en")
            out_format = request.form.get("out_format", "txt")
            result = recognizer.transcribe_file(
                tmp_filename,
                model_size=model_size,
                input_language=None if input_lang=="auto" else input_lang,
                output_language=output_lang,
                output_format=out_format,
                preprocess=True
            )
            os.remove(tmp_filename)
            return jsonify(result)
        except Exception as e:
            logging.error(f"Flask endpoint error: {str(e)}")
            return jsonify({"error": str(e)}), 500
    return app

# ---------------------- MAIN ENTRY POINT ---------------------- #
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enhanced Multilingual Speech Recognition System")
    parser.add_argument("--mode", choices=["file", "batch", "live", "mic"], help="Mode of operation (CLI)")
    parser.add_argument("--input", type=str, help="Input audio file path (for file mode)")
    parser.add_argument("--files", type=str, help="Comma-separated list of audio file paths (for batch mode)")
    parser.add_argument("--duration", type=int, default=10, help="Recording duration in seconds (for mic mode)")
    parser.add_argument("--model_size", choices=VALID_MODELS, default="medium", help="Whisper model size")
    parser.add_argument("--input_lang", default="auto", help="Input language (or 'auto')")
    parser.add_argument("--output_lang", default="en", help="Output language")
    parser.add_argument("--format", choices=["txt", "srt", "json"], default="txt", help="Output format")
    parser.add_argument("--preprocess", action="store_true", help="Enable audio preprocessing")
    parser.add_argument("--high_pass", action="store_true", help="Enable high-pass filtering in preprocessing")
    parser.add_argument("--hp_cutoff", type=float, default=80.0, help="High-pass filter cutoff frequency")
    parser.add_argument("--hp_order", type=int, default=4, help="High-pass filter order")
    parser.add_argument("--temperature", type=float, default=0.0, help="Decoding temperature (0.0 for greedy)")
    parser.add_argument("--advanced", action="store_true", help="Enable advanced processing techniques for all CLI options")
    parser.add_argument("--advanced_hp_cutoff", type=float, default=80.0, help="Advanced high-pass filter cutoff")
    parser.add_argument("--advanced_hp_order", type=int, default=4, help="Advanced high-pass filter order")
    parser.add_argument("--server", choices=["flask", "fastapi"], help="Run as a web server using Flask or FastAPI")
    return parser.parse_args()

def prompt_cli_args(args: argparse.Namespace) -> None:
    if args.mode == "file" and not args.input:
        args.input = input("Audio file path: ").strip()
    elif args.mode == "batch" and not args.files:
        args.files = input("File paths (comma-separated): ").strip()
    elif args.mode == "mic" and not args.duration:
        args.duration = int(input("Duration (seconds): ") or 10)
    if not args.model_size:
        args.model_size = input(f"Model size ({'/'.join(VALID_MODELS)}): ").strip().lower() or "medium"
    if args.model_size not in VALID_MODELS:
        print("Invalid model size. Using default: medium")
        args.model_size = "medium"
    if not args.input_lang:
        args.input_lang = input("Input language (auto for auto-detection): ").strip().lower() or "auto"
    if not args.output_lang:
        args.output_lang = input("Output language: ").strip().lower() or "en"
    if not args.format:
        args.format = input("Output format (txt/srt/json): ").strip().lower() or "txt"
    ans = input("Enable preprocessing (y/n): ").strip().lower()
    args.preprocess = (ans == 'y')
    ans = input("Enable high-pass filtering (y/n): ").strip().lower()
    args.high_pass = (ans == 'y')
    args.hp_cutoff = float(input("High-pass cutoff frequency (default 80.0): ") or "80.0")
    args.hp_order = int(input("High-pass filter order (default 4): ") or "4")
    args.temperature = float(input("Temperature (0.0 for greedy decoding): ") or "0.0")

def interactive_mode() -> None:
    recognizer = SpeechRecognizer()
    adv_choice = input("Enable advanced processing techniques for all options? (y/n): ").strip().lower()
    if adv_choice == "y":
        recognizer.advanced_processing = True
        try:
            cutoff = float(input("Enter advanced high-pass cutoff frequency (default 80.0): ") or "80.0")
            order = int(input("Enter advanced high-pass filter order (default 4): ") or "4")
            recognizer.advanced_hp_cutoff = cutoff
            recognizer.advanced_hp_order = order
        except Exception:
            logging.warning("Invalid input for advanced high-pass parameters. Using default values.")
    try:
        print(f"{Fore.CYAN}\nEnhanced Multilingual Speech Recognition System{Style.RESET_ALL}")
        print("1. Transcribe audio file")
        print("2. Batch process files")
        print("3. Live captioning (local microphone)")
        print("4. Record from microphone (fixed duration)")
        choice = input("Enter choice (1-4): ").strip()
        if choice == "1":
            path = input("Audio file path: ").strip()
            model_size = input(f"Model size ({'/'.join(VALID_MODELS)}): ").strip().lower() or "medium"
            input_lang = input("Input language (auto for auto-detection): ").strip().lower() or "auto"
            output_lang = input("Output language: ").strip().lower() or "en"
            out_format = input("Output format (txt/srt/json): ").strip().lower() or "txt"
            preprocess = input("Enable preprocessing (y/n): ").strip().lower() == 'y'
            hp_choice = input("Enable high-pass filtering (y/n): ").strip().lower() == 'y'
            cutoff = float(input("High-pass cutoff frequency (default 80.0): ") or "80.0")
            order = int(input("High-pass filter order (default 4): ") or "4")
            temp_val = float(input("Temperature (0.0 for greedy decoding): ") or "0.0")
            result = recognizer.transcribe_file(
                path,
                model_size=model_size,
                input_language=None if input_lang == "auto" else input_lang,
                output_language=output_lang,
                output_format=out_format,
                preprocess=preprocess,
                temperature=temp_val,
                apply_highpass=hp_choice,
                hp_cutoff=cutoff,
                hp_order=order
            )
            if result:
                print(f"\nTranscription completed. Output file: {result.get('output_file', 'N/A')}")
        elif choice == "2":
            paths = input("File paths (comma-separated): ").split(',')
            model_size = input(f"Model size ({'/'.join(VALID_MODELS)}): ").strip().lower() or "medium"
            input_lang = input("Input language (auto for auto-detection): ").strip().lower() or "auto"
            output_lang = input("Output language: ").strip().lower() or "en"
            out_format = input("Output format (txt/srt/json): ").strip().lower() or "txt"
            results = recognizer.transcribe_batch(
                [p.strip() for p in paths],
                model_size=model_size,
                input_language=None if input_lang == "auto" else input_lang,
                output_language=output_lang,
                output_format=out_format,
                preprocess=True
            )
            print("\nBatch Results:")
            for path, res in results.items():
                status = "Success" if res and 'segments' in res else "Failed"
                out_file = res.get("output_file", "N/A") if res else "N/A"
                print(f"{path}: {status} | Output file: {out_file}")
        elif choice == "3":
            model_size = input(f"Model size ({'/'.join(VALID_MODELS)}): ").strip().lower() or "medium"
            input_lang = input("Input language (auto for auto-detection): ").strip().lower() or "auto"
            output_lang = input("Output language: ").strip().lower() or "en"
            # For live captioning, stream transcript chunks to the console.
            def print_chunk(chunk: str):
                print(chunk)
            recognizer.live_caption_with_callback(
                {"modelSize": model_size, "inputLanguage": input_lang, "outputLanguage": output_lang, "beamSize": 15, "vadFilter": True},
                print_chunk
            )
        elif choice == "4":
            duration = int(input("Duration (seconds): ") or 10)
            model_size = input(f"Model size ({'/'.join(VALID_MODELS)}): ").strip().lower() or "medium"
            input_lang = input("Input language (auto for auto-detection): ").strip().lower() or "auto"
            output_lang = input("Output language: ").strip().lower() or "en"
            result = recognizer.transcribe_from_microphone(
                duration=duration,
                model_size=model_size,
                input_language=None if input_lang == "auto" else input_lang,
                output_language=output_lang
            )
            if result:
                print(f"\nTranscription completed. Output file: {result.get('output_file', 'N/A')}")
        else:
            print("Invalid choice")
    except Exception as e:
        logging.error(f"Operation failed: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
    finally:
        recognizer.shutdown()

def non_interactive_mode(args: argparse.Namespace) -> None:
    recognizer = SpeechRecognizer()
    if args.advanced:
        recognizer.advanced_processing = True
        recognizer.advanced_hp_cutoff = args.advanced_hp_cutoff
        recognizer.advanced_hp_order = args.advanced_hp_order
    prompt_cli_args(args)
    try:
        mode = args.mode
        input_lang = args.input_lang.lower()
        output_lang = args.output_lang.lower()
        model_size = args.model_size
        out_format = args.format.lower()
        if mode == "file":
            if not args.input:
                logging.error("For file mode, --input must be provided.")
                return
            result = recognizer.transcribe_file(
                args.input,
                model_size=model_size,
                input_language=None if input_lang == "auto" else input_lang,
                output_language=output_lang,
                output_format=out_format,
                preprocess=args.preprocess,
                apply_highpass=args.high_pass,
                hp_cutoff=args.hp_cutoff,
                hp_order=args.hp_order,
                temperature=args.temperature
            )
            print(json.dumps(result, indent=2, ensure_ascii=False))
        elif mode == "batch":
            if not args.files:
                logging.error("For batch mode, --files must be provided.")
                return
            file_list = [f.strip() for f in args.files.split(',')]
            results = recognizer.transcribe_batch(
                file_list,
                model_size=model_size,
                input_language=None if input_lang == "auto" else input_lang,
                output_language=output_lang,
                output_format=out_format,
                preprocess=args.preprocess
            )
            print(json.dumps(results, indent=2, ensure_ascii=False))
        elif mode == "live":
            recognizer.live_caption(
                model_size=model_size,
                input_language=None if input_lang == "auto" else input_lang,
                output_language=output_lang
            )
        elif mode == "mic":
            result = recognizer.transcribe_from_microphone(
                duration=args.duration,
                model_size=model_size,
                input_language=None if input_lang == "auto" else input_lang,
                output_language=output_lang
            )
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            logging.error("Invalid mode selected.")
    finally:
        recognizer.shutdown()

# ---------------------- FASTAPI SERVER ---------------------- #
def create_fastapi_app() -> "FastAPI":
    from fastapi import FastAPI, File, UploadFile, Form, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    recognizer = SpeechRecognizer()

    @app.post("/transcribe")
    async def transcribe(
        file: UploadFile = File(...),
        model_size: str = Form("medium"),
        input_lang: str = Form("auto"),
        output_lang: str = Form("en"),
        out_format: str = Form("txt")
    ):
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_filename = tmp.name
            result = recognizer.transcribe_file(
                tmp_filename,
                model_size=model_size,
                input_language=None if input_lang=="auto" else input_lang,
                output_language=output_lang,
                output_format=out_format,
                preprocess=True
            )
            os.remove(tmp_filename)
            if not result:
                raise HTTPException(status_code=500, detail="Transcription failed")
            return JSONResponse(content=result)
        except Exception as e:
            logging.error(f"FastAPI endpoint error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/transcribe-batch")
    async def transcribe_batch(
        files: List[UploadFile] = File(...),
        model_size: str = Form("medium"),
        input_lang: str = Form("auto"),
        output_lang: str = Form("en"),
        out_format: str = Form("txt")
    ):
        try:
            temp_files = []
            file_paths = []
            for f in files:
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                content = await f.read()
                tmp.write(content)
                tmp.flush()
                temp_files.append(tmp.name)
                file_paths.append(tmp.name)
            results = recognizer.transcribe_batch(
                file_paths,
                model_size=model_size,
                input_language=None if input_lang=="auto" else input_lang,
                output_language=output_lang,
                output_format=out_format,
                preprocess=True
            )
            for t in temp_files:
                if os.path.exists(t):
                    os.remove(t)
            return JSONResponse(content=results)
        except Exception as e:
            logging.error(f"Batch endpoint error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/start-recording")
    async def start_recording(options: dict):
        try:
            model_size = options.get("modelSize", "medium")
            input_lang = options.get("inputLanguage", "auto")
            output_lang = options.get("outputLanguage", "en")
            session_id = str(uuid.uuid4())
            recognizer.start_recording_sync(session_id)
            return JSONResponse(content={"sessionId": session_id})
        except Exception as e:
            logging.error(f"Start-recording endpoint error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/stop-recording")
    async def stop_recording(session_data: dict):
        try:
            session_id = session_data.get("sessionId")
            if not session_id:
                raise HTTPException(status_code=400, detail="No sessionId provided")
            result = recognizer.stop_recording(session_id)
            if not result:
                raise HTTPException(status_code=500, detail="Recording session not finished or not found")
            return JSONResponse(content=result)
        except Exception as e:
            logging.error(f"Stop-recording endpoint error: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.websocket("/live-captioning")
    async def live_captioning(websocket: WebSocket):
        session = None
        try:
            await websocket.accept()
            
            # Receive configuration
            data = await websocket.receive_text()
            options = json.loads(data)
            
            # Set default options if not provided
            options.setdefault("modelSize", "medium")
            options.setdefault("inputLanguage", None)  # Auto-detect
            options.setdefault("outputLanguage", "en")
            options.setdefault("vadFilter", True)
            options.setdefault("beamSize", 5)
            options.setdefault("preprocess", True)
            
            session = LiveCaptioningSession(websocket, options)
            await session.start()

            # Main WebSocket loop
            while not session.stop_event.is_set():
                try:
                    # Receive audio data
                    data = await websocket.receive_bytes()
                    await session.audio_buffer.add_with_backpressure(data)
                    
                except WebSocketDisconnect:
                    logging.info("WebSocket disconnected")
                    break
                except Exception as e:
                    logging.error(f"WebSocket error: {e}")
                    break

        except Exception as e:
            logging.error(f"Live captioning error: {e}")
        finally:
            if session:
                await session.stop()

    return app

# ---------------------- FLASK INTEGRATION (OPTIONAL) ---------------------- #
def create_flask_app() -> "Flask":
    from flask import Flask, request, jsonify
    app = Flask(__name__)
    recognizer = SpeechRecognizer()
    @app.route("/transcribe", methods=["POST"])
    def transcribe():
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400
        file = request.files['file']
        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                file.save(tmp)
                tmp_filename = tmp.name
            model_size = request.form.get("model_size", "medium")
            input_lang = request.form.get("input_lang", "auto")
            output_lang = request.form.get("output_lang", "en")
            out_format = request.form.get("out_format", "txt")
            result = recognizer.transcribe_file(
                tmp_filename,
                model_size=model_size,
                input_language=None if input_lang=="auto" else input_lang,
                output_language=output_lang,
                output_format=out_format,
                preprocess=True
            )
            os.remove(tmp_filename)
            return jsonify(result)
        except Exception as e:
            logging.error(f"Flask endpoint error: {str(e)}")
            return jsonify({"error": str(e)}), 500
    return app

# ---------------------- MAIN ENTRY POINT ---------------------- #
def main() -> None:
    args = parse_arguments()
    if args.server == "flask":
        app = create_flask_app()
        logging.info("Starting Flask server on http://0.0.0.0:5000")
        app.run(host="0.0.0.0", port=5000)
    elif args.server == "fastapi":
        import uvicorn
        app = create_fastapi_app()
        logging.info("Starting FastAPI server on http://0.0.0.0:8000")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        if len(sys.argv) > 1:
            non_interactive_mode(args)
        else:
            interactive_mode()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Exiting...")

# Add version checking for critical dependencies
def check_dependencies():
    required = {
        'faster-whisper': '0.5.0',
        'pyaudio': '0.2.11',
        'torch': '2.0.0',
        'librosa': '0.10.0'
    }
    for package, version in required.items():
        try:
            pkg_resources.require(f"{package}>={version}")
        except pkg_resources.VersionConflict:
            print(f"Warning: {package} version {version} or higher required")
        except pkg_resources.DistributionNotFound:
            print(f"Error: {package} not found")

# Add these new helper classes for live captioning stability
class AudioBuffer:
    def __init__(self, maxlen: int = 50):
        self.buffer = deque(maxlen=maxlen)
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.active = True

    def add(self, data: bytes) -> None:
        with self.lock:
            self.buffer.append(data)
            self.condition.notify()

    def get(self, chunk_size: int) -> List[bytes]:
        with self.lock:
            while len(self.buffer) < chunk_size and self.active:
                self.condition.wait(timeout=1.0)
            return [self.buffer.popleft() for _ in range(min(chunk_size, len(self.buffer)))]

    def clear(self) -> None:
        with self.lock:
            self.buffer.clear()

    def stop(self) -> None:
        with self.lock:
            self.active = False
            self.condition.notify_all()

    async def add_with_backpressure(self, data: bytes):
        while len(self.buffer) >= self.maxlen:
            await asyncio.sleep(0.1)
        self.add(data)

class LiveCaptioningSession:
    def __init__(self, websocket: WebSocket, options: dict):
        self.websocket = websocket
        self.options = options
        self.audio_buffer = AudioBuffer(maxlen=int(SAMPLE_RATE * 2))  # 2 seconds buffer
        self.stop_event = threading.Event()
        self.heartbeat_task = None
        self.processing_task = None
        self.translator = Translator()
        self._setup_processor()

    def _setup_processor(self):
        self.model = self._get_model(self.options.get("modelSize", "medium"))
        self.vad_threshold = self.options.get("vadThreshold", 0.5)
        self.min_silence_duration = self.options.get("minSilenceDuration", 0.5)
        self.last_voice_activity = time.time()
        self.current_chunk = []

    async def start(self):
        self.heartbeat_task = asyncio.create_task(self._heartbeat())
        self.processing_task = asyncio.create_task(self._process_audio())

    async def stop(self):
        self.stop_event.set()
        self.audio_buffer.stop()
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
        if self.processing_task:
            self.processing_task.cancel()
        await self.websocket.close()

    async def _heartbeat(self):
        while not self.stop_event.is_set():
            try:
                await self.websocket.send_json({"type": "ping"})
                await asyncio.sleep(30)
            except Exception as e:
                logging.error(f"Heartbeat error: {e}")
                break
        await self.stop()

    async def _process_audio(self):
        chunk_duration = 0.5  # Process in 0.5-second chunks
        chunk_size = int(SAMPLE_RATE * chunk_duration)
        
        while not self.stop_event.is_set():
            try:
                chunks = self.audio_buffer.get(chunk_size)
                if not chunks:
                    await asyncio.sleep(0.1)
                    continue

                # Process chunks
                result = await self._process_chunks(chunks)
                if result:
                    # Send result with confidence score
                    await self.websocket.send_json({
                        "type": "transcript",
                        "text": result,
                        "timestamp": time.time()
                    })

            except Exception as e:
                logging.error(f"Audio processing error: {e}")
                break

        await self.stop()

    async def _process_chunks(self, chunks: List[bytes]) -> Optional[str]:
        try:
            # Convert bytes to numpy array
            audio_data = np.frombuffer(b''.join(chunks), dtype=np.int16)
            audio_data = audio_data.astype(np.float32) / 32768.0  # Normalize to [-1, 1]

            # Apply preprocessing
            if self.options.get("preprocess", True):
                audio_data = AudioPreprocessor.apply_noise_reduction(audio_data, SAMPLE_RATE)
                if self.options.get("highPassFilter", True):
                    audio_data = AudioPreprocessor.apply_high_pass_filter(
                        audio_data,
                        sr=SAMPLE_RATE,
                        cutoff=self.options.get("hpCutoff", 80.0),
                        order=self.options.get("hpOrder", 4)
                    )

            # Get model and transcribe
            model = self._get_model(self.options.get("modelSize", "medium"))
            segments, info = await asyncio.to_thread(
                model.transcribe,
                audio_data,
                language=self.options.get("inputLanguage"),
                beam_size=self.options.get("beamSize", 5),
                vad_filter=self.options.get("vadFilter", True),
                word_timestamps=True
            )

            # Process segments
            text_parts = []
            for segment in segments:
                if segment.no_speech_prob < 0.5:  # Filter out segments that are likely not speech
                    text_parts.append(segment.text)

            # Translate if needed
            result = " ".join(text_parts).strip()
            if result and self.options.get("outputLanguage") != info.language:
                result = await asyncio.to_thread(
                    self.translator.translate,
                    result,
                    info.language,
                    self.options.get("outputLanguage")
                )

            return result

        except Exception as e:
            logging.error(f"Chunk processing error: {e}")
            return None

    async def reconnect(self):
        try:
            await self.websocket.close()
            await self.websocket.connect()
            await self.start()
        except Exception as e:
            logging.error(f"Reconnection failed: {e}")
