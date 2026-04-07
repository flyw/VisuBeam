import logging
import time
import wave
import numpy as np
import os
import threading
import shutil
import datetime 
import soundfile as sf # Add soundfile support

from src.core.config.settings import SystemConfiguration
from src.core.utils.log_manager import create_log_directory

logger = logging.getLogger(__name__)

class OutputSaver:
    def __init__(self,
                 config: SystemConfiguration,
                 sample_rate: int,
                 num_channels: int,
                 log_directory: str = None):
        """
        Initializes the DOA Output Saver.
        This class is responsible for saving intermediate audio data to WAV/FLAC files.

        Args:
            config: DOA configuration
            sample_rate: Audio sample rate
            num_channels: Number of audio channels
            log_directory: Directory to save output files to. If None, creates a new one.
        """
        self.config = config
        self._lock = threading.Lock()
        self.wpe_output_file = None
        self.denoised_output_file = None
        self.mcra_output_file = None
        self.original_audio_file = None
        self.mvdr_output_file = None
        self.apm_output_file = None # WebRTC APM output
        self.dtln_output_file = None

        # Default to output mono audio regardless of input channels
        self.output_as_mono = True  # Always output mono
        self.num_channels_to_save = 1

        # 保存初始化参数以供close时使用
        self.sample_rate = sample_rate

        # Use provided log directory or create a new one
        if log_directory:
            self.log_dir = log_directory
        else:
            self.log_dir = create_log_directory()

        # Helper to check if saving is enabled in either config
        def is_enabled(attr_path, default=False):
            # Check enhancement config first
            if hasattr(self.config, 'enhancement') and self.config.enhancement:
                obj = self.config.enhancement
                for part in attr_path.split('.'):
                    if hasattr(obj, part):
                        obj = getattr(obj, part)
                    else:
                        obj = None
                        break
                if obj is not None and obj is not False:
                    return True
            
            # Check doa config
            if hasattr(self.config, 'doa') and self.config.doa:
                obj = self.config.doa
                for part in attr_path.split('.'):
                    if hasattr(obj, part):
                        obj = getattr(obj, part)
                    else:
                        obj = None
                        break
                if obj is not None and obj is not False:
                    return True
            
            return default

        # Initialize output files if saving is enabled
        if self.config.wpe and self.config.wpe.save_output:
            filename = "wpe_output.wav"
            filepath = os.path.join(self.log_dir, filename)
            self.wpe_output_file = wave.open(filepath, 'wb')
            self.wpe_output_file.setsampwidth(2)  # 16-bit
            self.wpe_output_file.setframerate(sample_rate)
            self.wpe_output_file.setnchannels(self.num_channels_to_save)
            print(f"WPE output will be saved to: {filepath}")

        if is_enabled('save_denoised_output'):
            filename = "denoised_output.flac"
            filepath = os.path.join(self.log_dir, filename)
            self.denoised_output_file = sf.SoundFile(filepath, mode='w', samplerate=self.sample_rate, channels=self.num_channels_to_save, subtype='PCM_16')
            print(f"Denoised output will be saved to: {filepath}")

        if is_enabled('save_mcra_output'):
            filename = "mcra_output.wav"
            filepath = os.path.join(self.log_dir, filename)
            self.mcra_output_file = wave.open(filepath, 'wb')
            self.mcra_output_file.setsampwidth(2)  # 16-bit
            self.mcra_output_file.setframerate(sample_rate)
            self.mcra_output_file.setnchannels(self.num_channels_to_save)
            print(f"MCRA output will be saved to: {filepath}")

        if is_enabled('enable_mvdr_output_wav') or is_enabled('enable_mvdr_output'):
            filename = "mvdr_output.flac"
            filepath = os.path.join(self.log_dir, filename)
            self.mvdr_output_file = sf.SoundFile(filepath, mode='w', samplerate=self.sample_rate, channels=self.num_channels_to_save, subtype='PCM_16')
            print(f"MVDR output will be saved to: {filepath}")
        
        if is_enabled('save_apm_output'):
            filename = "apm_output.flac"
            filepath = os.path.join(self.log_dir, filename)
            self.apm_output_file = sf.SoundFile(filepath, mode='w', samplerate=self.sample_rate, channels=self.num_channels_to_save, subtype='PCM_16')
            print(f"APM output will be saved to: {filepath}")
        
        if is_enabled('dtln.save_output'):
            filename = "dtln_output.wav"
            filepath = os.path.join(self.log_dir, filename)
            self.dtln_output_file = wave.open(filepath, 'wb')
            self.dtln_output_file.setsampwidth(2)
            self.dtln_output_file.setframerate(self.sample_rate)
            self.dtln_output_file.setnchannels(1) # DTLN output is always mono
            print(f"DTLN output will be saved to: {filepath}")

        # Initialize MVDR log file path (no header written here for log format)
        self.mvdr_log_file_path = os.path.join(self.log_dir, "mvdr_results.log")

    def _convert_to_mono(self, audio_data: np.ndarray) -> np.ndarray:
        """将多通道音频数据转换为单声道"""
        if audio_data.ndim == 1:
            # 已经是单声道
            return audio_data
        elif audio_data.ndim == 2:
            # 多通道音频，取平均值合并为单声道
            mono_audio = np.mean(audio_data, axis=1, dtype=audio_data.dtype)
            return mono_audio
        else:
            # 异常情况，返回原始数据
            return audio_data

    def save_original_chunk(self, audio_chunk: np.ndarray):
        """Callback function to save a chunk of original audio."""
        with self._lock:
            if self.original_audio_file:
                # 由于现在总是输出mono，先转换
                audio_chunk = self._convert_to_mono(audio_chunk)
                # 确保音频数据是正确的格式
                if audio_chunk.ndim > 1 and audio_chunk.shape[1] > 1:
                    # 这是多通道音频
                    audio_bytes = audio_chunk.astype(np.int16).tobytes()
                else:
                    # 单通道音频
                    audio_bytes = audio_chunk.astype(np.int16).tobytes()
                self.original_audio_file.writeframes(audio_bytes)

    def save_wpe_chunk(self, audio_chunk: np.ndarray):
        """Callback function to save a chunk of WPE-processed audio."""
        with self._lock:
            if self.wpe_output_file:
                # Since we always output mono, convert first
                audio_chunk = self._convert_to_mono(audio_chunk)
                # Scale float32 audio to int16 range before saving
                audio_bytes = (audio_chunk * 32767).astype(np.int16).tobytes()
                self.wpe_output_file.writeframes(audio_bytes)

    def save_denoised_chunk(self, audio_chunk: np.ndarray):
        """Callback function to save a chunk of denoised audio."""
        with self._lock:
            if self.denoised_output_file:
                # Since we always output mono, convert first
                audio_chunk = self._convert_to_mono(audio_chunk)
                # SoundFile accepts float32, no need to scale to int16 manually
                self.denoised_output_file.write(audio_chunk)

    def save_mcra_chunk(self, audio_chunk: np.ndarray):
        """Callback function to save a chunk of MCRA-processed audio."""
        with self._lock:
            if self.mcra_output_file:
                # Since we always output mono, convert first
                audio_chunk = self._convert_to_mono(audio_chunk)
                # Scale float32 audio to int16 range before saving
                audio_bytes = (audio_chunk * 32767).astype(np.int16).tobytes()
                self.mcra_output_file.writeframes(audio_bytes)

    def save_mvdr_chunk(self, audio_chunk: np.ndarray, metadata=None):
        """Callback function to save a chunk of MVDR-processed audio."""
        with self._lock:
            if self.mvdr_output_file:
                # Since we always output mono, convert first
                audio_chunk = self._convert_to_mono(audio_chunk)
                # SoundFile accepts float32
                self.mvdr_output_file.write(audio_chunk)
    
    def save_apm_chunk(self, audio_chunk: np.ndarray):
        """Callback function to save a chunk of WebRTC APM-processed audio."""
        with self._lock:
            if self.apm_output_file:
                audio_chunk = self._convert_to_mono(audio_chunk)
                self.apm_output_file.write(audio_chunk)
            
    def save_dtln_chunk(self, audio_chunk: np.ndarray, metadata=None):
        """Callback function to save a chunk of DTLN-processed audio."""
        logger.debug(f"OS - Saving DTLN chunk: shape={audio_chunk.shape}, samples={audio_chunk.size}")
        with self._lock:
            if self.dtln_output_file:
                # DTLN output is mono, so no conversion needed
                audio_bytes = (audio_chunk * 32767).astype(np.int16).tobytes()
                self.dtln_output_file.writeframes(audio_bytes)

    def close(self):
        """Closes any open files."""
        with self._lock:
            if self.wpe_output_file:
                print("Closing WPE output file...")
                self.wpe_output_file.close()
                self.wpe_output_file = None
            if self.denoised_output_file:
                print("Closing Denoised output file...")
                self.denoised_output_file.close()
                self.denoised_output_file = None
            if self.mcra_output_file:
                print("Closing MCRA output file...")
                self.mcra_output_file.close()
                self.mcra_output_file = None
            if self.original_audio_file:
                print("Closing Original audio file...")
                self.original_audio_file.close()
                self.original_audio_file = None
            if self.mvdr_output_file:
                print("Closing MVDR output file...")
                self.mvdr_output_file.close()
                self.mvdr_output_file = None
            if self.apm_output_file:
                print("Closing APM output file...")
                self.apm_output_file.close()
                self.apm_output_file = None
            if self.dtln_output_file:
                print("Closing DTLN output file...")
                self.dtln_output_file.close()
                self.dtln_output_file = None

    def log_doa_result(self, timestamp: float, frame_idx: int, detected_angles: list):
        """Log DOA results to file"""
        import os
        log_file_path = os.path.join(self.log_dir, "doa_results.log")
        with open(log_file_path, "a", encoding="utf-8") as log_file:
            log_file.write(
                f"DOA - Time: {timestamp:.3f}s, Frame: {frame_idx}, Detected Angles: {detected_angles}\n")

    def log_mvdr_decision(self, timestamp: float, frame_idx: int, angle: float, energy: float, decision_type: str, locked_angle: float, tolerance: float):
        """Logs the MVDR decision for a single frame to a log file."""
        angle_str = f"{angle:.2f}" if angle is not None else "N/A"
        energy_str = f"{energy:.4f}" if energy is not None else "N/A"
        
        locked_angle_str = f"{locked_angle:.2f}" if locked_angle is not None else "N/A"
        tolerance_str = f"{tolerance:.2f}" if tolerance is not None else "N/A"

        dt_object_utc = datetime.datetime.fromtimestamp(timestamp, tz=datetime.timezone.utc)
        formatted_timestamp = dt_object_utc.strftime("%Y-%m-%dT%H:%M:%S") + f".{dt_object_utc.microsecond // 1000:03d}Z"

        log_line = (
            f"MVDR - Time: {formatted_timestamp}, Frame: {frame_idx}, "
            f"Detected Angle: {angle_str}, Energy: {energy_str}, "
            f"Decision: {decision_type}, "
            f"Locked Angle: {locked_angle_str} (Tolerance: {tolerance_str})\n"
        )
        with open(self.mvdr_log_file_path, "a", encoding="utf-8") as log_file:
            log_file.write(log_line)


class NoOpOutputSaver(OutputSaver):
    """
    A dummy OutputSaver that does nothing.
    Used in live mode to prevent saving intermediate files to the global log directory.
    """
    def __init__(self, *args, **kwargs):
        # Do not call super().__init__ to avoid creating directories or files
        self.log_dir = None
        pass

    def save_original_chunk(self, audio_chunk: np.ndarray):
        pass

    def save_wpe_chunk(self, audio_chunk: np.ndarray):
        pass

    def save_denoised_chunk(self, audio_chunk: np.ndarray):
        pass

    def save_mcra_chunk(self, audio_chunk: np.ndarray):
        pass

    def save_mvdr_chunk(self, audio_chunk: np.ndarray, metadata=None):
        pass

    def save_dtln_chunk(self, audio_chunk: np.ndarray, metadata=None):
        pass

    def close(self):
        pass

    def log_doa_result(self, timestamp: float, frame_idx: int, detected_angles: list):
        pass

    def log_mvdr_decision(self, timestamp: float, frame_idx: int, angle: float, energy: float, decision_type: str, locked_angle: float, tolerance: float):
        pass
