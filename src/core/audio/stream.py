import traceback

import pyaudio
import numpy as np
import threading
import time
import soundfile as sf
from typing import Optional, Any, TYPE_CHECKING
from .buffer import AudioChunk
from ..config.settings import SystemConfiguration
from ..processor.linear_aec_processor import LinearAECProcessor

if TYPE_CHECKING:
    from .buffer import SharedCircularBuffer


class AudioStreamPipeline:
    """音频流管道类，支持多通道音频流处理"""
    
    def __init__(self, config: SystemConfiguration, audio_file_path: Optional[str] = None, 
                 doa_service: Optional[Any] = None, 
                 shared_buffer: Optional['SharedCircularBuffer'] = None,
                 wpe_processor: Optional[Any] = None):
        """
        初始化音频流管道
        
        Args:
            config: 系统配置
            audio_file_path: 音频文件路径（如果为None则使用实时输入）
            doa_service: DOA服务实例（文件模式兼容）
            shared_buffer: 共享缓冲区（实时模式使用，解耦 I/O 和处理）
            wpe_processor: WPE处理器实例
        """
        self.config = config
        self.audio_file_path = audio_file_path
        self.doa_service = doa_service
        self.shared_buffer = shared_buffer
        self.wpe_processor = wpe_processor
        self.wpe_callbacks = []
        self.pipeline_id = f"pipeline-{int(time.time() * 1000)}"
        self.sample_rate = config.sample_rate
        self.channel_count = config.input_channels
        self.sample_rate = config.sample_rate
        self.channel_count = config.input_channels
        self.processing_buffer_size = config.buffer_size # Downstream requirement (e.g. 1024)
        
        # Determine processing channels from mic_positions if available
        self.processing_channels = self.channel_count
        if self.config.mic_positions:
            self.processing_channels = len(self.config.mic_positions)
            print(f"[AudioStreamPipeline] Configured for {self.processing_channels} processing channels (Hardware: {self.channel_count})")

        # AEC Initialization
        self.aec_processor = None
        self.aec_output_buffer = None
        self.capture_buffer_size = config.buffer_size # Default to processing size
        
        if config.aec and config.aec.enabled:
            # Initialize Linear AEC Processor (v3.0)
            try:
                self.aec_processor = LinearAECProcessor(
                    config=config.aec, 
                    sample_rate=self.sample_rate, 
                    input_channels=self.channel_count,
                    mic_positions=self.config.mic_positions
                )
                # AEC requires 10ms frames. At 16kHz, this is 160 samples.
                self.capture_buffer_size = int(self.sample_rate * 0.01)
                self.aec_output_buffer = np.zeros((0, self.processing_channels), dtype=np.int16)
                print(f"[AudioStreamPipeline] Linear AEC v3.0 Enabled. Capture chunk size set to {self.capture_buffer_size} samples (10ms).")
            except Exception as e:
                print(f"[AudioStreamPipeline] Failed to initialize Linear AEC: {e}")
                traceback.print_exc()
                self.aec_processor = None

        # self.buffer_size is used by PyAudio and File reader. 
        # We set this to 10ms (160 samples) to ensure minimum latency and smooth data flow.
        self.buffer_size = int(self.sample_rate * 0.01)
        self.device_index = config.device_index
        self.is_active = False
        self.processing_enabled = True
        self.created_at = time.time()
        self.processing_thread = None

        # 初始化PyAudio（仅实时模式）
        self.pyaudio_instance = None
        self.audio_stream = None

        if audio_file_path is None:
            self.pyaudio_instance = pyaudio.PyAudio()
            
            # Resolve device index by name if provided
            if self.config.device_name:
                found_index = self._find_device_by_name(self.config.device_name)
                if found_index is not None:
                    print(f"[AudioStreamPipeline] Found device '{self.config.device_name}' at index {found_index}")
                    self.device_index = found_index
                else:
                    print(f"[AudioStreamPipeline] Warning: Device '{self.config.device_name}' not found. Using default or configured index {self.device_index}")

        # This check is now less critical as FoundationService pre-populates these
        if self.channel_count is None:
            if audio_file_path is not None:
                try:
                    info = sf.info(audio_file_path)
                    self.channel_count = info.channels
                except Exception:
                    self.channel_count = 1
            elif self.pyaudio_instance:
                try:
                    if self.device_index is not None:
                        device_info = self.pyaudio_instance.get_device_info_by_index(self.device_index)
                    else:
                        device_info = self.pyaudio_instance.get_default_input_device_info()
                    self.channel_count = device_info['maxInputChannels']
                except Exception:
                    self.channel_count = 1

        # Buffer for WPE processing to handle non-aligned chunk sizes
        self.wpe_input_buffer = np.zeros((0, self.processing_channels), dtype=np.int16)

    def _find_device_by_name(self, device_name: str) -> Optional[int]:
        """Find device index by name."""
        if not self.pyaudio_instance:
            return None
            
        device_count = self.pyaudio_instance.get_device_count()
        for i in range(device_count):
            try:
                device_info = self.pyaudio_instance.get_device_info_by_index(i)
                # Check if name contains the requested name (case-insensitive partial match might be better, but exact for now)
                # The user config has "Mic Arry 8", let's do a partial match to be safe?
                # Or exact match? The user config comment says "Mic Arry 8".
                # Let's do exact match first, or 'in'.
                if device_name in device_info['name'] and device_info['maxInputChannels'] > 0:
                    return i
            except Exception:
                continue
        return None

    def add_wpe_callback(self, callback):
        self.wpe_callbacks.append(callback)

    def _invoke_callbacks(self, callbacks, *args, **kwargs):
        for callback in callbacks:
            callback(*args, **kwargs)

    def _process_aec(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Process audio data through Linear AEC v3.0 by splitting into 10ms chunks.
        
        Args:
            audio_data: Input audio data (samples, channels).
            
        Returns:
            Processed audio data (same size as input).
        """
        if not self.aec_processor:
             return audio_data

        try:
             # AEC requires 10ms frames. At 16kHz, this is 160 samples.
             chunk_size = int(self.sample_rate * 0.01)
             
             # If input is already 10ms, just process it
             if audio_data.shape[0] == chunk_size:
                 return self.aec_processor.process(audio_data)
                 
             # Otherwise, split into chunks
             total_samples = audio_data.shape[0]
             processed_chunks = []
             
             for i in range(0, total_samples, chunk_size):
                 chunk = audio_data[i:i + chunk_size]
                 # Only process full chunks (AEC expects fixed size)
                 if chunk.shape[0] == chunk_size:
                     processed_chunk = self.aec_processor.process(chunk)
                     processed_chunks.append(processed_chunk)
                 else:
                     # For partial chunks at the end (shouldn't happen with correct buffering upstream but safe to handle)
                     # Just pass through or zero-pad? Passing through is safer for now.
                     processed_chunks.append(chunk)
             
             if processed_chunks:
                 return np.vstack(processed_chunks)
             else:
                 return audio_data

        except Exception as e:
             print(f"Linear AEC Processing Error: {e}")
             traceback.print_exc()
             return audio_data

    def _process_wpe(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Process audio data through WPE with buffering to handle chunk sizes.
        
        Args:
            audio_data: Input audio data (samples, channels)
            
        Returns:
            Processed audio data (may be smaller or larger than input depending on buffering)
        """
        if not self.wpe_processor:
            return audio_data

        # Append new data to buffer
        if self.wpe_input_buffer.shape[0] > 0:
            self.wpe_input_buffer = np.vstack([self.wpe_input_buffer, audio_data])
        else:
            self.wpe_input_buffer = audio_data

        wpe_chunk_size = self.wpe_processor.stft_shift
        total_samples = self.wpe_input_buffer.shape[0]
        num_wpe_chunks = total_samples // wpe_chunk_size

        if num_wpe_chunks == 0:
            # Not enough data for a full chunk, return empty (downstream needs to handle this)
            # Actually, for the pipeline, returning empty might break things if downstream expects data.
            # But since we are buffering, we output what we can.
            return np.zeros((0, self.processing_channels), dtype=audio_data.dtype)

        wpe_sub_chunks = []
        for i in range(num_wpe_chunks):
            start = i * wpe_chunk_size
            end = start + wpe_chunk_size
            sub_chunk = self.wpe_input_buffer[start:end, :]
            
            # Normalize to float32 [-1.0, 1.0] for WPE processing
            sub_chunk_float = sub_chunk.astype(np.float32) / 32768.0
            
            dereverberated_chunk_float = self.wpe_processor.dereverberate(sub_chunk_float)
            
            # Pass normalized float data to callbacks (e.g. OutputSaver expects floats)
            self._invoke_callbacks(self.wpe_callbacks, dereverberated_chunk_float)
            
            # Denormalize back to int16 for the pipeline
            dereverberated_chunk_int16 = (np.clip(dereverberated_chunk_float, -1.0, 1.0) * 32767.0).astype(np.int16)
            wpe_sub_chunks.append(dereverberated_chunk_int16)
        
        # Update buffer with remainder
        remainder_start = num_wpe_chunks * wpe_chunk_size
        if remainder_start < total_samples:
            self.wpe_input_buffer = self.wpe_input_buffer[remainder_start:, :]
        else:
            self.wpe_input_buffer = np.zeros((0, self.processing_channels), dtype=audio_data.dtype)

        if wpe_sub_chunks:
            return np.vstack(wpe_sub_chunks)
        else:
            return np.zeros((0, self.processing_channels), dtype=audio_data.dtype)

    def start(self):
        """启动音频流处理"""
        if self.is_active:
            return True
        
        self.is_active = True
        if self.audio_file_path is None:
            self._start_realtime_mode()
        else:
            self._start_file_mode()

    def _start_realtime_mode(self):
        """启动实时音频模式"""
        try:
            # Print device info
            if self.device_index is not None:
                device_info = self.pyaudio_instance.get_device_info_by_index(self.device_index)
            else:
                device_info = self.pyaudio_instance.get_default_input_device_info()
            
            print(f"\n[AudioStreamPipeline] Starting Real-time Stream:")
            print(f"  - Device Name: {device_info['name']}")
            print(f"  - Device Index: {device_info['index']}")
            print(f"  - Max Input Channels: {device_info['maxInputChannels']}")
            print(f"  - Default Sample Rate: {device_info['defaultSampleRate']}\n")

            self.audio_stream = self.pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=self.channel_count,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.buffer_size,
                stream_callback=self._audio_callback
            )
            self.audio_stream.start_stream()
        except Exception as e:
            print(f"Failed to start real-time audio stream: {e}")
            raise
    
    def _start_file_mode(self):
        """启动文件音频模式"""
        print(f"[Log] Audio stream pipeline starting in file mode: {self.audio_file_path}")
        self.processing_thread = threading.Thread(target=self._file_processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def _file_processing_loop(self):
        """循环读取和处理音频文件"""
        print("[Log] File processing thread started.")
        start_time = time.time() # Start timing
        try:
            with sf.SoundFile(self.audio_file_path, 'r') as audio_file:
                print(f"[Log] Opened audio file: {self.audio_file_path}")
                if audio_file.samplerate != self.sample_rate:
                    print(f"Warning: File sample rate ({audio_file.samplerate}Hz) differs from config ({self.sample_rate}Hz).")

                file_channels = audio_file.channels
                sample_rate = audio_file.samplerate  # Use the actual file's sample rate for accurate timestamp
                samples_processed = 0  # Track total samples processed for accurate timestamp calculation

                loop_count = 0
                while self.is_active:
                    # print(f"[Log] Loop {loop_count}: Reading {self.buffer_size} samples from file...")
                    audio_data = audio_file.read(self.buffer_size, dtype='int16')

                    if audio_data is None or len(audio_data) == 0:
                        print("[Log] End of audio file reached.")
                        break

                    # print(f"[Log] Read {len(audio_data)} samples.")

                    if len(audio_data) < self.buffer_size:
                        print(f"[Log] Padding chunk from {len(audio_data)} to {self.buffer_size} samples.")
                        padding = np.zeros((self.buffer_size - len(audio_data), file_channels), dtype=np.int16)
                        audio_data = np.vstack([audio_data, padding])


                    # AEC Processing (which includes buffering if enabled)
                    if self.aec_processor:
                        audio_data = self._process_aec(audio_data)
                        if len(audio_data) == 0:
                            continue

                    # WPE processing with buffering
                    if self.wpe_processor:
                        audio_data = self._process_wpe(audio_data)

                    # If WPE buffering results in no output for this iteration, skip processing
                    if len(audio_data) == 0:
                        continue

                    # Calculate timestamp based on the position in the audio file
                    chunk_timestamp = samples_processed / sample_rate

                    chunk = AudioChunk(
                        data=audio_data,
                        num_channels=file_channels,
                        timestamp=chunk_timestamp
                    )

                    if self.doa_service:
                        # print(f"[Log] Calling DOA service with chunk {chunk.id} at timestamp {chunk_timestamp:.3f}s...")
                        self.doa_service.process_audio(chunk)
                    else:
                        print("[Log] DOA service not available, skipping processing.")

                    # Update the number of samples processed
                    samples_processed += len(audio_data)
                    loop_count += 1

        except Exception as e:
            print(f"Error during file processing: {e}")
            traceback.print_exc()
        finally:
            end_time = time.time() # End timing
            duration = end_time - start_time
            print(f"[Log] _file_processing_loop finished in {duration:.4f} seconds.")
            self.is_active = False
            print("[Log] File processing finished and thread is terminating.")

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """音频回调函数（仅实时模式）"""
        if self.processing_enabled:
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            
            if self.channel_count > 1:
                audio_data = audio_data.reshape(-1, self.channel_count)
            else:
                audio_data = audio_data.reshape(-1, 1)

            # Slice to processing channels if needed
            if self.processing_channels < self.channel_count:
                audio_data = audio_data[:, :self.processing_channels]
            
            # AEC Processing
            if self.aec_processor:
                audio_data = self._process_aec(audio_data)
                # If buffering, we might get empty output.
                if len(audio_data) == 0:
                    return (None, pyaudio.paContinue)

            # WPE processing with buffering
            if self.wpe_processor:
                audio_data = self._process_wpe(audio_data)
            
            # If WPE buffering results in no output, we can't write nothing to buffer manager if it expects something?
            # Actually buffer manager write_frame just appends. Writing empty is fine.
            if len(audio_data) > 0:
                # Priority 1: Write to shared buffer (new SPEC-009 architecture)
                if self.shared_buffer is not None:
                    # Convert int16 to float32 for SharedCircularBuffer
                    audio_data_float = audio_data.astype(np.float32) / 32768.0
                    written = self.shared_buffer.write(audio_data_float)
                    # Log periodically (every ~1 second worth of data at 16kHz)
                    if hasattr(self, '_write_counter'):
                        self._write_counter += written
                    else:
                        self._write_counter = written
                    if self._write_counter >= 16000:
                        print(f"SharedBuffer: wrote {self._write_counter} samples, available={self.shared_buffer.get_available_samples()}")
                        self._write_counter = 0
                
                # Priority 2: Direct callback (legacy file mode or fallback)
                elif self.doa_service:
                    chunk = AudioChunk(
                        data=audio_data,
                        num_channels=self.processing_channels,
                        timestamp=time.time()
                    )
                    self.doa_service.process_audio(chunk)
        
        return (None, pyaudio.paContinue)
    
    def stop(self):
        """停止音频流处理"""
        if not self.is_active:
            return
        
        self.is_active = False
        
        if self.audio_stream and self.audio_stream.is_active():
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        
        if self.pyaudio_instance:
            self.pyaudio_instance.terminate()

        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
        
        print("Audio stream pipeline stopped.")
    
    def is_running(self) -> bool:
        """检查音频流是否正在运行"""
        if self.audio_file_path is None:
            return self.is_active and self.audio_stream and self.audio_stream.is_active()
        else:
            return self.is_active and self.processing_thread and self.processing_thread.is_alive()
    
    def get_status(self) -> dict:
        return {
            'pipeline_id': self.pipeline_id,
            'sample_rate': self.sample_rate,
            'channel_count': self.channel_count,
            'buffer_size': self.buffer_size,
            'is_active': self.is_active,
            'processing_enabled': self.processing_enabled,
            'created_at': self.created_at,
            'input_source': 'FILE' if self.audio_file_path else 'REALTIME',
            'file_path': self.audio_file_path
        }
    
    def update_config(self, new_config: SystemConfiguration):
        config_changed = (
            self.sample_rate != new_config.sample_rate or
            self.channel_count != new_config.input_channels
        )
        
        self.config = new_config
        self.sample_rate = new_config.sample_rate
        self.buffer_size = new_config.buffer_size
        
        if new_config.input_channels is not None:
            self.channel_count = new_config.input_channels
        
        if config_changed and self.is_active:
            was_running = self.is_running()
            self.stop()
            if was_running:
                self.start()
