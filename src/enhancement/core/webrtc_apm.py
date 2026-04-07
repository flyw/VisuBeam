import logging
import numpy as np
from typing import Optional

try:
    import webrtc_audio_processing as ap
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
    ap = None

from src.enhancement.config.webrtc_apm_config import WebRtcApmConfig

logger = logging.getLogger(__name__)

class WebRtcApmProcessor:
    """
    WebRTC Audio Processing Module (APM) Middleware.
    Handles non-linear AEC, NS, TS, and AGC.
    Strictly processes 10ms (160 samples @ 16kHz) frames.
    Includes internal ring buffering to handle arbitrary input sizes.
    """

    def __init__(self, config: WebRtcApmConfig, fs: int = 16000):
        self.config = config
        self.fs = fs
        self.block_size = int(self.fs * 0.01) # 10ms = 160 samples
        self.apm = None
        
        # Ring buffers for input data that isn't a multiple of block_size
        self.mic_buffer = np.array([], dtype=np.float32)
        self.ref_buffer = np.array([], dtype=np.float32)
        self.output_buffer = np.array([], dtype=np.float32)

        if not WEBRTC_AVAILABLE:
            logger.warning("webrtc-audio-processing not installed. WebRTC APM will be bypassed.")
            return

        try:
            # Construct APM configuration
            # Based on the installed wrapper (likely xiongyihui/python-webrtc-audio-processing),
            # __init__ only accepts 'aec_type' and 'enable_ns'.
            
            apm_kwargs = {
                "enable_ns": self.config.enable_ns,
            }
            
            if self.config.enable_aec:
                apm_kwargs["aec_type"] = 2 # Desktop AEC
            else:
                 apm_kwargs["aec_type"] = 0 # No AEC (if supported by wrapper to disable via type?)
            
            # Initialize APM
            self.apm = ap.AudioProcessingModule(**apm_kwargs)
            
            # Set NS level
            if hasattr(self.apm, 'set_ns_level'):
                self.apm.set_ns_level(self.config.ns_level)
            
            # Set AGC (Mode)
            # The wrapper likely enables AGC by default. set_agc_level sets the mode.
            if hasattr(self.apm, 'set_agc_level'):
                self.apm.set_agc_level(self.config.agc_mode)
            
            # Set VAD
            # The wrapper likely enables VAD by default.
            if hasattr(self.apm, 'set_vad_level'):
                # Map boolean enable_vad? No, config has no level for VAD.
                # Just use a default level if enabled? 
                # If the wrapper enables it by default, we can't easily disable it if the wrapper doesn't expose Enable.
                # We'll just set a level if we can.
                pass

            # set_stream_format usually expects (in_fs, in_ch, out_fs, out_ch)
            if hasattr(self.apm, 'set_stream_format'):
                self.apm.set_stream_format(self.fs, 1, self.fs, 1)
            
            if hasattr(self.apm, 'set_reverse_stream_format'):
                self.apm.set_reverse_stream_format(self.fs, 1)
            
            logger.info("WebRtcApmProcessor initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize WebRTC APM: {e}")
            self.apm = None

    def _float_to_int16(self, audio: np.ndarray) -> np.ndarray:
        return (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)

    def _int16_to_float(self, audio: np.ndarray) -> np.ndarray:
        return audio.astype(np.float32) / 32767.0

    def process(self, mvdr_chunk: np.ndarray, ref_chunk: np.ndarray) -> np.ndarray:
        """
        Processes audio through WebRTC APM.
        Always returns an array of the same length as mvdr_chunk to maintain pipeline timing.
        Initial latency is handled by padding with zeros.
        """
        if self.apm is None:
            return mvdr_chunk

        target_len = len(mvdr_chunk)

        # Add to input buffers
        self.mic_buffer = np.append(self.mic_buffer, mvdr_chunk)
        self.ref_buffer = np.append(self.ref_buffer, ref_chunk)

        # Process as many 10ms blocks as available
        while len(self.mic_buffer) >= self.block_size and len(self.ref_buffer) >= self.block_size:
            # Extract 10ms blocks
            mic_block = self.mic_buffer[:self.block_size]
            ref_block = self.ref_buffer[:self.block_size]
            
            # Update buffers
            self.mic_buffer = self.mic_buffer[self.block_size:]
            self.ref_buffer = self.ref_buffer[self.block_size:]

            # Convert to Int16 for WebRTC
            mic_int16 = self._float_to_int16(mic_block)
            ref_int16 = self._float_to_int16(ref_block)

            try:
                # Feed Reference (Reverse Stream)
                self.apm.process_reverse_stream(ref_int16.tobytes())
                
                # Process Capture (Forward Stream)
                out_bytes = self.apm.process_stream(mic_int16.tobytes())
                
                # Convert back to Float32
                out_int16 = np.frombuffer(out_bytes, dtype=np.int16)
                out_float = self._int16_to_float(out_int16)
                
                self.output_buffer = np.append(self.output_buffer, out_float)
            except Exception as e:
                logger.error(f"WebRTC APM processing error: {e}")
                self.output_buffer = np.append(self.output_buffer, mic_block)

        # Return exactly target_len samples to maintain 1:1 timing
        if len(self.output_buffer) >= target_len:
            res = self.output_buffer[:target_len]
            self.output_buffer = self.output_buffer[target_len:]
            return res
        else:
            # Pad with zeros if we don't have enough yet (initial latency)
            padding_size = target_len - len(self.output_buffer)
            res = np.concatenate([self.output_buffer, np.zeros(padding_size, dtype=np.float32)])
            self.output_buffer = np.array([], dtype=np.float32)
            logger.debug(f"WebRtcApmProcessor: Output buffer short, padded {padding_size} zeros.")
            return res
