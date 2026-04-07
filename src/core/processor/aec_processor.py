import logging
import numpy as np
from typing import Optional, List, Dict, Any

try:
    from webrtc_audio_processing import AudioProcessingModule

    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
    AudioProcessingModule = None

from src.core.config.aec_config import AecConfig

logger = logging.getLogger(__name__)


class AecProcessor:
    """
    Acoustic Echo Cancellation (AEC) Processor.
    Wraps webrtc-audio-processing to perform echo cancellation on 10ms audio frames.
    Optimized for multi-channel input/output using a single APM instance.
    Selective processing based on channel roles.
    """

    def __init__(self, config: AecConfig, sample_rate: int, input_channels: int, mic_positions: Optional[List[Dict[str, Any]]] = None):
        self.config = config
        self.sample_rate = sample_rate
        self.input_channels = input_channels
        self.mic_positions = mic_positions
        self.apm = None
        self._debug_frame_count = 0 
        
        # Identify which channels to process (Role 1) and which is ref (Role 2)
        self.mic_indices = []
        self.ref_index = self.config.reference_channel_index
        
        if self.mic_positions:
            for i, pos in enumerate(self.mic_positions):
                role = pos.get('role', 1) # Default to 1 (Mic) if no role
                if role == 1:
                    self.mic_indices.append(i)
                elif role == 2:
                    self.ref_index = i # Override config ref index if found in mic_positions
        else:
            # Fallback: Process all channels except reference
            self.mic_indices = [i for i in range(self.input_channels) if i != self.ref_index]

        self.proc_channels = len(self.mic_indices)

        if self.config.enabled:
            if not WEBRTC_AVAILABLE:
                logger.warning("AEC enabled but 'webrtc-audio-processing' library not found. AEC will be bypassed.")
                return
            
            if self.ref_index is None:
                raise ValueError("AEC enabled but reference_channel_index is not set.")
                
            if self.ref_index >= self.input_channels:
                raise ValueError(f"AEC reference_channel_index {self.ref_index} out of bounds.")

            try:
                # Initialize ONE APM for processed channels
                # aec_type=2 (Desktop AEC) is critical for effective echo cancellation on PC
                # enable_ns=True (Noise Suppression) enabled to suppress residual echo artifacts
                self.apm = AudioProcessingModule(aec_type=2, enable_ns=True)
                
                # Configure: Process only selected mic channels
                if hasattr(self.apm, 'set_stream_format'):
                    self.apm.set_stream_format(self.sample_rate, self.proc_channels, self.sample_rate, self.proc_channels)
                
                if hasattr(self.apm, 'set_reverse_stream_format'):
                    self.apm.set_reverse_stream_format(self.sample_rate, 1) # Reference is mono
                    
                if hasattr(self.apm, 'set_aec_level'):
                    self.apm.set_aec_level(2) # Max level for desktop is 2
                
                if hasattr(self.apm, 'set_ns_level'):
                    self.apm.set_ns_level(1) # Moderate noise suppression

                logger.info(f"AEC Processor initialized.")
                logger.info(f"  - Total Channels: {self.input_channels}")
                logger.info(f"  - Processing Channels: {self.mic_indices} ({self.proc_channels} ch)")
                logger.info(f"  - Reference Channel: {self.ref_index}")
                logger.info(f"  - Delay: {self.config.system_delay_ms}ms")
            except Exception as e:
                logger.error(f"AEC Failed to initialize WebRTC APM: {e}")
                self.apm = None

    def process(self, audio_frame: bytes) -> bytes:
        if not self.apm:
            return audio_frame
            
        self._debug_frame_count += 1
        is_debug_frame = (self._debug_frame_count % 50 == 0)

        # 1. Validate frame length
        samples_per_10ms = int(self.sample_rate * 0.01)
        expected_bytes = samples_per_10ms * self.input_channels * 2
        
        if len(audio_frame) != expected_bytes:
            # Fallback if frame size is wrong
            return audio_frame
            
        # 2. Prepare data with gain
        # [Auto-Gain] Amplify signal slightly for AEC
        GAIN_FACTOR = 1.0 # Reset to 1.0 since we are using specific channels and trust hardware gain more now? 

        data_int16 = np.frombuffer(audio_frame, dtype=np.int16)
        data_boosted = np.clip(data_int16 * GAIN_FACTOR, -32768, 32767).astype(np.int16)
        
        # Reshape (samples, channels)
        data_reshaped = data_boosted.reshape(-1, self.input_channels)
        
        # 3. Extract Reference Stream (Mono)
        ref_col = data_reshaped[:, self.ref_index]
        ref_bytes = np.ascontiguousarray(ref_col).tobytes()
        
        # 4. Extract Mic Streams to Process
        mic_sub_frame = data_reshaped[:, self.mic_indices]
        mic_bytes = np.ascontiguousarray(mic_sub_frame).tobytes()

        # 5. Process through APM
        try:
            # Set delay
            if hasattr(self.apm, 'set_system_delay'):
                self.apm.set_system_delay(self.config.system_delay_ms)
            
            # Feed Reverse Stream (Reference)
            self.apm.process_reverse_stream(ref_bytes)
            
            # Feed Forward Stream (Only Mic channels)
            processed_bytes = self.apm.process_stream(mic_bytes)
            
            if not processed_bytes:
                return audio_frame

            # 6. Reconstruct Output
            # Start with a COPY of the boosted data (preserves unused channels and ref channel)
            out_data = data_reshaped.copy()
            
            # Overwrite only the processed mic channels
            processed_mics = np.frombuffer(processed_bytes, dtype=np.int16).reshape(-1, self.proc_channels)
            out_data[:, self.mic_indices] = processed_mics

            # 7. Diagnostics & Logging
            if is_debug_frame:
                max_ref = np.max(np.abs(ref_col))
                raw_ref = int(max_ref / GAIN_FACTOR)
                ref_status = "SILENCE" if max_ref < 1000 else "ACTIVE"
                
                # Check first processed mic (relative to mic_indices[0])
                cap_val = np.max(np.abs(mic_sub_frame[:, 0]))
                out_val = np.max(np.abs(processed_mics[:, 0]))
                delta = cap_val - out_val
                
                has_echo = self.apm.has_echo() if hasattr(self.apm, 'has_echo') else False
                echo_label = "ECHO_YES" if has_echo else "ECHO_NO"
                aec_lvl = self.apm.aec_level() if hasattr(self.apm, 'aec_level') else "?"
                
                action_msg = "No Change"
                if delta > 30: 
                    action_msg = f"Cancelling (Red: {delta})"
                elif ref_status == "ACTIVE":
                    action_msg = f"Converging ({echo_label})"
                
                print(f"[AEC DEBUG] Frame {self._debug_frame_count} | Ref: {max_ref} (Raw:{raw_ref}) [{ref_status}] | Ch{self.mic_indices[0]}: {cap_val} -> {out_val} ({action_msg}) [LVL:{aec_lvl}][Gain {GAIN_FACTOR}x][Delay {self.config.system_delay_ms}ms]")

            return out_data.tobytes()

        except Exception as e:
             if self._debug_frame_count % 100 == 0:
                 logger.error(f"AEC processing error: {e}")
             return audio_frame
