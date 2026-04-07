import numpy as np
from typing import Tuple, Callable, Optional, TYPE_CHECKING

from src.doa.config.doa_config import DOAConfig
# from src.core.config.wpe_config import WpeConfig # Avoid circular import if possible, utilize TYPE_CHECKING
if TYPE_CHECKING:
    from src.core.config.wpe_config import WpeConfig

from src.core.processor.mcra_reducer import MCRAReducer
from src.doa.core.doa_engine import DoaEngine
from src.core.audio.stft_engine import StftEngine
from src.core.processor.wpe_processor import WPEProcessor


class DOAProcessor:
    def __init__(self, config: DOAConfig, wpe_config: Optional['WpeConfig'] = None):
        """
        Initializes the DOA Processor, which now acts as the central controller
        for DOA estimation, MVDR beamforming, and DTLN post-processing.
        """
        self.config = config
        self.sample_rate = config.sample_rate

        # Identify active mics (Role 1) and Reference (Role 2)
        self.active_mic_indices = []
        self.ref_mic_index = None
        if self.config.mic_positions:
            for i, pos in enumerate(self.config.mic_positions):
                role = pos.get('role', 1) if isinstance(pos, dict) else 1
                if role == 1:
                    self.active_mic_indices.append(i)
                elif role == 2:
                    self.ref_mic_index = i

        # Default threshold for echo suppression
        self.echo_suppression_threshold = getattr(self.config, 'echo_suppression_threshold', 0.05)

        # If no explicit roles found or all unused, fallback to all channels
        if not self.active_mic_indices:
            self.active_mic_indices = list(range(config.num_mics))

        self.num_channels = len(self.active_mic_indices)

        # --- STFT Engine ---
        self.frame_length_samples = int(self.sample_rate * self.config.frame_length_ms / 1000)
        self.hop_length_samples = int(self.sample_rate * self.config.hop_length_ms / 1000)
        self.fft_length = self.frame_length_samples
        self.stft_engine = StftEngine(
            frame_len=self.frame_length_samples,
            hop_len=self.hop_length_samples,
            use_scipy=False,
            fs=self.sample_rate
        )
        


        # Internal buffer
        self.internal_buffer = np.array([]).reshape(0, self.num_channels)

        # Callbacks
        self.doa_logging_callback = None
        self.realtime_plot_callbacks = []

        self.mcra_callbacks = []  # 新增MCRA回调
        self.wpe_callbacks = []   # New WPE callbacks

        # --- WPE Initialization ---
        self.wpe_processor = None
        self.wpe_input_buffer = None
        if wpe_config and wpe_config.enable:
            print(f"Initializing WPE Processor (Taps: {wpe_config.taps}, Delay: {wpe_config.delay})")
            self.wpe_processor = WPEProcessor(
                sample_rate=self.sample_rate,
                channels=self.num_channels, # Use active channels count
                taps=wpe_config.taps,
                delay=wpe_config.delay,
                alpha=wpe_config.alpha,
                stft_size=wpe_config.stft_size,
                stft_shift=wpe_config.stft_shift
            )
            # Initialize buffer for WPE chunking
            self.wpe_input_buffer = np.zeros((0, self.num_channels), dtype=np.float32)

        if self.config.enable_mcra_denoise:
            self.mcra_reducer = MCRAReducer(
                sample_rate=self.sample_rate,
                channels=self.num_channels,
                mic_indices=self.active_mic_indices,
                stft_size=self.config.mcra.stft_size,
                stft_shift=self.config.mcra.stft_shift,
                mcra_alpha_s=self.config.mcra.alpha_s,
                mcra_alpha_d=self.config.mcra.alpha_d,
                mcra_l_window=self.config.mcra.l_window,
                mcra_gamma=self.config.mcra.gamma,
                mcra_delta=self.config.mcra.delta,
                mcra_gain_floor=self.config.mcra.gain_floor,
                mcra_gain_exponent=self.config.mcra.gain_exponent
            )
        else:
            self.mcra_reducer = None

        self.doa_engine = DoaEngine(config=self.config)

        self.frame_counter = 0
        print("DOA Processor initialized.")



    def add_mcra_callback(self, callback):
        """Adds a callback to be invoked with the MCRA-processed audio chunk."""
        self.mcra_callbacks.append(callback)

    def add_wpe_callback(self, callback):
        """Adds a callback to be invoked with the WPE-processed audio chunk."""
        self.wpe_callbacks.append(callback)

    def add_realtime_plot_callback(self, callback):
        self.realtime_plot_callbacks.append(callback)

    def _invoke_callbacks(self, callbacks, *args, **kwargs):
        for callback in callbacks:
            callback(*args, **kwargs)

    def _process_wpe(self, audio_data: np.ndarray) -> np.ndarray:
        """Helper to process audio through WPE with chunking."""
        if not self.wpe_processor:
            return audio_data

        # Ensure we are working with float32 for buffer and processing
        if audio_data.dtype != np.float32:
             audio_data = audio_data.astype(np.float32)

        # Append new data to buffer
        if self.wpe_input_buffer.shape[0] > 0:
            self.wpe_input_buffer = np.vstack([self.wpe_input_buffer, audio_data])
        else:
            self.wpe_input_buffer = audio_data

        wpe_chunk_size = self.wpe_processor.stft_shift
        total_samples = self.wpe_input_buffer.shape[0]
        num_wpe_chunks = total_samples // wpe_chunk_size

        if num_wpe_chunks == 0:
            # Not enough data for a full chunk, return empty (buffer holds it)
            return np.zeros((0, self.num_channels), dtype=np.float32)

        wpe_sub_chunks = []
        for i in range(num_wpe_chunks):
            start = i * wpe_chunk_size
            end = start + wpe_chunk_size
            sub_chunk = self.wpe_input_buffer[start:end, :]
            
            # WPE Processor expects float32 [-1, 1], which we already should have
            dereverberated_chunk = self.wpe_processor.dereverberate(sub_chunk)
            
            if self.wpe_callbacks:
                self._invoke_callbacks(self.wpe_callbacks, dereverberated_chunk)
            
            wpe_sub_chunks.append(dereverberated_chunk)
        
        # Update buffer with remainder
        remainder_start = num_wpe_chunks * wpe_chunk_size
        if remainder_start < total_samples:
            self.wpe_input_buffer = self.wpe_input_buffer[remainder_start:, :]
        else:
            self.wpe_input_buffer = np.zeros((0, self.num_channels), dtype=np.float32)

        if wpe_sub_chunks:
            return np.vstack(wpe_sub_chunks)
        else:
            return np.zeros((0, self.num_channels), dtype=np.float32)

    def process(self, audio_data: np.ndarray, timestamp: float):
        """
        Processes an audio chunk, runs the DOA estimation, and prints the detected angles.
        """
        # Normalize to float32 in range [-1, 1]
        # Only divide by 32768 if input is input type (int16)
        if np.issubdtype(audio_data.dtype, np.integer):
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype != np.float32:
            # Already normalized float (float32 from SharedCircularBuffer)
            audio_data = audio_data.astype(np.float32)
            
        # --- Echo Suppression (Reference Gating) ---
        is_echo_active = False
        if self.ref_mic_index is not None and self.ref_mic_index < audio_data.shape[1]:
            ref_energy = np.sqrt(np.mean(audio_data[:, self.ref_mic_index]**2))
            if ref_energy > self.echo_suppression_threshold:
                is_echo_active = True

        # Handle channel mismatch: Slice if input has more channels (e.g. 5th playback channel)
        # We only process the configured number of microphone channels for DOA
        # In file mode, input may have fewer channels than configured
        if audio_data.shape[1] >= max(self.active_mic_indices) + 1:
            # Use only the active microphone channels for processing
            processing_data = audio_data[:, self.active_mic_indices]
        else:
            # If input has fewer channels than expected, use all available channels
            # This handles file mode where only mic channels are present
            processing_data = audio_data

        # --- 0. WPE Preprocessing ---
        if self.wpe_processor:
             # Process through WPE. Note: This may return more or less data depending on buffering
             processing_data = self._process_wpe(processing_data)
             
             # If WPE buffering results in no output, we can't proceed with this chunk
             if processing_data.shape[0] == 0:
                 return [], np.zeros((0, self.num_channels), dtype=np.float32)

        # --- 1. MCRA Preprocessing ---
        all_results = []
        processed_data = processing_data
        if self.config.enable_mcra_denoise and self.mcra_reducer:
            mcra_chunk_size = self.config.mcra.stft_shift
            num_mcra_chunks = processing_data.shape[0] // mcra_chunk_size

            mcra_sub_chunks = []
            if num_mcra_chunks > 0:
                for i in range(num_mcra_chunks):
                    start = i * mcra_chunk_size
                    end = start + mcra_chunk_size
                    sub_chunk = processing_data[start:end, :]

                    denoised_chunk = self.mcra_reducer.reduce_noise(sub_chunk)
                    if self.mcra_callbacks:  # 新增MCRA回调
                        self._invoke_callbacks(self.mcra_callbacks, denoised_chunk)
                    mcra_sub_chunks.append(denoised_chunk)

                processed_data = np.vstack(mcra_sub_chunks)
            else:
                processed_data = np.array([]).reshape(0, processing_data.shape[1])

        # Check if internal buffer needs to be initialized or if channel count changed
        if self.internal_buffer.size == 0:
            # Initialize buffer with the correct number of channels
            self.internal_buffer = np.array([]).reshape(0, processed_data.shape[1])

        # Ensure internal buffer and processed_data have compatible shapes
        if self.internal_buffer.shape[1] != processed_data.shape[1]:
            # If channel count changed, reinitialize buffer
            self.internal_buffer = np.array([]).reshape(0, processed_data.shape[1])

        # Append new audio data to the internal buffer
        self.internal_buffer = np.vstack([self.internal_buffer, processed_data])
        num_samples_in_buffer = self.internal_buffer.shape[0]

        while num_samples_in_buffer >= self.frame_length_samples:
            frame_data = self.internal_buffer[:self.frame_length_samples, :]
            stfts = self.stft_engine.analysis_single_frame(frame_data.astype(np.float32))

            frame_timestamp = timestamp - ((num_samples_in_buffer - self.frame_length_samples) / self.sample_rate)

            # Get DOA results (with Echo Suppression)
            if is_echo_active:
                # Echo is active, suppress DOA detection to prevent locking onto the speaker
                detected_angles = []
                srp_db = np.zeros(len(self.doa_engine.angle_grid)) if hasattr(self.doa_engine, 'angle_grid') else []
                # Optional debug
                # if self.frame_counter % 20 == 0:
                #    print(f"Frame {self.frame_counter}: DOA Suppressed (Echo Active)")
            else:
                detected_angles, srp_db = self.doa_engine.get_doa_results(stfts, frame_timestamp, self.frame_counter)

            # Print the detected angles to console if found
            if detected_angles:
                print(f"Frame {self.frame_counter}: DOA detected angles: {detected_angles}")

            # --- Result and Callback Handling ---
            # Always append result to maintain frame synchronization with EnhancementService
            all_results.append({"timestamp": frame_timestamp, "doa": detected_angles})
            if self.doa_logging_callback:
                self.doa_logging_callback(frame_timestamp, self.frame_counter, detected_angles)
            if self.realtime_plot_callbacks:
                callback_data = {"timestamp": frame_timestamp, "spectrum_data": (self.doa_engine.angle_grid, srp_db), "results": detected_angles}
                self._invoke_callbacks(self.realtime_plot_callbacks, callback_data)

            self.frame_counter += 1
            self.internal_buffer = self.internal_buffer[self.hop_length_samples:, :]
            num_samples_in_buffer = self.internal_buffer.shape[0]

        return all_results, processed_data

    def close(self):
        """Closes the DOA processor."""
        print("Closing DOA Processor...")
