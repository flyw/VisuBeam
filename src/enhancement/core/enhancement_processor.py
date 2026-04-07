import logging
import numpy as np
import torch
from typing import Tuple, Callable
import scipy.signal
import time

from src.enhancement.config.enhancement_config import EnhancementConfig
# from src.enhancement.core.doa_engine import DoaEngine
from src.core.processor.mcra_reducer import MCRAReducer
from src.enhancement.core.mvdr_processor import MvdrProcessor
from src.enhancement.core.dtln_processor import DTLNProcessor
from src.enhancement.core.webrtc_apm import WebRtcApmProcessor
from src.core.audio.stft_engine import StftEngine
# from src.enhancement.core.model_loader import get_dtln_model

logger = logging.getLogger(__name__)


class EnhancementProcessor:
    def __init__(self, config: EnhancementConfig, dtln_model=None):
        """
        Initializes the DOA Processor, which now acts as the central controller
        for DOA estimation, MVDR beamforming, and DTLN post-processing.
        """
        self.config = config
        self.sample_rate = config.sample_rate
        self.num_channels = len(config.mic_positions)



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

        # --- MVDR/DTLN Streaming Synthesis Buffer ---
        self.synthesis_window = scipy.signal.windows.hann(self.frame_length_samples)
        self.mvdr_output_buffer = np.zeros(self.frame_length_samples, dtype=np.float32)

        # --- DTLN Batching ---
        self.DTLN_CHUNK_SIZE_FRAMES = self.config.dtln_chunk_size_frames
        self.dtln_stft_buffer = []
        self.dtln_doa_buffer = []
        self.ola_overlap_buffer = np.zeros(self.frame_length_samples - self.hop_length_samples, dtype=np.float32)
        self.ola_weight_overlap_buffer = np.zeros(self.frame_length_samples - self.hop_length_samples, dtype=np.float32)
        self.window_sq = self.synthesis_window ** 2

        # Identify active mics for MVDR based on role=1
        self.active_mic_indices = []
        self.reference_mic_index = None
        if self.config.mic_positions:
            for i, pos in enumerate(self.config.mic_positions):
                if isinstance(pos, dict):
                    role = pos.get('role', 1)
                    if role == 1:
                        self.active_mic_indices.append(i)
                    elif role == 2:
                        self.reference_mic_index = i

        # If no explicit roles found or all unused, fallback to all channels
        if not self.active_mic_indices:
            self.active_mic_indices = list(range(self.num_channels))

        self.num_active_channels = len(self.active_mic_indices)
        logger.info(f"EnhancementProcessor: Active MVDR channels indices: {self.active_mic_indices}")

        # --- WebRTC APM Middleware (Multi-channel Pre-processing) ---
        self.apm_processors = []
        if self.config.enable_webrtc_apm:
            logger.info(f"Initializing {self.num_active_channels} WebRTC APM processors for Pre-MVDR AEC...")
            for i in range(self.num_active_channels):
                self.apm_processors.append(WebRtcApmProcessor(self.config.webrtc_apm, fs=self.sample_rate))

        # --- MVDR Background Noise Model ---
        self.background_noise_alpha_slow = self.config.mvdr.background_noise_alpha_slow
        num_freq_bins = self.fft_length // 2 + 1
        self.psd_n_background = np.zeros((num_freq_bins, self.num_active_channels, self.num_active_channels), dtype=np.complex64)
        for k in range(num_freq_bins):
            self.psd_n_background[k] = np.eye(self.num_active_channels, dtype=np.complex64) * 1e-6

        # --- Inertial Steering State Variables ---
        self.miss_count = 0
        self.inertial_hold_frames = self.config.mvdr.inertial_hold_frames
        self.current_target_angle = self.config.mvdr.target_angle

        # Internal buffer - use only active channels
        self.internal_buffer = np.array([]).reshape(0, self.num_active_channels)

        # --- MVDR Processing Frequency Control ---
        self.mvdr_process_counter = 0  # Counter to control MVDR processing frequency
        self.mvdr_process_interval = 1  # Process every frame now that we have GPU acceleration

        # Callbacks
        self.mvdr_logging_callback = None
        self.dtln_callbacks = []
        self.mvdr_callbacks = [] # New callbacks for MVDR time-domain output
        self.apm_callbacks = [] # Callbacks for WebRTC APM output
        self.denoise_callbacks = []


        if self.config.enable_mcra_denoise:
            self.mcra_reducer = MCRAReducer(
                sample_rate=self.sample_rate,
                channels=self.num_active_channels,  # Use active channels count instead of total
                mic_indices=list(range(self.num_active_channels)),  # Process all active channels
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

        # self.doa_engine = DoaEngine(config=self.config) # Removed, using external DOA results
        self.mvdr_processor = MvdrProcessor(config=config) if self.config.enable_mvdr else None
        
        self.dtln_processor = None
        if self.config.dtln.enabled:
            if dtln_model is None:
                raise ValueError("DTLN is enabled but no model instance was provided.")
            self.dtln_processor = DTLNProcessor(config=self.config.dtln, dtln_model=dtln_model)

        self.frame_counter = 0

    def add_denoise_callback(self, callback):
        self.denoise_callbacks.append(callback)

    def add_dtln_callback(self, callback: Callable[[np.ndarray], None]):
        """Adds a callback to be invoked with the DTLN-processed audio chunk."""
        self.dtln_callbacks.append(callback)

    def add_mvdr_callback(self, callback: Callable[[np.ndarray], None]):
        """Adds a callback to be invoked with the MVDR time-domain audio chunk."""
        self.mvdr_callbacks.append(callback)
    
    def add_apm_callback(self, callback: Callable[[np.ndarray], None]):
        """Adds a callback to be invoked with the APM-processed audio chunk."""
        self.apm_callbacks.append(callback)

    def _invoke_callbacks(self, callbacks, *args, **kwargs):
        for callback in callbacks:
            callback(*args, **kwargs)
    
    def _process_dtln_chunk(self):
        """
        Processes a batch of buffered STFT frames using a streaming iSTFT
        (Overlap-Add) to synthesize the time-domain signal for MVDR saving and
        DTLN processing. This prevents the audio expansion bug.
        """
        logger.debug(f"EHP - Entering _process_dtln_chunk. Buffered STFT frames: {len(self.dtln_stft_buffer)}")
        if not self.dtln_stft_buffer:
            logger.debug("EHP - _process_dtln_chunk: No STFT frames to process.")
            return

        num_processed_frames = len(self.dtln_stft_buffer)
        
        # Calculate dimensions
        output_len = num_processed_frames * self.hop_length_samples
        overlap_len = self.frame_length_samples - self.hop_length_samples
        
        # The buffer needs to hold the output plus the new overlap tail
        total_len = output_len + overlap_len
        
        ola_buffer = np.zeros(total_len, dtype=np.float32)
        weight_buffer = np.zeros(total_len, dtype=np.float32)

        # 1. Restore overlap from previous chunk
        if not hasattr(self, 'ola_overlap_buffer'):
             self.ola_overlap_buffer = np.zeros(overlap_len, dtype=np.float32)
        if not hasattr(self, 'ola_weight_overlap_buffer'):
             self.ola_weight_overlap_buffer = np.zeros(overlap_len, dtype=np.float32)
             
        ola_buffer[:overlap_len] += self.ola_overlap_buffer
        weight_buffer[:overlap_len] += self.ola_weight_overlap_buffer

        # 2. Optimization: Batch transfer from GPU to CPU
        # Check if the buffer contains GPU tensors
        if self.dtln_stft_buffer and isinstance(self.dtln_stft_buffer[0], torch.Tensor):
            try:
                # Stack all frames into one tensor on GPU: (Batch, Freq)
                batch_gpu = torch.stack(self.dtln_stft_buffer)
                # Move entire batch to CPU once: (Batch, Freq)
                batch_cpu = batch_gpu.detach().cpu().numpy()
                frames_to_process = batch_cpu
            except Exception as e:
                logger.error(f"EHP - Batch GPU transfer failed: {e}. Falling back to list.")
                # Fallback mechanism if stack fails
                frames_to_process = [f.detach().cpu().numpy() for f in self.dtln_stft_buffer]
        else:
            # Already numpy arrays or empty
            frames_to_process = self.dtln_stft_buffer

        # 3. Accumulate new frames (Overlap-Add)
        for i, frame_stft in enumerate(frames_to_process):
            # Perform iFFT and apply synthesis window
            # frame_stft is now guaranteed to be numpy array
            frame_time = np.fft.irfft(frame_stft, n=self.frame_length_samples)
            windowed_frame = frame_time * self.synthesis_window

            # Overlap-add signal and weights
            start = i * self.hop_length_samples
            end = start + self.frame_length_samples
            ola_buffer[start:end] += windowed_frame
            weight_buffer[start:end] += self.window_sq

        # 4. Extract output and save new overlap
        # Normalize the output by the accumulated weights (WOLA)
        output_weights = weight_buffer[:output_len]
        mvdr_time_chunk = ola_buffer[:output_len] / (output_weights + 1e-12)
        logger.debug(f"EHP - _process_dtln_chunk: iSTFT (OLA) produced mvdr_time_chunk shape={mvdr_time_chunk.shape}")
        
        # Save overlaps for next chunk
        self.ola_overlap_buffer = ola_buffer[output_len:]
        self.ola_weight_overlap_buffer = weight_buffer[output_len:]

        # Extract metadata once for all callbacks
        doa_metadata = self.dtln_doa_buffer[:]

        # --- Callback: MVDR Output (Linear Beamforming Output, Pre-DTLN) ---
        if self.config.enable_mvdr_output_wav and self.mvdr_callbacks:
            logger.debug(f"EHP - _process_dtln_chunk: Invoking {len(self.mvdr_callbacks)} MVDR callbacks.")
            self._invoke_callbacks(self.mvdr_callbacks, mvdr_time_chunk, metadata=doa_metadata)

        # --- DTLN Processing ---
        enhanced_time_chunk = mvdr_time_chunk
        
        # Note: APM is now done in pre-processing (before MVDR)
        
        # --- Callback: APM Output (Deprecated/Mapped to MVDR output for now) ---
        if self.apm_callbacks:
             # logger.debug(f"EHP - _process_dtln_chunk: Invoking {len(self.apm_callbacks)} APM callbacks.")
             # self._invoke_callbacks(self.apm_callbacks, enhanced_time_chunk)
             pass

        if self.dtln_callbacks:
            output_chunk = enhanced_time_chunk
            if self.dtln_processor:
                logger.debug(f"EHP - _process_dtln_chunk: Calling DTLNProcessor with chunk shape={enhanced_time_chunk.shape}")
                t_dtln_start = time.perf_counter()
                processed_chunk = self.dtln_processor.process(enhanced_time_chunk)
                t_dtln_end = time.perf_counter()
                logger.info(f"DTLN Batch Perf: {(t_dtln_end - t_dtln_start)*1000:.2f}ms for {num_processed_frames} frames")

                if processed_chunk.size > 0:
                    output_chunk = processed_chunk
                else:
                    # DTLN might buffer internally, so an empty chunk is not an error,
                    # but we shouldn't send empty audio.
                    logger.warning("EHP - _process_dtln_chunk: DTLN returned empty chunk, skipping callback for this cycle.")
                    output_chunk = None
            
            if output_chunk is not None:
                logger.debug(f"EHP - _process_dtln_chunk: Invoking {len(self.dtln_callbacks)} enhanced audio callbacks with chunk size {output_chunk.size}")
                self._invoke_callbacks(self.dtln_callbacks, output_chunk, metadata=doa_metadata)
        
        # Clear the buffers now that they have been processed
        self.dtln_stft_buffer.clear()
        self.dtln_doa_buffer.clear()
        logger.debug("EHP - Exiting _process_dtln_chunk. STFT buffer cleared.")

    def process(self, audio_data: np.ndarray, timestamp: float, doa_results: list = None) -> Tuple[list, list]:
        """
        Processes an audio chunk, runs the full DOA, MVDR, and batched DTLN pipeline.
        """
        start_total = time.perf_counter()
        
        logger.debug(
            f"EHP - Entering process: timestamp={timestamp:.4f}s, audio_data_shape={audio_data.shape}, "
            f"current_buffer_size={self.internal_buffer.shape[0]}"
        )
        # Normalize to float32 in range [-1, 1]
        # Only divide by 32768 if input is integer type (int16)
        if np.issubdtype(audio_data.dtype, np.integer):
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype != np.float32:
            # Already normalized float (float32 from SharedCircularBuffer)
            audio_data = audio_data.astype(np.float32)

        # --- Pre-Processing: Multi-channel AEC via WebRTC APM ---
        ref_signal = None
        if self.apm_processors and self.reference_mic_index is not None:
            # Check if reference_mic_index is within bounds of original audio_data
            if self.reference_mic_index < audio_data.shape[1]:
                ref_signal = audio_data[:, self.reference_mic_index]
                # logger.debug(f"EHP - Extracted {len(ref_signal)} samples of reference signal.")
            else:
                logger.warning(f"EHP - reference_mic_index {self.reference_mic_index} out of bounds for audio_data shape {audio_data.shape}")

        # Filter audio data to only include active channels (role=1)
        if self.active_mic_indices:
            # Ensure indices are within bounds of the actual input data
            valid_indices = [idx for idx in self.active_mic_indices if idx < audio_data.shape[1]]
            if valid_indices:
                audio_data = audio_data[:, valid_indices]
                # logger.debug(f"EHP - Filtered audio data to active channels: {valid_indices}, new shape: {audio_data.shape}")
            else:
                logger.warning(f"EHP - No valid active channels found in input data with shape {audio_data.shape}")
                return [], []

        # Apply APM to each active channel if enabled
        if ref_signal is not None and self.apm_processors:
            if audio_data.shape[1] == len(self.apm_processors):
                t_apm_start = time.perf_counter()
                for i, apm in enumerate(self.apm_processors):
                    # Process channel i
                    # Note: apm.process returns a new array, so we assign it back
                    audio_data[:, i] = apm.process(audio_data[:, i], ref_signal)
                t_apm_end = time.perf_counter()
                # logger.debug(f"EHP - Multi-channel APM Perf: {(t_apm_end - t_apm_start)*1000:.2f}ms for {audio_data.shape[0]} samples")
            else:
                logger.error(f"EHP - Channel mismatch: Audio has {audio_data.shape[1]} channels, but {len(self.apm_processors)} APM processors configured.")

        # --- Callback: Pre-MVDR Multi-channel AEC Output ---
        # Invoke callback regardless of whether APM ran, so we save the "input to MVDR" (which is AEC output if ran)
        if self.apm_callbacks:
            self._invoke_callbacks(self.apm_callbacks, audio_data)

        # --- 1. MCRA Preprocessing ---
        processed_data = audio_data
        if self.config.enable_mcra_denoise and self.mcra_reducer:
            mcra_chunk_size = self.config.mcra.stft_shift
            num_mcra_chunks = audio_data.shape[0] // mcra_chunk_size

            mcra_sub_chunks = []
            if num_mcra_chunks > 0:
                for i in range(num_mcra_chunks):
                    start = i * mcra_chunk_size
                    end = start + mcra_chunk_size
                    sub_chunk = audio_data[start:end, :]

                    denoised_chunk = self.mcra_reducer.reduce_noise(sub_chunk)
                    if self.denoise_callbacks:
                        self._invoke_callbacks(self.denoise_callbacks, denoised_chunk)
                    # else:
                    #     print("Warning: No denoise callbacks registered!")

                    mcra_sub_chunks.append(denoised_chunk)

                processed_data = np.vstack(mcra_sub_chunks)
            else:
                processed_data = np.array([]).reshape(0, audio_data.shape[1])  # Use actual input channels
        # else:
        #     print(f"MCRA Disabled: {self.config.enable_mcra_denoise}, Reducer: {self.mcra_reducer is not None}")

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

        all_results = []
        all_enhanced_stfts = []

        while num_samples_in_buffer >= self.frame_length_samples:
            frame_data = self.internal_buffer[:self.frame_length_samples, :]
            
            # --- Log individual frame processing start ---
            logger.debug(
                f"EHP - Processing frame {self.frame_counter}: timestamp={timestamp:.4f}s, "
                f"frame_data_shape={frame_data.shape}, buffer_left={num_samples_in_buffer}"
            )

            t0 = time.perf_counter()
            stfts = self.stft_engine.analysis_single_frame(frame_data.astype(np.float32))
            t1 = time.perf_counter()
            logger.debug(f"EHP - Frame {self.frame_counter} STFTs shape: {stfts.shape}")
            
            frame_timestamp = timestamp - ((num_samples_in_buffer - self.frame_length_samples) / self.sample_rate)

            # Get DOA results from external source
            detected_angles = []
            if doa_results and len(doa_results) > 0:
                # Pop the first result (assuming FIFO order and synchronization)
                # We expect one result per processed frame
                current_result = doa_results.pop(0)
                detected_angles = current_result.get('doa', [])
            
            # detected_angles, srp_db = self.doa_engine.get_doa_results(stfts, frame_timestamp, self.frame_counter)
            srp_db = None # SRP-PHAT data not available/needed here anymore

            decision_type = ""
            process_angle = self.current_target_angle
            target_angle_detected = False
            raw_doa = None
            raw_energy = 0.0

            if detected_angles:
                for angle, energy in detected_angles:
                    if abs(angle - self.current_target_angle) <= self.config.mvdr.tolerance:
                        target_angle_detected = True
                        raw_doa = angle
                        raw_energy = energy
                        break
                if not target_angle_detected:
                    raw_doa = detected_angles[0][0]
                    raw_energy = detected_angles[0][1]

            if raw_doa is None:
                decision_type = "Background Noise"
                self.miss_count += 1
            else:
                if target_angle_detected or abs(raw_doa - self.current_target_angle) <= self.config.mvdr.tolerance:
                    self.miss_count = 0
                    decision_type = "Target DOA"
                else:
                    self.miss_count += 1
                    decision_type = "Inertial_Hold" if self.miss_count <= self.inertial_hold_frames else "Interference DOA"

            if decision_type == "Background Noise":
                # Since we've already filtered to active channels, use stfts directly
                stfts_active = stfts
                # Adjust background PSD tensor dimensions if needed
                if self.psd_n_background.size == 0 or self.psd_n_background.shape[1] != stfts_active.shape[1]:
                    # Reinitialize background PSD with correct dimensions
                    num_freq_bins = stfts_active.shape[0]
                    self.psd_n_background = np.zeros((num_freq_bins, stfts_active.shape[1], stfts_active.shape[1]), dtype=np.complex64)
                    for k in range(num_freq_bins):
                        self.psd_n_background[k] = np.eye(stfts_active.shape[1], dtype=np.complex64) * 1e-6

                current_psd = np.einsum('fk,fl->fkl', stfts_active, stfts_active.conj())
                self.psd_n_background = (self.background_noise_alpha_slow * self.psd_n_background +
                                         (1 - self.background_noise_alpha_slow) * current_psd)
            
            if self.mvdr_logging_callback:
                self.mvdr_logging_callback(
                    timestamp=frame_timestamp, frame_idx=self.frame_counter, angle=raw_doa,
                    energy=raw_energy, decision_type=decision_type,
                    locked_angle=self.current_target_angle, tolerance=self.config.mvdr.tolerance
                )

            # --- MVDR Processing ---
            enhanced_stft = stfts[:, 0]
            t2 = time.perf_counter()
            if self.mvdr_processor:
                # Control MVDR processing frequency to reduce CPU usage
                # Process MVDR only every mvdr_process_interval frames
                if self.mvdr_process_counter % self.mvdr_process_interval == 0:
                    logger.debug(f"EHP - Frame {self.frame_counter} MVDR processing with angle: {process_angle}")
                    # Since we've already filtered the audio data to active channels,
                    # stfts should already contain only the active channels
                    stfts_active = stfts

                    background_psd_tensor = torch.from_numpy(self.psd_n_background)
                    # Adjust background PSD tensor dimensions if needed
                    if self.psd_n_background.size == 0 or background_psd_tensor.shape[1] != stfts_active.shape[1]:
                        # Reinitialize background PSD with correct dimensions
                        num_freq_bins = stfts_active.shape[0]
                        self.psd_n_background = np.zeros((num_freq_bins, stfts_active.shape[1], stfts_active.shape[1]), dtype=np.complex64)
                        for k in range(num_freq_bins):
                            self.psd_n_background[k] = np.eye(stfts_active.shape[1], dtype=np.complex64) * 1e-6
                        background_psd_tensor = torch.from_numpy(self.psd_n_background)

                    enhanced_stft = self.mvdr_processor.process(
                        stft_data=stfts_active, decision_type=decision_type,
                        process_angle=process_angle, background_psd=background_psd_tensor
                    )
                else:
                    # When not processing MVDR, just return the first channel's STFT
                    logger.debug(f"EHP - Frame {self.frame_counter} MVDR skipped (frequency control)")
                    enhanced_stft = stfts[:, 0]
            t3 = time.perf_counter()

            # Increment the counter for each frame processed
            self.mvdr_process_counter += 1
            # logger.debug(f"EHP - Frame {self.frame_counter} MVDR output shape: {enhanced_stft.shape}")

            all_enhanced_stfts.append(enhanced_stft)

            # --- Output Synthesis & DTLN Post-Processing (Batched) ---
            self.dtln_stft_buffer.append(enhanced_stft)
            # Buffer the DOA result for this frame (if available) to pass to callbacks later
            # User Request: Just use the array index, remove 'timestamp' wrapper.
            self.dtln_doa_buffer.append(detected_angles if detected_angles else [])
            
            # logger.debug(f"EHP - Frame {self.frame_counter} added to DTLN buffer. Buffer size: {len(self.dtln_stft_buffer)}")

            if len(self.dtln_stft_buffer) >= self.DTLN_CHUNK_SIZE_FRAMES:
                # logger.debug(f"EHP - Triggering _process_dtln_chunk for {len(self.dtln_stft_buffer)} frames.")
                self._process_dtln_chunk()

            # --- Result and Callback Handling ---
            if detected_angles:
                all_results.append({"timestamp": frame_timestamp, "doa": detected_angles})

            t_end = time.perf_counter()
            # Log performance metrics if slow
            if (t_end - t0) * 1000 > 10.0 or self.frame_counter % 100 == 0:
                 logger.info(f"Frame {self.frame_counter} Perf: STFT={(t1-t0)*1000:.2f}ms, MVDR={(t3-t2)*1000:.2f}ms, Total={(t_end-t0)*1000:.2f}ms")

            self.frame_counter += 1
            self.internal_buffer = self.internal_buffer[self.hop_length_samples:, :]
            num_samples_in_buffer = self.internal_buffer.shape[0]

        logger.debug(f"EHP - Exiting process loop. Remaining buffer size: {num_samples_in_buffer}")

    def close(self):
        """Flushes any remaining data in the DTLN buffer."""
        print("Closing DOA Processor and flushing DTLN buffer...")
        if self.dtln_processor:
            self._process_dtln_chunk()

