import numpy as np
from scipy import signal
from .mcra import MCRAEstimator

class MCRAReducer:
    """
    A streaming noise reducer based on MCRA noise estimation and Wiener filtering.
    This class is designed as a drop-in replacement for the previous NoiseReducer.
    """
    def __init__(self,
                 sample_rate: int = 16000,
                 channels: int = 4,
                 mic_indices: list = None,
                 stft_size: int = 1024,
                 stft_shift: int = 256,
                 mcra_alpha_s: float = 0.8,
                 mcra_alpha_d: float = 0.95,
                 mcra_l_window: int = 15,
                 mcra_gamma: float = 1.67,
                 mcra_delta: float = 5.0,
                 mcra_gain_floor: float = 0.4,
                 mcra_gain_exponent: float = 0.6):
        """
        Initializes the streaming MCRA-based Noise Reducer.

        Args:
            sample_rate: Audio sample rate
            channels: Total number of input channels
            mic_indices: List of microphone channel indices to process (role=1). If None, processes all channels.
            stft_size: Size of STFT
            stft_shift: STFT shift size
            mcra_alpha_s: Power spectral smoothing factor
            mcra_alpha_d: Noise estimation update smoothing factor
            mcra_l_window: Noise tracking window size (frames)
            mcra_gamma: Noise estimation bias compensation
            mcra_delta: Speech decision threshold (dB)
            mcra_gain_floor: Gain floor
            mcra_gain_exponent: Gain exponent
        """
        self.sample_rate = sample_rate
        self.total_channels = channels
        self.mic_indices = mic_indices if mic_indices is not None else list(range(channels))
        self.num_mic_channels = len(self.mic_indices)
        self.stft_size = stft_size
        self.stft_shift = stft_shift
        self.gain_floor = mcra_gain_floor
        self.gain_exponent = mcra_gain_exponent

        # Initialize buffers only for microphone channels
        self.input_buffer = np.zeros((self.num_mic_channels, self.stft_size), dtype=np.float32)
        self.output_buffer = np.zeros((self.num_mic_channels, self.stft_size), dtype=np.float32)
        self.window = signal.windows.hann(self.stft_size)

        # Synthesis window for perfect reconstruction in overlap-add
        self.synthesis_window = self.window

        # Initialize one MCRA estimator per microphone channel
        self.mcra_estimators = [
            MCRAEstimator(
                n_fft=stft_size,
                alpha_s=mcra_alpha_s,
                alpha_d=mcra_alpha_d,
                l_window=mcra_l_window,
                gamma=mcra_gamma,
                delta=mcra_delta
            ) for _ in range(self.num_mic_channels)
        ]

        # State for decision-directed SNR estimation
        self.prev_clean_psd = None

        print(f"Streaming MCRA-based Noise Reducer Initialized (STFT size: {self.stft_size}, shift: {self.stft_shift}, "
              f"total channels: {self.total_channels}, mic channels: {self.num_mic_channels}).")

    def _calculate_prior_snr(self, ch, post_snr, noise_psd):
        alpha = 0.98
        prior_snr = alpha * (self.prev_clean_psd[ch] / (noise_psd + 1e-6)) + (1 - alpha) * np.maximum(post_snr - 1, 0)
        return prior_snr

    def reduce_noise(self, audio_chunk: np.ndarray) -> np.ndarray:
        """
        Processes a new chunk of audio samples to reduce noise using MCRA and Wiener gain.

        Args:
            audio_chunk: A new chunk of audio, shape (stft_shift, total_channels), as float32.

        Returns:
            The denoised audio chunk, shape (stft_shift, total_channels), as float32.
            Only microphone channels are processed; other channels are passed through unchanged.
        """
        if audio_chunk.shape[0] != self.stft_shift:
            raise ValueError(f"Input chunk size must be equal to the STFT shift size. "
                             f"Expected {self.stft_shift}, got {audio_chunk.shape[0]}")

        # Check if input has enough channels for our mic_indices
        if audio_chunk.shape[1] < max(self.mic_indices) + 1:
            raise ValueError(f"Input channel count ({audio_chunk.shape[1]}) is less than "
                             f"required for configured microphone indices {self.mic_indices}")

        # Extract only microphone channels for processing
        mic_audio_chunk = audio_chunk[:, self.mic_indices]  # shape: (stft_shift, num_mic_channels)
        # Transpose to get shape (num_mic_channels, stft_shift)
        audio_chunk_float = mic_audio_chunk.T

        # --- Analysis ---
        # Roll buffer and add new data
        self.input_buffer = np.roll(self.input_buffer, -self.stft_shift, axis=1)
        self.input_buffer[:, -self.stft_shift:] = audio_chunk_float

        # Apply analysis window and perform STFT
        windowed = self.input_buffer * self.window
        stft_frame = np.fft.rfft(windowed, n=self.stft_size, axis=1)

        # Initialize prev_clean_psd on the first run with the noisy PSD
        if self.prev_clean_psd is None:
            self.prev_clean_psd = np.abs(stft_frame)**2

        denoised_stft = np.zeros_like(stft_frame)

        # --- Processing per microphone channel ---
        for ch in range(self.num_mic_channels):
            stft_ch = stft_frame[ch, :]

            # 1. Calculate Power Spectral Density (PSD)
            psd = np.abs(stft_ch)**2

            # 2. Estimate noise PSD using the dedicated MCRA estimator for this channel
            noise_psd = self.mcra_estimators[ch].estimate(psd)

            # 3. Calculate gain using decision-directed approach
            post_snr = psd / (noise_psd + 1e-6)
            prior_snr = self._calculate_prior_snr(ch, post_snr, noise_psd)

            gain = prior_snr / (1 + prior_snr)
            if self.gain_exponent != 1.0:
                gain = gain ** self.gain_exponent
            gain = np.maximum(gain, self.gain_floor)

            # 4. Apply gain to the complex STFT frame
            denoised_stft_ch = stft_ch * gain
            denoised_stft[ch, :] = denoised_stft_ch

            # 5. Update previous clean PSD for next frame's prior SNR calculation
            self.prev_clean_psd[ch] = np.abs(denoised_stft_ch)**2

        # --- Synthesis ---
        # Inverse FFT
        time_frame = np.fft.irfft(denoised_stft, n=self.stft_size, axis=1)

        # Apply synthesis window and perform overlap-add
        self.output_buffer += time_frame * self.synthesis_window

        # Extract the processed chunk
        output_chunk_float = self.output_buffer[:, :self.stft_shift].copy()

        # Shift the output buffer for the next frame
        self.output_buffer = np.roll(self.output_buffer, -self.stft_shift, axis=1)
        self.output_buffer[:, -self.stft_shift:] = 0

        # Transpose back to (stft_shift, num_mic_channels)
        processed_mic_chunk = output_chunk_float.T

        # Create output array with the same shape as input
        output_chunk = audio_chunk.copy()  # shape: (stft_shift, input_channels)

        # Replace only the microphone channels with processed data
        # Make sure mic_indices are within the input/output channel range
        valid_mic_indices = [idx for idx in self.mic_indices if idx < audio_chunk.shape[1]]
        for i, orig_mic_idx in enumerate(valid_mic_indices):
            output_chunk[:, orig_mic_idx] = processed_mic_chunk[:, i]

        return output_chunk
