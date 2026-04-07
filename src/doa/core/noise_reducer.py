import numpy as np
from scipy import signal

class NoiseReducer:
    """
    A streaming noise reducer using spectral subtraction.
    This implementation is a simplified version and may require tuning for specific noise types.
    """
    def __init__(self,
                 sample_rate: int = 16000,
                 channels: int = 4,
                 stft_size: int = 1024,
                 stft_shift: int = 256,
                 noise_reduction_amount: float = 2.0,
                 noise_update_rate: float = 0.01):
        """
        Initializes the streaming Noise Reducer.
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.stft_size = stft_size
        self.stft_shift = stft_shift
        self.noise_reduction_amount = noise_reduction_amount
        self.noise_update_rate = noise_update_rate

        self.input_buffer = np.zeros((channels, self.stft_size), dtype=np.float32)
        self.output_buffer = np.zeros((channels, self.stft_size), dtype=np.float32)
        self.window = signal.windows.hann(self.stft_size)
        
        # Use a simple trick to get a synthesis window for perfect reconstruction
        self.synthesis_window = self.window

        self.noise_profile = np.zeros((channels, self.stft_size // 2 + 1), dtype=np.float32)
        self.frame_count = 0
        self.initial_noise_frames = 10  # Use first 10 frames to build initial noise profile

        print(f"Streaming Noise Reducer Initialized (STFT size: {self.stft_size}, shift: {self.stft_shift}).")

    def reduce_noise(self, audio_chunk: np.ndarray) -> np.ndarray:
        """
        Processes a new chunk of audio samples to reduce noise.

        Args:
            audio_chunk: A new chunk of audio, shape (256, channels).
              
        Returns:
            The denoised audio chunk, shape (256, channels).
        """
        if audio_chunk.shape[0] != self.stft_shift:
            raise ValueError(f"Input chunk size must be equal to the STFT shift size. "
                             f"Expected {self.stft_shift}, got {audio_chunk.shape[0]}")

        audio_chunk_float = (audio_chunk.astype(np.float32) / 32768.0).T

        self.input_buffer = np.roll(self.input_buffer, -self.stft_shift, axis=1)
        self.input_buffer[:, -self.stft_shift:] = audio_chunk_float

        windowed = self.input_buffer * self.window
        stft_frame = np.fft.rfft(windowed, n=self.stft_size, axis=1)
        stft_magnitude = np.abs(stft_frame)
        stft_phase = np.angle(stft_frame)

        if self.frame_count < self.initial_noise_frames:
            # Build initial noise profile
            self.noise_profile += stft_magnitude / self.initial_noise_frames
        else:
            # Update noise profile when signal is likely noise
            # This is a very simple voice activity detection (VAD)
            signal_power = np.mean(stft_magnitude**2)
            noise_power = np.mean(self.noise_profile**2)
            if signal_power < 1.5 * noise_power: # Simple threshold
                self.noise_profile = (1 - self.noise_update_rate) * self.noise_profile + self.noise_update_rate * stft_magnitude

        # Spectral subtraction
        denoised_magnitude = stft_magnitude - self.noise_profile * self.noise_reduction_amount
        denoised_magnitude = np.maximum(denoised_magnitude, 0) # Ensure non-negative

        # Reconstruct frame
        denoised_frame = denoised_magnitude * np.exp(1j * stft_phase)

        time_frame = np.fft.irfft(denoised_frame, n=self.stft_size, axis=1)
        time_frame = time_frame * self.synthesis_window

        self.output_buffer += time_frame
        output_chunk_float = self.output_buffer[:, :self.stft_shift].copy()

        self.output_buffer = np.roll(self.output_buffer, -self.stft_shift, axis=1)
        self.output_buffer[:, -self.stft_shift:] = 0

        output_int16 = (output_chunk_float.T * 32768.0).astype(np.int16)
        
        self.frame_count += 1

        return output_int16
