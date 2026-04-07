import numpy as np
import tensorflow as tf
from scipy import signal

# Monkeypatch for TF2 compatibility as nara_wpe might use tf.real/tf.imag which are removed in TF2
if not hasattr(tf, 'real'):
    try:
        tf.real = tf.math.real
    except AttributeError:
        pass
if not hasattr(tf, 'imag'):
    try:
        tf.imag = tf.math.imag
    except AttributeError:
        pass
if not hasattr(tf, 'conj'):
    try:
        tf.conj = tf.math.conj
    except AttributeError:
        pass

from nara_wpe.tf_wpe import online_wpe_step, get_power_online

class WPEProcessor:
    """
    A true streaming WPE processor that operates on a frame-by-frame basis
    using the Overlap-Add method for STFT/ISTFT, accelerated by TensorFlow.
    """
    def __init__(self,
                 sample_rate: int = 16000,
                 channels: int = 4,
                 taps: int = 10,
                 delay: int = 3,
                 alpha: float = 0.9999,
                 stft_size: int = 1024,
                 stft_shift: int = 512):
        """
        Initializes the streaming WPE Processor with TensorFlow backend.
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.stft_size = stft_size
        self.stft_shift = stft_shift
        self.taps = taps
        self.delay = delay
        self.alpha = alpha

        self.frequency_bins = self.stft_size // 2 + 1
        
        # Initialize WPE state variables
        self.Q = np.stack([np.identity(channels * taps) for _ in range(self.frequency_bins)]).astype(np.complex128)
        self.G = np.zeros((self.frequency_bins, channels * taps, channels), dtype=np.complex128)

        # STFT input buffer (sliding window) - needs to hold enough frames for WPE taps + delay
        # OnlineWPE requires a buffer of (taps + delay + 1) frames
        self.wpe_buffer_len = taps + delay + 1
        self.wpe_frame_buffer = np.zeros((self.wpe_buffer_len, self.frequency_bins, channels), dtype=np.complex128)

        # Time domain buffers
        self.input_buffer = np.zeros((channels, self.stft_size), dtype=np.float32)
        self.output_buffer = np.zeros((channels, self.stft_size), dtype=np.float32)

        # Windows
        self.window = signal.windows.hann(self.stft_size).astype(np.float32)
        from nara_wpe.utils import _biorthogonal_window_fastest
        self.synthesis_window = _biorthogonal_window_fastest(
            self.window, self.stft_shift, use_amplitude=False
        ).astype(np.float32)

        # TensorFlow Setup (using compat.v1 for nara_wpe compatibility if needed)
        tf.compat.v1.disable_eager_execution()
        self.graph = tf.Graph()
        
        # Configure session for potentially better performance on CPU/GPU
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.compat.v1.Session(graph=self.graph, config=config)
        
        with self.graph.as_default():
            self.Y_tf = tf.compat.v1.placeholder(tf.complex128, shape=(self.wpe_buffer_len, self.frequency_bins, self.channels))
            self.Q_tf = tf.compat.v1.placeholder(tf.complex128, shape=(self.frequency_bins, self.channels * self.taps, self.channels * self.taps))
            self.G_tf = tf.compat.v1.placeholder(tf.complex128, shape=(self.frequency_bins, self.channels * self.taps, self.channels))
            
            # Replicate get_power_online and online_wpe_step in graph
            # Note: get_power_online expects (bins, frames, channels) in reference, 
            # but tf_wpe.get_power_online implementation might vary. 
            # In reference: get_power_online(tf.transpose(Y_tf, (1, 0, 2)))
            power_tf = get_power_online(tf.transpose(self.Y_tf, (1, 0, 2)))
            self.wpe_op = online_wpe_step(
                self.Y_tf, power_tf, self.Q_tf, self.G_tf, 
                alpha=self.alpha, taps=self.taps, delay=self.delay
            )

        print(f"TensorFlow Streaming WPE Processor Initialized (Channels: {channels}, Taps: {taps}, Delay: {delay}).")

    def dereverberate(self, audio_chunk: np.ndarray) -> np.ndarray:
        """
        Processes a new chunk of audio samples in a streaming fashion using TF.

        Args:
            audio_chunk: A new chunk of audio, shape (self.stft_shift, channels).
              
        Returns:
            The dereverberated audio chunk, shape (self.stft_shift, channels).
        """
        if audio_chunk.shape[0] != self.stft_shift:
            raise ValueError(f"Input chunk size must be {self.stft_shift}, got {audio_chunk.shape[0]}")

        # Update time domain input buffer
        self.input_buffer = np.roll(self.input_buffer, -self.stft_shift, axis=1)
        self.input_buffer[:, -self.stft_shift:] = audio_chunk.astype(np.float32).T

        # STFT
        windowed = self.input_buffer * self.window
        stft_frame = np.fft.rfft(windowed, n=self.stft_size, axis=1) # (channels, freq_bins)
        
        # Update WPE frequency domain buffer
        self.wpe_frame_buffer = np.roll(self.wpe_frame_buffer, -1, axis=0)
        self.wpe_frame_buffer[-1, :, :] = stft_frame.T # (freq_bins, channels)

        # TF Inference
        feed_dict = {
            self.Y_tf: self.wpe_frame_buffer,
            self.Q_tf: self.Q,
            self.G_tf: self.G
        }
        
        Z, self.Q, self.G = self.session.run(self.wpe_op, feed_dict=feed_dict)
        # Z is the dereverberated frame for the current step (freq_bins, channels)

        # ISTFT
        dereverberated_frame = Z.T # (channels, freq_bins)
        time_frame = np.fft.irfft(dereverberated_frame, n=self.stft_size, axis=1)
        time_frame = time_frame * self.synthesis_window

        # Overlap-add
        self.output_buffer += time_frame
        output_chunk = self.output_buffer[:, :self.stft_shift].copy()

        # Shift output buffer
        self.output_buffer = np.roll(self.output_buffer, -self.stft_shift, axis=1)
        self.output_buffer[:, -self.stft_shift:] = 0

        return output_chunk.T

    def __del__(self):
        """Cleanup TF session if it exists."""
        if hasattr(self, 'session') and self.session:
            try:
                self.session.close()
            except:
                pass