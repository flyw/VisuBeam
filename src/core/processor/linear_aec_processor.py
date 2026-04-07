import numpy as np
from typing import Optional, List, Dict, Any
import collections
import threading
import scipy.signal

class LinearAECProcessor:
    """
    High-Precision DOA Oriented AEC v3.0 (Python Prototype)
    
    V3.2 Ultra-Stable Version:
    1. Divergence Protection: Resets weights if output energy > input energy.
    2. Minimal Step-size (mu=0.02): Prevents gradient explosion.
    3. Auto-Alignment: Starts with 0 delay to capture near-field loopback.
    """
    def __init__(self, config, sample_rate: int = 16000, input_channels: int = 4, 
                 mic_positions: Optional[List[Dict[str, Any]]] = None):
        self.config = config
        self.sample_rate = sample_rate
        self.input_channels = input_channels
        self.mic_positions = mic_positions
        
        # 1. Channel Roles
        self.mic_indices = []
        self.ref_index = getattr(config, 'reference_channel_index', None)
        if self.mic_positions:
            for i, pos in enumerate(self.mic_positions):
                role = pos.get('role', 1) 
                if role == 1: self.mic_indices.append(i)
                elif role == 2: self.ref_index = i 
        else:
            self.mic_indices = [i for i in range(self.input_channels) if i != self.ref_index]
        self.proc_channels = len(self.mic_indices)
        
        # 2. MDF Parameters
        self.block_size = int(sample_rate * 0.01) # 10ms = 160 samples
        self.fft_size = 2 * self.block_size
        self.num_bins = self.fft_size // 2 + 1
        self.M = 16 # 160ms filter length
        
        # Ultra-conservative Adaptation
        self.mu = 0.02 
        self.dtd_threshold = 0.4
        self.leakage = 0.999 # Stronger leakage to suppress divergence
        self.psd_floor = 1e-2
        
        # 3. Delay alignment - Force to 0 for initial auto-sync
        self.current_delay_samples = 0
        self.ref_delay_buffer = collections.deque(maxlen=int(sample_rate * 0.5))
        
        # 4. State
        self.X_history = np.zeros((self.M, self.num_bins), dtype=np.complex64)
        self.W = np.zeros((self.M, self.num_bins, self.proc_channels), dtype=np.complex64)
        self.P = np.ones(self.num_bins, dtype=np.float32) * self.psd_floor
        
        self.S_ee = np.zeros((self.num_bins, self.proc_channels), dtype=np.float32)
        self.S_dd = np.zeros((self.num_bins, self.proc_channels), dtype=np.float32)
        self.S_ed = np.zeros((self.num_bins, self.proc_channels), dtype=np.complex64)
        
        self.old_ref = np.zeros(self.block_size, dtype=np.float32)
        self._frame_count = 0

    def _process_block(self, mic_block: np.ndarray, ref_block: np.ndarray) -> np.ndarray:
        self._frame_count += 1
        
        # 1. Shared Reference FFT
        ref_fft_input = np.concatenate([self.old_ref, ref_block])
        X_curr = np.fft.rfft(ref_fft_input, axis=0)
        self.X_history = np.roll(self.X_history, 1, axis=0)
        self.X_history[0] = X_curr
        
        # PSD estimation
        self.P = 0.9 * self.P + 0.1 * np.abs(X_curr)**2
        P_safe = self.P + 0.1 # Add regularization
        
        # 2. Echo Estimation
        Y = np.zeros((self.num_bins, self.proc_channels), dtype=np.complex64)
        for m in range(self.M):
            Y += self.W[m] * self.X_history[m, :, np.newaxis]
            
        y_time = np.fft.irfft(Y, n=self.fft_size, axis=0)
        y_echo = y_time[self.block_size:]
        error = mic_block - y_echo
        
        # 3. Divergence Protection (IMPORTANT)
        mic_pwr = np.mean(mic_block**2)
        err_pwr = np.mean(error**2)
        if err_pwr > mic_pwr * 1.2 and mic_pwr > 1e-5:
            # If echo cancellation is increasing energy, it's diverging.
            # Reset weights and bypass for this frame.
            self.W.fill(0)
            if self._frame_count % 100 == 0:
                print(f"[LinearAEC] Divergence detected (Gain: {err_pwr/mic_pwr:.2f}x). Resetting weights.")
            return mic_block
        
        # 4. Adaptation
        ref_pwr = np.mean(ref_block**2)
        if ref_pwr > 1e-6:
            err_fft_input = np.zeros((self.fft_size, self.proc_channels), dtype=np.float32)
            err_fft_input[self.block_size:] = error
            E = np.fft.rfft(err_fft_input, axis=0)
            
            mic_fft_input = np.zeros((self.fft_size, self.proc_channels), dtype=np.float32)
            mic_fft_input[self.block_size:] = mic_block
            D = np.fft.rfft(mic_fft_input, axis=0)
            
            # Coherence-based DTD
            alpha = 0.9
            self.S_ee = alpha * self.S_ee + (1-alpha) * np.abs(E)**2
            self.S_dd = alpha * self.S_dd + (1-alpha) * np.abs(D)**2
            self.S_ed = alpha * self.S_ed + (1-alpha) * (E * np.conj(D))
            coherence = np.abs(self.S_ed)**2 / (self.S_ee * self.S_dd + 1e-10)
            
            gamma_global = np.min(coherence, axis=1)
            mu_scale = np.clip((gamma_global - self.dtd_threshold) / (1 - self.dtd_threshold), 0, 1)
            
            # Normalized Update
            mu_norm = (self.mu * mu_scale) / P_safe
            mu_norm = np.minimum(mu_norm, 0.1) # Hard cap on step size
            
            for m in range(self.M):
                G = mu_norm[:, np.newaxis] * np.conj(self.X_history[m, :, np.newaxis]) * E
                self.W[m] = self.leakage * self.W[m] + G
                # Weight Projection
                w_t = np.fft.irfft(self.W[m], n=self.fft_size, axis=0)
                w_t[self.block_size:] = 0
                self.W[m] = np.fft.rfft(w_t, axis=0)
                
            if self._frame_count % 500 == 0:
                print(f"[LinearAEC] Frame {self._frame_count} | Coherence: {np.mean(gamma_global):.2f} | Mu: {np.mean(mu_norm):.4f}")
        
        self.old_ref = ref_block.copy()
        return error

    def process(self, audio_chunk: np.ndarray) -> np.ndarray:
        if self.ref_index is None: return audio_chunk
        
        # 1. Normalize
        data_float = audio_chunk.astype(np.float32) / 32768.0
        ref_sig = data_float[:, self.ref_index]
        mic_sigs = data_float[:, self.mic_indices]
        
        # 2. Process
        error_float = self._process_block(mic_sigs, ref_sig)
        
        # 3. Output Reconstruct
        out_data = audio_chunk.copy()
        out_data[:, self.mic_indices] = (np.clip(error_float, -1.0, 1.0) * 32767.0).astype(np.int16)
        return out_data
