import torch
import numpy as np
import logging
from torch import Tensor

from src.enhancement.config.enhancement_config import EnhancementConfig

logger = logging.getLogger(__name__)

# For type hinting
ComplexTensor = torch.Tensor

class MvdrProcessor:
    """
    MVDR (Minimum Variance Distortionless Response) Beamformer.
    This class is a pure computational engine that receives instructions on how to process each frame.
    It estimates Power Spectral Density (PSD) matrices, computes the MVDR spatial filter based on
    an externally provided angle, and applies it.
    """

    def __init__(self, config: EnhancementConfig):
        """
        Initializes the MVDR processor with configuration parameters.
        """
        self.config = config

        # Device detection for Ubuntu/Intel (CUDA preferred)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        logger.info(f"MvdrProcessor initialized on device: {self.device}")

        # EMA (Exponential Moving Average) factors for PSD updates.
        self.alpha_speech = 0.85
        self.alpha_noise_fast = 0.80

        # Placeholders for the estimated PSD matrices (will be moved to device on first update)
        self.psd_speech = None
        self.speech_psd_frames = 0 # Counter for warm-up
        self.psd_noise_slow = None # This will be updated from the controller
        self.psd_noise_fast = None

        # Microphone array and signal properties from config
        raw_positions = self.config.mic_positions
        active_mics = []
        if raw_positions and isinstance(raw_positions[0], dict) and 'role' in raw_positions[0]:
             for pos in raw_positions:
                 if pos.get('role') == 1:
                     active_mics.append([pos['x'], pos['y'], pos['z']])
        else:
             active_mics = [[pos['x'], pos['y'], pos['z']] for pos in raw_positions]

        # Pre-move mic positions to GPU
        self.mic_positions = torch.tensor(active_mics, dtype=torch.float32, device=self.device)
        self.num_mics = len(self.mic_positions)

        self.sample_rate = self.config.sample_rate
        self.fft_length = self.config.fft_length
        self.speed_of_sound = self.config.speed_of_sound

        # Pre-calculate frequencies on GPU
        freqs_np = np.fft.rfftfreq(self.fft_length, 1.0 / self.sample_rate).astype(np.float32)
        self.freqs = torch.from_numpy(freqs_np).to(self.device)
        
        self.cached_steering_vector = None
        self.cached_angle = None

    def _calculate_psd_matrix(self, stft_frame: ComplexTensor) -> ComplexTensor:
        """
        Calculates the Power Spectral Density (PSD) matrix for a single STFT frame on GPU.
        """
        # stft_frame: (F, C) -> Output: (F, C, C)
        return torch.einsum('fc,fd->fcd', stft_frame, stft_frame.conj())

    def _update_psd_matrix(self, current_psd: ComplexTensor, new_frame_psd: ComplexTensor, alpha: float) -> ComplexTensor:
        """
        Updates a PSD matrix using Exponential Moving Average (EMA).
        """
        if current_psd is None:
            return new_frame_psd
        else:
            return alpha * current_psd + (1 - alpha) * new_frame_psd

    def update_speech_estimate(self, stft_data: ComplexTensor):
        """Updates the speech PSD matrix with a new frame."""
        psd_frame = self._calculate_psd_matrix(stft_data)
        self.psd_speech = self._update_psd_matrix(self.psd_speech, psd_frame, self.alpha_speech)
        self.speech_psd_frames += 1

    def update_noise_estimate_slow(self, stft_data: ComplexTensor):
        """Updates the slow-adapting noise PSD matrix (ambient noise)."""
        psd_frame = self._calculate_psd_matrix(stft_data)
        self.psd_noise_slow = self._update_psd_matrix(self.psd_noise_slow, psd_frame, self.alpha_noise_slow)

    def update_noise_estimate_fast(self, stft_data: ComplexTensor):
        """Updates the fast-adapting noise PSD matrix (interferers)."""
        psd_frame = self._calculate_psd_matrix(stft_data)
        self.psd_noise_fast = self._update_psd_matrix(self.psd_noise_fast, psd_frame, self.alpha_noise_fast)

    def get_combined_noise_psd(self) -> ComplexTensor:
        """
        Combines slow and fast noise estimates. Prioritizes the fast estimate.
        """
        if self.psd_noise_fast is not None:
            return self.psd_noise_fast
        return self.psd_noise_slow
        
    def _calculate_steering_vector(self, angle_deg: float) -> ComplexTensor:
        """
        Calculates the steering vector on GPU.
        Caches the result to avoid re-computation if the angle hasn't changed.
        """
        if self.cached_angle is not None and abs(angle_deg - self.cached_angle) < 1e-5 and self.cached_steering_vector is not None:
            return self.cached_steering_vector

        # All calculations on GPU
        target_angle_rad = torch.tensor(np.deg2rad(angle_deg), dtype=torch.float32, device=self.device)
        direction_vector = torch.tensor([torch.cos(target_angle_rad), torch.sin(target_angle_rad), 0.0], dtype=torch.float32, device=self.device)
        
        # time_delays = (mic_positions @ direction_vector) / c
        time_delays = torch.mv(self.mic_positions, direction_vector) / self.speed_of_sound
        
        # steering = exp(-2j * pi * f * tau)
        # arg: (F, 1) * (1, M) -> (F, M)
        arg = -2j * np.pi * self.freqs.unsqueeze(1) * time_delays.unsqueeze(0)
        steering_vector = torch.exp(arg).to(dtype=torch.complex64)

        self.cached_steering_vector = steering_vector
        self.cached_angle = angle_deg

        return steering_vector

    def process(self, stft_data: ComplexTensor, decision_type: str, process_angle: float, background_psd: ComplexTensor) -> ComplexTensor:
        """
        Processes a single STFT frame with MVDR beamforming on GPU.
        """
        if not self.config.enable_mvdr:
            return stft_data[:, 0]

        # 0. Data migration to GPU
        if not isinstance(stft_data, Tensor):
            stft_data = torch.from_numpy(stft_data).to(self.device)
        else:
            stft_data = stft_data.to(self.device)

        # Ensure we are not using float64/complex128 on MPS
        if self.device.type == 'mps':
            if stft_data.dtype == torch.float64:
                stft_data = stft_data.to(torch.float32)
            elif stft_data.dtype == torch.complex128:
                stft_data = stft_data.to(torch.complex64)

        if background_psd is not None:
             background_psd = background_psd.to(self.device)
             if self.device.type == 'mps':
                 if background_psd.dtype == torch.complex128:
                     background_psd = background_psd.to(torch.complex64)
                 elif background_psd.dtype == torch.float64:
                     background_psd = background_psd.to(torch.float32)
             
             if stft_data.dtype != background_psd.dtype:
                 stft_data = stft_data.to(background_psd.dtype)
            
        self.psd_noise_slow = background_psd

        # 1. Update PSDs
        if decision_type in ["Target DOA", "Tracking", "Switching"]:
            self.update_speech_estimate(stft_data)
        elif decision_type == "Interference DOA":
            self.update_noise_estimate_fast(stft_data)

        # 2. Checks
        if self.psd_noise_slow is None:
            return stft_data[:, 0]
        
        psd_n = self.get_combined_noise_psd()
        if psd_n is None:
            return stft_data[:, 0]

        # 3. Beamforming
        reference_vector = self._calculate_steering_vector(process_angle)
        reference_vector = reference_vector.to(psd_n.dtype)

        if self.psd_speech is not None and self.speech_psd_frames > 10:
            beamformer = self.get_mvdr_vector(self.psd_speech, psd_n, reference_vector)
        else:
            beamformer = self.get_standard_mvdr_vector(psd_n, reference_vector)
        
        # 4. Apply
        beamformer = beamformer.to(stft_data.dtype)
        enhanced_stft = torch.einsum('fc,fc->f', beamformer.conj(), stft_data)

        return enhanced_stft

    @staticmethod
    def _solve_system(A: ComplexTensor, B: ComplexTensor) -> ComplexTensor:
        """
        Solves AX = B using torch.linalg.solve, with a fallback to CPU for MPS devices
        handling complex types, as MPS currently doesn't support complex LU factorization.
        """
        if A.device.type == 'mps' and A.is_complex():
            return torch.linalg.solve(A.cpu(), B.cpu()).to(A.device)
        return torch.linalg.solve(A, B)

    @staticmethod
    def get_mvdr_vector(
        psd_s: ComplexTensor,
        psd_n: ComplexTensor,
        reference_vector: ComplexTensor,
        eps: float = 1e-8,
    ) -> ComplexTensor:
        """
        Computes the MVDR-Souden beamforming vector.
        h = (Npsd^-1 @ Spsd) / (Tr(Npsd^-1 @ Spsd)) @ u
        """
        C = psd_n.size(-1)
        eye = torch.eye(C, dtype=torch.complex64 if psd_n.dtype.is_complex else torch.float32, device=psd_n.device)
        psd_n_stable = psd_n + eps * eye.view(1, C, C)

        # Handle solve on CPU if using MPS and complex types
        numerator = MvdrProcessor._solve_system(psd_n_stable, psd_s)
        
        trace = torch.einsum('...fii->...f', numerator).real
        norm_factor = trace[..., None, None] + eps
        phi = numerator / norm_factor
        beamform_vector = torch.einsum("...fec,...fc->...fe", phi, reference_vector)

        return beamform_vector

    @staticmethod
    def get_standard_mvdr_vector(
        psd_n: ComplexTensor,
        steering_vector: ComplexTensor,
        eps: float = 1e-8
    ) -> ComplexTensor:
        """
        Computes the Standard MVDR (Capon) beamforming vector.
        w = (Pn^-1 @ d) / (d^H @ Pn^-1 @ d)
        """
        C = psd_n.size(-1)
        eye = torch.eye(C, dtype=torch.complex64 if psd_n.dtype.is_complex else torch.float32, device=psd_n.device)
        psd_n_stable = psd_n + eps * eye.view(1, C, C)

        # numerator = Pn^-1 @ d
        # Ensure steering_vector has the same dtype as psd_n
        steering_vector = steering_vector.to(psd_n_stable.dtype)
        
        # Handle solve on CPU if using MPS and complex types
        numerator = MvdrProcessor._solve_system(psd_n_stable, steering_vector.unsqueeze(-1)).squeeze(-1)

        # denominator = d^H @ Pn^-1 @ d
        # d^H @ numerator
        denominator = torch.einsum('...c,...c->...', steering_vector.conj(), numerator)

        # w = numerator / denominator
        beamform_vector = numerator / (denominator.unsqueeze(-1) + eps)

        return beamform_vector
