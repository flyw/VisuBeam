import numpy as np
import scipy.signal
import torch
from scipy.signal import windows


class StftEngine:
    """
    STFT/iSTFT 封装器 (Stage 2.2)
    负责将时域信号转换为频域 (analysis) 或将频域信号转换回时域 (synthesis)。
    """

    def __init__(self, frame_len: int, hop_len: int, use_scipy=True, fs=16000):
        # 参数应通过配置中心 (DoaConfig) 间接传入
        self.frame_len = frame_len
        self.hop_len = hop_len
        self.fs = fs  # 采样率默认为 16000 [5, 6]

        # 汉宁窗的计算基于帧长 [5, 8]
        self.window = scipy.signal.windows.hann(self.frame_len).astype(np.float32)
        self.use_scipy = use_scipy

        # 频率分箱数 (N_Freq)
        self.n_freq = self.frame_len // 2 + 1
    
    def analysis_single_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        对单个多通道音频帧执行STFT。
        专为流式处理设计，其中分帧逻辑在外部处理。

        Args:
            frame (np.ndarray): 一个音频帧，形状为 (n_samples, n_channels)。

        Returns:
            np.ndarray: STFT结果，形状为 (n_bins, n_channels)。
        """
        # 确保帧是二维的
        if frame.ndim != 2:
            raise ValueError(f"Input frame must be 2D (n_samples, n_channels), but got shape {frame.shape}")
        
        # 将一维窗口广播到所有通道
        windowed_frame = frame * self.window[:, np.newaxis]
        
        # 沿时间轴 (axis=0) 对每个通道执行FFT
        stft_result = np.fft.rfft(windowed_frame, n=self.frame_len, axis=0)
        
        return stft_result

    def analysis(self, signal: np.ndarray) -> np.ndarray:
        """
        对完整的单通道信号执行短时傅里叶变换 (STFT)。
        源文件 [3] 中的实现。
        """
        if signal.ndim != 1:
            raise ValueError(f"Input signal for full analysis must be 1D, but got shape {signal.shape}")

        if self.use_scipy:
            # 使用 scipy.signal.stft，注意使用 self.fs [3]
            # 返回值 Zxx 形状为 (n_freq, n_frames)
            _, _, Zxx = scipy.signal.stft(signal, fs=self.fs, window=self.window,
                                          nperseg=self.frame_len, noverlap=self.hop_len)
            return Zxx
        else:
            # 手动实现 STFT (NumPy) [4]
            n_frames = 1 + (len(signal) - self.frame_len) // self.hop_len
            stft_matrix = np.empty((self.n_freq, n_frames), dtype=np.complex64)

            for i in range(n_frames):
                start = i * self.hop_len
                end = start + self.frame_len
                frame = signal[start:end] * self.window
                stft_matrix[:, i] = np.fft.rfft(frame, self.frame_len)

            return stft_matrix

    def synthesis(self, stft_matrix: np.ndarray) -> np.ndarray:
        """
        执行逆短时傅里叶变换 (iSTFT)。
        源文件 [4, 7] 中的实现。
        """
        if self.use_scipy:
            # 使用 scipy.signal.istft，注意使用 self.fs [4]
            _, x = scipy.signal.istft(stft_matrix, fs=self.fs, window=self.window,
                                      nperseg=self.frame_len, noverlap=self.hop_len)
            return x
        else:
            # 手动实现 iSTFT (NumPy - Overlap-Add) [7]
            n_freq, n_frames = stft_matrix.shape
            expected_len = self.frame_len + (n_frames - 1) * self.hop_len
            signal = np.zeros(expected_len)
            window_correction = np.zeros(expected_len)

            for i in range(n_frames):
                frame = np.fft.irfft(stft_matrix[:, i], self.frame_len)
                start = i * self.hop_len
                end = start + self.frame_len

                # 累加和加窗
                signal[start:end] += frame * self.window
                window_correction[start:end] += self.window ** 2

            # 窗口修正 [7]
            window_correction[window_correction < 1e-6] = 1.0
            signal /= window_correction

            return signal

    def synthesis_multi_frame(self, stft_frames: list) -> np.ndarray:
        """
        执行多通道逆短时傅里叶变换 (iSTFT)，用于流式帧列表。
        此方法现在可以处理单通道（1D 帧）和多通道（2D 帧）输入。
        
        Args:
            stft_frames (list): STFT帧的列表，每个帧的形状为 (n_freq,) 或 (n_freq, n_channels)。

        Returns:
            np.ndarray: 重建的时域信号，单通道时为1D数组，多通道时为2D数组。
        """
        if not stft_frames:
            return np.array([])

        # Convert all frames to numpy if they are torch tensors
        stft_frames = [f.numpy() if isinstance(f, torch.Tensor) else f for f in stft_frames]

        # Handle single-channel case (from MVDR) by reshaping
        if stft_frames[0].ndim == 1:
            stft_frames = [frame[:, np.newaxis] for frame in stft_frames]

        # 从第一个帧获取通道数和帧数
        n_channels = stft_frames[0].shape[1]
        n_frames = len(stft_frames)

        # 计算期望的输出信号长度
        expected_len = self.frame_len + (n_frames - 1) * self.hop_len
        
        # 初始化输出信号和窗口校正数组
        signal = np.zeros((expected_len, n_channels), dtype=np.float32)
        window_correction = np.zeros(expected_len, dtype=np.float32)

        for i, frame_complex in enumerate(stft_frames):
            # 对每个通道执行irfft
            frame_time = np.fft.irfft(frame_complex, self.frame_len, axis=0)

            start = i * self.hop_len
            end = start + self.frame_len

            # 重叠相加
            signal[start:end, :] += frame_time * self.window[:, np.newaxis]
            window_correction[start:end] += self.window ** 2

        # 窗口校正，避免除以零
        window_correction[window_correction < 1e-12] = 1.0
        signal /= window_correction[:, np.newaxis]

        # 如果结果是单通道，则移除多余的维度
        return signal.squeeze()
