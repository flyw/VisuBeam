import numpy as np


class MCRAEstimator:
    """
    MCRA (Minima Controlled Recursive Averaging) 噪声估计算法的一个简化实现。
    该类为单个通道维护状态。
    """

    def __init__(self, n_fft, alpha_s, alpha_d, l_window, gamma, delta):
        """
        初始化 MCRA 噪声估计器。
        """
        self.n_fft = n_fft
        self.alpha_s = alpha_s
        self.alpha_d = alpha_d
        self.L = l_window
        self.gamma = gamma
        self.delta = delta

        # 状态变量
        n_bins = self.n_fft // 2 + 1
        self.S = np.zeros(n_bins)  # 当前帧的平滑功率谱
        self.S_min = np.full(n_bins, np.inf)  # 平滑功率谱的最小值
        self.noise_psd = None  # 噪声功率谱的估计
        self.p = np.zeros(n_bins)  # 语音存在概率 (简化的二进制)
        self.frame_psds_buffer = []  # 存储最近L帧的平滑功率谱

    def estimate(self, frame_psd: np.ndarray) -> np.ndarray:
        """
        估计给定帧的噪声功率谱 (PSD)。

        Args:
            frame_psd: 当前帧的功率谱密度 (单通道)。

        Returns:
            估计的噪声功率谱密度。
        """
        if self.noise_psd is None:
            # 使用第一帧进行初始化
            self.noise_psd = frame_psd.copy()
            self.S = frame_psd.copy()

        # 1. 平滑当前帧的功率谱
        self.S = self.alpha_s * self.S + (1 - self.alpha_s) * frame_psd

        # 2. 跟踪最近 L 帧的最小平滑功率谱
        if len(self.frame_psds_buffer) < self.L:
            self.frame_psds_buffer.append(self.S)
        else:
            # 缓冲区已满，移除最旧的并添加新的
            self.frame_psds_buffer.pop(0)
            self.frame_psds_buffer.append(self.S)

        self.S_min = np.min(np.array(self.frame_psds_buffer), axis=0)

        # 3. 估计语音存在概率 (简化为二进制决策)
        # ratio > delta 表示可能存在语音
        ratio = self.S / (self.gamma * self.S_min + 1e-6)
        self.p = np.where(ratio > self.delta, 1.0, 0.0)

        # 4. 递归更新噪声估计 (只在语音不存在的频段更新)
        # 使用一个更标准的条件更新方法
        is_noise_bin = (self.p < 0.5)
        self.noise_psd[is_noise_bin] = self.alpha_d * self.noise_psd[is_noise_bin] + \
                                       (1 - self.alpha_d) * self.S[is_noise_bin]

        # 5. 确保噪声估计不会超过当前帧的平滑谱
        self.noise_psd = np.minimum(self.noise_psd, self.S)

        return self.noise_psd
