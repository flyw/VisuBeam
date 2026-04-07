import numpy as np
from scipy.signal import find_peaks, peak_prominences, peak_prominences
from ..config.enhancement_config import EnhancementConfig


class DoaEngine:
    """
    基于SRP-PHAT算法的核心DOA（声源到达方向）定位引擎。

    本类负责DOA计算的核心部分。它接收一个经过预处理的频域音频帧（STFT），
    通过计算空间谱来找出声源的方向。

    主要功能：
    - 在初始化时，为所有麦克风对和扫描角度，预先计算TDOA（到达时间差）查找表。
    - 根据输入的STFT帧，计算GCC-PHAT（广义互相关相位变换）矩阵。
    - 使用GCC矩阵和TDOA查找表，计算SRP（导向响应功率）空间谱。
    - 在SRP空间谱中检测峰值，以确定DOA角度。
    - 将每次计算的结果以日志形式记录到文件中。

    注意：本引擎不负责处理音频的输入、分帧、加窗或任何预处理（如WPE去混响、降噪等）。
    它是一个纯粹的计算引擎，仅对单帧的STFT数据进行操作。
    """

    def __init__(self, config: EnhancementConfig):
        self.config = config
        print(config)
        # 1. 预计算频率掩码 (只需一次)
        freq_bins = np.fft.rfftfreq(config.fft_length, 1.0 / config.sample_rate)
        self.freq_mask = (freq_bins >= config.freq_min_hz) & (freq_bins <= config.freq_max_hz)

        # 2. 生成麦克风对
        self.mic_pairs = [(i, j) for i in range(config.num_mics) for j in range(i + 1, config.num_mics)]

        # 3. 预计算SRP查找表：所有角度的理论延迟
        self.angle_grid = np.arange(0, 181, self.config.scan_step_deg)
        self.theoretical_delays = DoaEngine._calculate_all_theoretical_delays(
            self.angle_grid, self.mic_pairs, self.config
        )

    def _calculate_gcc_matrix(self, stft_frame, is_clean_spectrum=False) -> np.ndarray:
        """计算包含可选频率滤波和可选平滑的GCC矩阵。"""
        gcc_matrix = []
        n_gcc = self.config.fft_length * self.config.interpolation_rate

        for mic1, mic2 in self.mic_pairs:
            R = stft_frame[:, mic1] * np.conj(stft_frame[:, mic2])

            R_processed = R

            R_phat = R_processed / (np.abs(R_processed) + 1e-10)
            gcc = np.fft.irfft(R_phat, n=n_gcc)
            gcc_matrix.append(gcc)

        return np.array(gcc_matrix)

    def _calculate_srp_spectrum(self, gcc_matrix: np.ndarray) -> np.ndarray:
        """从 GCC 矩阵通过延迟求和计算 SRP 频谱。"""
        srp_spectrum = np.zeros(len(self.angle_grid))
        for i, angle in enumerate(self.angle_grid):
            srp_sum = 0
            for pair_idx, pair in enumerate(self.mic_pairs):
                delay_s = self.theoretical_delays[pair][i]
                gcc_idx = DoaEngine._convert_delays_to_gcc_indices(np.array([delay_s]), self.config)[0]
                if gcc_idx < gcc_matrix.shape[1]:
                    srp_sum += gcc_matrix[pair_idx, gcc_idx]
            srp_spectrum[i] = srp_sum
        return srp_spectrum

    def _find_peaks(self, srp_fused: np.ndarray) -> tuple:
        """在最终的融合SRP谱上寻找峰值。"""
        # 计算距离阈值（以索引为单位）：peak_distance_deg / scan_step_deg
        distance_in_indices = int(self.config.peak_distance_deg / self.config.scan_step_deg)
        peaks, properties = find_peaks(
            srp_fused,
            height=self.config.peak_height_threshold,
            prominence=self.config.peak_prominence,  # 新增：使用显著性作为更强的判断标准
            distance=max(1, distance_in_indices)  # 至少为1个索引点
        )
        sorted_peaks = sorted(zip(peaks, properties['peak_heights']), key=lambda x: x[1], reverse=True)
        top_peaks = sorted_peaks[:self.config.num_sources_expected]
        detected_angles = [(self.angle_grid[p[0]], p[1]) for p in top_peaks]
        return detected_angles, srp_fused

    def get_doa_results(self, enhanced_stft_frame: np.ndarray, current_time: float, current_frame_idx: int) -> tuple:
        """执行完整的定位、融合和峰值检测流程。"""
        # 1. 锐利谱路径
        gcc_sharp = self._calculate_gcc_matrix(enhanced_stft_frame, is_clean_spectrum=False)

        gcc_sharp_filtered = gcc_sharp

        # 4. 计算SRP谱
        srp_sharp = self._calculate_srp_spectrum(gcc_sharp_filtered)

        srp_fused = srp_sharp  # 不融合，直接使用锐利谱

        # 6. 在线性功率谱上进行峰值检测
        detected_angles, _ = self._find_peaks(srp_fused)

        # 7. 将最终的SRP功率谱转换为dB单位用于热力图
        srp_power = srp_fused.copy()
        # 将SRP中的负值（可能由GCC的负旁瓣累加而来）处理为0
        srp_power[srp_power < 0] = 0
        srp_db = 10 * np.log10(srp_power + 1e-10)  # 加上一个小的epsilon防止log(0)

        return detected_angles, srp_db

    @staticmethod
    def _calculate_all_theoretical_delays(angle_grid_deg: np.ndarray, mic_pairs: list, config: EnhancementConfig) -> dict:
        """
        为给定的角度网格预先计算所有麦克风对的理论时间延迟 (TDOA)。
        这个函数通常在定位开始前调用一次，以生成一个查找表。

        Args:
            angle_grid_deg: 一维数组，包含所有待计算方向的角度 (单位：度)。
                            约定：90度为阵列法线方向 (broadside)。
            mic_pairs: 麦克风对的列表，例如 [(0, 1), (0, 2), ...]。
            config: DoaConfig 配置对象，包含麦克风位置和声速。

        Returns:
            一个字典，键是麦克风对元组，值是对应于每个角度的理论延迟时间 (秒) 的一维数组。
        """
        angle_grid_rad = np.deg2rad(angle_grid_deg)
        delays_dict = {}

        for pair in mic_pairs:
            mic1_pos = config.mic_positions[pair[0]]
            mic2_pos = config.mic_positions[pair[1]]
            distance = mic2_pos['x'] - mic1_pos['x']

            # 理论延迟计算公式: tau = (d * cos(theta)) / c
            # theta 是声源方向与麦克风阵列轴线之间的物理夹角。
            # 用户约定: 0度为最左边, 90度为正前方(broadside), 180度为最右边。
            # 这意味着用户的角度 phi 与物理角度 theta 的关系是: theta = 180 - phi。
            # 因此, 我们需要使用 cos(pi - phi_rad)。
            delays = distance * np.cos(np.pi - angle_grid_rad) / config.speed_of_sound
            delays_dict[pair] = delays

        return delays_dict

    @staticmethod
    def _convert_delays_to_gcc_indices(delays_s: np.ndarray, config: EnhancementConfig) -> np.ndarray:
        """
        将以秒为单位的时间延迟数组转换为 GCC (广义互相关) 数组中的整数索引。
        GCC 数组是通过对互功率谱进行插值 IFFT 得到的。

        Args:
            delays_s: 以秒为单位的时间延迟的一维数组。
            config: DoaConfig 配置对象，包含 FFT 长度、采样率和插值率。

        Returns:
            一个一维整数数组，包含 GCC 数组中对应的样本索引。
        """
        # GCC 数组的总长度
        n_gcc = config.fft_length * config.interpolation_rate

        # 由于频域插值，GCC 信号的有效采样率
        effective_fs = config.sample_rate * config.interpolation_rate

        # 将时间延迟转换为样本索引
        indices = np.round(delays_s * effective_fs).astype(int)

        # 处理负延迟 (对应于 GCC 数组的后半部分)
        # 这是 IFFT 的属性，负频率/延迟会环绕到数组的末尾
        indices[indices < 0] += n_gcc

        return indices
