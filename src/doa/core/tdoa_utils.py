import numpy as np
from src.doa.config.doa_config import DOAConfig


def calculate_all_theoretical_delays(
    angle_grid_deg: np.ndarray, mic_pairs: list, config: DOAConfig
) -> dict:
    """
    为给定的角度网格预先计算所有麦克风对的理论时间延迟 (TDOA)。
    这个函数通常在定位开始前调用一次，以生成一个查找表。

    Args:
        angle_grid_deg: 一维数组，包含所有待计算方向的角度 (单位：度)。
                        约定：90度为阵列法线方向 (broadside)。
        mic_pairs: 麦克风对的列表，例如 [(0, 1), (0, 2), ...]。
        config: DOAConfig 配置对象，包含麦克风位置和声速。

    Returns:
        一个字典，键是麦克风对元组，值是对应于每个角度的理论延迟时间 (秒) 的一维数组。
    """
    angle_grid_rad = np.deg2rad(angle_grid_deg)
    delays_dict = {}

    for pair in mic_pairs:
        # 获取麦克风位置 - 在doa_processor中mic_positions是np.array格式
        # 麦克风位置数组的每一行是[x, y, z]，我们需要x坐标
        mic1_pos = config.mic_positions[pair[0]][0]  # x坐标
        mic2_pos = config.mic_positions[pair[1]][0]  # x坐标
        distance = mic2_pos - mic1_pos

        # 理论延迟计算公式: tau = (d * cos(theta)) / c
        # theta 是声源方向与麦克风阵列轴线之间的物理夹角。
        # 用户约定: 0度为最左边, 90度为正前方(broadside), 180度为最右边。
        # 这意味着用户的角度 phi 与物理角度 theta 的关系是: theta = 180 - phi。
        # 因此, 我们需要使用 cos(pi - phi_rad)。
        delays = distance * np.cos(np.pi - angle_grid_rad) / config.speed_of_sound
        delays_dict[pair] = delays

    return delays_dict


def convert_delays_to_gcc_indices(
    delays_s: np.ndarray, config: DOAConfig
) -> np.ndarray:
    """
    将以秒为单位的时间延迟数组转换为 GCC (广义互相关) 数组中的整数索引。
    GCC 数组是通过对互功率谱进行插值 IFFT 得到的。

    Args:
        delays_s: 以秒为单位的时间延迟的一维数组。
        config: DOAConfig 配置对象，包含 FFT 长度、采样率和插值率。

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
