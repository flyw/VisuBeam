from src.core.config.mcra_config import McraConfig
from .mvdr_config import MvdrConfig
from .dtln_config import DTLNConfig
from .webrtc_apm_config import WebRtcApmConfig


class EnhancementConfig:
    def __init__(self, config_data: dict):
        """
        Initializes and validates the DOA configuration from a dictionary.
        """
        if not config_data:
            # Allow empty config, will use defaults
            config_data = {}

        # Handle both full config and doa-only config
        if "enhancement" in config_data:
            enhancement_section = config_data["enhancement"]
        else:
            enhancement_section = config_data

        # Algorithm switches
        self.enable_mcra_denoise = enhancement_section.get("enable_mcra_denoise", False)
        self.enable_resolution_enhance = enhancement_section.get("enable_resolution_enhance", False)
        self.enable_fusion = enhancement_section.get("enable_fusion", False)
        self.save_wpe_output = enhancement_section.get("save_wpe_output", False)
        self.save_denoised_output = enhancement_section.get("save_denoised_output", False)
        self.enable_mvdr = enhancement_section.get("enable_mvdr", False)
        self.enable_mvdr_output_wav = enhancement_section.get("enable_mvdr_output", False) # 保存MVDR处理后的音频
        self.enable_webrtc_apm = enhancement_section.get("enable_webrtc_apm", False)
        self.save_apm_output = enhancement_section.get("save_apm_output", False)
        self.dtln_pool_size = enhancement_section.get("dtln_pool_size", 4)
        # Output batching (latency control)
        self.dtln_chunk_size_frames = enhancement_section.get("dtln_chunk_size_frames", 1)


        # Instantiate modular configs
        self.mcra = McraConfig(enhancement_section.get("mcra", {}))
        self.mvdr = MvdrConfig(enhancement_section.get("mvdr", {}))
        self.dtln = DTLNConfig.from_dict(enhancement_section)
        self.webrtc_apm = WebRtcApmConfig.from_dict(enhancement_section.get("webrtc_apm", {}))

        # DOA parameters
        self.frame_length_ms = enhancement_section.get("frame_length_ms", 64)
        self.hop_length_ms = enhancement_section.get("hop_length_ms", 32)
        self.interpolation_rate = enhancement_section.get("interpolation_rate", 4)
        self.freq_min_hz = enhancement_section.get("freq_min_hz", 300)
        self.freq_max_hz = enhancement_section.get("freq_max_hz", 4900)
        self.use_freq_filter = enhancement_section.get("use_freq_filter", True)
        self.num_mics = enhancement_section.get("num_mics", 4) # Default to 4 microphones
        self.scan_step_deg = enhancement_section.get("scan_step_deg", 10.0)  # 从DOA_Oct_15移植
        self.peak_distance_deg = enhancement_section.get("peak_distance_deg", 25)
        self.peak_height_threshold = enhancement_section.get("peak_height_threshold", 0.45)
        self.peak_prominence = enhancement_section.get("peak_prominence", 0.2)  # 从DOA_Oct_15移植
        self.num_sources_expected = enhancement_section.get("num_sources_expected", 1)







        # 音频配置参数现在直接在顶层或已注入到doa_config中
        audio_config = config_data.get("audio", {})
        self.mic_positions = audio_config.get("mic_positions", [])
        self.sample_rate = audio_config.get("sample_rate", 16000)  # 采样率（从audio部分获取）
        self.speed_of_sound = audio_config.get("speed_of_sound", 343.0)  # 声速

        # 角度更新超时配置
        self.angle_update_timeout = enhancement_section.get("angle_update_timeout", 10.0)



        # FFT参数（从DOA_Oct_15移植）
        self.fft_length = int(self.sample_rate * self.frame_length_ms / 1000)
        self.hop_length = int(self.sample_rate * self.hop_length_ms / 1000)

        print(f"Enhancement Config loaded successfully.")
