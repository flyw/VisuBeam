import yaml
from src.core.config.mcra_config import McraConfig
from src.core.config.wpe_config import WpeConfig


class DOAConfig:
    def __init__(self, config_data: dict):
        """
        Initializes and validates the DOA configuration from a dictionary.
        """
        if not config_data:
            # Allow empty config, will use defaults
            config_data = {}

        # Handle both full config and doa-only config
        if "doa" in config_data:
            doa_section = config_data["doa"]
            # If passing full config, we might want access to other sections if needed
        else:
            doa_section = config_data

        # Algorithm switches
        self.enable_wpe = doa_section.get("enable_wpe", False)
        self.enable_mcra_denoise = doa_section.get("enable_mcra_denoise", False)
        self.enable_resolution_enhance = doa_section.get("enable_resolution_enhance", False)
        self.enable_fusion = doa_section.get("enable_fusion", False)
        self.save_wpe_output = doa_section.get("save_wpe_output", False)
        self.save_mcra_output = doa_section.get("save_mcra_output", False)
        # Instantiate modular configs
        self.wpe = WpeConfig(doa_section.get("wpe", {}))
        self.mcra = McraConfig(doa_section.get("mcra", {}))

        # DOA parameters
        self.frame_length_ms = doa_section.get("frame_length_ms", 64)
        self.hop_length_ms = doa_section.get("hop_length_ms", 32)
        self.interpolation_rate = doa_section.get("interpolation_rate", 4)
        self.scan_step_deg = doa_section.get("scan_step_deg", 10.0)  # 从DOA_Oct_15移植
        self.num_sources_expected = doa_section.get("num_sources_expected", 1)
        self.doa_precision = doa_section.get("doa_precision", 10.0)
        self.peak_height_threshold = doa_section.get("peak_height_threshold", 0.45)
        self.peak_prominence = doa_section.get("peak_prominence", 0.2)  # 从DOA_Oct_15移植
        self.peak_distance_deg = doa_section.get("peak_distance_deg", 25)
        self.energy_gate_threshold = doa_section.get("energy_gate_threshold", 0.4)
        self.freq_min_hz = doa_section.get("freq_min_hz", 300)
        self.freq_max_hz = doa_section.get("freq_max_hz", 4900)
        self.use_freq_filter = doa_section.get("use_freq_filter", True)
        self.num_mics = doa_section.get("num_mics", 4) # Default to 4 microphones

        # 音频配置参数现在直接在顶层或已注入到doa_config中
        audio_config = config_data.get("audio", {})
        self.mic_positions = audio_config.get("mic_positions", [])
        self.sample_rate = audio_config.get("sample_rate", 16000)  # 采样率（从audio部分获取）
        self.speed_of_sound = audio_config.get("speed_of_sound", 343.0)  # 声速

        # FFT参数（从DOA_Oct_15移植）
        self.fft_length = int(self.sample_rate * self.frame_length_ms / 1000)
        self.hop_length = int(self.sample_rate * self.hop_length_ms / 1000)

        # Saving parameters
        self.save_original_audio = doa_section.get("save_original_audio", False)

        # Visualization parameters - check both at the top level (complete config) and under doa section
        # First, try to get visualization from config root (for when full config is passed)
        visualization_params = doa_section.get("visualization", {})

        self.realtime_plot_enabled = visualization_params.get("realtime_plot_enabled", False)
        self.global_heatmap_enabled = visualization_params.get("global_heatmap_enabled", False)
        self.global_doa_plot_enabled = visualization_params.get("global_doa_plot_enabled", False)

        print(f"DOA Config loaded successfully.")

def load_doa_config(config_path="config.yaml") -> DOAConfig:
    """
    Loads the DOA configuration from the main YAML file.
    """
    try:
        with open(config_path, 'r') as f:
            full_config = yaml.safe_load(f)
            # Pass full config to allow access to audio section etc.
            return DOAConfig(full_config)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{config_path}'. Using default DOA config.")
        return DOAConfig({})
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}. Using default DOA config.")
        return DOAConfig({})
