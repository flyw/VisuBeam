class McraConfig:
    def __init__(self, mcra_data: dict):
        self.stft_size = mcra_data.get("stft_size", 1024)
        self.stft_shift = mcra_data.get("stft_shift", 512)
        self.alpha_s = mcra_data.get("alpha_s", 0.8)
        self.alpha_d = mcra_data.get("alpha_d", 0.95)
        self.l_window = mcra_data.get("l_window", 15)
        self.gamma = mcra_data.get("gamma", 1.67)
        self.delta = mcra_data.get("delta", 5.0)
        self.gain_floor = mcra_data.get("gain_floor", 0.4)
        self.gain_exponent = mcra_data.get("gain_exponent", 0.6)
