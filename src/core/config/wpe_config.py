class WpeConfig:
    def __init__(self, wpe_data: dict):
        self.enable = wpe_data.get("enable", False)
        self.save_output = wpe_data.get("save_output", False)
        self.taps = wpe_data.get("taps", 10)
        self.delay = wpe_data.get("delay", 3)
        self.alpha = wpe_data.get("alpha", 0.99)
        self.stft_size = wpe_data.get("stft_size", 1024)
        self.stft_shift = wpe_data.get("stft_shift", 512)
