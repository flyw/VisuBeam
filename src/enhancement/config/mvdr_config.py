class MvdrConfig:
    def __init__(self, mvdr_data: dict):
        self.target_angle = mvdr_data.get("target_angle", 0.0)
        self.tolerance = mvdr_data.get("tolerance", 10.0)
        self.background_noise_alpha_slow = mvdr_data.get("background_noise_alpha_slow", 0.99)
        self.inertial_hold_frames = mvdr_data.get("inertial_hold_frames", 3)
