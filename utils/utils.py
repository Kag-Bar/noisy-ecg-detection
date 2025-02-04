import numpy as np

def compute_snr(signal):
    signal_power = np.mean(signal ** 2)
    noise_power = np.var(signal - np.mean(signal))
    return 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 0

def signal_to_image(signal):
    return signal.reshape(50, 60)