import cv2
import numpy as np
# import matplotlib.pyplot as plt
from scipy.signal import resample


def preprocess_ecg_image(image_path):
    """Load and preprocess the ECG image."""
    img = cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (1000, 800))
    img = cv2.GaussianBlur(img, (5, 5), 0)
    _, img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
    return img


def extract_leads(img):
    """Extract the 12 lead waveforms from the image."""
    lead_height = img.shape[0] // 12
    leads = [img[i*lead_height:(i+1)*lead_height, :] for i in range(12)]
    return leads


def convert_to_signal(lead_img):
    """Convert a single lead image into a 1D waveform."""
    height, width = lead_img.shape
    signal = np.zeros(width)
    for x in range(width):
        y_values = np.where(lead_img[:, x] == 255)[0]
        if len(y_values) > 0:
            signal[x] = np.mean(y_values)

    signal = (height - signal) / height
    return signal


def process_ecg_image(image_path):
    """Complete pipeline: image to numpy array (12, 5000)."""
    img = preprocess_ecg_image(image_path)
    leads = extract_leads(img)

    signals = [convert_to_signal(lead) for lead in leads]
    signals_resampled = [resample(sig, 5000) for sig in signals]
    return np.array(signals_resampled)


image_path = "ecg_sample.png"
ecg_array = process_ecg_image(image_path)
np.save("ecg_output.npy", ecg_array)
