import os
import numpy as np
import pandas as pd
import pyedflib
from scipy.signal import stft
from tqdm import tqdm

print("Script started")

# Parameters
WINDOW_SIZE = 12  # seconds
STFT_WIN = 1      # seconds
STFT_OVERLAP = 0.5
FS = 256  # Typical neonatal EEG sampling rate; will be read from file

# Noise bands to remove (50Hz and harmonics)
NOISE_BANDS = [(47, 53), (97, 103), (57, 63), (117, 123)]

# Paths
ANNOTATION_FILE = 'annotations_2017_A.csv'
CLINICAL_FILE = 'clinical_information.csv'
EEG_DIR = '.'
PREPROCESSED_DIR = 'preprocessed'
os.makedirs(PREPROCESSED_DIR, exist_ok=True)

# Load clinical info for mapping
clinical_df = pd.read_csv(CLINICAL_FILE)

# Load annotations (rows: time windows, cols: patients)
ann = pd.read_csv(ANNOTATION_FILE, header=None)
ann = ann.values  # shape: (n_windows, n_patients)

# Map patient index to EEG file
patient_files = clinical_df['EEG file'].tolist()
print(f"Found {len(patient_files)} patient files: {patient_files}")

for idx, eeg_file in enumerate(patient_files):
    print(f"\nProcessing patient {idx+1}/{len(patient_files)}: {eeg_file}")
    if pd.isna(eeg_file):
        print("  Skipping: EEG file is NaN")
        continue
    eeg_path = os.path.join(EEG_DIR, eeg_file + '.edf')
    if not os.path.exists(eeg_path):
        print(f"  Skipping: Missing file {eeg_path}")
        continue
    try:
        print(f"  Attempting to load {eeg_path} with pyedflib...")
        f = pyedflib.EdfReader(eeg_path)
        n_channels = f.signals_in_file
        fs = int(f.getSampleFrequency(0))
        print(f"  Number of channels: {n_channels}, Sampling frequency: {fs}")
        n_samples = min([f.getNSamples()[ch] for ch in range(n_channels)])
        data = np.zeros((n_channels, n_samples))
        for ch in range(n_channels):
            data[ch, :] = f.readSignal(ch)[:n_samples]
        f.close()
        print(f"  Data shape: {data.shape}")
    except Exception as e:
        print(f"  Error reading {eeg_path}: {e}")
        continue
    # Remove DC
    data = data - np.mean(data, axis=1, keepdims=True)
    print("  DC removed.")
    # Segment into 12s windows
    win_samples = int(WINDOW_SIZE * fs)
    n_windows = n_samples // win_samples
    print(f"  Segmenting into {n_windows} windows of {win_samples} samples each")
    X = []
    for w in range(n_windows):
        seg = data[:, w*win_samples:(w+1)*win_samples]
        f_stft, t, Zxx = stft(seg, fs=fs, nperseg=int(STFT_WIN*fs), noverlap=int(STFT_WIN*fs*STFT_OVERLAP), axis=-1)
        freq_mask = np.ones_like(f_stft, dtype=bool)
        freq_mask &= (f_stft > 0)
        for band in NOISE_BANDS:
            freq_mask &= ~((f_stft >= band[0]) & (f_stft <= band[1]))
        Zxx = Zxx[:, freq_mask, :]
        X.append(np.abs(Zxx))
    if len(X) == 0:
        print("  No windows extracted, skipping save.")
        continue
    X = np.stack(X)
    y = ann[:n_windows, idx] if idx < ann.shape[1] else np.zeros(n_windows)
    print(f"  Saving {X.shape[0]} windows to {os.path.join(PREPROCESSED_DIR, f'{eeg_file}_data.npz')}")
    np.savez_compressed(os.path.join(PREPROCESSED_DIR, f'{eeg_file}_data.npz'), X=X, y=y)
print("Preprocessing complete.") 