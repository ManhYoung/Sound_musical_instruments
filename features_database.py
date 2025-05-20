# Tạo cơ sở dữ liệu đặc trưng features_database.pkl từ thư mục dataset
import os
import pickle
import numpy as np
import pandas as pd
from module import (
    load_wav_file,
    extract_mfcc,
    extract_pitch,
    extract_rms_energy,
    extract_zero_crossing_rate,
    extract_spectral_centroid,
    extract_spectral_bandwidth
)
from sklearn.preprocessing import StandardScaler

data_path = "./dataset" 
output_path = "features_database.pkl"

features_list = []
file_paths = []
labels = []

instrument_types = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]

for instrument in instrument_types:
    folder = os.path.join(data_path, instrument)
    for fname in os.listdir(folder):
        if fname.endswith(".wav"):
            fpath = os.path.join(folder, fname)
            try:
                signal, sr = load_wav_file(fpath)
                n_fft = 2048
                hop_length = 512
                n_mfcc = 13

                mfccs = extract_mfcc(signal, sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
                mfccs_mean = np.mean(mfccs, axis=0)

                pitch = extract_pitch(signal, sr, frame_size=n_fft, hop_length=hop_length)
                pitch_nonzero = pitch[pitch > 0]
                pitch_mean = np.mean(pitch_nonzero) if len(pitch_nonzero) > 0 else 0

                rms = extract_rms_energy(signal, frame_size=n_fft, hop_length=hop_length)
                zcr = extract_zero_crossing_rate(signal, frame_size=n_fft, hop_length=hop_length)
                sc = extract_spectral_centroid(signal, sr, n_fft=n_fft, hop_length=hop_length)
                sb = extract_spectral_bandwidth(signal, sr, n_fft=n_fft, hop_length=hop_length)

                feature_vec = np.hstack([
                    mfccs_mean,
                    np.mean(pitch),
                    np.mean(rms),
                    np.mean(zcr),
                    np.mean(sc),
                    np.mean(sb)
                ])
                features_list.append(feature_vec)
                file_paths.append(fpath)
                labels.append(instrument)
            except Exception as e:
                print(f"Lỗi với file {fpath}: {e}")

# Chuẩn hóa đặc trưng
scaler = StandardScaler()
features_array = scaler.fit_transform(np.array(features_list))

# Lưu CSDL
df = pd.DataFrame({
    'file_path': file_paths,
    'instrument': labels,
    'features': list(features_array)
})

with open(output_path, "wb") as f:
    pickle.dump({
        'features_db': df,
        'scaler': scaler
    }, f)

output_path
