import numpy as np
import math
from scipy.fft import fft, ifft
import scipy.signal as signal
import wave
import struct

def load_wav_file(file_path):
    """
    Đọc file âm thanh định dạng WAV
    
    Parameters:
    -----------
    file_path : str
        Đường dẫn đến file âm thanh
        
    Returns:
    --------
    y : np.ndarray
        Dữ liệu âm thanh đã chuẩn hóa về giá trị từ -1 đến 1
    sr : int
        Tần số lấy mẫu (sample rate)
    """
    with wave.open(file_path, 'rb') as wav_file:
        # Lấy thông tin file
        n_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        
        # Đọc dữ liệu raw từ file
        raw_data = wav_file.readframes(n_frames)
        
        # Chuyển đổi raw data thành mảng numpy
        if sample_width == 1:  # 8-bit, unsigned
            data = np.array(struct.unpack(f"{n_frames * n_channels}B", raw_data), dtype=np.uint8)
            # Chuyển đổi từ uint8 về dạng -1 đến 1
            data = (data.astype(np.float32) - 128) / 128
        elif sample_width == 2:  # 16-bit, signed
            data = np.array(struct.unpack(f"{n_frames * n_channels}h", raw_data), dtype=np.int16)
            # Chuyển đổi từ int16 về dạng -1 đến 1
            data = data.astype(np.float32) / 32768
        elif sample_width == 4:  # 32-bit, signed
            data = np.array(struct.unpack(f"{n_frames * n_channels}i", raw_data), dtype=np.int32)
            # Chuyển đổi từ int32 về dạng -1 đến 1
            data = data.astype(np.float32) / 2147483648
        else:
            raise ValueError(f"Không hỗ trợ sample width {sample_width}")
        
        # Nếu là stereo, chuyển về mono bằng cách lấy trung bình các kênh
        if n_channels > 1:
            data = data.reshape(-1, n_channels)
            data = np.mean(data, axis=1)
    
    return data, sample_rate

def frame_signal(signal, frame_size, hop_size):
    """
    Chia tín hiệu thành các khung (frames)
    
    Parameters:
    -----------
    signal : np.ndarray
        Tín hiệu âm thanh
    frame_size : int
        Kích thước mỗi khung (số mẫu)
    hop_size : int
        Khoảng cách giữa các khung liên tiếp (số mẫu)
        
    Returns:
    --------
    frames : np.ndarray
        Mảng 2D chứa các khung, mỗi hàng là một khung
    """
    n_samples = len(signal)
    n_frames = 1 + int(np.floor((n_samples - frame_size) / hop_size))
    
    frames = np.zeros((n_frames, frame_size))
    
    for i in range(n_frames):
        start = i * hop_size
        end = start + frame_size
        frames[i] = signal[start:end]
    
    return frames

def apply_window(frames, window_type='hann'):
    """
    Áp dụng hàm cửa sổ cho các khung
    
    Parameters:
    -----------
    frames : np.ndarray
        Mảng 2D chứa các khung, mỗi hàng là một khung
    window_type : str
        Loại cửa sổ ('hann', 'hamming', 'blackman')
        
    Returns:
    --------
    windowed_frames : np.ndarray
        Mảng các khung đã áp dụng cửa sổ
    """
    frame_size = frames.shape[1]
    
    if window_type == 'hann':
        window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(frame_size) / (frame_size - 1))
    elif window_type == 'hamming':
        window = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(frame_size) / (frame_size - 1))
    elif window_type == 'blackman':
        window = 0.42 - 0.5 * np.cos(2 * np.pi * np.arange(frame_size) / (frame_size - 1)) + 0.08 * np.cos(4 * np.pi * np.arange(frame_size) / (frame_size - 1))
    else:
        window = np.ones(frame_size)  # rectangular window
    
    # Áp dụng cửa sổ cho từng khung
    return frames * window

def compute_stft(frames):
    """
    Tính toán Short-Time Fourier Transform
    
    Parameters:
    -----------
    frames : np.ndarray
        Mảng 2D chứa các khung đã áp dụng cửa sổ
        
    Returns:
    --------
    stft : np.ndarray
        Mảng 2D chứa phổ tần số của các khung
    """
    n_frames = frames.shape[0]
    n_fft = frames.shape[1]
    
    # Tính FFT cho từng khung
    stft = np.zeros((n_frames, n_fft // 2 + 1), dtype=complex)
    
    for i in range(n_frames):
        fft_result = fft(frames[i])
        # Chỉ lấy nửa đầu (phần dương của phổ tần số)
        stft[i] = fft_result[:n_fft // 2 + 1]
    
    return stft

def compute_power_spectrogram(stft):
    """
    Tính toán Power Spectrogram từ STFT
    
    Parameters:
    -----------
    stft : np.ndarray
        Mảng 2D chứa phổ tần số của các khung
        
    Returns:
    --------
    power_spec : np.ndarray
        Mảng 2D chứa phổ công suất
    """
    return np.abs(stft) ** 2

def compute_mel_filterbank(n_filters, n_fft, sample_rate, fmin=0, fmax=None):
    """
    Tạo bộ lọc Mel
    
    Parameters:
    -----------
    n_filters : int
        Số lượng bộ lọc Mel
    n_fft : int
        Kích thước FFT
    sample_rate : int
        Tần số lấy mẫu
    fmin : float
        Tần số thấp nhất (Hz)
    fmax : float
        Tần số cao nhất (Hz)
        
    Returns:
    --------
    filterbank : np.ndarray
        Mảng 2D chứa bộ lọc Mel
    """
    if fmax is None:
        fmax = sample_rate / 2
    
    # Chuyển đổi Hz sang Mel
    def hz_to_mel(f):
        return 2595 * np.log10(1 + f / 700)
    
    # Chuyển đổi Mel sang Hz
    def mel_to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)
    
    # Tính toán các điểm Mel
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_filters + 2)
    
    # Chuyển đổi về Hz
    freq_points = mel_to_hz(mel_points)
    
    # Chuyển đổi Hz sang FFT bins
    fft_bins = np.floor((n_fft + 1) * freq_points / sample_rate).astype(int)
    
    # Tạo bộ lọc Mel
    filterbank = np.zeros((n_filters, n_fft // 2 + 1))
    
    for i in range(n_filters):
        # Tính tam giác cho mỗi bộ lọc
        for j in range(fft_bins[i], fft_bins[i + 1]):
            filterbank[i, j] = (j - fft_bins[i]) / (fft_bins[i + 1] - fft_bins[i])
        
        for j in range(fft_bins[i + 1], fft_bins[i + 2]):
            filterbank[i, j] = (fft_bins[i + 2] - j) / (fft_bins[i + 2] - fft_bins[i + 1])
    
    return filterbank

def compute_mel_spectrogram(power_spec, mel_filterbank):
    """
    Tính toán Mel Spectrogram
    
    Parameters:
    -----------
    power_spec : np.ndarray
        Mảng 2D chứa phổ công suất
    mel_filterbank : np.ndarray
        Bộ lọc Mel
        
    Returns:
    --------
    mel_spec : np.ndarray
        Mảng 2D chứa phổ mel
    """
    return np.dot(power_spec, mel_filterbank.T)

def compute_dct(mel_spec):
    """
    Tính toán Discrete Cosine Transform
    
    Parameters:
    -----------
    mel_spec : np.ndarray
        Mảng 2D chứa phổ mel
        
    Returns:
    --------
    dct_result : np.ndarray
        Kết quả của phép biến đổi DCT
    """
    n_frames, n_mels = mel_spec.shape
    dct_result = np.zeros_like(mel_spec)
    
    for i in range(n_frames):
        # Lấy logarithm của mel spectrogram
        log_mel_spec = np.log(mel_spec[i] + 1e-10)  # Add small value to avoid log(0)
        
        # Tính DCT
        for j in range(n_mels):
            sum_val = 0
            for k in range(n_mels):
                sum_val += log_mel_spec[k] * np.cos(np.pi * j * (k + 0.5) / n_mels)
            dct_result[i, j] = sum_val
    
    return dct_result

def extract_mfcc(signal, sample_rate, n_mfcc=13, n_fft=2048, hop_length=512, n_mels=40):
    """
    Trích xuất MFCC từ tín hiệu âm thanh
    
    Parameters:
    -----------
    signal : np.ndarray
        Tín hiệu âm thanh
    sample_rate : int
        Tần số lấy mẫu
    n_mfcc : int
        Số lượng hệ số MFCC muốn trích xuất
    n_fft : int
        Kích thước khung FFT
    hop_length : int
        Khoảng cách giữa các khung liên tiếp
    n_mels : int
        Số lượng bộ lọc Mel
        
    Returns:
    --------
    mfccs : np.ndarray
        Mảng chứa các hệ số MFCC
    """
    # Chia tín hiệu thành các khung
    frames = frame_signal(signal, n_fft, hop_length)
    
    # Áp dụng hàm cửa sổ
    windowed_frames = apply_window(frames)
    
    # Tính STFT
    stft = compute_stft(windowed_frames)
    
    # Tính Power Spectrogram
    power_spec = compute_power_spectrogram(stft)
    
    # Tạo bộ lọc Mel
    mel_filterbank = compute_mel_filterbank(n_mels, n_fft, sample_rate)
    
    # Tính Mel Spectrogram
    mel_spec = compute_mel_spectrogram(power_spec, mel_filterbank)
    
    # Tính DCT
    dct_result = compute_dct(mel_spec)
    
    # Lấy n_mfcc hệ số đầu tiên
    mfccs = dct_result[:, :n_mfcc]
    
    return mfccs

def extract_spectral_centroid(signal, sample_rate, n_fft=2048, hop_length=512):
    """
    Trích xuất Spectral Centroid từ tín hiệu âm thanh
    
    Parameters:
    -----------
    signal : np.ndarray
        Tín hiệu âm thanh
    sample_rate : int
        Tần số lấy mẫu
    n_fft : int
        Kích thước khung FFT
    hop_length : int
        Khoảng cách giữa các khung liên tiếp
        
    Returns:
    --------
    centroid : np.ndarray
        Mảng chứa giá trị Spectral Centroid
    """
    # Chia tín hiệu thành các khung
    frames = frame_signal(signal, n_fft, hop_length)
    
    # Áp dụng hàm cửa sổ
    windowed_frames = apply_window(frames)
    
    # Tính STFT
    stft = compute_stft(windowed_frames)
    
    # Tính Power Spectrogram
    power_spec = compute_power_spectrogram(stft)
    
    # Tính tần số trung tâm
    freqs = np.fft.rfftfreq(n_fft, d=1/sample_rate)
    
    # Tính Spectral Centroid
    centroids = np.sum(power_spec * freqs, axis=1) / (np.sum(power_spec, axis=1) + 1e-10)
    
    return centroids

def extract_spectral_bandwidth(signal, sample_rate, n_fft=2048, hop_length=512):
    """
    Trích xuất Spectral Bandwidth từ tín hiệu âm thanh
    
    Parameters:
    -----------
    signal : np.ndarray
        Tín hiệu âm thanh
    sample_rate : int
        Tần số lấy mẫu
    n_fft : int
        Kích thước khung FFT
    hop_length : int
        Khoảng cách giữa các khung liên tiếp
        
    Returns:
    --------
    bandwidth : np.ndarray
        Mảng chứa giá trị Spectral Bandwidth
    """
    # Chia tín hiệu thành các khung
    frames = frame_signal(signal, n_fft, hop_length)
    
    # Áp dụng hàm cửa sổ
    windowed_frames = apply_window(frames)
    
    # Tính STFT
    stft = compute_stft(windowed_frames)
    
    # Tính Power Spectrogram
    power_spec = compute_power_spectrogram(stft)
    
    # Tính tần số 
    freqs = np.fft.rfftfreq(n_fft, d=1/sample_rate)
    
    # Tính Spectral Centroid
    centroids = np.sum(power_spec * freqs, axis=1) / (np.sum(power_spec, axis=1) + 1e-10)
    
    # Tính Spectral Bandwidth (spread)
    sq_diff = (freqs - centroids[:, np.newaxis]) ** 2
    bandwidth = np.sqrt(np.sum(sq_diff * power_spec, axis=1) / (np.sum(power_spec, axis=1) + 1e-10))
    
    return bandwidth

def extract_zero_crossing_rate(signal, frame_size=2048, hop_length=512):
    """
    Trích xuất Zero Crossing Rate từ tín hiệu âm thanh
    
    Parameters:
    -----------
    signal : np.ndarray
        Tín hiệu âm thanh
    frame_size : int
        Kích thước khung
    hop_length : int
        Khoảng cách giữa các khung liên tiếp
        
    Returns:
    --------
    zcr : np.ndarray
        Mảng chứa giá trị Zero Crossing Rate
    """
    # Chia tín hiệu thành các khung
    frames = frame_signal(signal, frame_size, hop_length)
    
    # Tính Zero Crossing Rate
    zcr = np.zeros(frames.shape[0])
    
    for i in range(frames.shape[0]):
        frame = frames[i]
        # Đếm số lần tín hiệu đổi dấu
        signs = np.sign(frame[:-1]) != np.sign(frame[1:])
        zcr[i] = np.sum(signs) / (frame_size - 1)
    
    return zcr

def extract_rms_energy(signal, frame_size=2048, hop_length=512):
    """
    Trích xuất RMS Energy từ tín hiệu âm thanh
    
    Parameters:
    -----------
    signal : np.ndarray
        Tín hiệu âm thanh
    frame_size : int
        Kích thước khung
    hop_length : int
        Khoảng cách giữa các khung liên tiếp
        
    Returns:
    --------
    rms : np.ndarray
        Mảng chứa giá trị RMS Energy
    """
    # Chia tín hiệu thành các khung
    frames = frame_signal(signal, frame_size, hop_length)
    
    # Tính RMS Energy
    rms = np.sqrt(np.mean(frames**2, axis=1))
    
    return rms

def extract_pitch(signal, sample_rate, frame_size=2048, hop_length=512, fmin=50, fmax=2000):
    """
    Trích xuất Pitch (F0) từ tín hiệu âm thanh sử dụng phương pháp tự tương quan
    
    Parameters:
    -----------
    signal : np.ndarray
        Tín hiệu âm thanh
    sample_rate : int
        Tần số lấy mẫu
    frame_size : int
        Kích thước khung
    hop_length : int
        Khoảng cách giữa các khung liên tiếp
    fmin : float
        Tần số thấp nhất (Hz)
    fmax : float
        Tần số cao nhất (Hz)
        
    Returns:
    --------
    pitches : np.ndarray
        Mảng chứa giá trị Pitch (F0)
    """
    # Chia tín hiệu thành các khung
    frames = frame_signal(signal, frame_size, hop_length)
    
    # Áp dụng hàm cửa sổ
    windowed_frames = apply_window(frames)
    
    # Chuyển đổi tần số thành độ trễ (lag)
    min_lag = int(sample_rate / fmax)
    max_lag = int(sample_rate / fmin)
    
    # Đảm bảo max_lag không vượt quá kích thước khung
    max_lag = min(max_lag, frame_size - 1)
    
    pitches = np.zeros(frames.shape[0])
    
    for i in range(frames.shape[0]):
        frame = windowed_frames[i]
        
        # Tính tự tương quan
        autocorr = np.correlate(frame, frame, mode='full')
        autocorr = autocorr[frame_size-1:]  # Chỉ lấy phần dương
        
        # Tìm đỉnh trong khoảng min_lag đến max_lag
        if min_lag >= len(autocorr) or max_lag >= len(autocorr):
            pitches[i] = 0
            continue
            
        candidate_peaks = autocorr[min_lag:max_lag]
        peak_idx = np.argmax(candidate_peaks) + min_lag
        
        # Kiểm tra đỉnh có đủ mạnh không
        if autocorr[peak_idx] > 0.1 * autocorr[0]:
            pitches[i] = sample_rate / peak_idx
        else:
            pitches[i] = 0
    
    return pitches
