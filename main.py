import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import pygame
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import io
from module import (
    load_wav_file,
    extract_mfcc,
    extract_pitch,
    extract_rms_energy,
    extract_zero_crossing_rate,
    extract_spectral_centroid,
    extract_spectral_bandwidth
)

class StringInstrumentSystem:
    def __init__(self, data_path='./dataset'):
        self.data_path = data_path
        self.features_db = None
        self.scaler = StandardScaler()

    def extract_features(self, file_path):
        signal, sr = load_wav_file(file_path)
        n_fft = 2048
        hop_length = 512
        n_mfcc = 13
        mfccs = extract_mfcc(signal, sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfccs_mean = np.mean(mfccs, axis=0)
        pitch = extract_pitch(signal, sr, frame_size=n_fft, hop_length=hop_length)
        pitch_mean = np.mean(pitch[pitch > 0]) if len(pitch[pitch > 0]) > 0 else 0
        rms_mean = np.mean(extract_rms_energy(signal, frame_size=n_fft, hop_length=hop_length))
        zcr_mean = np.mean(extract_zero_crossing_rate(signal, frame_size=n_fft, hop_length=hop_length))
        sc_mean = np.mean(extract_spectral_centroid(signal, sr, n_fft=n_fft, hop_length=hop_length))
        sb_mean = np.mean(extract_spectral_bandwidth(signal, sr, n_fft=n_fft, hop_length=hop_length))
        return np.hstack([mfccs_mean, pitch_mean, rms_mean, zcr_mean, sc_mean, sb_mean])

    def load_database(self, db_path='features_database.pkl'):
        try:
            with open(db_path, 'rb') as f:
                data = pickle.load(f)
                self.features_db = data['features_db']
                self.scaler = data['scaler']
            return True
        except:
            return False

    def search(self, query_file_path, top_k=3):
        if self.features_db is None:
            return []
        try:
            query_features = self.extract_features(query_file_path)
            query_features_norm = self.scaler.transform(query_features.reshape(1, -1))
            similarities = []
            for i, row in self.features_db.iterrows():
                db_features = np.array(row['features']).reshape(1, -1)
                similarity = cosine_similarity(query_features_norm, db_features)[0, 0]
                similarities.append((i, similarity))
            similarities.sort(key=lambda x: x[1], reverse=True)
            top_indices = [idx for idx, _ in similarities[:top_k]]
            results = self.features_db.iloc[top_indices].copy()
            results['similarity'] = [sim for _, sim in similarities[:top_k]]
            return results
        except:
            return []

    def visualize_audio(self, file_path):
        signal, sr = load_wav_file(file_path)
        duration = len(signal) / sr
        time = np.linspace(0, duration, len(signal))
        pitch = extract_pitch(signal, sr)
        mfccs = extract_mfcc(signal, sr, n_mfcc=13, n_fft=2048, hop_length=512)
        fig = Figure(figsize=(7, 6))
        ax1 = fig.add_subplot(3, 1, 1)
        ax1.plot(time, signal)
        ax1.set_title("Dạng sóng")
        ax2 = fig.add_subplot(3, 1, 2)
        ax2.plot(pitch)
        ax2.set_title("Đường cong Pitch")
        ax3 = fig.add_subplot(3, 1, 3)
        ax3.set_title("MFCCs")
        cax = ax3.imshow(mfccs.T, aspect='auto', origin='lower', cmap='viridis')
        fig.colorbar(cax, ax=ax3)
        fig.tight_layout()
        return fig

    def play_audio(self, file_path):
        try:
            pygame.mixer.init()
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
        except:
            pass

class AudioSearchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hệ thống tìm kiếm âm thanh nhạc cụ dây")
        self.root.geometry("1000x700")
        self.system = StringInstrumentSystem()
        self.system.load_database('features_database.pkl')

        self.main_frame = ttk.Frame(root, padding=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Thiết lập grid co giãn
        self.main_frame.rowconfigure(1, weight=1)
        self.main_frame.columnconfigure(0, weight=1)

        self.top_frame = ttk.Frame(self.main_frame)
        self.top_frame.grid(row=0, column=0, sticky="ew")

        self.select_file_btn = ttk.Button(self.top_frame, text="Chọn file âm thanh", command=self.select_file)
        self.select_file_btn.pack(side=tk.LEFT, padx=5)
        self.search_btn = ttk.Button(self.top_frame, text="Tìm kiếm", command=self.search_audio)
        self.search_btn.pack(side=tk.LEFT, padx=5)
        self.file_label = ttk.Label(self.top_frame, text="Chưa chọn file")
        self.file_label.pack(side=tk.LEFT, padx=20)

        self.result_frame = ttk.Frame(self.main_frame)
        self.result_frame.grid(row=1, column=0, sticky="nsew")
        self.result_frame.columnconfigure(0, weight=1)
        self.result_frame.columnconfigure(1, weight=1)
        self.result_frame.rowconfigure(0, weight=1)

        self.query_frame = ttk.LabelFrame(self.result_frame, text="File truy vấn")
        self.query_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        self.query_frame.rowconfigure(0, weight=1)
        self.query_frame.columnconfigure(0, weight=1)

        self.query_canvas = tk.Canvas(self.query_frame)
        self.query_canvas.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.play_query_btn = ttk.Button(self.query_frame, text="Phát âm thanh", command=lambda: self.play_audio('query'))
        self.play_query_btn.grid(row=1, column=0, pady=5)

        self.results_display_frame = ttk.LabelFrame(self.result_frame, text="Kết quả tìm kiếm")
        self.results_display_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        self.top_results_frame = ttk.Frame(self.results_display_frame)
        self.top_results_frame.pack(fill=tk.BOTH, expand=True)

        self.status_var = tk.StringVar(value="Sẵn sàng")
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.current_query_file = None
        self.result_files = []

    def update_status(self, message):
        self.status_var.set(message)
        self.root.update_idletasks()

    def select_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if file_path:
            self.current_query_file = file_path
            self.file_label.config(text=os.path.basename(file_path))
            self.update_query_visualization(file_path)

    def update_query_visualization(self, file_path):
        fig = self.system.visualize_audio(file_path)
        buf = io.BytesIO()
        FigureCanvas(fig).print_png(buf)
        img = Image.open(buf)
        self.query_img = ImageTk.PhotoImage(img)
        self.query_canvas.create_image(0, 0, anchor=tk.NW, image=self.query_img)
        self.query_canvas.config(scrollregion=self.query_canvas.bbox(tk.ALL))

    def search_audio(self):
        if not self.current_query_file:
            self.update_status("Vui lòng chọn file âm thanh")
            return
        self.update_status("Đang tìm kiếm...")
        results = self.system.search(self.current_query_file)
        self.display_results(results)
        self.update_status("Hoàn tất")

    def display_results(self, results):
        for widget in self.top_results_frame.winfo_children():
            widget.destroy()
        self.result_files = []
        for i, (_, row) in enumerate(results.iterrows()):
            frame = ttk.LabelFrame(self.top_results_frame, text=f"#{i+1} - {row['instrument']} ({row['similarity']:.2f})")
            frame.pack(fill=tk.X, expand=True, pady=5)
            canvas = tk.Canvas(frame, height=100)
            canvas.pack(fill=tk.X, padx=5)
            y, sr = librosa.load(row['file_path'], sr=None)
            fig = Figure(figsize=(6, 1.5))
            ax = fig.add_subplot(1, 1, 1)
            ax.set_title("Dạng sóng")
            ax.plot(np.linspace(0, len(y)/sr, len(y)), y)
            ax.set_xlim(0, len(y)/sr)
            fig.tight_layout()
            buf = io.BytesIO()
            FigureCanvas(fig).print_png(buf)
            img = Image.open(buf)
            photo = ImageTk.PhotoImage(img)
            canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            canvas.image = photo
            btn = ttk.Button(frame, text="Phát âm thanh", command=lambda f=row['file_path']: self.system.play_audio(f))
            btn.pack()

    def play_audio(self, idx):
        if idx == 'query' and self.current_query_file:
            self.system.play_audio(self.current_query_file)

# Chạy ứng dụng
if __name__ == "__main__":
    root = tk.Tk()
    app = AudioSearchApp(root)
    root.mainloop()
