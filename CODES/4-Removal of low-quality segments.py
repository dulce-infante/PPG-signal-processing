#libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks, welch
from scipy.stats import skew, kurtosis

#load data
df = pd.read_csv("Data PPG.csv")
y_col = "PPG"
ppg = df[y_col].dropna().values  #remove NaNs

#acquisition parameters
N = len(ppg)
tiempo_seg = 122
fs = N / tiempo_seg
print(f"Sampling frequency: {fs:.2f} Hz")

#create time data
t = np.linspace(0, tiempo_seg, N)

#Butterworth filter
fc = 10         
orden = 2
nyq = fs / 2
fc_norm = fc / nyq
b, a = butter(orden, fc_norm, btype="low")
ppg_filtrada = filtfilt(b, a, ppg)

#normalization
def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))
ppg_norm = normalize(ppg)
ppg_filtrada_norm = normalize(ppg_filtrada)
ppg_filtrada_norm = normalize(ppg_filtrada)

#SQI metrics function
def get_sqi_metrics(segment, fs):
    metrics = {}
    #energy in cardiac band (0.5-4Hz)
    f, Pxx = welch(segment, fs, nperseg=len(segment))
    cardiac_band = np.sum(Pxx[(f >= 0.5) & (f <= 4)])
    total_pwr = np.sum(Pxx)
    metrics['pwr_ratio'] = cardiac_band / total_pwr if total_pwr > 0 else 0
    #number of peaks
    peaks, _ = find_peaks(segment, distance=fs*0.4, prominence=np.std(segment)*0.5)
    metrics['num_peaks'] = len(peaks)
    
    if len(peaks) > 2:
        intervals = np.diff(peaks)
        #RR variability
        metrics['rr_variability'] = np.std(intervals) / np.mean(intervals)
        peak_amps = segment[peaks]
        #amplitude variability
        metrics['amp_variability'] = np.std(peak_amps) / np.mean(peak_amps)
    else:
        metrics['rr_variability'] = 1.0
        metrics['amp_variability'] = 1.0
    #skewness
    metrics['skew'] = skew(segment)
    return metrics

#signal segmentation
WINDOW_SEC = 5       
STEP_SEC = 1         
window_size = int(WINDOW_SEC * fs)
step_size = int(STEP_SEC * fs)

votos_positivos = np.zeros(N)
ventanas_totales = np.zeros(N)

#analysis and quality classification by windows
for start in range(0, N - window_size, step_size):
    end = start + window_size
    segment = ppg_filtrada[start:end]
    m = get_sqi_metrics(segment, fs)
    
    score = 0
    if np.std(segment) > 0.001 and m['num_peaks'] >= 2:
        if m['rr_variability'] < 0.18: score += 1 
        if m['amp_variability'] < 0.25: score += 1
        if m['pwr_ratio'] > 0.45: score += 1
        if 0 < m['skew'] < 2.0: score += 1

    #evaluation record
    ventanas_totales[start:end] += 1
    if score >= 3:
        votos_positivos[start:end] += 1

#strict mask
umbral_consenso = 0.8 
mask_buena = np.divide(votos_positivos, ventanas_totales, 
                       out=np.zeros_like(votos_positivos), 
                       where=ventanas_totales!=0) >= umbral_consenso

ppg_curada = ppg_filtrada_norm.copy()
ppg_curada[~mask_buena] = np.nan

#signal summary
porcentaje_conservado = (np.sum(mask_buena) / N) * 100
print(f"Accepted signal percentage: {porcentaje_conservado:.2f}%")

#plot signal
plt.figure(figsize=(15, 6))
plt.plot(t, ppg_filtrada_norm, color='lightgrey', linewidth=1, label="Removed segments", alpha=0.6)
plt.plot(t, ppg_curada, color='green', linewidth=1.2, label="Good quality signal")
plt.title("Signal without low-quality segments")
plt.xlabel("Time (s)")
plt.ylabel("Normalized amplitude")
plt.xlim(0, tiempo_seg)
plt.grid(True, alpha=0.2)
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()