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
tiempo_seg = 60
fs = N / tiempo_seg

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

#signal segmentation
WINDOW_SEC = 5       
STEP_SEC = 1         
window_size = int(WINDOW_SEC * fs)
step_size = int(STEP_SEC * fs)

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

#analysis and quality classification by windows
results = []
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

    if score >= 3:
        calidad_visual = "Good"
    else:
        calidad_visual = "Not Acceptable"
    
    results.append({
        "Start_s": start / fs,
        "End_s": end / fs,
        "Quality": calidad_visual
    })

df_results = pd.DataFrame(results)
print(df_results.head(N))

#plot signal
plt.figure(figsize=(15, 6))
plt.plot(t, ppg_filtrada_norm, color='black', alpha=0.9, linewidth=1)
color_map = {"Good": "green", "Not Acceptable": "red"}

for _, row in df_results.iterrows():
    plt.axvspan(row['Start_s'], row['End_s'], color=color_map[row['Quality']], alpha=0.2)

plt.title("PPG Quality Analysis (Green: Good | Red: Not Acceptable)")
plt.xlabel("Time (s)")
plt.ylabel("Normalized amplitude")
plt.show()
