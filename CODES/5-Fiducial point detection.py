#libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks, welch, savgol_filter
from scipy.stats import skew, kurtosis

#load data
df = pd.read_csv("Data PPG.csv")
y_col = "PPG"
ppg = df[y_col].dropna().values  #remove NaNs

#acquisition parameters
N = len(ppg)
tiempo_seg = 122
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

ppg_filtrada_norm = normalize(ppg_filtrada)

#SQI metrics function
def get_sqi_metrics(segment, fs):
    metrics = {}
    f, Pxx = welch(segment, fs, nperseg=len(segment))
    cardiac_band = np.sum(Pxx[(f >= 0.5) & (f <= 4)])
    total_pwr = np.sum(Pxx)
    metrics['pwr_ratio'] = cardiac_band / total_pwr if total_pwr > 0 else 0
    try:
        p_sqi, _ = find_peaks(segment, distance=int(0.4*fs), prominence=np.std(segment)*0.5)
        metrics['num_peaks'] = len(p_sqi)
        if len(p_sqi) > 2:
            intervals = np.diff(p_sqi)
            metrics['rr_var'] = np.std(intervals) / np.mean(intervals)
            peak_amps = segment[p_sqi]
            metrics['amp_var'] = np.std(peak_amps) / np.mean(peak_amps)
        else:
            metrics['rr_var'], metrics['amp_var'] = 1.0, 1.0
    except:
        metrics['num_peaks'], metrics['rr_var'], metrics['amp_var'] = 0, 1.0, 1.0
    metrics['skew'] = skew(segment)
    return metrics

#signal segmentation
WINDOW_SEC, STEP_SEC = 5, 1
window_size, step_size = int(WINDOW_SEC * fs), int(STEP_SEC * fs)
mask_buena = np.zeros(N, dtype=bool)

for start in range(0, N - window_size, step_size):
    end = start + window_size
    seg_sqi = ppg_filtrada[start:end]
    m = get_sqi_metrics(seg_sqi, fs)
    score = 0
    if m['num_peaks'] >= 2:
        if m['pwr_ratio'] > 0.50: score += 1
        if m['rr_var'] < 0.15: score += 1
        if m['amp_var'] < 0.20: score += 1
        if 0 < m['skew'] < 2.0: score += 1
    if score >= 3:
        mask_buena[start:end] = True

ppg_curada = ppg_filtrada_norm.copy()
ppg_curada[~mask_buena] = np.nan

#derivatives and point localization
ppg_smooth = savgol_filter(ppg_filtrada_norm, 11, 3)
vpg = np.gradient(ppg_smooth, 1/fs)
apg = np.gradient(vpg, 1/fs)
vpg_smooth = savgol_filter(vpg, 11, 3)
apg_smooth = savgol_filter(apg, 11, 3)

peaks_locs, _ = find_peaks(ppg_curada, distance=int(0.4*fs), prominence=0.05)
vpg_max, _ = find_peaks(vpg_smooth, distance=int(0.05*fs))
vpg_min, _ = find_peaks(-vpg_smooth, distance=int(0.05*fs))
apg_max, _ = find_peaks(apg_smooth, distance=int(0.05*fs))
apg_min, _ = find_peaks(-apg_smooth, distance=int(0.05*fs))

fid_vpg = {"u": [], "v": [], "w": []}
fid_apg = {"a": [], "b": [], "c": [], "d": [], "e": []}

for peak in peaks_locs:
    u = vpg_max[vpg_max < peak][-1] if np.any(vpg_max < peak) else None
    v = vpg_min[vpg_min > peak][0] if np.any(vpg_min > peak) else None
    w = vpg_max[vpg_max > v][0] if v is not None and np.any(vpg_max > v) else None
    if u is not None: 
        fid_vpg["u"].append(u); fid_vpg["v"].append(v); fid_vpg["w"].append(w)

    a = apg_max[apg_max < peak][-1] if np.any(apg_max < peak) else None
    b = apg_min[apg_min > a][0] if a is not None and np.any(apg_min > a) else None
    c = apg_max[apg_max > b][0] if b is not None and np.any(apg_max > b) else None
    d = apg_min[apg_min > c][0] if c is not None and np.any(apg_min > c) else None
    e = apg_max[apg_max > d][0] if d is not None and np.any(apg_max > d) else (apg_max[apg_max > a][0] if a is not None else None)
    if a is not None: 
        fid_apg["a"].append(a); fid_apg["b"].append(b); fid_apg["c"].append(c); fid_apg["d"].append(d); fid_apg["e"].append(e)

#plot signals
plt.figure(figsize=(15, 12))
t_z1, t_z2 = 6, 16 

#PPG
plt.subplot(3, 1, 1)
plt.plot(t, ppg_filtrada_norm, color='lightgrey', alpha=0.5)
plt.plot(t, ppg_curada, color='black')
plt.scatter(t[peaks_locs], ppg_curada[peaks_locs], color='black', s=20, zorder=3)
plt.title("Systolic peaks detected in PPG")
plt.ylabel("PPG")
plt.xlim(t_z1, t_z2)
plt.legend()

#VPG
plt.subplot(3, 1, 2)
plt.plot(t, vpg_smooth, 'k-', alpha=0.3)
v_colors = {'u': 'blue', 'v': 'orange', 'w': 'green'}
for k, idxs in fid_vpg.items():
    clean_idxs = [i for i in idxs if i is not None]
    plt.plot(t[clean_idxs], vpg_smooth[clean_idxs], 'o', markersize=5, label=f"Point {k}", color=v_colors[k])
    for i in clean_idxs:
        if t_z1 <= t[i] <= t_z2:
            plt.text(t[i], vpg_smooth[i] + 0.05, f"{vpg_smooth[i]:.2f}", color=v_colors[k], fontsize=8, ha='center')
plt.title("Points u, v and w detected in VPG signal")
plt.ylabel("VPG")
plt.xlim(t_z1, t_z2)
plt.legend()

#APG
plt.subplot(3, 1, 3)
plt.plot(t, apg_smooth, 'k-', alpha=0.3)
a_colors = {'a': 'red', 'b': 'blue', 'c': 'green', 'd': 'purple', 'e': 'brown'}
for k, idxs in fid_apg.items():
    clean_idxs = [i for i in idxs if i is not None]
    plt.plot(t[clean_idxs], apg_smooth[clean_idxs], 's', markersize=4, label=f"Point {k}", color=a_colors[k])
    for i in clean_idxs:
        if t_z1 <= t[i] <= t_z2:
            v_offset = 0.2 if k in ['a', 'c', 'e'] else -0.4
            plt.text(t[i], apg_smooth[i] + v_offset, f"{apg_smooth[i]:.2f}", color=a_colors[k], fontsize=8, ha='center')
plt.title("Points a, b, c, d and e detected in APG signal")
plt.xlabel("Time [s]")
plt.ylabel("APG")
plt.xlim(t_z1, t_z2)
plt.legend(ncol=5)
plt.tight_layout()
plt.show()