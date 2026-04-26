#libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

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

#RMSE
rmse = np.sqrt(np.mean((ppg_norm - ppg_filtrada_norm)**2))

#plot signal
plt.figure(figsize=(12, 6))
plt.plot(t, ppg_norm, label="Original (Normalized 0-1)", color="gray", alpha=0.4)
plt.plot(t, ppg_filtrada_norm, label=f"Low-pass filtered ({fc} Hz)", color="black", linewidth=1.5)
plt.title(f"Filtered and normalized PPG signal - RMSE: {rmse:.4f}")
plt.legend(loc="upper right")
plt.xlabel("Time (s)")         
plt.ylabel("Normalized amplitude") 
plt.show()