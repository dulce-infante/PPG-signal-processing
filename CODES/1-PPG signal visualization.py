#libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

#plot the signal
fig, ax1 = plt.subplots(1, 1, figsize=(12, 5))
ax1.plot(t, ppg, color="black", linewidth=0.8)
ax1.set_title(f"PPG Signal")
ax1.set_ylabel("Amplitude")
ax1.set_xlabel("Time (s)")
ax1.grid(True, alpha=0.3)
plt.show()
