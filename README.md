# PPG-signal-processing

Repository for photoplethysmographic (PPG) signal processing; the objective is to develop a robust analysis model for the detection of fiducial points by conditioning the signal through filtering and normalization, evaluating its quality using signal quality indices (SQI), and obtaining its derivatives—volume derivative (VPG) and acceleration derivative (APG)—to facilitate the accurate identification of morphological features of the signal.

##How to use:

1. Load the dataset: Place your PPG .csv file and update the path in the script.
2. Install dependencies: Ensure numpy, pandas, matplotlib, scipy are installed.
3. Set acquisition parameters: Verify sampling frequency (fs) based on your data.
4. Run the preprocessing: The script applies Butterworth filtering and normalization.
5. Execute SQI analysis: Signal quality is evaluated using sliding windows to remove noisy segments.
6. Compute derivatives: VPG and APG signals are obtained from the filtered PPG.
7. Detect fiducial points: Peaks and characteristic points (u, v, w, a, b, c, d, e) are automatically extracted.
8. Visualize results: The script plots PPG, VPG, and APG signals with detected points.
