import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib import patches

# Load the CSV file
file_path = "/Users/kaiden/Downloads/Archive/red_808quarter.csv"  # change path if needed
data = pd.read_csv(file_path)

# Clean up column names
data.columns = [col.strip() for col in data.columns]

# Extract polariser angles (index is the angle)
polariserAngles = data.index.astype(float)

# Identify y-data columns
y_columns = [col for col in data.columns if col.startswith("y_")]

# Define model function for ellipse fit
def model_f(theta, p1, p2, p3):
    theta_rad = np.deg2rad(theta)  # convert degrees to radians
    return (p1 * np.cos(theta_rad - p3))**2 + (p2 * np.sin(theta_rad - p3))**2

# Create the figure — make the right subplot bigger using gridspec
fig = plt.figure(figsize=(14, 6))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.3])
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax2.set_aspect('equal', 'box')

# --- 1️⃣ Polariser Angle vs Intensity plot ---
colors = plt.cm.tab20(np.linspace(0, 1, len(y_columns)))  # distinct colours

for i, col in enumerate(y_columns):
    powers = data[col].values
    
    # Fit the ellipse model to each dataset
    try:
        popt, pcov = curve_fit(
            model_f,
            polariserAngles,
            powers,
            bounds=([-5, -5, 0], [5, 5, np.pi])
        )
        Emax, Emin, alpha = popt
    except RuntimeError:
        print(f"Fit failed for {col}")
        continue

    fittingAngles = np.arange(0, 180, 1)
    colour = colors[i]

    # Plot raw data and fitted model on left subplot
    ax1.plot(polariserAngles, powers, 'o', markersize=3, color=colour, label=f'{col} data')
    ax1.plot(fittingAngles, model_f(fittingAngles, Emax, Emin, alpha), '--', color=colour, linewidth=1)

    # Draw corresponding ellipse on right subplot
    e = patches.Ellipse(
        (0, 0),
        width=Emax*1.5,
        height=Emin*1.5,
        angle=np.degrees(alpha),
        linewidth=1.8,
        fill=False,
        alpha=0.8,
        edgecolor=colour
    )
    ax2.add_patch(e)

# --- Styling for main graph ---
ax1.set_xlabel("Polariser Angle (deg)")
ax1.set_ylabel("Intensity (a.u.)")
ax1.set_title("Polariser Angle vs Intensity (All 18 Datasets)")
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(fontsize=7, loc='best')

# --- 2️⃣ Polarisation Ellipses plot ---
ax2.set_xlim([-1.3, 1.3])
ax2.set_ylim([-1.3, 1.3])
ax2.set_title("Polarisation Ellipses (All Fits)", fontsize=11)
ax2.axis('off')

plt.tight_layout()
plt.show()