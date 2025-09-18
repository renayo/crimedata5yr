import pandas as pd
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# Load the dataset
try:
    data = pd.read_pickle('full_astro_crime_data.pkl')
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: 'full_astro_crime_data.pkl' not found. Make sure the file is in the correct directory.")
    exit()

# --- Generating Fuller Autocorrelation Plots ---
print("\n--- Generating Extended Autocorrelation Plots ---")

# Set the number of lags to display
number_of_lags = 24 * 31 # Approximately one month of hourly data
days = number_of_lags // 24

# Create figure and axes, stacked vertically for better readability
fig, axes = plt.subplots(2, 1, figsize=(16, 12)) # 2 rows, 1 column

# --- Autocorrelation Function (ACF) ---
plot_acf(data['sum_severity'], ax=axes[0], lags=number_of_lags)
axes[0].set_title(f'Autocorrelation (ACF) for Lags up to {number_of_lags} Hours (i.e., {days} days)')
axes[0].set_xlabel('Lag (Hours)')
axes[0].set_ylabel('Autocorrelation')
axes[0].grid(True, linestyle='--', alpha=0.6)

# --- Partial Autocorrelation Function (PACF) ---
# PACF shows the direct relationship between an observation and its lag.
# It removes the effects of the intervening lags, unlike ACF.
plot_pacf(data['sum_severity'], ax=axes[1], lags=number_of_lags)
axes[1].set_title(f'Partial Autocorrelation (PACF) for Lags up to {number_of_lags} Hours (i.e., {days} days)')
axes[1].set_xlabel('Lag (Hours)')
axes[1].set_ylabel('Partial Autocorrelation')
axes[1].grid(True, linestyle='--', alpha=0.6)

# Define the figure caption text
caption_text = (
    "Figure 1: Autocorrelation Analysis of Crime Severity.\n\n"
    "The figure displays the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) for the `sum_severity` time series, "
    "with lags shown up to 744 hours (31 days).\n\n"
    "The ACF plot (top) shows the total correlation between the time series and its past values.\n\n"
    "The PACF plot (bottom) isolates the direct correlation between the series and a specific lag, removing the influence of any shorter, intervening lags.\n\n"
    "The prominent spikes recurring at multiples of 24 hours in both plots strongly indicate a daily seasonal pattern in the data.\n\n"
    "Spikes extending beyond the blue shaded area represent statistically significant correlations at the 95% confidence level."
)

# --- Save the plot and the caption text ---
plt.suptitle('Extended Autocorrelation Analysis of Summed Severity Measure', fontsize=18)
plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout to prevent title overlap

# Save the figure without the caption box
output_image_filename = 'autocorrelation_plots_extended_vertical.png'
plt.savefig(output_image_filename)
print(f"Extended autocorrelation plots saved as '{output_image_filename}'")
plt.show()
plt.close()

# Save the caption text to a separate .txt file
caption_filename = 'figure_1_caption.txt'
try:
    with open(caption_filename, 'w') as f:
        f.write(caption_text)
    print(f"Figure caption saved to '{caption_filename}'")
except IOError as e:
    print(f"Error writing to file {caption_filename}: {e}")