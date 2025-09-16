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

# Plot ACF and PACF with the specified number of lags
fig, axes = plt.subplots(1, 2, figsize=(20, 6)) # Increased figure size for readability
plot_acf(data['sum_severity'], ax=axes[0], lags=number_of_lags)
plot_pacf(data['sum_severity'], ax=axes[1], lags=number_of_lags)

axes[0].set_title('Autocorrelation (ACF) to {} lags'.format(number_of_lags))
axes[1].set_title('Partial Autocorrelation (PACF) to {} lags'.format(number_of_lags))

plt.suptitle('Extended Autocorrelation Analysis of sum_severity', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
plt.savefig('autocorrelation_plots_extended.png')
print(f"Extended autocorrelation plots saved as 'autocorrelation_plots_extended.png'")
plt.close()