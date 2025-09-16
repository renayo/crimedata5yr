import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
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

# --- 1. Autocorrelation Analysis ---
print("\n--- Generating Autocorrelation Plots ---")

# Plot ACF and PACF
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
plot_acf(data['sum_severity'], ax=axes[0])
plot_pacf(data['sum_severity'], ax=axes[1])
plt.suptitle('Autocorrelation and Partial Autocorrelation of sum_severity')
plt.savefig('autocorrelation_plots.png')
print("Autocorrelation plots saved as 'autocorrelation_plots.png'")
plt.close()


# --- 2. Model Comparison ---
print("\n--- Comparing Time Series (ARIMA) and Random Forest Models ---")

# Split data into training and testing sets (80/20)
train_size = int(len(data) * 0.8)
train, test = data[0:train_size], data[train_size:len(data)]

# --- ARIMA Model ---
print("\nTraining ARIMA model...")
# These (p,d,q) parameters are a starting point. You may need to tune them based on the ACF/PACF plots.
# p: from PACF plot, d: order of differencing, q: from ACF plot
history = [x for x in train['sum_severity']]
predictions_arima = []

for t in range(len(test)):
    model_arima = ARIMA(history, order=(5,1,0)) # Example order
    model_fit = model_arima.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions_arima.append(yhat)
    obs = test['sum_severity'][t]
    history.append(obs)

rmse_arima = np.sqrt(mean_squared_error(test['sum_severity'], predictions_arima))
print(f"ARIMA Model RMSE: {rmse_arima}")


# --- Random Forest Model ---
print("\nTraining Random Forest model...")
features = [col for col in data.columns if col not in ['mean_severity', 'median_severity', 'sum_severity', 'crime_count']]
X_train, y_train = train[features], train['sum_severity']
X_test, y_test = test[features], test['sum_severity']

model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)
predictions_rf = model_rf.predict(X_test)

rmse_rf = np.sqrt(mean_squared_error(y_test, predictions_rf))
print(f"Random Forest Model RMSE: {rmse_rf}")


# --- 3. Conclusion ---
print("\n--- Model Performance Comparison ---")
if rmse_arima < rmse_rf:
    print(f"The ARIMA model (RMSE: {rmse_arima:.4f}) performed better than the Random Forest model (RMSE: {rmse_rf:.4f}).")
    print("This suggests that the time series nature of the data is a strong predictor.")
else:
    print(f"The Random Forest model (RMSE: {rmse_rf:.4f}) performed better than the ARIMA model (RMSE: {rmse_arima:.4f}).")
    print("This suggests that the astrological features are more predictive than the time series sequence alone.")