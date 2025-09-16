import pandas as pd

# Load the dataset
try:
    data = pd.read_pickle('full_astro_crime_data.pkl')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: 'full_astro_crime_data.pkl' not found. Make sure the file is in the correct directory.")
    exit()

# Extract the 'sum_severity' column
sum_severity = data['sum_severity']

# Calculate the split point
split_point = int(len(sum_severity) * 0.8)

# Get the first 80% of the data
first_80_percent = sum_severity.iloc[:split_point]

# Get the last 20% of the data
last_20_percent = sum_severity.iloc[split_point:]

# Save the first 80% to a CSV file
first_80_percent.to_csv('sum_severity_first_80.csv', index=False, header=['sum_severity'])
print("First 80% of 'sum_severity' saved to sum_severity_first_80.csv")

# Save the last 20% to a CSV file
last_20_percent.to_csv('sum_severity_last_20.csv', index=False, header=['sum_severity'])
print("Last 20% of 'sum_severity' saved to sum_severity_last_20.csv")