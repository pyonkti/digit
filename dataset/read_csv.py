import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a pandas DataFrame
# Replace 'your_file.csv' with the path to your actual CSV file
df = pd.read_csv('param_id,trial_id,param_name,param_value')

pivot_df = df.pivot(index='trial_id', columns='param_name', values='param_value')

# Sort the DataFrame by trial_id
pivot_df.sort_index(inplace=True)

# Define parameter groups
group1 = ['gaussian_kernel', 'median_kernel', 'line_threshold']
group2 = [param for param in pivot_df.columns if param not in group1]

# Plot the first group of parameters
plt.figure(figsize=(12, 6))
for column in group1:
    plt.plot(pivot_df.index, pivot_df[column], marker='o', label=column)

plt.xlabel('Trial ID')
plt.ylabel('Parameter Value')
plt.title('Trend of Parameters with Closer Intervals Over Trials')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot the second group of parameters
plt.figure(figsize=(12, 6))
for column in group2:
    plt.plot(pivot_df.index, pivot_df[column], marker='o', label=column)

plt.xlabel('Trial ID')
plt.ylabel('Parameter Value')
plt.title('Trend of Other Parameters Over Trials')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
