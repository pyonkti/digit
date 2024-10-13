import re

# Read the entire input as a string
with open("canny_log.txt", "r") as file:
    data = file.read()

# Use regex to find all numerical values that come after the word "after"
time_values = re.findall(r"after (\d+\.\d+) seconds", data)

# Convert the extracted values to float and use set to remove duplicates
time_values = list(set(float(value) for value in time_values))

# Sort the unique values in ascending order
time_values.sort()

# Export as an array
print(time_values)
