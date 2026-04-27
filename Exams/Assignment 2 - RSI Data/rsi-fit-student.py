import pandas as pd
import numpy as np
from scipy.stats import norm, chisquare, ttest_ind, ttest_1samp
import matplotlib.pyplot as plt

"""
Preamble: Load data from source CSV file
"""
path_to_datafile = r"C:\Users\willi\Desktop\ENGR 315\ENGR315-sp2026-student\data\drop-jump/all_participant_data_rsi.csv"

# Load CSV file
data = pd.read_csv(path_to_datafile)

# Extract force plate and accelerometer data
force_plate_rsi = data['force_plate_rsi']
accelerometer_rsi = data['accelerometer_rsi']

"""
Question 1: Load the force plate and acceleration based RSI data for all participants. Map each data set (accel and FP)
to a normal distribution. Clearly report the distribution parameters (mu and std) and generate a graph two each curve's 
probability distribution function. Include appropriate labels, titles, and legends.
"""
print('-----Question 1-----')

# Apply normal distribution fits to both force plate and accelerometer data
mu_force_plate, std_force_plate = norm.fit(force_plate_rsi)
mu_accelerometer, std_accelerometer = norm.fit(accelerometer_rsi)

# Report mu and std from both data columns
print(f"Force Plate RSI -> mu: {mu_force_plate}, std: {std_force_plate}")
print(f"Accelerometer RSI -> mu: {mu_accelerometer}, std: {std_accelerometer}")

# Create x range for plotting, use larger number for smoother plot curve
x_force_plate = np.linspace(min(force_plate_rsi), max(force_plate_rsi), 1000)
x_accelerometer = np.linspace(min(accelerometer_rsi), max(accelerometer_rsi), 1000)

# Create probability density functions
pdf_force_plate = norm.pdf(x_force_plate, mu_force_plate, std_force_plate)
pdf_accelerometer = norm.pdf(x_accelerometer, mu_accelerometer, std_accelerometer)

# Create a plot
plt.figure()

# Put x and y data on appropriate axis and label each line correctly
plt.plot(x_force_plate, pdf_force_plate, label='Force Plate RSI PDF')
plt.plot(x_accelerometer, pdf_accelerometer, label='Accelerometer RSI PDF')

# Label title and axes, and show legend
plt.title('Normal Distribution Fit for RSI Data')
plt.xlabel('RSI Value')
plt.ylabel('Probability Density')
plt.legend()
plt.grid()

# Show resulting plot
plt.show()

"""
Question 2: Conduct a Chi2 Goodness of Fit Test for each dataset to test whether the data is a good fit
for the derived normal distribution. Clearly print out the p-value, chi2 stat, and an indication of whether it is 
a fit or not. Do this for both acceleration and force plate distributions. It is suggested to generate 9 bins between 
[0,2), add append -inf and +inf to both ends of the bins. An alpha=0.05 is suitable for these tests.
"""
print('\n\n-----Question 2-----')

"""
Acceleration
"""
# Define bins that were suggested: (-inf, 0), [0,2) split into 9 bins, (2, inf)
bins = np.concatenate(([-np.inf], np.linspace(0, 2, 10), [np.inf]))

# Use histogram function
# Uses data array and bins created earlier
# Outputs histogram array and the bin edges used
obs_accelerometer, bin_edges1 = np.histogram(accelerometer_rsi, bins=bins)

# Get expected counts using cumulative distribution function
# Array to keep track of results
expected_accelerometer = []
# Number of items in dataset
n_accelerometer= len(accelerometer_rsi)

# Loop through bins, calculate probability for each bin by counting how many fall into each bin
for i in range(len(bins) - 1):
    prob = norm.cdf(bins[i+1], mu_accelerometer, std_accelerometer) - \
           norm.cdf(bins[i], mu_accelerometer, std_accelerometer)
    expected_accelerometer.append(prob * n_accelerometer)

# Change into array so it can be properly recognized by other functions
expected_accelerometer = np.array(expected_accelerometer)

# Normalize data by getting ratio of sum of observations and sum of expected probabilities
expected_accelerometer *= np.sum(obs_accelerometer) / np.sum(expected_accelerometer)

# Chisquare test
chi2_accelerometer, p_accelerometer= chisquare(obs_accelerometer, expected_accelerometer)

# Print results
print("Acceleration Data:")
print(f"Chi2 Stat = {chi2_accelerometer}, p-value = {p_accelerometer}")

if p_accelerometer> 0.05:
    print("Good fit")
else:
    print("Not a good fit")


"""
Force Plate
"""
# Use histogram function again, use same bins and different dataset
obs_force_plate, bin_edges2 = np.histogram(force_plate_rsi, bins=bins)

# Get expected counts using cumulative distribution function
# Array to keep track of results
expected_force_plate = []
# Number of items in dataset
n_force_plate = len(force_plate_rsi)

# Loop through bins, calculate probability for each bin by counting how many fall into each bin
for i in range(len(bins) - 1):
    prob = norm.cdf(bins[i+1], mu_force_plate, std_force_plate) - \
           norm.cdf(bins[i], mu_force_plate, std_force_plate)
    expected_force_plate.append(prob * n_force_plate)

# Change into array so it can be properly recognized by other functions
expected_force_plate = np.array(expected_force_plate)

# Normalize data by getting ratio of sum of observations and sum of expected probabilities
expected_force_plate *= np.sum(obs_force_plate) / np.sum(expected_force_plate)

# Chisquare test
chi2_force_plate, p_force_plate = chisquare(obs_force_plate, expected_force_plate)

# Print results
print("\nForce Plate Data:")
print(f"Chi2 Stat = {chi2_force_plate}, p-value = {p_force_plate}")

if p_force_plate > 0.05:
    print("Good fit")
else:
    print("Not a good fit")

"""
Question 3: Perform a t-test to determine whether the RSI means for the acceleration and force plate data are equivalent 
or not. Clearly report the p-value for the t-test and make a clear determination as to whether they are equal or not.
An alpha=0.05 is suitable for these tests.
"""
print('\n\n-----Question 3-----')

# Perform independent two-sample t-test
# Use equal_var=False since variances may differ between the two data sets
t_stat, p_value = ttest_ind(accelerometer_rsi, force_plate_rsi, equal_var=False)

# Report p value
print(f"p-value = {p_value}")

# Alpha of 0.05 is noted to be suitable for this test
alpha = 0.05

# Determine if data sets are equivalent or not
if p_value > alpha:
    print("Means are not significantly different")
else:
    print("Means are significantly different")

# Verification of whether mean values are significantly different or not
print(f"Mean Force Plate RSI = {np.mean(force_plate_rsi)}")
print(f"Mean Accelerometer RSI = {np.mean(accelerometer_rsi)}")

"""
Question 4: Calculate the RSI Error for the dataset where error is expressed as the difference between the 
Force Plate RSI measurement and the Accelerometer RSI measurement. Fit this error distribution to a normal curve and 
plot a histogram of the data on the same plot showing the fitted normal curve. Include appropriate labels, titles, and 
legends. The default binning approach from matplot lib with 16 bins is sufficient.
"""

### YOUR CODE HERE