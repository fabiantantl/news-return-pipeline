import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Read data from file
x = []
y = []

with open("prelim_data.txt", "r") as file:
    for line in file:
        # Split the line by whitespace and convert to floats
        values = line.split()
        if len(values) == 2:
            x_val, y_val = float(values[0]), float(values[1])
            # Only add non-zero values
            if x_val != 0 or y_val != 0:
                x.append(x_val)
                y.append(y_val)

# Convert x and y to numpy arrays
x = np.array(x)
y = np.array(y)

# Perform linear regression to find trendline
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

# Calculate the trendline values
y_trendline = slope * x + intercept

# Plot the scatter plot
plt.scatter(x, y, color="blue", label="Data Points")

# Plot the trendline
plt.plot(x, y_trendline, color="red", label=f"Trendline (R² = {r_value**2:.4f})")

# Labels and title
plt.xlabel("5 Days Forward Return (Index Funds)")
plt.ylabel("Aggregated News Headline Sentiment (Daily)")
plt.title("Index Funds vs News Sentiment")  # Clean title without "with trendline"
plt.legend()

# Save the plot as a PNG image
plt.savefig("scatter_plot_with_trendline.png")

# Show the plot (optional)
plt.show()
