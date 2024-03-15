#import matplotlib.pyplot as plt
#
## Data for the four classes
#classes = ["Non", "Very Mild", "Mild", "Moderate"]
#average_values = [299006, 286164, 264562, 260600]
#standard_deviations = [29806, 29854, 39447, 42501]
#
## Create a bar plot
#plt.figure(figsize=(10, 6))
#plt.bar(classes, average_values, yerr=standard_deviations, capsize=6, color="skyblue", edgecolor="black")
#plt.xlabel("Classes")
#plt.ylim(200000,350000)
#plt.ylabel("Average")
##plt.title("Average and Standard Deviation for Different Classes")
#plt.grid(axis="y", linestyle="--", alpha=0.7)
#
## Show the plot
#plt.show()
#
import matplotlib.pyplot as plt

# Data for the four classes
classes = ["Non", "Very Mild", "Mild", "Moderate"]
average_values = [299006, 286164, 264562, 260600]
standard_deviations = [29806, 29854, 39447, 42501]

# Create a bar plot
plt.figure(figsize=(6, 4))  # Adjust the figure size for a two-column layout
plt.bar(classes, average_values, yerr=standard_deviations, capsize=6, color="skyblue", edgecolor="black")
plt.xlabel("Severity Classes")
plt.ylabel("Average Value")
plt.ylim(200000, 350000)  # Set the y-axis limits for better readability
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Save the plot as an image (optional)
plt.savefig("bar_plot.pdf", dpi=300, bbox_inches="tight")  # Adjust the filename and DPI as needed

# Show the plot (optional)
plt.show()

