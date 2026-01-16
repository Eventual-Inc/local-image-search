import csv
import matplotlib.pyplot as plt
import numpy as np

# Read benchmark results
with open("benchmark_results.csv", "r") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    data = list(reader)

# Extract run columns
run_cols = [f for f in fieldnames if f.startswith("run_")]

# Calculate averages
images = []
averages = []
std_devs = []

for row in data:
    n = int(row["images"])
    times = [float(row[col]) for col in run_cols]
    avg = np.mean(times)
    std = np.std(times)

    images.append(n)
    averages.append(avg)
    std_devs.append(std)

# Plot
plt.figure(figsize=(10, 6))
plt.errorbar(images, averages, yerr=std_devs, marker='o', capsize=5, capthick=2, linewidth=2, markersize=8)
plt.xlabel("Number of Images", fontsize=12)
plt.ylabel("Time (seconds)", fontsize=12)
plt.title("MLX CLIP Embedding Time vs Number of Images\n(M4 Max, average of 10 runs, error bars = std dev)", fontsize=14)
plt.grid(True, alpha=0.3)

# Add data labels
for x, y in zip(images, averages):
    plt.annotate(f"{y:.2f}s", (x, y), textcoords="offset points", xytext=(0, 10), ha='center')

plt.tight_layout()
plt.savefig("benchmark_plot.png", dpi=150)
print("Plot saved to benchmark_plot.png")

# Print summary
print("\nSummary (average of 10 runs):")
for n, avg, std in zip(images, averages, std_devs):
    print(f"  {n:>4} images: {avg:.2f}s (±{std:.2f}s)")
