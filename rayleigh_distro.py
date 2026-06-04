import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Adjustable parameters
mean1 = 0          # Mean of first distribution
variance1 = 0.08**2    # Variance of first distribution
std1 = np.sqrt(variance1)

mean2 = 0          # Mean of second distribution  
variance2 = 0.086**2    # Variance of second distribution
std2 = np.sqrt(variance2)

n_samples = 10000    # Number of samples to generate

# Generate concrete samples from both distributions
samples1 = np.random.normal(mean1, std1, n_samples)
samples2 = np.random.normal(mean2, std2, n_samples)

# Calculate pairwise RMSE between the two distributions
# RMSE for each pair: sqrt((x1 - x2)^2)
pairwise_rmse = np.sqrt(samples1**2 - samples2**2)  # This is just |x1 - x2| for single pairs

# Create figure with 3 subplots in a row (triptych)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: Distribution 1
ax1.hist(samples1, bins=50, density=True, alpha=0.7, color='blue', edgecolor='black')
ax1.axvline(x=np.mean(samples1), color='darkblue', linestyle='--', linewidth=2, 
            label=f'Mean: {np.mean(samples1):.3f}')
ax1.axvline(x=np.median(samples1), color='navy', linestyle=':', linewidth=2,
            label=f'Median: {np.median(samples1):.3f}')

# Overlay theoretical distribution
x_range1 = np.linspace(samples1.min(), samples1.max(), 200)
pdf1 = stats.norm.pdf(x_range1, mean1, std1)
ax1.plot(x_range1, pdf1, 'b-', linewidth=2, label=f'Theory: μ={mean1}, σ²={variance1}')

ax1.set_xlabel('Value', fontsize=12)
ax1.set_ylabel('Probability Density', fontsize=12)
ax1.set_title(f'Distribution 1\n(n={n_samples} samples)', fontsize=14, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Plot 2: Distribution 2
ax2.hist(samples2, bins=50, density=True, alpha=0.7, color='red', edgecolor='black')
ax2.axvline(x=np.mean(samples2), color='darkred', linestyle='--', linewidth=2,
            label=f'Mean: {np.mean(samples2):.3f}')
ax2.axvline(x=np.median(samples2), color='maroon', linestyle=':', linewidth=2,
            label=f'Median: {np.median(samples2):.3f}')

# Overlay theoretical distribution
x_range2 = np.linspace(samples2.min(), samples2.max(), 200)
pdf2 = stats.norm.pdf(x_range2, mean2, std2)
ax2.plot(x_range2, pdf2, 'r-', linewidth=2, label=f'Theory: μ={mean2}, σ²={variance2}')

ax2.set_xlabel('Value', fontsize=12)
ax2.set_ylabel('Probability Density', fontsize=12)
ax2.set_title(f'Distribution 2\n(n={n_samples} samples)', fontsize=14, fontweight='bold')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

# Plot 3: Pairwise RMSE Distribution
ax3.hist(pairwise_rmse, bins=50, density=True, alpha=0.7, color='green', edgecolor='black')
ax3.axvline(x=np.mean(pairwise_rmse), color='darkgreen', linestyle='--', linewidth=2,
            label=f'Mean: {np.mean(pairwise_rmse):.3f}')
ax3.axvline(x=np.median(pairwise_rmse), color='forestgreen', linestyle=':', linewidth=2,
            label=f'Median: {np.median(pairwise_rmse):.3f}')

# Add KDE for smooth curve
# kde = stats.gaussian_kde(pairwise_rmse)
# x_kde = np.linspace(pairwise_rmse.min(), pairwise_rmse.max(), 200)
# ax3.plot(x_kde, kde(x_kde), 'g-', linewidth=2, label='KDE')

ax3.set_xlabel('RMSE', fontsize=12)
ax3.set_ylabel('Probability Density', fontsize=12)
ax3.set_title(f'Pairwise RMSE Distribution\n(n={n_samples} pairs)', fontsize=14, fontweight='bold')
ax3.legend(loc='upper right')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print statistics
print("=" * 60)
print("DISTRIBUTION STATISTICS")
print("=" * 60)

print("\nDistribution 1 (Blue):")
print(f"  Theoretical: μ={mean1}, σ²={variance1}, σ={std1:.4f}")
print(f"  Empirical:   μ={np.mean(samples1):.4f}, σ²={np.var(samples1):.4f}, σ={np.std(samples1):.4f}")

print("\nDistribution 2 (Red):")
print(f"  Theoretical: μ={mean2}, σ²={variance2}, σ={std2:.4f}")
print(f"  Empirical:   μ={np.mean(samples2):.4f}, σ²={np.var(samples2):.4f}, σ={np.std(samples2):.4f}")

print("\nPairwise RMSE Distribution (Green):")
print(f"  Mean: {np.mean(pairwise_rmse):.4f}")
print(f"  Median: {np.median(pairwise_rmse):.4f}")
print(f"  Std Dev: {np.std(pairwise_rmse):.4f}")
print(f"  Min: {np.min(pairwise_rmse):.4f}")
print(f"  Max: {np.max(pairwise_rmse):.4f}")
print(f"  25th percentile: {np.percentile(pairwise_rmse, 25):.4f}")
print(f"  75th percentile: {np.percentile(pairwise_rmse, 75):.4f}")

# Theoretical expectation for difference of two normal distributions
# If X1 ~ N(μ1, σ1²) and X2 ~ N(μ2, σ2²), then X1-X2 ~ N(μ1-μ2, σ1²+σ2²)
# For |X1-X2|, this follows a folded normal distribution
theoretical_diff_std = np.sqrt(variance1 + variance2)
print(f"\nTheoretical std of (X1-X2): {theoretical_diff_std:.4f}")
print(f"Expected mean of |X1-X2| (folded normal): {theoretical_diff_std * np.sqrt(2/np.pi):.4f}")