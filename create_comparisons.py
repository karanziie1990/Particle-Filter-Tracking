import numpy as np
import matplotlib.pyplot as plt
from particle_filter_tracking import run_simulation

# Set random seed for reproducibility
np.random.seed(42)

# Define experiments
experiments = [
    {
        'name': 'Baseline',
        'num_particles': 100,
        'motion_noise': {'forward': 0.1, 'turn': 0.05, 'drift': 0.02},
        'measurement_noise': 0.5,
        'color': 'blue'
    },
    {
        'name': 'Few Particles',
        'num_particles': 20,
        'motion_noise': {'forward': 0.1, 'turn': 0.05, 'drift': 0.02},
        'measurement_noise': 0.5,
        'color': 'red'
    },
    {
        'name': 'High Motion Noise',
        'num_particles': 100,
        'motion_noise': {'forward': 0.5, 'turn': 0.2, 'drift': 0.1},
        'measurement_noise': 0.5,
        'color': 'green'
    }
]

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle('Particle Filter Performance Comparison', fontsize=16, fontweight='bold')

# Run simulations and collect data
print("Running trajectory comparison simulations...")
results = []
for exp in experiments:
    np.random.seed(42)  # Same seed for fair comparison
    pf, robot, landmarks = run_simulation(
        exp['num_particles'],
        exp['motion_noise'],
        exp['measurement_noise'],
        num_steps=50
    )
    results.append({
        'name': exp['name'],
        'pf': pf,
        'robot': robot,
        'landmarks': landmarks,
        'color': exp['color']
    })

# Plot 1: Trajectories Comparison
ax1 = axes[0, 0]
for result in results:
    estimated = np.array(result['pf'].history['estimated_pose'])
    ax1.plot(estimated[:, 0], estimated[:, 1], 
            color=result['color'], linewidth=2, alpha=0.7, 
            label=result['name'])

# Plot true trajectory
true_traj = np.array(results[0]['robot'].history)
ax1.plot(true_traj[:, 0], true_traj[:, 1], 
        'k--', linewidth=2, label='True Trajectory')

# Plot landmarks
landmarks = results[0]['landmarks']
ax1.scatter(landmarks[:, 0], landmarks[:, 1], 
           c='orange', s=300, marker='*', 
           edgecolors='darkorange', linewidths=2,
           label='Landmarks', zorder=5)

ax1.set_xlabel('X Position (m)', fontsize=11)
ax1.set_ylabel('Y Position (m)', fontsize=11)
ax1.set_title('Estimated Trajectories', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal')

# Plot 2: Error Over Time
ax2 = axes[0, 1]
for result in results:
    estimated = np.array(result['pf'].history['estimated_pose'])
    true_poses = np.array(result['pf'].history['true_pose'])
    errors = np.linalg.norm(estimated[:, :2] - true_poses[:, :2], axis=1)
    
    ax2.plot(range(len(errors)), errors, 
            color=result['color'], linewidth=2, 
            label=result['name'])

ax2.set_xlabel('Time Step', fontsize=11)
ax2.set_ylabel('Position Error (m)', fontsize=11)
ax2.set_title('Tracking Error Over Time', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(bottom=0)

# Plot 3: Cumulative Error
ax3 = axes[1, 0]
for result in results:
    estimated = np.array(result['pf'].history['estimated_pose'])
    true_poses = np.array(result['pf'].history['true_pose'])
    errors = np.linalg.norm(estimated[:, :2] - true_poses[:, :2], axis=1)
    cumulative_error = np.cumsum(errors)
    
    ax3.plot(range(len(cumulative_error)), cumulative_error, 
            color=result['color'], linewidth=2, 
            label=result['name'])

ax3.set_xlabel('Time Step', fontsize=11)
ax3.set_ylabel('Cumulative Error (m)', fontsize=11)
ax3.set_title('Cumulative Tracking Error', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# Plot 4: Variance Over Time
ax4 = axes[1, 1]
for result in results:
    variances = result['pf'].history['variance']
    var_total = [v['x'] + v['y'] for v in variances]
    
    ax4.plot(range(len(var_total)), var_total, 
            color=result['color'], linewidth=2, 
            label=result['name'])

ax4.set_xlabel('Time Step', fontsize=11)
ax4.set_ylabel('Total Position Variance', fontsize=11)
ax4.set_title('Uncertainty Over Time', fontsize=12, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)
ax4.set_yscale('log')

plt.tight_layout()
plt.savefig('outputs/trajectory_comparison.png', dpi=150, bbox_inches='tight')
print("Trajectory comparison plot saved!")

# Create detailed error statistics table
print("\n" + "="*80)
print("DETAILED ERROR STATISTICS")
print("="*80)

for result in results:
    estimated = np.array(result['pf'].history['estimated_pose'])
    true_poses = np.array(result['pf'].history['true_pose'])
    errors = np.linalg.norm(estimated[:, :2] - true_poses[:, :2], axis=1)
    
    print(f"\n{result['name']}:")
    print(f"  Mean Error:        {np.mean(errors):.3f}m")
    print(f"  Median Error:      {np.median(errors):.3f}m")
    print(f"  Std Dev:           {np.std(errors):.3f}m")
    print(f"  Min Error:         {np.min(errors):.3f}m")
    print(f"  Max Error:         {np.max(errors):.3f}m")
    print(f"  Final Error:       {errors[-1]:.3f}m")
    print(f"  Total Distance:    {np.sum(errors):.3f}m")
    
    # Compute RMSE
    rmse = np.sqrt(np.mean(errors**2))
    print(f"  RMSE:              {rmse:.3f}m")
    
    # Compute percentage of time error < 1m
    pct_good = np.sum(errors < 1.0) / len(errors) * 100
    print(f"  Error < 1m:        {pct_good:.1f}% of time")

print("\n" + "="*80)

# Create a final summary visualization
fig2, ax = plt.subplots(figsize=(12, 8))

# Box plot of errors
error_data = []
labels = []
for result in results:
    estimated = np.array(result['pf'].history['estimated_pose'])
    true_poses = np.array(result['pf'].history['true_pose'])
    errors = np.linalg.norm(estimated[:, :2] - true_poses[:, :2], axis=1)
    error_data.append(errors)
    labels.append(result['name'])

bp = ax.boxplot(error_data, labels=labels, patch_artist=True,
                showmeans=True, meanline=True)

# Color the boxes
colors = [r['color'] for r in results]
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax.set_ylabel('Position Error (m)', fontsize=12)
ax.set_title('Error Distribution Comparison', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Add legend for mean vs median
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='green', linewidth=2, linestyle='--', label='Median'),
    Line2D([0], [0], color='green', linewidth=2, linestyle='-', label='Mean')
]
ax.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.savefig('outputs/error_distribution.png', dpi=150, bbox_inches='tight')
print("Error distribution plot saved!")

print("\nAll visualizations complete!")