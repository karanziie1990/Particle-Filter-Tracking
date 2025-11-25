import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
import seaborn as sns
from scipy.stats import norm

class ParticleFilter:
    """
    Particle Filter for robot tracking with nonlinear motion model
    """
    
    def __init__(self, num_particles, landmarks, motion_noise, measurement_noise):
        """
        Initialize the particle filter
        
        Parameters:
        -----------
        num_particles : int
            Number of particles to use
        landmarks : np.array 
            Array of landmark positions (N x 2)
        motion_noise : dict
            Dictionary containing 'forward', 'turn', and 'drift' noise std deviations
        measurement_noise : float
            Standard deviation of measurement noise
        """
        self.num_particles = num_particles
        self.landmarks = landmarks
        self.motion_noise = motion_noise
        self.measurement_noise = measurement_noise
        
        # Initialize particles randomly in the space
        self.particles = np.zeros((num_particles, 3))  # [x, y, theta]
        self.particles[:, 0] = np.random.uniform(0, 100, num_particles)  # x
        self.particles[:, 1] = np.random.uniform(0, 100, num_particles)  # y
        self.particles[:, 2] = np.random.uniform(0, 2*np.pi, num_particles)  # theta
        
        # Initialize weights uniformly
        self.weights = np.ones(num_particles) / num_particles
        
        # Store history for analysis
        self.history = {
            'particles': [],
            'weights': [],
            'estimated_pose': [],
            'true_pose': [],
            'variance': []
        }
    
    def propagate(self, command_forward=1.0, command_turn=np.deg2rad(10)):
        """
        Propagate particles according to the motion model with noise
        
        Parameters:
        -----------
        command_forward : float
            Forward movement command (meters)
        command_turn : float
            Turn command (radians, positive = left turn)
        """
        # Add Gaussian noise to the commands
        forward_noise = np.random.normal(0, self.motion_noise['forward'], self.num_particles)
        turn_noise = np.random.normal(0, self.motion_noise['turn'], self.num_particles)
        drift_noise = np.random.normal(0, self.motion_noise['drift'], self.num_particles)
        
        # Apply noisy motion model to each particle
        actual_forward = command_forward + forward_noise
        actual_turn = command_turn + turn_noise
        
        # Update particle positions
        # First update orientation
        self.particles[:, 2] += actual_turn + drift_noise
        
        # Normalize angles to [0, 2π)
        self.particles[:, 2] = np.mod(self.particles[:, 2], 2*np.pi)
        
        # Then update position based on new orientation
        self.particles[:, 0] += actual_forward * np.cos(self.particles[:, 2])
        self.particles[:, 1] += actual_forward * np.sin(self.particles[:, 2])
    
    def compute_measurement(self, pose):
        """
        Compute expected measurements (distances to landmarks) from a given pose
        
        Parameters:
        -----------
        pose : np.array
            Robot pose [x, y, theta]
        
        Returns:
        --------
        distances : np.array
            Distances to each landmark
        """
        distances = np.sqrt(
            (self.landmarks[:, 0] - pose[0])**2 + 
            (self.landmarks[:, 1] - pose[1])**2
        )
        return distances
    
    def update(self, measurements):
        """
        Update particle weights based on sensor measurements
        
        Parameters:
        -----------
        measurements : np.array
            Actual measurements (distances to landmarks)
        """
        # Compute likelihood for each particle
        for i in range(self.num_particles):
            # Expected measurements for this particle
            expected = self.compute_measurement(self.particles[i])
            
            # Compute likelihood using Gaussian model
            # P(z|x) = product of P(z_i|x) for each measurement
            likelihood = 1.0
            for j in range(len(measurements)):
                # Probability density of observed measurement given expected
                likelihood *= norm.pdf(measurements[j], expected[j], self.measurement_noise)
            
            self.weights[i] *= likelihood
        
        # Normalize weights
        weight_sum = np.sum(self.weights)
        if weight_sum > 0:
            self.weights /= weight_sum
        else:
            # If all weights are zero, reset to uniform
            self.weights = np.ones(self.num_particles) / self.num_particles
    
    def resample(self):
        """
        Resample particles based on their weights (systematic resampling)
        """
        # Systematic resampling
        positions = (np.arange(self.num_particles) + np.random.random()) / self.num_particles
        
        cumulative_sum = np.cumsum(self.weights)
        i, j = 0, 0
        new_particles = np.zeros_like(self.particles)
        
        while i < self.num_particles:
            if positions[i] < cumulative_sum[j]:
                new_particles[i] = self.particles[j]
                i += 1
            else:
                j += 1
        
        self.particles = new_particles
        self.weights = np.ones(self.num_particles) / self.num_particles
    
    def estimate_pose(self):
        """
        Estimate the robot's pose as weighted average of particles
        
        Returns:
        --------
        estimated_pose : np.array
            Estimated [x, y, theta]
        """
        # Weighted average for x and y
        x_est = np.sum(self.weights * self.particles[:, 0])
        y_est = np.sum(self.weights * self.particles[:, 1])
        
        # For angle, use circular mean
        sin_sum = np.sum(self.weights * np.sin(self.particles[:, 2]))
        cos_sum = np.sum(self.weights * np.cos(self.particles[:, 2]))
        theta_est = np.arctan2(sin_sum, cos_sum)
        
        return np.array([x_est, y_est, theta_est])
    
    def compute_variance(self):
        """
        Compute variance in particle positions
        
        Returns:
        --------
        variance : dict
            Variances in x, y, and theta
        """
        estimated_pose = self.estimate_pose()
        
        var_x = np.sum(self.weights * (self.particles[:, 0] - estimated_pose[0])**2)
        var_y = np.sum(self.weights * (self.particles[:, 1] - estimated_pose[1])**2)
        
        # Circular variance for angle
        var_theta = 1 - np.sqrt(
            (np.sum(self.weights * np.cos(self.particles[:, 2])))**2 +
            (np.sum(self.weights * np.sin(self.particles[:, 2])))**2
        )
        
        return {'x': var_x, 'y': var_y, 'theta': var_theta}
    
    def store_history(self, true_pose):
        """
        Store current state for later analysis
        """
        self.history['particles'].append(self.particles.copy())
        self.history['weights'].append(self.weights.copy())
        self.history['estimated_pose'].append(self.estimate_pose())
        self.history['true_pose'].append(true_pose.copy())
        self.history['variance'].append(self.compute_variance())


class Robot:
    """
    Simulated robot with nonlinear motion
    """
    
    def __init__(self, x=50, y=50, theta=0):
        self.pose = np.array([x, y, theta], dtype=float)
        self.history = [self.pose.copy()]
    
    def move(self, forward, turn, motion_noise):
        """
        Move the robot with noise
        
        Parameters:
        -----------
        forward : float
            Forward command
        turn : float
            Turn command (radians)
        motion_noise : dict
            Noise parameters
        """
        # Add noise to motion
        actual_forward = forward + np.random.normal(0, motion_noise['forward'])
        actual_turn = turn + np.random.normal(0, motion_noise['turn'])
        drift = np.random.normal(0, motion_noise['drift'])
        
        # Update orientation
        self.pose[2] += actual_turn + drift
        self.pose[2] = np.mod(self.pose[2], 2*np.pi)
        
        # Update position
        self.pose[0] += actual_forward * np.cos(self.pose[2])
        self.pose[1] += actual_forward * np.sin(self.pose[2])
        
        self.history.append(self.pose.copy())
    
    def sense(self, landmarks, measurement_noise):
        """
        Measure distances to landmarks with noise
        
        Parameters:
        -----------
        landmarks : np.array
            Landmark positions
        measurement_noise : float
            Measurement noise std deviation
        
        Returns:
        --------
        measurements : np.array
            Noisy distance measurements
        """
        true_distances = np.sqrt(
            (landmarks[:, 0] - self.pose[0])**2 + 
            (landmarks[:, 1] - self.pose[1])**2
        )
        
        # Add Gaussian noise
        noisy_distances = true_distances + np.random.normal(0, measurement_noise, len(landmarks))
        
        return noisy_distances


def run_simulation(num_particles, motion_noise, measurement_noise, num_steps=50):
    """
    Run a complete particle filter simulation
    
    Parameters:
    -----------
    num_particles : int
        Number of particles
    motion_noise : dict
        Motion noise parameters
    measurement_noise : float
        Measurement noise std deviation
    num_steps : int
        Number of simulation steps
    
    Returns:
    --------
    pf : ParticleFilter
        Particle filter with history
    robot : Robot
        Robot with history
    """
    # Set up landmarks (fixed points in the environment)
    landmarks = np.array([
        [20, 20],
        [80, 20],
        [80, 80],
        [20, 80],
        [50, 50]
    ])
    
    # Initialize robot and particle filter
    robot = Robot(x=50, y=50, theta=0)
    pf = ParticleFilter(num_particles, landmarks, motion_noise, measurement_noise)
    
    # Commands
    command_forward = 1.0  # meters
    command_turn = np.deg2rad(10)  # 10 degrees left
    
    # Run simulation
    for step in range(num_steps):
        # Move robot
        robot.move(command_forward, command_turn, motion_noise)
        
        # Get sensor measurements
        measurements = robot.sense(landmarks, measurement_noise)
        
        # Particle filter steps
        pf.propagate(command_forward, command_turn)
        pf.update(measurements)
        pf.resample()
        
        # Store history
        pf.store_history(robot.pose)
    
    return pf, robot, landmarks


def visualize_step(pf, robot, landmarks, step, save_path=None):
    """
    Visualize a single time step
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Get data for this step
    particles = pf.history['particles'][step]
    weights = pf.history['weights'][step]
    estimated_pose = pf.history['estimated_pose'][step]
    true_pose = pf.history['true_pose'][step]
    
    # Normalize weights for visualization
    weights_normalized = weights / np.max(weights) if np.max(weights) > 0 else weights
    
    # Plot landmarks
    ax.scatter(landmarks[:, 0], landmarks[:, 1], 
              c='green', s=300, marker='*', 
              edgecolors='darkgreen', linewidths=2,
              label='Landmarks', zorder=5)
    
    # Plot particles with size proportional to weight
    scatter = ax.scatter(particles[:, 0], particles[:, 1],
                        c='blue', s=weights_normalized * 500,
                        alpha=0.5, label='Particles')
    
    # Plot estimated position
    ax.scatter(estimated_pose[0], estimated_pose[1],
              c='orange', s=300, marker='D',
              edgecolors='darkorange', linewidths=2,
              label='Estimated Position', zorder=4)
    
    # Plot true robot position
    ax.scatter(true_pose[0], true_pose[1],
              c='red', s=300, marker='o',
              edgecolors='darkred', linewidths=2,
              label='True Position', zorder=4)
    
    # Plot orientation arrows
    arrow_length = 5
    ax.arrow(estimated_pose[0], estimated_pose[1],
            arrow_length * np.cos(estimated_pose[2]),
            arrow_length * np.sin(estimated_pose[2]),
            head_width=2, head_length=1.5, fc='orange', ec='darkorange',
            linewidth=2, zorder=4)
    
    ax.arrow(true_pose[0], true_pose[1],
            arrow_length * np.cos(true_pose[2]),
            arrow_length * np.sin(true_pose[2]),
            head_width=2, head_length=1.5, fc='red', ec='darkred',
            linewidth=2, zorder=4)
    
    # Plot trajectory
    true_trajectory = np.array(robot.history[:step+2])
    estimated_trajectory = np.array(pf.history['estimated_pose'][:step+1])
    
    ax.plot(true_trajectory[:, 0], true_trajectory[:, 1],
           'r--', linewidth=2, alpha=0.7, label='True Trajectory')
    ax.plot(estimated_trajectory[:, 0], estimated_trajectory[:, 1],
           'orange', linewidth=2, alpha=0.7, label='Estimated Trajectory')
    
    # Compute and display statistics
    variance = pf.history['variance'][step]
    error = np.linalg.norm(estimated_pose[:2] - true_pose[:2])
    
    stats_text = f'Step: {step}\n'
    stats_text += f'Particles: {len(particles)}\n'
    stats_text += f'Position Error: {error:.2f}m\n'
    stats_text += f'Var(x): {variance["x"]:.2f}\n'
    stats_text += f'Var(y): {variance["y"]:.2f}\n'
    stats_text += f'Var(θ): {variance["theta"]:.3f}'
    
    ax.text(0.02, 0.98, stats_text,
           transform=ax.transAxes,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
           fontsize=10, family='monospace')
    
    ax.set_xlim(-10, 110)
    ax.set_ylim(-10, 110)
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title(f'Particle Filter - Step {step}', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, ax


def plot_analysis(pf, robot, save_path=None):
    """
    Create comprehensive analysis plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Extract data
    steps = range(len(pf.history['estimated_pose']))
    estimated_poses = np.array(pf.history['estimated_pose'])
    true_poses = np.array(pf.history['true_pose'])
    variances = pf.history['variance']
    
    # 1. Position error over time
    position_errors = np.linalg.norm(estimated_poses[:, :2] - true_poses[:, :2], axis=1)
    axes[0, 0].plot(steps, position_errors, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Time Step', fontsize=11)
    axes[0, 0].set_ylabel('Position Error (m)', fontsize=11)
    axes[0, 0].set_title('Tracking Error Over Time', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=np.mean(position_errors), color='r', 
                       linestyle='--', label=f'Mean: {np.mean(position_errors):.2f}m')
    axes[0, 0].legend()
    
    # 2. Variance over time
    var_x = [v['x'] for v in variances]
    var_y = [v['y'] for v in variances]
    axes[0, 1].plot(steps, var_x, 'r-', linewidth=2, label='Var(x)')
    axes[0, 1].plot(steps, var_y, 'b-', linewidth=2, label='Var(y)')
    axes[0, 1].set_xlabel('Time Step', fontsize=11)
    axes[0, 1].set_ylabel('Variance', fontsize=11)
    axes[0, 1].set_title('Position Variance Over Time', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # 3. Angle error over time
    angle_errors = np.abs(estimated_poses[:, 2] - true_poses[:, 2])
    # Wrap angle errors to [-π, π]
    angle_errors = np.arctan2(np.sin(angle_errors), np.cos(angle_errors))
    angle_errors = np.abs(angle_errors)
    axes[1, 0].plot(steps, np.rad2deg(angle_errors), 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Time Step', fontsize=11)
    axes[1, 0].set_ylabel('Orientation Error (degrees)', fontsize=11)
    axes[1, 0].set_title('Orientation Error Over Time', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=np.mean(np.rad2deg(angle_errors)), color='r',
                       linestyle='--', label=f'Mean: {np.mean(np.rad2deg(angle_errors)):.2f}°')
    axes[1, 0].legend()
    
    # 4. Error distribution
    axes[1, 1].hist(position_errors, bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[1, 1].axvline(x=np.mean(position_errors), color='r', 
                       linestyle='--', linewidth=2, label=f'Mean: {np.mean(position_errors):.2f}m')
    axes[1, 1].axvline(x=np.median(position_errors), color='orange',
                       linestyle='--', linewidth=2, label=f'Median: {np.median(position_errors):.2f}m')
    axes[1, 1].set_xlabel('Position Error (m)', fontsize=11)
    axes[1, 1].set_ylabel('Frequency', fontsize=11)
    axes[1, 1].set_title('Position Error Distribution', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def compare_experiments():
    """
    Run multiple experiments with different parameters
    """
    # Define experiments
    experiments = [
        {
            'name': 'Baseline (100 particles)',
            'num_particles': 100,
            'motion_noise': {'forward': 0.1, 'turn': 0.05, 'drift': 0.02},
            'measurement_noise': 0.5
        },
        {
            'name': 'Few Particles (20)',
            'num_particles': 20,
            'motion_noise': {'forward': 0.1, 'turn': 0.05, 'drift': 0.02},
            'measurement_noise': 0.5
        },
        {
            'name': 'Many Particles (500)',
            'num_particles': 500,
            'motion_noise': {'forward': 0.1, 'turn': 0.05, 'drift': 0.02},
            'measurement_noise': 0.5
        },
        {
            'name': 'High Motion Noise',
            'num_particles': 100,
            'motion_noise': {'forward': 0.5, 'turn': 0.2, 'drift': 0.1},
            'measurement_noise': 0.5
        },
        {
            'name': 'High Measurement Noise',
            'num_particles': 100,
            'motion_noise': {'forward': 0.1, 'turn': 0.05, 'drift': 0.02},
            'measurement_noise': 2.0
        },
        {
            'name': 'Low Noise',
            'num_particles': 100,
            'motion_noise': {'forward': 0.02, 'turn': 0.01, 'drift': 0.005},
            'measurement_noise': 0.1
        }
    ]
    
    results = []
    
    print("Running experiments...")
    for i, exp in enumerate(experiments):
        print(f"\nExperiment {i+1}/{len(experiments)}: {exp['name']}")
        
        # Run simulation with fixed seed for reproducibility
        np.random.seed(42)
        pf, robot, landmarks = run_simulation(
            exp['num_particles'],
            exp['motion_noise'],
            exp['measurement_noise'],
            num_steps=50
        )
        
        # Calculate statistics
        estimated_poses = np.array(pf.history['estimated_pose'])
        true_poses = np.array(pf.history['true_pose'])
        position_errors = np.linalg.norm(estimated_poses[:, :2] - true_poses[:, :2], axis=1)
        
        results.append({
            'name': exp['name'],
            'mean_error': np.mean(position_errors),
            'std_error': np.std(position_errors),
            'max_error': np.max(position_errors),
            'median_error': np.median(position_errors),
            'final_error': position_errors[-1]
        })
        
        print(f"  Mean Error: {results[-1]['mean_error']:.3f}m")
        print(f"  Std Error: {results[-1]['std_error']:.3f}m")
        print(f"  Max Error: {results[-1]['max_error']:.3f}m")
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    names = [r['name'] for r in results]
    mean_errors = [r['mean_error'] for r in results]
    std_errors = [r['std_error'] for r in results]
    max_errors = [r['max_error'] for r in results]
    median_errors = [r['median_error'] for r in results]
    
    # Mean error comparison
    axes[0, 0].barh(names, mean_errors, color='skyblue', edgecolor='black')
    axes[0, 0].set_xlabel('Mean Position Error (m)', fontsize=11)
    axes[0, 0].set_title('Mean Tracking Error', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='x')
    
    # Standard deviation comparison
    axes[0, 1].barh(names, std_errors, color='lightcoral', edgecolor='black')
    axes[0, 1].set_xlabel('Std Dev of Error (m)', fontsize=11)
    axes[0, 1].set_title('Error Standard Deviation', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='x')
    
    # Max error comparison
    axes[1, 0].barh(names, max_errors, color='lightgreen', edgecolor='black')
    axes[1, 0].set_xlabel('Max Position Error (m)', fontsize=11)
    axes[1, 0].set_title('Maximum Tracking Error', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    # Median error comparison
    axes[1, 1].barh(names, median_errors, color='plum', edgecolor='black')
    axes[1, 1].set_xlabel('Median Position Error (m)', fontsize=11)
    axes[1, 1].set_title('Median Tracking Error', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('outputs/experiment_comparison.png', dpi=150, bbox_inches='tight')
    
    return results, fig


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run baseline simulation
    print("Running baseline simulation...")
    motion_noise = {'forward': 0.1, 'turn': 0.05, 'drift': 0.02}
    measurement_noise = 0.5
    num_particles = 100
    
    pf, robot, landmarks = run_simulation(num_particles, motion_noise, measurement_noise)
    
    # Visualize selected steps
    print("Creating visualizations...")
    steps_to_plot = [0, 10, 25, 49]
    for step in steps_to_plot:
        visualize_step(pf, robot, landmarks, step, 
                      save_path=f'outputs/step_{step:02d}.png')
        plt.close()
    
    # Create analysis plot
    plot_analysis(pf, robot, save_path='outputs/analysis.png')
    plt.close()
    
    # Run comparative experiments
    print("\nRunning comparative experiments...")
    results, comp_fig = compare_experiments()
    plt.close()
    
    # Print summary
    print("\n" + "="*60)
    print("SIMULATION COMPLETE")
    print("="*60)
    print("\nFiles generated:")
    print("  - step_00.png, step_10.png, step_25.png, step_49.png")
    print("  - analysis.png")
    print("  - experiment_comparison.png")
    print("\nAll files saved to outputs/")
