"""
Trajectory and Flight Profile Plotter
Generates 4D trajectory and 2D altitude vs time plots from state log files
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import os

def load_state_log(filepath):
    """
    Load state log file and return parsed dataframe
    
    Args:
        filepath: Path to the state log file
        
    Returns:
        DataFrame with columns: simt, id, lat, lon, distflown, alt, cas, tas, gs
    """
    df = pd.read_csv(filepath, comment='#', header=None)
    df.columns = ['simt', 'id', 'lat', 'lon', 'distflown', 'alt', 'cas', 'tas', 'gs']
    # Convert altitude from meters to feet
    df['alt'] = df['alt'] * 3.28084
    # Convert distance from meters to nautical miles (1 NM = 1852 meters)
    df['distflown'] = df['distflown'] / 1852.0
    return df

def plot_4d_trajectory(df, output_dir=None, filename='4d_trajectory.pdf'):
    """
    Create 4D trajectory plot (3D trajectory with altitude encoded as color)
    
    Args:
        df: DataFrame with trajectory data
        output_dir: Directory to save plot (if None, uses current directory)
        filename: Output filename for the plot
    """
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract coordinates
    lat = df['lat'].values
    lon = df['lon'].values
    alt = df['alt'].values
    time = df['simt'].values
    
    # Create scatter plot with altitude as color
    scatter = ax.scatter(lon, lat, alt, c=time, cmap='viridis', 
                        s=20, alpha=0.6, edgecolors='w', linewidth=0.5)
    
    # Also plot the trajectory as a line
    ax.plot(lon, lat, alt, 'r-', alpha=0.3, linewidth=1, label='Flight Path')
    
    # Labels and title
    ax.set_xlabel('Longitude', fontsize=11, fontweight='bold')
    ax.set_ylabel('Latitude', fontsize=11, fontweight='bold')
    ax.set_zlabel('Altitude (ft)', fontsize=11, fontweight='bold')
    ax.set_title('4D Flight Trajectory (Lat, Lon, Alt, Time)', 
                fontsize=13, fontweight='bold', pad=20)
    
    # Add colorbar for time
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Simulation Time (s)', fontsize=10, fontweight='bold')
    
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Adjust viewing angle
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    
    if output_dir:
        filepath = os.path.join(output_dir, filename)
    else:
        filepath = filename
    
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ 4D Trajectory plot saved to: {filepath}")
    plt.close()

def plot_4d_trajectory_alternative(df, output_dir=None, filename='4d_trajectory_alt.pdf'):
    """
    Alternative 4D trajectory plot with multiple subplots
    
    Args:
        df: DataFrame with trajectory data
        output_dir: Directory to save plot
        filename: Output filename for the plot
    """
    fig = plt.figure(figsize=(16, 10))
    
    # 3D plot with time coloring
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    time = df['simt'].values
    lat = df['lat'].values
    lon = df['lon'].values
    alt = df['alt'].values
    
    scatter = ax1.scatter(lon, lat, alt, c=time, cmap='plasma', s=30, alpha=0.7)
    ax1.plot(lon, lat, alt, 'b-', alpha=0.3, linewidth=0.8)
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_zlabel('Altitude (ft)')
    ax1.set_title('3D Trajectory (Time encoded as color)')
    plt.colorbar(scatter, ax=ax1, label='Time (s)', shrink=0.6)
    
    # Lat vs Lon (top view)
    ax2 = fig.add_subplot(2, 2, 2)
    scatter2 = ax2.scatter(lon, lat, c=time, cmap='plasma', s=30, alpha=0.7)
    ax2.plot(lon, lat, 'b-', alpha=0.3, linewidth=0.8)
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    ax2.set_title('Top View (Horizontal Trajectory)')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax2, label='Time (s)')
    
    # Lon vs Alt (side view)
    ax3 = fig.add_subplot(2, 2, 3)
    scatter3 = ax3.scatter(lon, alt, c=time, cmap='plasma', s=30, alpha=0.7)
    ax3.plot(lon, alt, 'b-', alpha=0.3, linewidth=0.8)
    ax3.set_xlabel('Longitude')
    ax3.set_ylabel('Altitude (ft)')
    ax3.set_title('Longitude vs Altitude')
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter3, ax=ax3, label='Time (s)')
    
    # Lat vs Alt (side view)
    ax4 = fig.add_subplot(2, 2, 4)
    scatter4 = ax4.scatter(lat, alt, c=time, cmap='plasma', s=30, alpha=0.7)
    ax4.plot(lat, alt, 'b-', alpha=0.3, linewidth=0.8)
    ax4.set_xlabel('Latitude')
    ax4.set_ylabel('Altitude (ft)')
    ax4.set_title('Latitude vs Altitude')
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter4, ax=ax4, label='Time (s)')
    
    fig.suptitle('4D Flight Trajectory - Multiple Views', 
                fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if output_dir:
        filepath = os.path.join(output_dir, filename)
    else:
        filepath = filename
    
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Alternative 4D Trajectory plot saved to: {filepath}")
    plt.close()

def plot_flight_profile_2d(df, output_dir=None, filename='flight_profile_2d.pdf'):
    """
    Create 2D flight profile plot (altitude vs time)
    
    Args:
        df: DataFrame with trajectory data
        output_dir: Directory to save plot
        filename: Output filename for the plot
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    time = df['simt'].values
    alt = df['alt'].values
    cas = df['cas'].values  # Include airspeed as additional metric
    
    # Plot altitude vs time
    line1 = ax.plot(time, alt, 'b-', linewidth=2.5, label='Altitude', marker='o', 
                   markersize=3, markevery=max(1, len(time)//50))
    ax.fill_between(time, alt, alpha=0.2, color='blue')
    
    ax.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Altitude (ft)', fontsize=12, fontweight='bold', color='b')
    ax.set_title('Flight Profile - Altitude vs Time', fontsize=13, fontweight='bold', pad=15)
    ax.tick_params(axis='y', labelcolor='b')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add statistics box
    max_alt = alt.max()
    min_alt = alt.min()
    total_time = time.max()
    avg_alt = alt.mean()
    
    stats_text = f'Max Altitude: {max_alt:.0f} ft\nMin Altitude: {min_alt:.0f} ft\n' \
                f'Avg Altitude: {avg_alt:.0f} ft\nTotal Time: {total_time:.2f} s'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if output_dir:
        filepath = os.path.join(output_dir, filename)
    else:
        filepath = filename
    
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Flight Profile (2D) plot saved to: {filepath}")
    plt.close()

def plot_altitude_vs_distance(df, output_dir=None, filename='altitude_vs_distance.pdf'):
    """
    Create altitude vs distance flown plot (hover phases appear as vertical lines)
    
    Args:
        df: DataFrame with trajectory data
        output_dir: Directory to save plot
        filename: Output filename for the plot
    """
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    
    dist = df['distflown'].values
    alt = df['alt'].values
    time = df['simt'].values
    
    # Plot altitude vs distance
    line1 = ax.plot(dist, alt, 'b-', linewidth=2.0, label='Altitude', marker='o', 
                   markersize=2.5, markevery=max(1, len(dist)//50))
    ax.fill_between(dist, alt, alpha=0.2, color='blue')
    
    ax.set_xlabel('Distance Flown (NM)', fontsize=13, color='black')
    ax.set_ylabel('Altitude (ft)', fontsize=13, color='black')
    ax.set_title('Flight Profile', fontsize=13, fontweight='bold', pad=8)
    ax.tick_params(axis='both', labelsize=10)
    ax.tick_params(axis='y', labelcolor='black')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add statistics box
    max_alt = alt.max()
    min_alt = alt.min()
    total_dist = dist.max()
    avg_alt = alt.mean()
    total_time = time.max()
    
    stats_text = f'Max Altitude: {max_alt:.0f} ft\nMin Altitude: {min_alt:.0f} ft\n' \
                f'Avg Altitude: {avg_alt:.0f} ft\nTotal Distance: {total_dist:.1f} NM\nTotal Time: {total_time:.1f} s'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if output_dir:
        filepath = os.path.join(output_dir, filename)
    else:
        filepath = filename
    
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Altitude vs Distance plot saved to: {filepath}")
    plt.close()

def plot_detailed_flight_profile(df, output_dir=None, filename='flight_profile_detailed.pdf'):
    """
    Create detailed flight profile with multiple parameters
    
    Args:
        df: DataFrame with trajectory data
        output_dir: Directory to save plot
        filename: Output filename for the plot
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    time = df['simt'].values
    alt = df['alt'].values
    cas = df['cas'].values
    tas = df['tas'].values
    gs = df['gs'].values
    distflown = df['distflown'].values
    
    # Plot 1: Altitude
    axes[0].plot(time, alt, 'b-', linewidth=2, marker='o', markersize=3, 
                markevery=max(1, len(time)//50), label='Altitude')
    axes[0].fill_between(time, alt, alpha=0.2, color='blue')
    axes[0].set_ylabel('Altitude (ft)', fontsize=11, fontweight='bold')
    axes[0].set_title('Flight Profile - Detailed View', fontsize=13, fontweight='bold', pad=15)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].legend(loc='upper left', fontsize=10)
    
    # Plot 2: Airspeeds
    axes[1].plot(time, cas, 'r-', linewidth=1.5, label='CAS (Calibrated)', marker='o', 
                markersize=2, markevery=max(1, len(time)//50))
    axes[1].plot(time, tas, 'g--', linewidth=1.5, label='TAS (True)', marker='s', 
                markersize=2, markevery=max(1, len(time)//50))
    axes[1].plot(time, gs, 'orange', linestyle=':', linewidth=1.5, label='GS (Ground Speed)', 
                marker='^', markersize=2, markevery=max(1, len(time)//50))
    axes[1].set_ylabel('Airspeed (m/s)', fontsize=11, fontweight='bold')
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].legend(loc='upper left', fontsize=10)
    
    # Plot 3: Distance Flown
    axes[2].plot(time, distflown, 'purple', linewidth=2, marker='o', markersize=3,
                markevery=max(1, len(time)//50), label='Distance Flown')
    axes[2].fill_between(time, distflown, alpha=0.2, color='purple')
    axes[2].set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
    axes[2].set_ylabel('Distance Flown (NM)', fontsize=11, fontweight='bold')
    axes[2].grid(True, alpha=0.3, linestyle='--')
    axes[2].legend(loc='upper left', fontsize=10)
    
    plt.tight_layout()
    
    if output_dir:
        filepath = os.path.join(output_dir, filename)
    else:
        filepath = filename
    
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✓ Detailed Flight Profile plot saved to: {filepath}")
    plt.close()

def main():
    """Main execution function"""
    
    # Path to the log file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(script_dir, 'profile_implementation', 'SBGR_DEST', 'Logs', 'STATELOG_SBSP-SBGR_20260310_09-16-55.log')
    output_dir = os.path.join(script_dir, 'trajectory_plots')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if file exists
    if not os.path.exists(log_file):
        print(f"Error: Log file not found at {log_file}")
        return
    
    print(f"Loading data from: {log_file}")
    print("-" * 70)
    
    # Load the data
    df = load_state_log(log_file)
    
    print(f"Data loaded successfully!")
    print(f"Number of data points: {len(df)}")
    print(f"Time range: {df['simt'].min():.2f}s to {df['simt'].max():.2f}s")
    print(f"Altitude range: {df['alt'].min():.0f}ft to {df['alt'].max():.0f}ft")
    print(f"Latitude range: {df['lat'].min():.6f}° to {df['lat'].max():.6f}°")
    print(f"Longitude range: {df['lon'].min():.6f}° to {df['lon'].max():.6f}°")
    print("-" * 70)
    
    # Generate plots
    print("\nGenerating plots...")
    print("-" * 70)
    
    # Plot 1: Main 4D trajectory
    plot_4d_trajectory(df, output_dir)
    
    # Plot 2: Alternative 4D views
    plot_4d_trajectory_alternative(df, output_dir)
    
    # Plot 3: Simple 2D flight profile (altitude vs time)
    plot_flight_profile_2d(df, output_dir)
    
    # Plot 4: Altitude vs distance (hover phases visible as vertical lines)
    plot_altitude_vs_distance(df, output_dir)
    
    # Plot 5: Detailed flight profile with multiple parameters
    plot_detailed_flight_profile(df, output_dir)
    
    print("-" * 70)
    print(f"\n✓ All plots completed successfully!")
    print(f"Plots saved to: {output_dir}")

if __name__ == "__main__":
    main()
