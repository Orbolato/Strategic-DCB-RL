"""
UAM Capacity Curves Generator

Generates capacity curves for two operational sides of a single vertiport.
Both sides are perpendicular to SBGR runway, representing alternative flight corridors.
Generates curves across three weather conditions (VMC, LMC, LIMC) using VertiportCapacityAnalyzer.

Outputs:
- Dictionaries: u (arrivals), v_min (min departures), v_max (max departures)
- Visualization plots for each operational side and weather condition
- CSV files with feasible solutions
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from vertiport_feasible_region_solver import VertiportCapacityAnalyzer

###############################################################################
# UAM Operational Sides and Weather Conditions
###############################################################################
# Single vertiport with two operational sides perpendicular to SBGR runway
# Aircraft can arrive/depart through Side A or Side B

UAM_CONFIGS = {
    1: 'Vertiport_Side_A (Perpendicular to SBGR)',
    2: 'Vertiport_Side_B (Perpendicular to SBGR)'
}

WEATHER_CONDITIONS = {
    0: 'VMC (Visual Meteorological Conditions)',
    1: 'IMC (Instrument Meteorological Conditions)', 
    2: 'LIMC (Low Instrument Meteorological Conditions)'
}

###############################################################################
# Weather-Specific Capacity Parameters for UAM
###############################################################################

# Define capacity constraints for each weather condition and operational side
# Side A: Optimized for arrivals (more arrivals, fewer departures)
# Side B: Optimized for departures (fewer arrivals, more departures)
# This creates operational tradeoff for wind/weather management

WEATHER_PARAMS = {
    0: {  # VMC - Best conditions
        1: {  # Side A: optimized for arrivals
            'max_arrivals': 9,
            'max_departures': 7,
            'max_capacity': 13,
            'lower_diagonal_bound': -4,
            'upper_diagonal_bound': 5
        },
        2: {  # Side B: optimized for departures
            'max_arrivals': 7,
            'max_departures': 9,
            'max_capacity': 13,
            'lower_diagonal_bound': -4,
            'upper_diagonal_bound': 5
        }
    },
    1: {  # IMC - Moderate conditions (reduced capacity)
        1: {  # Side A: optimized for arrivals
            'max_arrivals': 7,
            'max_departures': 5,
            'max_capacity': 10,
            'lower_diagonal_bound': -3,
            'upper_diagonal_bound': 4
        },
        2: {  # Side B: optimized for departures
            'max_arrivals': 5,
            'max_departures': 7,
            'max_capacity': 10,
            'lower_diagonal_bound': -3,
            'upper_diagonal_bound': 4
        }
    },
    2: {  # LIMC - Worst conditions (further reduced)
        1: {  # Side A: optimized for arrivals
            'max_arrivals': 5,
            'max_departures': 3,
            'max_capacity': 7,
            'lower_diagonal_bound': -2,
            'upper_diagonal_bound': 3
        },
        2: {  # Side B: optimized for departures
            'max_arrivals': 3,
            'max_departures': 5,
            'max_capacity': 7,
            'lower_diagonal_bound': -2,
            'upper_diagonal_bound': 3
        }
    }
}

###############################################################################
# Generate UAM Capacity Curves
###############################################################################

def generate_uam_curves():
    """
    Generate capacity curves for both operational sides of the vertiport across weather conditions.
    
    Returns:
    --------
    curves : dict
        Nested dictionary: curves[side_id][weather_id] = {
            'u': arrivals array,
            'v_min': minimum departures array,
            'v_max': maximum departures array,
            'analyzer': VertiportCapacityAnalyzer instance,
            'feasible_count': number of feasible solutions
        }
    """
    
    curves = {}
    
    print('\n' + '='*80)
    print('GENERATING UAM CAPACITY CURVES - SINGLE VERTIPORT, TWO OPERATIONAL SIDES')
    print('='*80)
    
    for config_id in UAM_CONFIGS.keys():
        print(f"\n{UAM_CONFIGS[config_id]}")
        print('-' * 80)
        
        curves[config_id] = {}
        
        for wc_id in WEATHER_CONDITIONS.keys():
            print(f"  {WEATHER_CONDITIONS[wc_id]}...", end=' ')
            
            # Initialize dictionary for this weather condition
            curves[config_id][wc_id] = {}
            
            # Get parameters for this weather condition and sides
            params_side_a = WEATHER_PARAMS[wc_id][1]
            params_side_b = WEATHER_PARAMS[wc_id][2]
            
            # Store parameters by side ID
            side_params = {1: params_side_a, 2: params_side_b}
            
            # Generate curves for both operational sides
            for side_id in [1, 2]:
                params = side_params[side_id]
                analyzer = VertiportCapacityAnalyzer(
                    min_arrivals=0,
                    min_departures=0,
                    max_arrivals=params['max_arrivals'],
                    max_departures=params['max_departures'],
                    max_capacity=params['max_capacity'],
                    lower_diagonal_bound=params['lower_diagonal_bound'],
                    upper_diagonal_bound=params['upper_diagonal_bound']
                )
                
                # Find feasible region
                analyzer.find_feasible_region()
                
                # Get bounds for each arrival count
                bounds = analyzer.get_feasible_bounds()
                
                if bounds is None:
                    print('FAILED - No feasible solutions')
                    continue
                
                arrivals_bounds = bounds['arrivals']
                
                # Extract u, v_min, v_max arrays
                u_values = np.array(sorted(list(arrivals_bounds.keys())), dtype=np.int64)
                v_min_values = np.array([arrivals_bounds[arr]['v_inf'] for arr in u_values], dtype=np.int64)
                v_max_values = np.array([arrivals_bounds[arr]['v_sup'] for arr in u_values], dtype=np.int64)
                
                # Store curves for this side
                curves[config_id][wc_id][side_id] = {
                    'u': u_values,
                    'v_min': v_min_values,
                    'v_max': v_max_values,
                    'analyzer': analyzer,
                    'feasible_count': len(analyzer.feasible_solutions)
                }
                
                side_label = 'A(Arr)' if side_id == 1 else 'B(Dep)'
                print(f"S{side_label}({u_values.max()},{v_max_values.max()}) ", end='')
            
            print()  # New line after each weather condition
    
    print('\n' + '='*80 + '\n')
    return curves


###############################################################################
# Visualization Functions
###############################################################################

def plot_single_curve(config_id, wc_id, side_id, curve_data, save_path=None):
    """
    Plot a single UAM capacity curve for one operational side and weather condition.
    Shows v_min and v_max departure bounds for each arrival count.
    
    Parameters:
    -----------
    config_id : int
        Configuration ID 
    wc_id : int
        Weather condition ID (0=VMC, 1=IMC, 2=LIMC)
    side_id : int
        Operational side ID (1=Side_A or 2=Side_B)
    curve_data : dict
        Dictionary containing 'u', 'v_min', 'v_max', 'analyzer'
    save_path : str, optional
        Path to save the figure
    """
    
    u = curve_data['u']
    v_min = curve_data['v_min']
    v_max = curve_data['v_max']
    analyzer = curve_data['analyzer']
    feasible_count = curve_data['feasible_count']
    
    # Get feasible solutions
    feasible_solutions = analyzer.feasible_solutions
    u_points = np.array([sol['u'] for sol in feasible_solutions])
    v_points = np.array([sol['v'] for sol in feasible_solutions])
    w_points = np.array([sol['w'] for sol in feasible_solutions])
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot feasible region as filled area between v_min and v_max
    ax.fill_between(u, v_min, v_max, alpha=0.3, color='cyan', 
                    label='Feasible Region', zorder=1, edgecolor='darkblue', linewidth=2)
    
    # Plot v_min boundary (lower diagonal)
    ax.plot(u, v_min, 'r-', linewidth=2.5, label=f'Min Departures (D ≥ A {analyzer.lower_diagonal_bound:+d})', zorder=2)
    
    # Plot v_max boundary (upper diagonal)
    ax.plot(u, v_max, 'g-', linewidth=2.5, label=f'Max Departures (D ≤ A {analyzer.upper_diagonal_bound:+d})', zorder=2)
    
    # Plot constraint boundaries
    ax_extended = np.linspace(-1, max(u) + 2, 200)
    
    # Additional constraints
    if analyzer.max_arrivals < max(u) + 2:
        ax.axvline(x=analyzer.max_arrivals, color='orange', linestyle='--', linewidth=2, 
                  label=f'Max Arrivals (A ≤ {analyzer.max_arrivals})', zorder=2)
    
    if analyzer.max_departures < max(v_max) + 2:
        ax.axhline(y=analyzer.max_departures, color='purple', linestyle='--', linewidth=2,
                  label=f'Max Departures (D ≤ {analyzer.max_departures})', zorder=2)
    
    # Plot feasible integer points
    scatter = ax.scatter(u_points, v_points, c=w_points, cmap='YlOrRd', 
                        s=100, alpha=0.8, edgecolors='darkred', linewidth=0.8,
                        label='Feasible integer points', zorder=4)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('TLOF Utilization (w)', fontsize=11, fontweight='bold')
    
    # Labels and formatting
    ax.set_xlabel('u: Number of Arrivals', fontsize=13, fontweight='bold')
    ax.set_ylabel('v: Number of Departures (Min-Max Bounds)', fontsize=13, fontweight='bold')
    side_label = 'Side A (Optimized for Arrivals)' if side_id == 1 else 'Side B (Optimized for Departures)'
    ax.set_title(f'Vertiport Capacity - {side_label}\n{WEATHER_CONDITIONS[wc_id]}', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    ax.set_xlim(left=-0.5, right=max(u) + 1)
    ax.set_ylim(bottom=-0.5, top=max(v_max) + 1)
    
    # Add statistics box
    stats_text = f'Feasible Solutions: {feasible_count}\n'
    stats_text += f'Arrivals Range: [0, {u.max()}]\n'
    stats_text += f'Departures Range: [0, {v_max.max()}]'
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='bottom', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"    Saved: {save_path}")
    
    plt.close()


def plot_all_curves(curves):
    """
    Generate visualization plots for all operational sides and weather conditions.
    """
    
    print('\n' + '='*80)
    print('GENERATING VISUALIZATIONS')
    print('='*80 + '\n')
    
    for config_id in curves.keys():
        print(f"{UAM_CONFIGS[config_id]}")
        for wc_id in curves[config_id].keys():
            for side_id in curves[config_id][wc_id].keys():
                save_path = f'uam_capacity_curve_config{config_id}_wc{wc_id}_side{side_id}.pdf'
                plot_single_curve(config_id, wc_id, side_id, curves[config_id][wc_id][side_id], save_path)
    
    print('\n' + '='*80 + '\n')


def plot_comparison_by_weather(curves):
    """
    Create comparison plots showing both operational sides for each weather condition.
    """
    
    print('\nGenerating weather condition comparison plots (Side A vs Side B)...\n')
    
    for wc_id in WEATHER_CONDITIONS.keys():
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Use config_id = 1 (any config, since all have the same sides)
        config_id = 1
        
        for idx, side_id in enumerate([1, 2]):
            if wc_id not in curves[config_id]:
                continue
            
            if side_id not in curves[config_id][wc_id]:
                continue
            
            curve_data = curves[config_id][wc_id][side_id]
            u = curve_data['u']
            v_min = curve_data['v_min']
            v_max = curve_data['v_max']
            
            # Get feasible solutions
            feasible_solutions = curve_data['analyzer'].feasible_solutions
            u_points = np.array([sol['u'] for sol in feasible_solutions])
            v_points = np.array([sol['v'] for sol in feasible_solutions])
            w_points = np.array([sol['w'] for sol in feasible_solutions])
            
            ax = axes[idx]
            
            # Fill feasible region
            ax.fill_between(u, v_min, v_max, alpha=0.3, color='cyan', 
                           label='Feasible Region', zorder=1, edgecolor='darkblue', linewidth=2)
            
            # Plot v_min and v_max
            ax.plot(u, v_min, 'r-', linewidth=2.5, label='Min Departures', zorder=2)
            ax.plot(u, v_max, 'g-', linewidth=2.5, label='Max Departures', zorder=2)
            
            # Plot feasible points
            scatter = ax.scatter(u_points, v_points, c=w_points, cmap='YlOrRd', 
                                s=80, alpha=0.8, edgecolors='darkred', linewidth=0.8,
                                zorder=4)
            
            ax.set_xlabel('Arrivals (u)', fontsize=11, fontweight='bold')
            ax.set_ylabel('Departures (v)', fontsize=11, fontweight='bold')
            side_name = 'Side A (Optimized for Arrivals)' if side_id == 1 else 'Side B (Optimized for Departures)'
            ax.set_title(f'{side_name}', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9, loc='upper left', framealpha=0.95)
            ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
        
        fig.suptitle(f'UAM Capacity Curves - {WEATHER_CONDITIONS[wc_id]}', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        save_path = f'uam_capacity_comparison_wc{wc_id}.pdf'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        plt.close()
    
    print()


def plot_all_curves_combined(curves):
    """
    Create a single plot showing all capacity curves (all weather conditions and sides).
    Shows both v_min (lower bound) and v_max (upper bound) for complete feasible envelope.
    Each curve set is shown with a unique color.
    """
    
    print('\nGenerating combined plot with all curves...\n')
    
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    
    # Define line styles for each operational side
    line_styles = {1: '-', 2: '--'}  # Solid for Side A, dashed for Side B
    line_widths = {1: 2.5, 2: 2.5}
    
    # Define unique colors for each weather-side combination (6 total colors)
    # Using a diverse color palette for maximum distinction
    colors_palette = {
        (0, 1): '#1f77b4',  # VMC - Side A - blue
        (0, 2): '#0055CC',  # VMC - Side B - dark blue
        (1, 1): '#ff7f0e',  # IMC - Side A - orange
        (1, 2): '#cc6600',  # IMC - Side B - dark orange
        (2, 1): '#d62728',  # LIMC - Side A - red
        (2, 2): '#8b0000'   # LIMC - Side B - dark red
    }
    
    weather_names = {
        0: 'VMC',
        1: 'IMC',
        2: 'LIMC'
    }
    
    side_names = {
        1: 'Arr.',
        2: 'Dep.'
    }
    
    # Use config_id = 1 (any config, since all have the same sides)
    config_id = 1
    
    # Plot both v_min and v_max curves for all weather and side combinations
    for wc_id in sorted(curves[config_id].keys()):
        for side_id in sorted(curves[config_id][wc_id].keys()):
            curve_data = curves[config_id][wc_id][side_id]
            u = curve_data['u']
            v_min = curve_data['v_min']
            v_max = curve_data['v_max']
            
            # Create label with weather + side info
            label = f"{weather_names[wc_id]} - {side_names[side_id]}"
            
            # Plot styling - get unique color for this weather-side combination
            color = colors_palette[(wc_id, side_id)]
            line_style = line_styles[side_id]
            marker = 'o' if side_id == 1 else 's'  # Circle for Side A, square for Side B
            
            # Plot v_max (upper bound) - solid or dashed line
            ax.plot(u, v_max, linewidth=2.0, color=color, 
                   linestyle=line_style, marker=marker, markersize=4.5, 
                   label=label, alpha=0.85, zorder=3)
            
            # Plot v_min (lower bound) - same color and line style, but with dots
            ax.plot(u, v_min, linewidth=1.4, color=color, 
                   linestyle=line_style, marker=marker, markersize=3.5, 
                   alpha=0.6, zorder=2)
            
            # Fill between v_min and v_max to show feasible envelope
            ax.fill_between(u, v_min, v_max, alpha=0.1, color=color, zorder=1)
    
    # Labels and formatting
    ax.set_xlabel('u: Number of Arrivals', fontsize=13)
    ax.set_ylabel('v: Number of Departures', fontsize=13)
    ax.set_title('Vertiport Capacity Curves', 
                fontsize=13, fontweight='bold', pad=8)
    ax.tick_params(axis='both', labelsize=10)
    
    # Create custom legend with both line styles and colors
    legend = ax.legend(fontsize=13, loc='center left', bbox_to_anchor=(1.02, 0.5),
                      framealpha=0.95, title='Weather - Side',
                      title_fontsize=13, ncol=1, borderaxespad=0.0,
                      handlelength=1.2)
    legend.get_frame().set_linewidth(1.5)
    
    ax.grid(True, alpha=0.4, linestyle=':', linewidth=0.8)
    ax.set_xlim(left=-0.5)
    ax.set_ylim(bottom=-0.5)
    
    plt.tight_layout()
    
    save_path = 'uam_capacity_curves_all_combined.pdf'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


###############################################################################
# Export Functions for Use in Main Code
###############################################################################

def export_curves_to_dicts(curves):
    """
    Export curves to the format expected by maskable_ppo_test_uam.py.
    
    Returns dictionaries structured as:
    u[wc][side_id] = array of arrivals
    v_min[wc][side_id] = array of minimum departures  
    v_max[wc][side_id] = array of maximum departures
    
    where side_id is 1 (A) or 2 (B)
    """
    
    u = {wc: {} for wc in WEATHER_CONDITIONS.keys()}
    v_min = {wc: {} for wc in WEATHER_CONDITIONS.keys()}
    v_max = {wc: {} for wc in WEATHER_CONDITIONS.keys()}
    
    for config_id in curves.keys():
        for wc_id in curves[config_id].keys():
            for side_id in curves[config_id][wc_id].keys():
                u[wc_id][side_id] = curves[config_id][wc_id][side_id]['u']
                v_min[wc_id][side_id] = curves[config_id][wc_id][side_id]['v_min']
                v_max[wc_id][side_id] = curves[config_id][wc_id][side_id]['v_max']
    
    return u, v_min, v_max


def export_curves_to_csv(curves):
    """
    Export all feasible solutions to CSV files for each side and weather.
    """
    
    print('\nExporting feasible solutions to CSV files...\n')
    
    for config_id in curves.keys():
        for wc_id in curves[config_id].keys():
            for side_id in curves[config_id][wc_id].keys():
                analyzer = curves[config_id][wc_id][side_id]['analyzer']
                df = pd.DataFrame(analyzer.feasible_solutions)
                df.columns = ['u (Arrivals)', 'v (Departures)', 'w (TLOF Util.)', 'w_gate (Gate Util.)']
                df = df.sort_values(by=['u (Arrivals)', 'v (Departures)']).reset_index(drop=True)
                
                save_path = f'uam_feasible_solutions_wc{wc_id}_side{side_id}.csv'
                df.to_csv(save_path, index=True, index_label='Index')
                print(f"  Saved: {save_path} ({len(df)} solutions)")


###############################################################################
# Main Execution
###############################################################################

if __name__ == "__main__":
    
    # Generate curves for all configurations and weather conditions
    curves = generate_uam_curves()
    
    # Create visualizations
    plot_all_curves(curves)
    plot_comparison_by_weather(curves)
    plot_all_curves_combined(curves)
    
    # Export to formats needed by main code
    u_dict, v_min_dict, v_max_dict = export_curves_to_dicts(curves)
    export_curves_to_csv(curves)
    
    # Print summary statistics
    print('='*80)
    print('SUMMARY OF GENERATED CURVES')
    print('='*80)
    
    for config_id in curves.keys():
        print(f"\n{UAM_CONFIGS[config_id]}")
        for wc_id in curves[config_id].keys():
            print(f"  {WEATHER_CONDITIONS[wc_id]:50s}")
            for side_id in curves[config_id][wc_id].keys():
                data = curves[config_id][wc_id][side_id]
                side_name = "Side A (Arrivals-opt)" if side_id == 1 else "Side B (Departures-opt)"
                print(f"    {side_name:35s}: u[0-{data['u'].max()}], v[{data['v_min'].min()}-{data['v_max'].max()}], {data['feasible_count']} solutions")
    
    print('\n' + '='*80)
    print('UAM Capacity Curves Generation Complete!')
    print('='*80 + '\n')
    
    # Display how to use the exported data
    print('To use these curves in maskable_ppo_test_uam.py:')
    print('\nfrom uam_capacity_curves import generate_uam_curves, export_curves_to_dicts')
    print('curves = generate_uam_curves()')
    print('u, v_min, v_max = export_curves_to_dicts(curves)')
    print('\nThen access curves as:')
    print('u[weather_cond][config_id]      # Array of arrivals')
    print('v_min[weather_cond][config_id]  # Array of minimum departures')
    print('v_max[weather_cond][config_id]  # Array of maximum departures')
