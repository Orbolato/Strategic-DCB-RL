"""
Vertiport Capacity Feasible Region Solver (Simulation-Based)

This script finds the feasible region using a simulation approach based on
the paper "Design Criteria and Accommodating Capacity Analysis of Vertiports 
for Urban Air Mobility".

Key Constraints:
1. Sum(Arrivals) - Sum(Departures) >= 0 (arrivals >= departures)
2. TLOF processing time: 60s (arrival) + 15s (taxi-in)
3. Gate processing time: 300s (turnaround) + 15s (taxi-out) + 60s (departure)
4. N_trd(t,n) <= (N_arr(t,n) + N_taxi(t))/2 (turnaround constraint)
5. Sequential operation constraints from simulation
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import deque

# Constants from the document
TIME_SEGMENTS = {
    'arrival': 60,      # seconds
    'taxi_in': 15,
    'turnaround': 300,
    'taxi_out': 15,
    'departure': 60
}

N_TLOF = 1  # Number of TLOF pads (paper example)
N_GATES = 4  # Number of gates (paper example)
OPERATION_TIME = 900  # seconds (15 minutes - matches paper example)

# Total cycle time per aircraft at TLOF
TLOF_CYCLE_TIME = (TIME_SEGMENTS['arrival'] + TIME_SEGMENTS['taxi_in'] + 
                   TIME_SEGMENTS['taxi_out'] + TIME_SEGMENTS['departure'])

# Total time per gate (excluding turnaround which happens at gate)
GATE_PROCESS_TIME = (TIME_SEGMENTS['taxi_in'] + TIME_SEGMENTS['turnaround'] + 
                     TIME_SEGMENTS['taxi_out'])

class VertiportCapacityAnalyzer:
    def __init__(self, operation_time=OPERATION_TIME, 
                 min_arrivals=0, min_departures=0,
                 max_arrivals=8, max_departures=8,
                 max_capacity=13,
                 lower_diagonal_bound=-4, upper_diagonal_bound=5):
        """
        Initialize the vertiport capacity analyzer with parametrized constraints.
        
        Parameters:
        -----------
        operation_time : int
            Total operation time in seconds (default: 900)
        
        min_arrivals : int
            Minimum number of arrivals allowed (default: 0)
            Constraint: A >= min_arrivals
        
        min_departures : int
            Minimum number of departures allowed (default: 0)
            Constraint: D >= min_departures
        
        max_arrivals : int
            Maximum number of arrivals allowed (default: 8)
            Constraint: A <= max_arrivals
        
        max_departures : int
            Maximum number of departures allowed (default: 8)
            Constraint: D <= max_departures
        
        max_capacity : int
            Maximum total operations (arrivals + departures) (default: 13)
            Constraint: A + D <= max_capacity
        
        lower_diagonal_bound : int
            Lower diagonal constraint bound (default: -4)
            Constraint: D >= A + lower_diagonal_bound
            Example: D >= A - 4 implies lower_diagonal_bound = -4
        
        upper_diagonal_bound : int
            Upper diagonal constraint bound (default: 5)
            Constraint: D <= A + upper_diagonal_bound
            Example: D <= A + 5 implies upper_diagonal_bound = 5
        """
        self.operation_time = operation_time
        
        # Bound constraints (inequality constraints on single variables)
        self.min_arrivals = min_arrivals
        self.min_departures = min_departures
        self.max_arrivals = max_arrivals
        self.max_departures = max_departures
        
        # Capacity and diagonal constraints
        self.max_capacity = max_capacity
        self.lower_diagonal_bound = lower_diagonal_bound      # D >= A + lower_diagonal_bound
        self.upper_diagonal_bound = upper_diagonal_bound      # D <= A + upper_diagonal_bound
        
        self.feasible_solutions = []
    
    def can_accommodate_operations(self, n_arrivals, n_departures):
        """
        Check feasibility using parametrized linear constraints.
        
        Constraints (all can be modified via constructor parameters):
        1. A >= min_arrivals (minimum arrivals)
        2. D >= min_departures (minimum departures)
        3. D >= A + lower_diagonal_bound (lower diagonal constraint)
        4. A <= max_arrivals (maximum arrivals)
        5. A + D <= max_capacity (maximum total capacity)
        6. D <= max_departures (maximum departures)
        7. D <= A + upper_diagonal_bound (upper diagonal constraint)
        
        All constraint values can be customized when instantiating the analyzer.
        """
        
        # Constraint 1: A >= min_arrivals
        if n_arrivals < self.min_arrivals:
            return False
        
        # Constraint 2: D >= min_departures
        if n_departures < self.min_departures:
            return False
        
        # Constraint 3: D >= A + lower_diagonal_bound (lower diagonal constraint)
        # Usually D >= A - 4, so lower_diagonal_bound = -4
        if n_departures < n_arrivals + self.lower_diagonal_bound:
            return False
        
        # Constraint 4: A <= max_arrivals
        # Usually A <= 8
        if n_arrivals > self.max_arrivals:
            return False
        
        # Constraint 5: A + D <= max_capacity (maximum total capacity)
        # Usually A + D <= 13
        if n_arrivals + n_departures > self.max_capacity:
            return False
        
        # Constraint 6: D <= max_departures
        # Usually D <= 8
        if n_departures > self.max_departures:
            return False
        
        # Constraint 7: D <= A + upper_diagonal_bound (upper diagonal constraint)
        # Usually D <= A + 5, so upper_diagonal_bound = 5
        if n_departures > n_arrivals + self.upper_diagonal_bound:
            return False
        
        return True
    
    def is_feasible(self, n_arrivals, n_departures):
        """Check if a solution is feasible"""
        return self.can_accommodate_operations(n_arrivals, n_departures)
    
    def find_feasible_region(self, max_arrivals=None, max_departures=None):
        """Find all feasible solutions within bounds"""
        
        if max_arrivals is None:
            # Maximum arrivals from constraint 4: A <= 8
            max_arrivals = 12  # Search a bit beyond to be safe
        
        if max_departures is None:
            # Maximum departures from constraint 5: D <= 8
            max_departures = 12  # Search a bit beyond to be safe
        
        self.feasible_solutions = []
        
        for n_arr in range(0, max_arrivals + 1):
            for n_dep in range(0, max_departures + 1):
                if self.is_feasible(n_arr, n_dep):
                    # Calculate TLOF utilization (w)
                    # w = (time for arrivals + time for departures) / total TLOF time
                    tlof_time_used = (n_arr * (TIME_SEGMENTS['arrival'] + TIME_SEGMENTS['taxi_in']) +
                                     n_dep * (TIME_SEGMENTS['taxi_out'] + TIME_SEGMENTS['departure']))
                    total_tlof_time = N_TLOF * self.operation_time
                    w_tlof = tlof_time_used / total_tlof_time
                    
                    # Also calculate gate utilization for reference
                    gate_time_used = (n_arr * TIME_SEGMENTS['turnaround'] +
                                     n_dep * (TIME_SEGMENTS['taxi_out'] + TIME_SEGMENTS['departure']))
                    total_gate_time = N_GATES * self.operation_time
                    w_gate = gate_time_used / total_gate_time
                    
                    self.feasible_solutions.append({
                        'u': n_arr,              # u: number of arrivals
                        'v': n_dep,              # v: number of departures  
                        'w': w_tlof,             # w: TLOF utilization rate
                        'w_gate': w_gate,        # gate utilization for reference
                    })
        
        return self.feasible_solutions
    
    def get_feasible_bounds(self):
        """
        Extract feasible departure bounds for each arrival count.
        
        Returns a dictionary with structure:
        {
            'u_max': maximum_arrivals,
            'v_max': maximum_departures,
            'arrivals': {
                0: {'v_inf': min_dep_at_0_arr, 'v_sup': max_dep_at_0_arr},
                1: {'v_inf': min_dep_at_1_arr, 'v_sup': max_dep_at_1_arr},
                ...
            }
        }
        
        This is useful for creating capacity curves where for each arrival count (u),
        there is a range of feasible departure counts [v_inf, v_sup].
        """
        if not self.feasible_solutions:
            print("Warning: No feasible solutions found. Run find_feasible_region() first.")
            return None
        
        # Group solutions by number of arrivals
        arrivals_dict = {}
        for sol in self.feasible_solutions:
            u = sol['u']
            v = sol['v']
            
            if u not in arrivals_dict:
                arrivals_dict[u] = {'v_inf': v, 'v_sup': v}
            else:
                arrivals_dict[u]['v_inf'] = min(arrivals_dict[u]['v_inf'], v)
                arrivals_dict[u]['v_sup'] = max(arrivals_dict[u]['v_sup'], v)
        
        # Get maximum values
        u_max = max(arrivals_dict.keys()) if arrivals_dict else 0
        v_max_overall = max(sol['v'] for sol in self.feasible_solutions) if self.feasible_solutions else 0
        
        return {
            'u_max': u_max,
            'v_max': v_max_overall,
            'arrivals': arrivals_dict
        }
    
    def plot_feasible_region(self, save_path='vertiport_feasible_region.png'):
        """
        Plot the feasible region bounded by 7 parametrized linear constraints.
        
        All constraints are dynamic based on instance parameters:
        - Constraint 1: A >= min_arrivals
        - Constraint 2: D >= min_departures
        - Constraint 3: D >= A + lower_diagonal_bound
        - Constraint 4: A <= max_arrivals
        - Constraint 5: A + D <= max_capacity
        - Constraint 6: D <= max_departures
        - Constraint 7: D <= A + upper_diagonal_bound
        """
        if not self.feasible_solutions:
            print("No feasible solutions found. Run find_feasible_region() first.")
            return
        
        u_values = [sol['u'] for sol in self.feasible_solutions]
        v_values = [sol['v'] for sol in self.feasible_solutions]
        w_values = [sol['w'] for sol in self.feasible_solutions]
        
        fig, ax = plt.subplots(figsize=(14, 11))
        
        # Create scatter plot with color gradient
        scatter = ax.scatter(u_values, v_values, c=w_values, cmap='YlOrRd', 
                            s=120, alpha=0.85, edgecolors='darkred', linewidth=1.0,
                            label='Feasible integer points', zorder=4)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('TLOF Utilization (w)', fontsize=11, fontweight='bold')
        
        # Helper function to find line intersections
        def line_intersection(a1, b1, c1, a2, b2, c2):
            """
            Find intersection of two lines: a1*x + b1*y = c1 and a2*x + b2*y = c2
            Returns (x, y) or None if parallel
            """
            det = a1 * b2 - a2 * b1
            if abs(det) < 1e-10:
                return None
            x = (c1 * b2 - c2 * b1) / det
            y = (a1 * c2 - a2 * c1) / det
            return (x, y)
        
        # Calculate vertices by finding intersections of constraint boundaries
        # Line equations in form: a*A + b*D = c
        
        # Constraint equations:
        # 1. A >= min_arrivals => A = min_arrivals (line: A - 0*D = min_arrivals)
        # 2. D >= min_departures => D = min_departures (line: 0*A + D = min_departures)
        # 3. D >= A + lower_diagonal => D - A = lower_diagonal (line: -A + D = lower_diagonal)
        # 4. A <= max_arrivals => A = max_arrivals (line: A - 0*D = max_arrivals)
        # 5. A + D <= max_capacity => A + D = max_capacity (line: A + D = max_capacity)
        # 6. D <= max_departures => D = max_departures (line: 0*A + D = max_departures)
        # 7. D <= A + upper_diagonal => D - A = upper_diagonal (line: -A + D = upper_diagonal)
        
        vertices = []
        
        # Intersection of constraint 2 (D=min_dep) and constraint 3 (D=A+lower_diag)
        p = line_intersection(1, -1, -self.lower_diagonal_bound, 0, 1, self.min_departures)
        if p: vertices.append(p)
        
        # Intersection of constraint 3 (D=A+lower_diag) and constraint 4 (A=max_arr)
        p = line_intersection(1, -1, -self.lower_diagonal_bound, 1, 0, self.max_arrivals)
        if p: vertices.append(p)
        
        # Intersection of constraint 4 (A=max_arr) and constraint 5 (A+D=max_cap)
        p = line_intersection(1, 0, self.max_arrivals, 1, 1, self.max_capacity)
        if p: vertices.append(p)
        
        # Intersection of constraint 5 (A+D=max_cap) and constraint 6 (D=max_dep)
        p = line_intersection(1, 1, self.max_capacity, 0, 1, self.max_departures)
        if p: vertices.append(p)
        
        # Intersection of constraint 6 (D=max_dep) and constraint 7 (D=A+upper_diag)
        p = line_intersection(0, 1, self.max_departures, 1, -1, -self.upper_diagonal_bound)
        if p: vertices.append(p)
        
        # Intersection of constraint 7 (D=A+upper_diag) and constraint 1 (A=min_arr)
        p = line_intersection(1, -1, -self.upper_diagonal_bound, 1, 0, self.min_arrivals)
        if p: vertices.append(p)
        
        # Intersection of constraint 1 (A=min_arr) and constraint 2 (D=min_dep)
        p = line_intersection(1, 0, self.min_arrivals, 0, 1, self.min_departures)
        if p: vertices.append(p)
        
        # Filter out invalid vertices (outside reasonable bounds)
        vertices = [v for v in vertices if v and v[0] >= -0.5 and v[1] >= -0.5 and 
                   v[0] <= self.max_arrivals + 2 and v[1] <= self.max_departures + 2]
        
        if vertices:
            vertices = np.array(vertices)
            
            # Fill region with more prominent opacity
            ax.fill(vertices[:, 0], vertices[:, 1], alpha=0.25, color='cyan', 
                   label='Feasible region (Dynamic polygon)', zorder=1, edgecolor='darkblue', linewidth=2.0)
            
            # Mark vertices with larger dots and labels
            for i, v in enumerate(vertices):
                ax.plot(v[0], v[1], 'ko', markersize=16, zorder=6, 
                       label='Polygon vertices' if i == 0 else '')
                ax.annotate(f'V{i+1}\n({v[0]:.1f}, {v[1]:.1f})', 
                           xy=v, xytext=(v[0]+0.5, v[1]+0.5), 
                           fontsize=8, fontweight='bold', ha='left',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8),
                           zorder=7)
        
        # Draw constraint boundaries
        ax_min = -1
        ax_max = max(self.max_arrivals, self.max_departures) + 2
        ax_range = np.linspace(ax_min, ax_max, 200)
        
        # Constraint 1: A = min_arrivals (vertical line)
        if self.min_arrivals > 0:
            ax.axvline(x=self.min_arrivals, color='black', linestyle='--', linewidth=2.0, zorder=3)
        
        # Constraint 2: D = min_departures (horizontal line)
        if self.min_departures > 0:
            ax.axhline(y=self.min_departures, color='black', linestyle='--', linewidth=2.0, zorder=3)
        
        # Constraint 3: D >= A + lower_diagonal_bound (lower diagonal) - RED
        d_lower = ax_range + self.lower_diagonal_bound
        ax.plot(ax_range, d_lower, 'r-', linewidth=2.5, 
               label=f'Constraint: D ≥ A {self.lower_diagonal_bound:+d} (Lower diagonal)', zorder=3)
        
        # Constraint 4: A <= max_arrivals (vertical line) - GREEN
        ax.axvline(x=self.max_arrivals, color='green', linestyle='-', linewidth=2.5, 
                  label=f'Constraint: A ≤ {self.max_arrivals} (Max arrivals)', zorder=3)
        
        # Constraint 5: D <= max_departures (horizontal line) - PURPLE
        ax.axhline(y=self.max_departures, color='purple', linestyle='-', linewidth=2.5,
                  label=f'Constraint: D ≤ {self.max_departures} (Max departures)', zorder=3)
        
        # Constraint 6: A + D <= max_capacity (diagonal line) - ORANGE
        a_total = np.linspace(ax_min, ax_max, 100)
        d_total = self.max_capacity - a_total
        ax.plot(a_total, d_total, 'orange', linewidth=2.5, linestyle='-', 
               label=f'Constraint: A + D ≤ {self.max_capacity} (Max capacity)', zorder=3)
        
        # Constraint 7: D <= A + upper_diagonal_bound (upper diagonal) - BROWN/MAROON
        d_upper = ax_range + self.upper_diagonal_bound
        ax.plot(ax_range, d_upper, color='brown', linewidth=2.5, linestyle='-',
               label=f'Constraint: D ≤ A {self.upper_diagonal_bound:+d} (Upper diagonal)', zorder=3)
        
        ax.set_xlabel('A: Number of Arrivals', fontsize=13, fontweight='bold')
        ax.set_ylabel('D: Number of Departures', fontsize=13, fontweight='bold')
        ax.set_title(f'Vertiport Capacity Feasible Region (Parametrized)\n({N_TLOF} TLOF, {N_GATES} Gates, {self.operation_time/60:.0f} Min Operation)', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right', framealpha=0.95)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
        ax.set_xlim(left=ax_min, right=ax_max)
        ax.set_ylim(bottom=ax_min, top=ax_max)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nFeasible region plot saved to: {save_path}")
        plt.close()
        
        return fig, ax
    
    
    def generate_solution_table(self, save_path='vertiport_feasible_solutions.csv'):
        """Generate a CSV table with all feasible solutions"""
        if not self.feasible_solutions:
            print("No feasible solutions found. Run find_feasible_region() first.")
            return
        
        df = pd.DataFrame(self.feasible_solutions)
        df.columns = ['u (Arrivals)', 'v (Departures)', 'w (TLOF Util.)', 'w_gate (Gate Util.)']
        
        # Sort by arrivals then departures
        df = df.sort_values(by=['u (Arrivals)', 'v (Departures)']).reset_index(drop=True)
        
        # Save to CSV
        df.to_csv(save_path, index=True, index_label='Index')
        print(f"Feasible solutions table saved to: {save_path}")
        
        return df
    
    def generate_summary_report(self):
        """Generate a summary report of the feasible region"""
        if not self.feasible_solutions:
            print("No feasible solutions found.")
            return
        
        df = pd.DataFrame(self.feasible_solutions)
        
        print("\n" + "="*70)
        print("VERTIPORT CAPACITY FEASIBLE REGION ANALYSIS")
        print("="*70)
        print(f"\nOperation Time: {self.operation_time} seconds ({self.operation_time/3600:.2f} hours)")
        print(f"TLOF Pads: {N_TLOF}")
        print(f"Gates: {N_GATES}")
        print(f"\nTime Segments:")
        for segment, time in TIME_SEGMENTS.items():
            print(f"  {segment}: {time} seconds")
        
        print(f"\nTotal feasible solutions: {len(self.feasible_solutions)}")
        
        print(f"\nArrivals (u):")
        print(f"  Minimum: {int(df['u'].min())}")
        print(f"  Maximum: {int(df['u'].max())}")
        print(f"  Average: {df['u'].mean():.2f}")
        
        print(f"\nDepartures (v):")
        print(f"  Minimum: {int(df['v'].min())}")
        print(f"  Maximum: {int(df['v'].max())}")
        print(f"  Average: {df['v'].mean():.2f}")
        
        print(f"\nTLOF Utilization (w):")
        print(f"  Minimum: {df['w'].min():.4f}")
        print(f"  Maximum: {df['w'].max():.4f}")
        print(f"  Average: {df['w'].mean():.4f}")
        
        print("\n" + "="*70)
        print("ARRIVALS vs MAX/MIN DEPARTURES")
        print("="*70)
        
        arrivals_groups = df.groupby('u')['v'].agg(['min', 'max'])
        for arrivals, (min_dep, max_dep) in arrivals_groups.iterrows():
            print(f"Arrivals (u): {int(arrivals):2d} -> Departures (v): min={int(min_dep):2d}, max={int(max_dep):2d}")


def main():
    """Main execution with default paper parameters"""
    print("\n" + "="*80)
    print("PARAMETRIZED CONSTRAINT EXAMPLE - Using Default Paper Values")
    print("="*80)
    
    # Initialize analyzer with DEFAULT PAPER parameters
    # You can easily change these parameters to get different feasible regions!
    analyzer = VertiportCapacityAnalyzer(
        operation_time=OPERATION_TIME,
        min_arrivals=0,                 # A >= P
        min_departures=0,               # D >= Q
        max_arrivals=8,                 # A <= R
        max_departures=6,               # D <= S
        max_capacity=13,                # A + D <= T
        lower_diagonal_bound=-4,        # D >= A + L
        upper_diagonal_bound=0          # D <= A + U
    )
    
    # Find feasible region
    print("\nFinding feasible region with current constraints...")
    analyzer.find_feasible_region()
    
    # Generate summary report
    analyzer.generate_summary_report()
    
    # Save solutions to CSV
    df = analyzer.generate_solution_table('vertiport_feasible_solutions.csv')
    print("\nFirst 10 solutions:")
    print(df.head(10))
    
    # Plot feasible region
    analyzer.plot_feasible_region('vertiport_feasible_region.png')
    
    print("\nAnalysis complete!")
    
    # Print instructions for custom constraints
    print("\n" + "="*80)
    print("HOW TO USE CUSTOM CONSTRAINTS")
    print("="*80)
    print("""
EXAMPLE 1: Change max capacity from 13 to 15
    analyzer = VertiportCapacityAnalyzer(max_capacity=15)
    analyzer.find_feasible_region()
    analyzer.plot_feasible_region('custom_region.png')

EXAMPLE 2: Restrict to only 5 departures maximum
    analyzer = VertiportCapacityAnalyzer(max_departures=5)
    analyzer.find_feasible_region()
    analyzer.plot_feasible_region('custom_region.png')

EXAMPLE 3: Change diagonal constraints
    analyzer = VertiportCapacityAnalyzer(
        lower_diagonal_bound=-3,        # D >= A - 3 (less restrictive)
        upper_diagonal_bound=6          # D <= A + 6 (more permissive)
    )
    analyzer.find_feasible_region()
    analyzer.plot_feasible_region('custom_region.png')

EXAMPLE 4: Completely different configuration
    analyzer = VertiportCapacityAnalyzer(
        min_arrivals=1,                 # Start from 1 arrival
        max_arrivals=10,                # Up to 10 arrivals
        max_departures=10,              # Up to 10 departures
        max_capacity=15,                # Max capacity is 15
        lower_diagonal_bound=-5,        # D >= A - 5
        upper_diagonal_bound=3          # D <= A + 3
    )
    analyzer.find_feasible_region()
    analyzer.plot_feasible_region('custom_region.png')

AVAILABLE PARAMETERS:
    - operation_time: Total operation time (seconds)
    - min_arrivals: Minimum arrivals (A >= value)
    - min_departures: Minimum departures (D >= value)
    - max_arrivals: Maximum arrivals (A <= value)
    - max_departures: Maximum departures (D <= value)
    - max_capacity: Maximum total (A + D <= value)
    - lower_diagonal_bound: Lower diagonal (D >= A + value)
    - upper_diagonal_bound: Upper diagonal (D <= A + value)
""")


if __name__ == "__main__":
    main()
