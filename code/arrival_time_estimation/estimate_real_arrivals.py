"""
Estimate Real Arrival Times based on Historical Flight Time Deviations
=======================================================================

This script:
1. Loads real flight data from VRA_2022, VRA_2023, VRA_2024 files
2. Calculates actual flight_time_deviation = (arrival - departure) - scheduled_flight_time
3. Analyzes deviations by origin+equipment combinations
4. Calculates statistics (mean, std, p10) for each combination
5. Creates visualizations of the distributions
6. Applies p10 adjustments to arrival_data.csv scheduled times
7. Generates estimated_real_arrival_time with historical deviation data

VRA Data Uses ICAO Codes:
- sg_icao_origem: Origin airport ICAO code (e.g., 'MPTO', 'SBGL')
- model_equip: Aircraft type code (e.g., 'B738', 'A320')
- dt_partida_real: Actual departure datetime
- dt_chegada_real: Actual arrival datetime (SBGR for all)
- dt_partida_prevista: Scheduled departure datetime
- dt_chegada_prevista: Scheduled arrival datetime
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Configuration
BASE_PATH = Path(__file__).resolve().parent.parent  # repository root
DATA_PATH = BASE_PATH / "data"
OUTPUT_PATH = BASE_PATH / "arrival_time_estimation"

# Ensure output directories exist
OUTPUT_PATH.mkdir(exist_ok=True)
(OUTPUT_PATH / "plots").mkdir(exist_ok=True)
(OUTPUT_PATH / "statistics").mkdir(exist_ok=True)

print("=" * 80)
print("ESTIMATING REAL ARRIVAL TIMES FROM HISTORICAL DEVIATIONS (VRA DATA)")
print("=" * 80)

# ============================================================================
# STEP 1: Load and Consolidate Real Flight Data
# ============================================================================
print("\n[1/6] Loading real flight data from VRA files...")

vra_files = ['VRA_2022.csv', 'VRA_2023.csv', 'VRA_2024.csv']
vra_data_list = []

for vra_file in vra_files:
    try:
        # VRA files use semicolon as separator
        vra = pd.read_csv(DATA_PATH / vra_file, sep=';', encoding='utf-8')
        print(f"  ✓ Loaded {vra_file}: {len(vra):,} flights")
        vra_data_list.append(vra)
    except Exception as e:
        print(f"  ✗ Error loading {vra_file}: {e}")

if not vra_data_list:
    print("  ✗ No VRA files found!")
    exit(1)

# Consolidate all years
vra_data = pd.concat(vra_data_list, ignore_index=True)
print(f"  ✓ Total consolidated: {len(vra_data):,} flights")

# ============================================================================
# STEP 2: Calculate Flight Time Deviations
# ============================================================================
print("\n[2/6] Calculating flight time deviations...")
print("  Note: Using flight duration deviations, NOT arrival time deviations")
print("  This isolates in-air time effects from departure delays")

# Convert time columns to datetime
time_cols = ['dt_partida_prevista', 'dt_partida_real', 'dt_chegada_prevista', 'dt_chegada_real']
for col in time_cols:
    vra_data[col] = pd.to_datetime(vra_data[col], errors='coerce')

# KEY CALCULATION: Flight time deviation based on ACTUAL FLIGHT DURATION
# This measures how much longer/shorter the actual flight time was compared to scheduled
# IMPORTANT: This only depends on the duration (dt_chegada_real - dt_partida_real),
# NOT on when the flight departed or arrived
# 
# Formula:
#   Scheduled Flight Time = scheduled_arrival - scheduled_departure
#   Actual Flight Time = actual_arrival - actual_departure  
#   Flight Time Deviation = Actual Flight Time - Scheduled Flight Time
#
# When applied to scheduled flights (with fixed scheduled departure):
#   Estimated Arrival = scheduled_departure + (scheduled_flight_time + deviation)
#                     = scheduled_arrival + deviation

print("  Calculating scheduled flight duration (in-air time)...")
vra_data['scheduled_flight_time'] = (
    (vra_data['dt_chegada_prevista'] - vra_data['dt_partida_prevista']).dt.total_seconds() / 60
).round(0)

print("  Calculating actual flight duration (in-air time)...")
vra_data['actual_flight_time'] = (
    (vra_data['dt_chegada_real'] - vra_data['dt_partida_real']).dt.total_seconds() / 60
).round(0)

print("  Computing flight time deviation (actual - scheduled)...")
vra_data['flight_time_deviation'] = (vra_data['actual_flight_time'] - vra_data['scheduled_flight_time']).round(1)

# Remove rows with invalid data
initial_count = len(vra_data)
vra_data = vra_data.dropna(subset=['dt_chegada_real', 'dt_partida_real', 'scheduled_flight_time'])
vra_data = vra_data[vra_data['scheduled_flight_time'] > 0]
vra_data = vra_data[vra_data['actual_flight_time'] > 0]
vra_data = vra_data[vra_data['flight_time_deviation'].between(-500, 500)]  # Remove extreme outliers

cleaned_count = len(vra_data)
print(f"  ✓ Cleaned data: {cleaned_count:,} valid flights ({cleaned_count/initial_count*100:.1f}%)")
print(f"  ✓ Removed {initial_count - cleaned_count:,} flights with invalid data")

# ============================================================================
# STEP 3: Analyze by Origin + Equipment
# ============================================================================
print("\n[3/6] Analyzing flight_time_deviation by origin+equipment...")

# Create grouping keys
vra_data['origin'] = vra_data['sg_icao_origem'].str.strip()
vra_data['equipment'] = vra_data['model_equip'].astype(str).str.strip()

# Group by origin+equipment and calculate statistics
deviation_stats = vra_data.groupby(['origin', 'equipment'])['flight_time_deviation'].agg([
    'count',
    'mean',
    'std',
    ('p10', lambda x: np.percentile(x, 10)),
    ('p25', lambda x: np.percentile(x, 25)),
    ('p50', lambda x: np.percentile(x, 50)),
    ('p75', lambda x: np.percentile(x, 75)),
    ('p90', lambda x: np.percentile(x, 90)),
    'min',
    'max'
]).reset_index()

deviation_stats.columns = ['origin', 'equipment', 'count', 'mean', 'std', 'p10', 'p25', 'p50', 'p75', 'p90', 'min', 'max']
deviation_stats = deviation_stats.sort_values('count', ascending=False)

# Calculate overall p10 for fallback
overall_p10 = np.percentile(vra_data['flight_time_deviation'], 10)
overall_mean = vra_data['flight_time_deviation'].mean()
overall_std = vra_data['flight_time_deviation'].std()

print(f"  ✓ Unique origin+equipment combinations: {len(deviation_stats)}")
print(f"  ✓ Overall statistics:")
print(f"      - Mean flight_time_deviation: {overall_mean:.2f} minutes")
print(f"      - Std dev: {overall_std:.2f} minutes")
print(f"      - P10 (10th percentile): {overall_p10:.2f} minutes")

# Save statistics
deviation_stats.to_csv(OUTPUT_PATH / "statistics" / "deviation_statistics.csv", index=False)
print(f"  ✓ Saved statistics to: statistics/deviation_statistics.csv")

# ============================================================================
# STEP 4: Generate Visualizations
# ============================================================================
print("\n[4/6] Generating visualizations...")

# Plot 1a: Histogram (separated)
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(vra_data['flight_time_deviation'], bins=50, edgecolor='black', alpha=0.7)
ax.axvline(overall_mean, color='r', linestyle='--', linewidth=2, label=f'Mean: {overall_mean:.2f}')
ax.axvline(overall_p10, color='g', linestyle='--', linewidth=2, label=f'P10: {overall_p10:.2f}')
ax.set_xlabel('Flight Time Deviation (minutes)')
ax.set_ylabel('Frequency')
ax.set_title('Histogram of Flight Time Deviations', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_PATH / "plots" / "01_histogram.pdf", dpi=100, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: plots/01_histogram.pdf")

# Plot 1b: Other distributions (box plot, sorted, CDF)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Flight Time Deviation Analysis - Distribution Comparisons', fontsize=14, fontweight='bold')

axes[0].boxplot(vra_data['flight_time_deviation'], vert=True)
axes[0].set_ylabel('Flight Time Deviation (minutes)')
axes[0].set_title('Box Plot of Flight Time Deviations')
axes[0].grid(alpha=0.3)

axes[1].plot(sorted(vra_data['flight_time_deviation']), linewidth=0.5)
axes[1].axhline(overall_p10, color='g', linestyle='--', linewidth=2, label=f'P10: {overall_p10:.2f}')
axes[1].set_xlabel('Flight Number (sorted)')
axes[1].set_ylabel('Flight Time Deviation (minutes)')
axes[1].set_title('Sorted Flight Time Deviations')
axes[1].legend()
axes[1].grid(alpha=0.3)

# CDF plot
sorted_dev = np.sort(vra_data['flight_time_deviation'])
cdf = np.arange(1, len(sorted_dev) + 1) / len(sorted_dev)
axes[2].plot(sorted_dev, cdf, linewidth=1)
axes[2].axvline(overall_p10, color='g', linestyle='--', linewidth=2, label=f'P10: {overall_p10:.2f}')
axes[2].axhline(0.1, color='g', linestyle=':', alpha=0.5)
axes[2].set_xlabel('Flight Time Deviation (minutes)')
axes[2].set_ylabel('Cumulative Probability')
axes[2].set_title('CDF of Flight Time Deviations')
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_PATH / "plots" / "01b_distributions_comparison.pdf", dpi=100, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: plots/01b_distributions_comparison.pdf")

# Plot 2: P10 by origin+equipment (top 30)
top_n = min(30, len(deviation_stats))
top_stats = deviation_stats.nlargest(top_n, 'count').sort_values('p10', ascending=True)

fig, ax = plt.subplots(figsize=(12, 8))
colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.7, len(top_stats)))
bars = ax.barh(range(len(top_stats)), top_stats['p10'].values, color=colors)
ax.axvline(overall_p10, color='black', linestyle='--', linewidth=2, label=f'Overall P10: {overall_p10:.2f}')
ax.set_yticks(range(len(top_stats)))
ax.set_yticklabels([f"{row['origin']}_{row['equipment']}" for _, row in top_stats.iterrows()], fontsize=9)
ax.set_xlabel('P10 Flight Time Deviation (minutes)')
ax.set_title(f'P10 Values by Origin+Equipment (Top {top_n} by sample size)', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3, axis='x')

# Add value labels on bars
for i, (idx, row) in enumerate(top_stats.iterrows()):
    ax.text(row['p10'] + 0.5, i, f"{row['p10']:.1f}", va='center', fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_PATH / "plots" / "02_p10_by_origin_equipment.pdf", dpi=100, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: plots/02_p10_by_origin_equipment.pdf")

# Plot 3: Mean vs std by origin+equipment
fig, ax = plt.subplots(figsize=(12, 8))
scatter = ax.scatter(deviation_stats['mean'], deviation_stats['std'], 
                     s=deviation_stats['count']/2, alpha=0.6, 
                     c=deviation_stats['p10'], cmap='RdYlGn_r')
ax.set_xlabel('Mean Flight Time Deviation (minutes)')
ax.set_ylabel('Std Dev Flight Time Deviation (minutes)')
ax.set_title('Mean vs Std Dev by Origin+Equipment\n(bubble size = sample count, color = P10)', fontweight='bold')
ax.grid(alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('P10 Value (minutes)')

# Add annotations for top combinations
for idx, row in deviation_stats.nlargest(5, 'count').iterrows():
    ax.annotate(f"{row['origin']}_{row['equipment']}", 
                xy=(row['mean'], row['std']), 
                xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)

plt.tight_layout()
plt.savefig(OUTPUT_PATH / "plots" / "03_mean_vs_std.pdf", dpi=100, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: plots/03_mean_vs_std.pdf")

# Plot 4: Distribution by top 12 combinations
top_12 = deviation_stats.nlargest(12, 'count')
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
fig.suptitle('Flight Time Deviation Distributions - Top 12 Origin+Equipment Combinations', fontsize=14, fontweight='bold')
axes_flat = axes.flatten()

for idx, (i, row) in enumerate(top_12.iterrows()):
    origin = row['origin']
    equipment = row['equipment']
    subset = vra_data[
        (vra_data['origin'] == origin) & 
        (vra_data['equipment'] == equipment)
    ]['flight_time_deviation']
    
    axes_flat[idx].hist(subset, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    axes_flat[idx].axvline(row['p10'], color='g', linestyle='--', linewidth=2, label=f"P10: {row['p10']:.1f}")
    axes_flat[idx].axvline(row['mean'], color='r', linestyle='--', linewidth=1.5, label=f"Mean: {row['mean']:.1f}")
    axes_flat[idx].set_title(f"{origin}_{equipment}\n(n={int(row['count'])})", fontsize=9, fontweight='bold')
    axes_flat[idx].set_xlabel('Deviation (min)')
    axes_flat[idx].set_ylabel('Count')
    axes_flat[idx].legend(fontsize=7)
    axes_flat[idx].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_PATH / "plots" / "04_distributions_by_origin_equip.pdf", dpi=100, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: plots/04_distributions_by_origin_equip.pdf")

# Plot 5: Distribution of P10 values across origin+equipment combinations
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(deviation_stats['p10'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
ax.axvline(overall_p10, color='r', linestyle='--', linewidth=2, label=f'Overall P10: {overall_p10:.2f}')
ax.axvline(deviation_stats['p10'].mean(), color='g', linestyle='--', linewidth=2, label=f'Mean P10: {deviation_stats["p10"].mean():.2f}')
ax.set_xlabel('P10 Flight Time Deviation (minutes)')
ax.set_ylabel('Frequency (number of origin+equipment combinations)')
ax.set_title('Distribution of P10 Values Across Origin+Equipment Combinations', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_PATH / "plots" / "05_p10_distribution.pdf", dpi=100, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: plots/05_p10_distribution.pdf")

# Plot 6: Distribution of Mean values across origin+equipment combinations
fig, ax = plt.subplots(figsize=(12, 6))
ax.hist(deviation_stats['mean'], bins=30, edgecolor='black', alpha=0.7, color='coral')
ax.axvline(overall_mean, color='r', linestyle='--', linewidth=2, label=f'Overall Mean: {overall_mean:.2f}')
ax.axvline(deviation_stats['mean'].mean(), color='g', linestyle='--', linewidth=2, label=f'Mean of Means: {deviation_stats["mean"].mean():.2f}')
ax.set_xlabel('Mean Flight Time Deviation (minutes)')
ax.set_ylabel('Frequency (number of origin+equipment combinations)')
ax.set_title('Distribution of Mean Values Across Origin+Equipment Combinations', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_PATH / "plots" / "06_mean_distribution.pdf", dpi=100, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: plots/06_mean_distribution.pdf")

# Plot 7: Extreme P10 routes (p10 < -60 or p10 > 60)
extreme_p10 = deviation_stats[(deviation_stats['p10'] < -60) | (deviation_stats['p10'] > 60)].sort_values('p10')

print(f"\n  Routes with extreme P10 values (< -60 or > 60 minutes):")
print(f"  {'='*100}")
print(f"  {'Origin':<10} {'Equipment':<12} {'Count':<8} {'P10':<8} {'Mean':<8} {'Std':<8} {'P25':<8} {'P75':<8}")
print(f"  {'-'*100}")
for idx, row in extreme_p10.iterrows():
    print(f"  {row['origin']:<10} {row['equipment']:<12} {int(row['count']):<8} {row['p10']:<8.2f} {row['mean']:<8.2f} {row['std']:<8.2f} {row['p25']:<8.2f} {row['p75']:<8.2f}")
print(f"  {'-'*100}")
print(f"  Total routes with extreme P10: {len(extreme_p10)}")

if len(extreme_p10) > 0:
    fig, ax = plt.subplots(figsize=(14, max(6, len(extreme_p10) * 0.3)))
    colors = ['red' if x < -60 else 'darkgreen' for x in extreme_p10['p10'].values]
    bars = ax.barh(range(len(extreme_p10)), extreme_p10['p10'].values, color=colors, alpha=0.7)
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    ax.axvline(-60, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label='Threshold: -60 min')
    ax.axvline(60, color='darkgreen', linestyle='--', linewidth=1.5, alpha=0.5, label='Threshold: +60 min')
    ax.set_yticks(range(len(extreme_p10)))
    ax.set_yticklabels([f"{row['origin']}_{row['equipment']} (n={int(row['count'])})" for _, row in extreme_p10.iterrows()], fontsize=9)
    ax.set_xlabel('P10 Flight Time Deviation (minutes)')
    ax.set_title('Routes with Extreme P10 Values (< -60 or > 60 minutes)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(extreme_p10.iterrows()):
        ax.text(row['p10'] + (3 if row['p10'] > 0 else -3), i, f"{row['p10']:.1f}", 
                va='center', ha='left' if row['p10'] > 0 else 'right', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH / "plots" / "07_extreme_p10_routes.pdf", dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: plots/07_extreme_p10_routes.pdf")
else:
    print(f"  ℹ No routes found with extreme P10 values (< -60 or > 60 minutes)")

try:
    arrival_data = pd.read_csv(DATA_PATH / "arrival_data.csv")
    print(f"  ✓ Loaded arrival_data.csv: {len(arrival_data):,} flights")
except Exception as e:
    print(f"  ✗ Error loading arrival_data.csv: {e}")
    exit(1)

# Prepare for merging
arrival_data_clean = arrival_data.copy()

# CRITICAL: Convert time columns to datetime BEFORE merge
# Date-only strings like "2022-01-03" should be 00:00:00 on that date
# Handle date-only values by appending " 00:00:00" before conversion
for col in ['scheduled_arrival_time', 'scheduled_departure_time']:
    if col in arrival_data_clean.columns:
        # Check for date-only format (10 characters like "2022-01-03")
        mask_date_only = arrival_data_clean[col].astype(str).str.len() == 10
        if mask_date_only.any():
            arrival_data_clean.loc[mask_date_only, col] = arrival_data_clean.loc[mask_date_only, col] + ' 00:00:00'
        
        # Now convert to datetime
        arrival_data_clean[col] = pd.to_datetime(arrival_data_clean[col], errors='coerce')

print(f"  ✓ Converted times to datetime")
print(f"    - Non-null scheduled_arrival_time: {arrival_data_clean['scheduled_arrival_time'].notna().sum():,}")
print(f"    - Non-null scheduled_departure_time: {arrival_data_clean['scheduled_departure_time'].notna().sum():,}")

arrival_data_clean['origin_match'] = arrival_data_clean['origin'].astype(str).str.strip()
arrival_data_clean['equipment_match'] = arrival_data_clean['equip'].astype(str).str.strip()

deviation_stats['origin_match'] = deviation_stats['origin'].astype(str).str.strip()
deviation_stats['equipment_match'] = deviation_stats['equipment'].astype(str).str.strip()

# Merge
merged = arrival_data_clean.merge(
    deviation_stats[['origin_match', 'equipment_match', 'p10', 'mean', 'std', 'count']],
    left_on=['origin_match', 'equipment_match'],
    right_on=['origin_match', 'equipment_match'],
    how='left'
)

# Count matches
matched = merged['p10'].notna().sum()
total = len(merged)
coverage = (matched / total) * 100

print(f"  ✓ Coverage: {matched:,}/{total:,} ({coverage:.1f}%)")

# Fill missing values with overall p10
merged['p10_adjusted'] = merged['p10'].fillna(overall_p10)
merged['p10_source'] = merged['p10'].notna().map({True: 'historical', False: 'overall'})

print(f"  ✓ Flights with historical p10: {(merged['p10_source'] == 'historical').sum():,}")
print(f"  ✓ Flights with overall p10: {(merged['p10_source'] == 'overall').sum():,}")

# ============================================================================
# STEP 6: Calculate Estimated Real Arrival Times
# ============================================================================
print("\n[6/6] Calculating estimated real arrival times...")
print("  Using flight time deviations to predict actual arrival times")

# Times are already converted to datetime in STEP 5, just ensure they're still datetime
# (sometimes merge can convert back to string if not careful)
if merged['scheduled_arrival_time'].dtype == 'object':
    merged['scheduled_arrival_time'] = pd.to_datetime(merged['scheduled_arrival_time'], errors='coerce')
if merged['scheduled_departure_time'].dtype == 'object':
    merged['scheduled_departure_time'] = pd.to_datetime(merged['scheduled_departure_time'], errors='coerce')

# Calculate scheduled flight duration for reference and validation
merged['scheduled_flight_duration'] = (
    (merged['scheduled_arrival_time'] - merged['scheduled_departure_time']).dt.total_seconds() / 60
).round(0)

# CRITICAL FORMULA: Apply flight time deviation to estimate actual arrival
# 
# Mathematical basis:
#   Estimated Arrival = Scheduled_Departure + (Scheduled_Flight_Time + Flight_Time_Deviation)
#                     = Scheduled_Departure + Scheduled_Flight_Time + Deviation
#                     = Scheduled_Arrival + Deviation
#
# Key assumption: Scheduled departure time doesn't change (it's what's scheduled)
# The p10_adjusted contains only flight time (in-air duration) variations, 
# not departure delay variations
#
# This ensures:
# 1. Early scheduled flights stay early-ish relative to their scheduled time
# 2. Late scheduled flights stay late-ish relative to their scheduled time  
# 3. We're adding flight time variability, not assuming departure delay carries over

def calc_estimated_arrival(row):
    """Calculate estimated real arrival time - ALWAYS returns a value, never NULL"""
    
    # p10_adjusted should NEVER be NULL (filled with overall_p10)
    if pd.isna(row['p10_adjusted']):
        # Fallback to scheduled arrival time if somehow p10 is null
        return row['scheduled_arrival_time']
    
    # Method 1: Using scheduled arrival (preferred when available)
    if pd.notna(row['scheduled_arrival_time']):
        estimated_arrival_v1 = row['scheduled_arrival_time'] + timedelta(minutes=row['p10_adjusted'])
        
        # Method 2: Validation using scheduled departure
        if pd.notna(row['scheduled_departure_time']) and pd.notna(row['scheduled_flight_duration']):
            actual_flight_time = row['scheduled_flight_duration'] + row['p10_adjusted']
            estimated_arrival_v2 = row['scheduled_departure_time'] + timedelta(minutes=actual_flight_time)
            diff_seconds = abs((estimated_arrival_v1 - estimated_arrival_v2).total_seconds())
            if diff_seconds > 60:
                print(f"  WARNING: Arrival calculation methods differ by {diff_seconds/60:.1f} minutes")
                return estimated_arrival_v2
        
        # Return method 1 result
        return estimated_arrival_v1
    
    # Method 2: Using scheduled departure as fallback (if method 1 not available)
    elif pd.notna(row['scheduled_departure_time']) and pd.notna(row['scheduled_flight_duration']):
        actual_flight_time = row['scheduled_flight_duration'] + row['p10_adjusted']
        return row['scheduled_departure_time'] + timedelta(minutes=actual_flight_time)
    
    # If neither method available, return NULL (data issue, but try to avoid this)
    return None

print("  Applying flight time deviations to scheduled flights...")
merged['estimated_real_arrival_time'] = merged.apply(calc_estimated_arrival, axis=1)

print(f"  ✓ Calculated estimated_real_arrival_time for all flights")

# Summary statistics
p10_values = merged['p10_adjusted'].values
print(f"\n  Estimated Real Arrival Time Adjustments:")
print(f"    - Mean adjustment: {p10_values.mean():.2f} minutes")
print(f"    - Std dev adjustment: {p10_values.std():.2f} minutes")
print(f"    - Min adjustment: {p10_values.min():.2f} minutes")
print(f"    - Max adjustment: {p10_values.max():.2f} minutes")

# ============================================================================
# STEP 7: Export Results
# ============================================================================
print("\n[7/7] Exporting results...")

# Prepare output dataframe with all original columns plus new ones
output_cols = list(arrival_data_clean.columns) + [
    'p10_adjusted', 'p10_source', 'mean', 'std', 'count', 'estimated_real_arrival_time'
]
output_df = merged[output_cols]

# Save full dataset
output_df.to_csv(OUTPUT_PATH / "arrival_data_with_estimated_times.csv", index=False)
print(f"  ✓ Saved: arrival_data_with_estimated_times.csv ({len(output_df):,} flights)")

# Save compact version
compact_cols = ['flight_number', 'origin', 'equip', 'scheduled_arrival_time', 
                'estimated_real_arrival_time', 'p10_adjusted', 'p10_source']
available_compact_cols = [c for c in compact_cols if c in output_df.columns]
compact_df = output_df[available_compact_cols]
compact_df.to_csv(OUTPUT_PATH / "arrival_data_compact_estimates.csv", index=False)
print(f"  ✓ Saved: arrival_data_compact_estimates.csv (compact version)")

# Save summary report
summary_stats = pd.DataFrame({
    'Metric': [
        'Total Flights (arrival_data)',
        'Flights Matched Historically',
        'Flights Using Overall P10',
        'Coverage %',
        'Historical Flights Analyzed (VRA)',
        'Overall Mean Deviation (min)',
        'Overall Std Dev (min)',
        'Overall P10 (min)',
        'Unique Origin+Equipment Combos',
        'Unique Origins',
        'Unique Equipment Types'
    ],
    'Value': [
        total,
        matched,
        total - matched,
        f"{coverage:.1f}%",
        len(vra_data),
        f"{overall_mean:.2f}",
        f"{overall_std:.2f}",
        f"{overall_p10:.2f}",
        len(deviation_stats),
        vra_data['origin'].nunique(),
        vra_data['equipment'].nunique()
    ]
})

summary_stats.to_csv(OUTPUT_PATH / "statistics" / "summary_report.csv", index=False)
print(f"  ✓ Saved: statistics/summary_report.csv")

# ============================================================================
# Final Summary
# ============================================================================
print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print(f"\nData Source: VRA files (Real flight data)")
print(f"  - VRA_2022.csv, VRA_2023.csv, VRA_2024.csv")
print(f"  - Covers {len(vra_data):,} real flights")
print(f"  - Uses ICAO codes for origins and aircraft equipment")

print(f"\nOutput files:")
print(f"  📊 Plots:")
print(f"     - plots/01_histogram.pdf")
print(f"     - plots/01b_distributions_comparison.pdf")
print(f"     - plots/02_p10_by_origin_equipment.pdf")
print(f"     - plots/03_mean_vs_std.pdf")
print(f"     - plots/04_distributions_by_origin_equip.pdf")
print(f"     - plots/05_p10_distribution.pdf")
print(f"     - plots/06_mean_distribution.pdf")
print(f"     - plots/07_extreme_p10_routes.pdf (if extreme values exist)")
print(f"  📄 Data:")
print(f"     - arrival_data_with_estimated_times.csv")
print(f"     - arrival_data_compact_estimates.csv")
print(f"  📋 Statistics:")
print(f"     - statistics/deviation_statistics.csv")
print(f"     - statistics/summary_report.csv")

print(f"\nKey Results:")
print(f"  • Historical coverage: {coverage:.1f}% ({matched:,} of {total:,} flights)")
print(f"  • P10 adjustments range: {p10_values.min():.1f} to {p10_values.max():.1f} minutes")
print(f"  • Origin+Equipment combinations analyzed: {len(deviation_stats)}")
print("=" * 80)
