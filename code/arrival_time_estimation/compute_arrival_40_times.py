"""
Compute 40 NM (nautical miles) advance arrival times and recalculate slot assignments
=======================================================================================

Steps:
  1. Create estimated_real_arrival_40_time = estimated_real_arrival_time - 15 min
     (40 NM point is approximately 15 minutes before touchdown for typical descent profiles)
  2. Calculate new slot_time and day_slot based on 40 NM advance times
  3. Recalculate scheduled_arrivals_40 from the new day_slot
  
Day structure: 6:00 AM D+0 to 5:45 AM D+1 (96 slots of 15 minutes each, indices 0-95)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

BASE_PATH = Path(__file__).resolve().parent.parent  # repository root
OUTPUT_PATH = BASE_PATH / "arrival_time_estimation"

print("=" * 100)
print("COMPUTING 40 NM ADVANCE ARRIVAL TIMES AND NEW SLOT ASSIGNMENTS")
print("=" * 100)

# Load data
print("\n[1] Loading estimated arrival times...")
df = pd.read_csv(OUTPUT_PATH / "arrival_data_with_estimated_times.csv")
print(f"✓ Loaded {len(df):,} flights")

# Display initial columns
print(f"\nOriginal columns: {list(df.columns)[:10]}...")

# ============================================================================
# STEP 1: Create 40 NM advance arrival times
# ============================================================================
print("\n[2] Creating estimated_real_arrival_40_time (40 NM from destination)...")

# Convert to datetime if needed
if df['estimated_real_arrival_time'].dtype == 'object':
    df['estimated_real_arrival_time'] = pd.to_datetime(df['estimated_real_arrival_time'], errors='coerce')

# Ensure we have scheduled_arrival_time as a datetime too
if df['scheduled_arrival_time'].dtype == 'object':
    df['scheduled_arrival_time'] = pd.to_datetime(df['scheduled_arrival_time'], errors='coerce')

# Compute 40 NM advance with fallback to scheduled arrival time
# For flights with estimated_real_arrival_time: use estimated - 15 min
# For flights without estimate: use scheduled - 15 min (conservative)
df['estimated_real_arrival_40_time'] = df.apply(
    lambda row: (row['estimated_real_arrival_time'] if pd.notna(row['estimated_real_arrival_time']) 
                 else row['scheduled_arrival_time']) - pd.Timedelta(minutes=15),
    axis=1
)

print(f"✓ Created estimated_real_arrival_40_time")
print(f"  Non-null values: {df['estimated_real_arrival_40_time'].notna().sum():,}")
if df['estimated_real_arrival_40_time'].isna().sum() > 0:
    print(f"  WARNING: {df['estimated_real_arrival_40_time'].isna().sum():,} flights still missing 40 NM time")

# ============================================================================
# STEP 2: Calculate slot_time and day_slot from 40 NM advance times
# ============================================================================
print("\n[3] Computing slot_time and day_slot from 40 NM times...")

def compute_slot_from_time(arrival_time):
    """
    Compute slot_time and day_slot from an arrival time.
    
    Day structure: 6:00 AM to 5:45 AM next day (in 15-min intervals)
    Slots: 0-95 (96 total slots)
    
    Returns: (slot_time, day_slot) where slot_time is a datetime representing the slot START time
    """
    if pd.isna(arrival_time):
        return None, None
    
    # Extract hour and minute
    hour = arrival_time.hour
    minute = arrival_time.minute
    
    # Days are defined from 6 AM to 5:45 AM next day
    # So if it's before 6 AM, it belongs to previous day
    if hour < 6:
        # Before 6 AM: use previous day's definition
        # Time in minutes since 6 AM previous day
        minutes_since_day_start = (hour - 6) * 60 + minute + 24 * 60  # Add 24 hours
        # Base date is previous day (for 6 AM reference)
        base_date = (arrival_time - timedelta(days=1)).date()
    else:
        # 6 AM or later: current day
        minutes_since_day_start = (hour - 6) * 60 + minute
        # Base date is current day (for 6 AM reference)
        base_date = arrival_time.date()
    
    # Each slot is 15 minutes
    slot_index = minutes_since_day_start // 15
    
    # Clamp to valid range [0, 95]
    slot_index = min(95, max(0, slot_index))
    
    # slot_time_40 should be a datetime representing the START of the slot
    # (matching the format of original slot_time column)
    slot_time = pd.Timestamp(base_date) + timedelta(hours=6, minutes=slot_index*15)
    
    return slot_time, slot_index

print("  Computing slots for each flight...")
slots = df['estimated_real_arrival_40_time'].apply(compute_slot_from_time)
df['slot_time_40'] = [s[0] for s in slots]
df['day_slot_40'] = [s[1] for s in slots]

print(f"✓ Computed slot_time_40 and day_slot_40")

# ============================================================================
# STEP 3: Count scheduled_arrivals_40 (number of aircraft in each slot)
# ============================================================================
print("\n[4] Counting aircraft per slot (scheduled_arrivals_40)...")

# Group by (slot_time_40, day_slot_40) to count aircraft per slot
# This ensures we count aircraft in the same 15-minute slot on the same operational day
slot_counts = df.groupby(['slot_time_40', 'day_slot_40']).size()

# Create a mapping from (slot_time_40, day_slot_40) to count
slot_count_dict = slot_counts.to_dict()

# Map the count back to each flight
df['scheduled_arrivals_40'] = df.apply(
    lambda row: slot_count_dict.get((row['slot_time_40'], row['day_slot_40']), None) 
                if pd.notna(row['slot_time_40']) else None,
    axis=1
)

print(f"✓ Counted aircraft per time slot")

# ============================================================================
# VALIDATION
# ============================================================================
print("\n[5] Validation and Summary...")

print(f"\nData quality check:")
print(f"  • Non-null estimated_real_arrival_40_time: {df['estimated_real_arrival_40_time'].notna().sum():,}")
print(f"  • Non-null day_slot_40: {df['day_slot_40'].notna().sum():,}")
print(f"  • Non-null scheduled_arrivals_40: {df['scheduled_arrivals_40'].notna().sum():,}")

print(f"\nSlot distribution:")
slot_dist = df['day_slot_40'].value_counts().sort_index()
print(f"  • Min slot: {slot_dist.index.min()}")
print(f"  • Max slot: {slot_dist.index.max()}")
print(f"  • Slots with data: {len(slot_dist)}")
print(f"  • Total slots available: 96 (0-95)")

print(f"\nTime differences (estimated_real_arrival - estimated_real_arrival_40):")
time_diff = (df['estimated_real_arrival_time'] - df['estimated_real_arrival_40_time']).dt.total_seconds() / 60
print(f"  • Mean difference: {time_diff.mean():.2f} minutes")
print(f"  • Std dev: {time_diff.std():.2f} minutes")
print(f"  • Min: {time_diff.min():.2f} min, Max: {time_diff.max():.2f} min")

# Sample rows
print(f"\nSample rows:")
sample_cols = ['flight_number', 'scheduled_arrival_time', 'estimated_real_arrival_time', 
               'estimated_real_arrival_40_time', 'day_slot_40', 'slot_time_40', 'scheduled_arrivals_40']
sample = df[sample_cols].head(5)
for idx, row in sample.iterrows():
    print(f"\n  Flight: {row['flight_number']}")
    print(f"    Scheduled arrival: {row['scheduled_arrival_time']}")
    print(f"    Estimated real arrival: {row['estimated_real_arrival_time']}")
    print(f"    Estimated arrival at 40 NM: {row['estimated_real_arrival_40_time']}")
    print(f"    Day slot (0-95): {int(row['day_slot_40']) if pd.notna(row['day_slot_40']) else 'N/A'}")
    print(f"    Slot time (40 NM): {row['slot_time_40']}")
    print(f"    Aircraft in this slot (scheduled_arrivals_40): {int(row['scheduled_arrivals_40']) if pd.notna(row['scheduled_arrivals_40']) else 'N/A'}")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n[6] Saving results...")

output_file = OUTPUT_PATH / "arrival_data_with_estimated_times_40.csv"
df.to_csv(output_file, index=False)
print(f"✓ Saved: {output_file.name} ({len(df):,} flights)")

# Also save compact version
compact_cols = ['flight_number', 'origin', 'equip', 'scheduled_arrival_time', 
                'estimated_real_arrival_time', 'estimated_real_arrival_40_time',
                'day_slot_40', 'slot_time_40', 'scheduled_arrivals_40', 'p10_source']

df_compact = df[compact_cols]
compact_file = OUTPUT_PATH / "arrival_data_compact_40_estimates.csv"
df_compact.to_csv(compact_file, index=False)
print(f"✓ Saved: {compact_file.name} (compact version)")

print("\n" + "=" * 100)
print("COMPLETE: 40 NM advance arrival times computed and slots recalculated")
print("=" * 100)
