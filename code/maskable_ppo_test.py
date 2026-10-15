
# %%
from sb3_contrib import MaskablePPO
import improved_gym_sdcboptimizer_specific as sdcb
# from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from os import listdir
from os.path import isfile, join
import scipy.stats as stats
import random
from itertools import product
import multiprocessing
from sklearn.linear_model import LinearRegression
from pathlib import Path
import datetime
# import seaborn as sns
PAPER_LABEL_FONTSIZE = 18
PAPER_TITLE_FONTSIZE = 20
PAPER_TICK_FONTSIZE = 16
PAPER_LEGEND_FONTSIZE = 16
PAPER_BAR_FIGSIZE = (5.2, 3.2)
PAPER_BAR_LABEL_FONTSIZE = 13
PAPER_BAR_TITLE_FONTSIZE = 13
PAPER_BAR_TICK_FONTSIZE = 11
PAPER_FOCUS_LABEL_FONTSIZE = PAPER_LABEL_FONTSIZE + 2
PAPER_FOCUS_TITLE_FONTSIZE = PAPER_TITLE_FONTSIZE + 2
PAPER_FOCUS_TICK_FONTSIZE = PAPER_TICK_FONTSIZE + 2
plt.rcParams.update({'font.size': PAPER_TICK_FONTSIZE})
plt.switch_backend('Agg')

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
from torch.distributions import Distribution
Distribution.set_default_validate_args(False)
torch.set_num_threads(1)  # single-threaded training: multi-threaded sums would make results depend on the thread count

###############################################################################
# Training setting and deployment variant (command-line options, see README.md)
###############################################################################

import argparse
_parser = argparse.ArgumentParser(description='Strategic DCB experiment, case 1: conventional ATM at SBGR')
_parser.add_argument('--setting', choices=['S1', 'S2'], default='S1',
                     help='S1 (default): entropy coefficient 0 and queue/configuration-switch training penalties linear '
                          'in the queue lengths; S2: entropy coefficient 0.01 and quadratic-form training penalties '
                          '(the structure of the evaluation cost, without its factor of 45)')
_parser.add_argument('--forecast-check', action='store_true',
                     help="keep each day's ground delays only if they reduce the tactical cost of the day on the "
                          'forecast (TAF); otherwise keep the original schedule')
_args, _ = _parser.parse_known_args()
SETTING, FORECAST_CHECK = _args.setting, _args.forecast_check
TRAINING_REWARD = {'S1': 'linear', 'S2': 'quadratic'}[SETTING]
ENT_COEF = {'S1': 0.0, 'S2': 0.01}[SETTING]
MODEL_DIR = {'S1': 'Results', 'S2': 'Results_S2'}[SETTING]  # trained models of the setting
OUT_DIR = MODEL_DIR + ('_check' if FORECAST_CHECK else '')  # result figures of this run
print(f'Training setting {SETTING} (training reward: {TRAINING_REWARD}, entropy coefficient: {ENT_COEF}), '
      f'forecast check: {"on" if FORECAST_CHECK else "off"}; models in {MODEL_DIR}/, figures in {OUT_DIR}/')

###############################################################################
# Function for day counter
###############################################################################

# def day_of_the_year(day_string):
    
#     day_str_split = day_string.split('-')
#     year = int(day_str_split[0])
#     month = int(day_str_split[1])
#     day = int(day_str_split[2])
    
#     if year % 4 == 0:
#         if year % 100 == 0 and year % 400 != 0:
#             month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
#         else:
#             month_days = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
#     else:
#         month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        
#     counter = 0
#     for i in range(month - 1):
#         counter += month_days[i]
#     counter += day - 1
    
#     return counter

# def previous_day(day_string):
    
#     day_str_split = day_string.split('-')
#     year = int(day_str_split[0])
#     month = int(day_str_split[1])
#     day = int(day_str_split[2])
    
#     if year % 4 == 0:
#         if year % 100 == 0 and year % 400 != 0:
#             month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
#         else:
#             month_days = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
#     else:
#         month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    
#     if day == 1:
#         if month == 1:
#             prev_year = year - 1
#             prev_month = 12
#             prev_day = 31
#         else:
#             prev_year = year
#             prev_month = month - 1
#             prev_day = month_days[prev_month - 1]
#     else:
#         prev_year = year
#         prev_month = month
#         prev_day = day - 1
    
#     prev_day_string = str(prev_year) + '-' + str(prev_month) + '-' + str(prev_day)
#     return prev_day_string

###############################################################################
# Runway Configuration Capacity Envelope (RCCE)
###############################################################################

def f(x,y,u): # Departures upper bound for the VMC curve
    v_ub = [] # Upper bound of departures v
    for i in range(len(u)): # For each arrival u_i
        for j in range(len(x)-1):  # For each intersecting point x_j
            if x[j] <= u[i] < x[j+1]: # If x_j <= u_i <= x_(j + 1)
                dy = y[j+1] - y[j]
                dx = x[j+1] - x[j]
                if dx == 0:
                    v_ub.append(y[j])
                else:
                    v_ub.append(dy/dx*(u[i] - x[j]) + y[j])
        if u[i] == x[-1]: # If u_i is the final point of the curve
            dy = y[-1] - y[-2]
            dx = x[-1] - x[-2]
            if dx == 0: # Vertical line
                v_ub.append(y[-2])
            else:
                v_ub.append(0)
        if u[i] > x[-1]: # If u_i is after the final point of the curve
            v_ub.append(0)
        if u[i] < 0: # If u_i is before the initial point of the curve
            v_ub.append(0)
    return np.array(v_ub, dtype=np.int64)

###############################################################################
# Data preprocessing
###############################################################################

# Importing data preprocessed in R
print('\nLoading METAR and traffic data...')
data = pd.read_csv('data/data.csv')
data['time'] = pd.to_datetime(data['time'], format='mixed')
print('Data loaded.\n')

# Importing data preprocessed in R
print('\nLoading flight data...')
arrival_data = pd.read_csv('arrival_time_estimation/arrival_data_with_estimated_times_40.csv')
# Convert slot_time_40 to datetime for proper matching
arrival_data['slot_time_40'] = pd.to_datetime(arrival_data['slot_time_40'], format='mixed')
arrival_data['slot_time'] = arrival_data['slot_time_40']
arrival_data['day_slot'] = arrival_data['day_slot_40']
arrival_data['scheduled_departure_time'] = pd.to_datetime(arrival_data['scheduled_departure_time'], format='mixed')
arrival_data['scheduled_arrival_time'] = pd.to_datetime(arrival_data['estimated_real_arrival_time'], format='mixed')
arrival_data['scheduled_arrival_40_time'] = pd.to_datetime(arrival_data['estimated_real_arrival_40_time'], format='mixed')
print('Data loaded.\n')

# Sort arrival_data by scheduled_arrival_40_time to ensure flights are processed in chronological order
arrival_data = arrival_data.sort_values('scheduled_arrival_40_time').reset_index(drop=True)
print('Arrival data sorted by scheduled_arrival_40_time.\n')

# Recalculate day column based on 40 NM arrival times
print('Recalculating day assignments based on 40 NM times...')
arrival_data['date'] = arrival_data['scheduled_arrival_40_time'].dt.date
for row in range(len(arrival_data)):
    if arrival_data.loc[row, 'scheduled_arrival_40_time'].hour < 6:
        arrival_data.loc[row, 'date'] -= datetime.timedelta(days=1)

# Create day column in format 'YYYY-M-D'
arrival_data['day'] = ''
for row in range(len(arrival_data)):
    year = arrival_data.loc[row, 'date'].year
    month = arrival_data.loc[row, 'date'].month
    day = arrival_data.loc[row, 'date'].day
    arrival_data.loc[row, 'day'] = str(year) + '-' + str(month) + '-' + str(day)
print('Day assignments recalculated based on 40 NM times.\n')

# ============================================================================
# REVISE scheduled_arrivals_40 IN data.csv BASED ON arrival_data
# ============================================================================
print('Revising scheduled_arrivals_40 based on 40 NM flight data...')

# Create mapping from slot_time_40 to revised scheduled_arrivals_40
# Group by slot_time_40 (datetime) and get the count value
# Note: all flights in the same slot have the same scheduled_arrivals_40 value
slot_arrivals_revised_map = arrival_data.groupby('slot_time_40')['scheduled_arrivals_40'].first().to_dict()

# Map the revised values to data using the time column
# Convert data['time'] to match the format of arrival_data['slot_time_40']
data['scheduled_arrivals_40_revised'] = data['time'].map(slot_arrivals_revised_map)

# Fill NaN with 0 - meaning no arrivals in that slot (no real 40 NM flight data for this time slot)
data['scheduled_arrivals_40_revised'] = data['scheduled_arrivals_40_revised'].fillna(0).astype(int)
print(f'Scheduled arrivals (40 NM) revised successfully.')
print(f'  Slots with real 40 NM data: {(data["scheduled_arrivals_40_revised"] > 0).sum():,} / {len(data):,}')
print(f'  Slots with no arrivals (40 NM): {(data["scheduled_arrivals_40_revised"] == 0).sum():,}\n')

# Show statistics comparing original and revised scheduled_arrivals_40
print('='*100)
print('COMPARISON: Original vs Revised scheduled_arrivals_40 (40 NM)')
print('='*100)

# Create comparison dataframe
comparison_df = data[['time', 'scheduled_arrivals_40', 'scheduled_arrivals_40_revised']].copy()
comparison_df['difference'] = comparison_df['scheduled_arrivals_40_revised'] - comparison_df['scheduled_arrivals_40']
comparison_df['pct_change'] = (comparison_df['difference'] / comparison_df['scheduled_arrivals_40'].replace(0, 1) * 100).round(2)

print(f"\n[1] Data Overview:")
print(f"    Total time slots: {len(comparison_df)}")
print(f"    Slots with revised data: {comparison_df['scheduled_arrivals_40_revised'].notna().sum()}")

print(f"\n[2] Original scheduled_arrivals_40:")
print(f"    Min: {data['scheduled_arrivals_40'].min():.0f}")
print(f"    Max: {data['scheduled_arrivals_40'].max():.0f}")
print(f"    Mean: {data['scheduled_arrivals_40'].mean():.2f}")
print(f"    Std Dev: {data['scheduled_arrivals_40'].std():.2f}")
print(f"    Median: {data['scheduled_arrivals_40'].median():.0f}")

print(f"\n[3] Revised scheduled_arrivals_40 (from 40 NM flight data):")
revised_data = data['scheduled_arrivals_40_revised'].dropna()
if len(revised_data) > 0:
    print(f"    Min: {revised_data.min():.0f}")
    print(f"    Max: {revised_data.max():.0f}")
    print(f"    Mean: {revised_data.mean():.2f}")
    print(f"    Std Dev: {revised_data.std():.2f}")
    print(f"    Median: {revised_data.median():.0f}")
else:
    print(f"    No matched data available")

differences = comparison_df['difference']
print(f"\n[4] Differences (Revised - Original):")
matched_mask = ~differences.isna()
unmatched_mask = differences.isna()
print(f"    Matched slots: {matched_mask.sum()}")
print(f"    Unmatched slots: {unmatched_mask.sum()}")

if matched_mask.sum() > 0:
    diffs_matched = differences[matched_mask]
    print(f"\n    For MATCHED slots:")
    print(f"      Slots with changes: {(diffs_matched != 0).sum()}")
    print(f"      Slots unchanged: {(diffs_matched == 0).sum()}")
    if (diffs_matched != 0).sum() > 0:
        print(f"      Mean difference: {diffs_matched[diffs_matched != 0].mean():.2f}")
        print(f"      Std Dev: {diffs_matched[diffs_matched != 0].std():.2f}")
        print(f"      Min difference: {diffs_matched.min():.0f}")
        print(f"      Max difference: {diffs_matched.max():.0f}")
        print(f"      Median difference: {diffs_matched.median():.0f}")

print(f"\n[5] Change Distribution (for matched slots only):")
if matched_mask.sum() > 0:
    diffs_matched = differences[matched_mask]
    change_bins = [-100, -10, -5, -1, 0, 1, 5, 10, 100]
    change_counts = pd.cut(diffs_matched, bins=change_bins).value_counts().sort_index()
    for interval, count in change_counts.items():
        pct = (count / matched_mask.sum() * 100) if matched_mask.sum() > 0 else 0
        print(f"    {interval}: {count:,} slots ({pct:.1f}%)")

print(f"\n[6] Sample rows with data (first 15):")
sample = comparison_df[matched_mask].head(15)
if len(sample) > 0:
    print(sample.to_string(index=False))
else:
    print("    No rows with matched data found")

print(f"\n[7] Sample rows with maximum changes (with matched data):")
matched_comparison = comparison_df[matched_mask]
if len(matched_comparison) > 0:
    max_changes = matched_comparison.loc[matched_comparison['difference'].abs().nlargest(5).index]
    print(max_changes.to_string(index=False))

print('\n' + '='*100 + '\n')

# Importing TAF data preprocessed in R
print('\nLoading TAF data...')
taf_data = pd.read_csv('data/new_taf_list.csv')
taf_data['time'] = pd.to_datetime(taf_data['time'], format='mixed')
print('Data loaded.\n')

# Importing standard TAF data preprocessed in R
print('\nLoading standard TAF data...')
std_taf_data = pd.read_csv('data/std_taf.csv')
std_taf_data['time'] = pd.to_datetime(std_taf_data['time'], format='mixed')
print('Data loaded.\n')

# print('Creating baseline...')
# for i in range(len(data)-1):
#     data.loc[i,'base_arrivals_40'] = data.loc[i+1,'actual_arrivals']
#     data.loc[i,'base_departures'] = data.loc[i+1,'actual_departures']
#     data.loc[i,'base_configuration'] = data.loc[i+1,'configuration']
# print('Done.\n')

# Introducing ATFM measure information
print('Loading ATFM measure information...')
file_path = "data/CGNA_MEDIDAS_ATFM_20180101_20231121_processed.csv"
df = pd.read_csv(file_path,sep=';')
print('Done.\n')

print('Processing ATFM measure information...')

# Converting columns from date to datetime
df['time_begin'] = pd.to_datetime(df['time_begin'], format='mixed')
df['time_end'] = pd.to_datetime(df['time_end'], format='mixed')

# Extracting date (without time) to the next grouping
df['date'] = df['time_begin'].dt.date

# Calculate duration in minutes
df['duration_min'] = (df['time_end'] - df['time_begin']).dt.total_seconds() / 60

# Extracting the numeric value of the column TMI
df['separation_nm'] = df['tmi_code'].str.extract(r'(\d+)\s*NM')[0]
df['separation_nm'] = pd.to_numeric(df['separation_nm'])

# Filter only SBGR data
df_sbgr = df[df['destination'] == 'SBGR'].copy()
df_sbgr = df_sbgr[df_sbgr['TMI'] == 'MIT']
# df_sbgr = df_sbgr[df_sbgr['year'] >= 2022]
# df_sbgr = df_sbgr[df_sbgr['year'] <= 2023]

# Sort by begin time
df_sbgr = df_sbgr.sort_values('time_begin')

# Create a list of consolidated intervals
intervals = []
current_start = None
current_end = None
current_max_sep = 0

for _, row in df_sbgr.iterrows():
    if current_start is None:
        # Firs interval
        current_start = row['time_begin']
        current_end = row['time_end']
        current_max_sep = row['separation_nm']
    else:
        if row['time_begin'] <= current_end:
            # Overlapping of intervals - extend the current interval
            current_end = max(current_end, row['time_end'])
            current_max_sep = max(current_max_sep, row['separation_nm'])
        else:
            # New interval without overlapping
            intervals.append({
                'time_begin': current_start,
                'time_end': current_end,
                'duration_min': (current_end - current_start).total_seconds() / 60,
                'max_separation_nm': current_max_sep,
                'num_origins': len(df_sbgr[(df_sbgr['time_begin'] <= current_end) & 
                               (df_sbgr['time_end'] >= current_start)])
            })
            current_start = row['time_begin']
            current_end = row['time_end']
            current_max_sep = row['separation_nm']

# Add the last interval
if current_start is not None:
    intervals.append({
        'time_begin': current_start,
        'time_end': current_end,
        'duration_min': (current_end - current_start).total_seconds() / 60,
        'max_separation_nm': current_max_sep,
        'num_origins': len(df_sbgr[(df_sbgr['time_begin'] <= current_end) & 
                         (df_sbgr['time_end'] >= current_start)])
    })

# Create consolidated DataFrame
df_intervals = pd.DataFrame(intervals)

# Extract data to a new column
df_intervals['date'] = df_intervals['time_begin'].dt.date

print('Done.\n')

print('Introducing ATFM measure information into data...')
sep_min = 5
sep_max = 30
sep_step = 5
data['base_sep'] = sep_min

def get_mit_info(start, end, intervals_df):
    # Encontrar intervalos que sobrepõem com o período de 15 minutos
    mask = (
        (intervals_df['time_begin'] <= start) & 
        (intervals_df['time_end'] >= end)
    )
    overlapping = intervals_df[mask]
    
    if not overlapping.empty:
        # Se houve sobreposição, retornar True e a separação máxima
        return True, overlapping['max_separation_nm'].max()
    else:
        # Se não houve sobreposição, retornar False e NaN
        return False, None
    
for row in range(len(data)):
    mit_applied, sep = get_mit_info(data.loc[row,'time'],
                                    data.loc[row,'time']+datetime.timedelta(minutes=15),
                                    df_intervals)
    if mit_applied:
        data.loc[row,'base_sep'] = sep
print('Done.\n')

df_intervals = df_intervals[df_intervals['time_end'].dt.year >= 2022]
df_intervals = df_intervals[df_intervals['time_begin'].dt.year <= 2023]

print('Plotting ATFM measure statistics...')

# Daily compilation
df_diario = df_intervals.groupby('date').agg({
    'duration_min': 'sum',
    'max_separation_nm': 'max',
    'num_origins': 'max'
}).reset_index()

# Filter relevant data
df_diario = df_diario[df_diario['duration_min'] <= 60*24]
data1 = df_diario['duration_min']
data2 = df_diario['max_separation_nm']

# Plotting statistics
plt.figure()
plt.xlabel('Total duration (min) per day')
plt.ylabel('Frequency')
plt.hist(data1, bins=50, color='blue')
plt.grid(axis='y',linestyle='dashed')
plt.savefig('mit_duration.pdf', format='pdf')
plt.show()

plt.figure()
plt.xlabel('Mean separation (NM) per day')
plt.ylabel('Frequency')
plt.hist(data2, bins=50, color='red')
plt.yticks(list(range(0,250,20)))
plt.grid(axis='y',linestyle='dashed')
plt.xticks(list(range(15,35,5)))
plt.savefig('mit_separation.pdf', format='pdf')
plt.show()

print('Done.\n')

print('Removing partial periods...')
# while(data.loc[0,'time'] != data.loc[0,'time'].replace(data.loc[0,'time'][-8:],'06:00:00')):
#     data.drop(0, axis='rows', inplace=True)
#     data.reset_index(drop=True, inplace=True)
# while(data.loc[len(data)-1,'time'] != data.loc[len(data)-1,'time'].replace(data.loc[len(data)-1,'time'][-8:],'05:45:00')):
#     data.drop(len(data)-1, axis='rows', inplace=True)
#     data.reset_index(drop=True, inplace=True)
while(data.loc[0,'time'].hour != 6):
    data.drop(0, axis='rows', inplace=True)
    data.reset_index(drop=True, inplace=True)
while(data.loc[len(data)-1,'time'].hour != 5):
    data.drop(len(data)-1, axis='rows', inplace=True)
    data.reset_index(drop=True, inplace=True)
print('Done.\n')

print('Determining periods...')
# data['day'] = data['time']
# for row in range(len(data)):
#     time_split = data.loc[row,'time'].split(' ')
#     day_string = time_split[0].split('-')
#     year = int(day_string[0])
#     month = int(day_string[1])
#     day = int(day_string[2])
#     day_string = str(year) + '-' + str(month) + '-' + str(day)
#     if len(time_split) > 1:
#         hour = int(time_split[1][0:2])
#     else:
#         hour = 0
#     if hour < 6:
#       data.loc[row,'day'] = previous_day(day_string)
#     else:
#       data.loc[row,'day'] = day_string
data['date'] = data['time'].dt.date
for row in range(len(data)):
    if data.loc[row,'time'].hour < 6:
        data.loc[row,'date'] -= datetime.timedelta(days=1)
data['day'] = str(data['date'])
for row in range(len(data)):
    year = data.loc[row,'date'].year
    month = data.loc[row,'date'].month
    day = data.loc[row,'date'].day
    data.loc[row,'day'] = str(year) + '-' + str(month) + '-' + str(day)
print('Done.\n')

print('Plotting separation...')
plt.figure()
plt.xlabel('Time steps')
plt.ylabel('Separation (NM)')
plt.yticks(list(range(5,35,5)))
plt.grid(axis='y', linestyle='dashed')
plt.plot(list(range(len(data))), data['base_sep'], color='red')

# Set scientific notation for x-axis
plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
plt.gca().xaxis.get_major_formatter().set_scientific(True)
plt.gca().xaxis.get_major_formatter().set_powerlimits((0, 0))  # Force scientific notation

plt.savefig('base_separation.pdf', format='pdf')
plt.show()
print('Done.\n')

print('Plotting separation...')
plt.figure()
plt.xlabel('Time steps')
plt.ylabel('Separation (NM)')
plt.yticks(list(range(5,35,5)))
plt.grid(axis='y', linestyle='dashed')
plt.plot(list(range(5*96+1)),data.loc[:5*96,'base_sep'],color='red')
plt.savefig('base_separation.pdf', format='pdf')
plt.show()
print('Done.\n')

# print('Plotting traffic data...')
# plt.figure()
# plt.xlabel('Time steps')
# plt.ylabel('Movements')
# plt.plot(range(len(data)),data['departure_demand'],'tab:blue')
# plt.plot(range(len(data)),data['arrival_demand_40'],'tab:green')
# plt.show()
# print('Done.\n')

# Introducing GDP information
print('Loading GDP information...')
df = pd.read_csv(file_path, sep=';')
print('Done.\n')

print('Processing GDP information...')

# Converting columns from date to datetime
df['time_begin'] = pd.to_datetime(df['time_begin'], format='mixed')
df['time_end'] = pd.to_datetime(df['time_end'], format='mixed')

# Extracting date (without time) for grouping
df['date'] = df['time_begin'].dt.date

# Calculate duration in minutes
df['duration_min'] = (df['time_end'] - df['time_begin']).dt.total_seconds() / 60

# Filter only GDP measures
df_gdp = df[df['TMI'] == 'GDP'].copy()

# Filter only data for a specific airport (e.g., SBGR)
df_sbgr = df_gdp[(df_gdp['regulated_element'] == 'SBGR') | (df_gdp['regulated_element'] == 'SBXP')].copy()

# Sort by begin time
df_sbgr = df_sbgr.sort_values('time_begin')

# Create a list of consolidated GDP periods
gdp_periods = []
current_start = None
current_end = None
current_min_rate = None

for _, row in df_sbgr.iterrows():
    if current_start is None:
        # First GDP period
        current_start = row['time_begin']
        current_end = row['time_end']
        current_min_rate = row['rate']
    else:
        if row['time_begin'] <= current_end:
            # Overlapping periods - extend the current GDP
            current_end = max(current_end, row['time_end'])
            current_min_rate = min(current_min_rate, row['rate']) if current_min_rate else row['rate']
        else:
            # New GDP period without overlapping
            gdp_periods.append({
                'begin': current_start,
                'end': current_end,
                'duration_min': (current_end - current_start).total_seconds() / 60,
                'min_rate': current_min_rate,
                'num_affected_flights': len(df_sbgr[(df_sbgr['time_begin'] <= current_end) & 
                                       (df_sbgr['time_end'] >= current_start)])
            })
            current_start = row['time_begin']
            current_end = row['time_end']
            current_min_rate = row['rate']

# Add the last GDP period
if current_start is not None:
    gdp_periods.append({
        'begin': current_start,
        'end': current_end,
        'duration_min': (current_end - current_start).total_seconds() / 60,
        'min_rate': current_min_rate,
        'num_affected_flights': len(df_sbgr[(df_sbgr['time_begin'] <= current_end) & 
                               (df_sbgr['time_end'] >= current_start)])
    })

# Create consolidated DataFrame
df_gdp_periods = pd.DataFrame(gdp_periods)

# Extract date to a new column
df_gdp_periods['date'] = df_gdp_periods['begin'].dt.date

print('Done.\n')

print('Introducing GDP information into data...')
default_rate = 60  # Default flights/hour rate when no GDP is in effect

def get_gdp_info(start, end, gdp_df):
    # Find GDP periods that overlap with the 15-minute window
    mask = (
        (gdp_df['begin'] <= start) & 
        (gdp_df['end'] >= end)
    )
    overlapping = gdp_df[mask]
    
    if not overlapping.empty:
        # If GDP was active, return True and the minimum rate
        return True, overlapping['min_rate'].min()
    else:
        # If no GDP, return False and None
        return False, None
    
for row in range(len(data)):
    gdp_active, rate = get_gdp_info(data.loc[row,'time'],
                                   data.loc[row,'time']+datetime.timedelta(minutes=15),
                                   df_gdp_periods)
    if gdp_active:
        data.loc[row,'current_rate'] = rate
    else:
        data.loc[row,'current_rate'] = default_rate
print('Done.\n')

# Filter for specific years if needed
df_gdp_periods = df_gdp_periods[df_gdp_periods['end'].dt.year >= 2022]
df_gdp_periods = df_gdp_periods[df_gdp_periods['begin'].dt.year <= 2023]

print('Plotting GDP statistics...')

# Daily compilation
df_daily = df_gdp_periods.groupby('date').agg({
    'duration_min': 'sum',
    'min_rate': 'min',
    'num_affected_flights': 'sum'
}).reset_index()

# Filter relevant data
df_daily = df_daily[df_daily['duration_min'] <= 60*24]
duration_data = df_daily['duration_min']
rate_data = df_daily['min_rate']

# Plotting statistics
plt.figure()
plt.xlabel('Total GDP duration (min) per day')
plt.ylabel('Frequency')
plt.hist(duration_data, bins=50, color='blue')
plt.grid(axis='y', linestyle='dashed')
plt.savefig('gdp_duration.pdf', format='pdf')
plt.show()

plt.figure()
plt.xlabel('Minimum GDP rate (flights/hour) per day')
plt.ylabel('Frequency')
plt.hist(rate_data, bins=20, color='green')
plt.grid(axis='y', linestyle='dashed')
plt.xticks(list(range(0, max(rate_data)+5, 5)))
plt.savefig('gdp_rates.pdf', format='pdf')
plt.show()

print('Done.\n')

# Calculating demand error
data['arrival_error_40'] = data['arrival_demand_40'] - data['scheduled_arrivals_40_revised']
data['arrival_error_40'] = data['arrival_error_40'].astype(int)
data['departure_error'] = data['departure_demand'] - data['scheduled_departures']
data['departure_error'] = data['departure_error'].astype(int)

# Error in function of scheduled arrivals (40 NM)
y1_mean = []
y1_std = []
data1 = []
sched_arr_40 = np.sort(data['scheduled_arrivals_40_revised'].unique())
x = []
for i in sched_arr_40:
    y1 = data.loc[data['scheduled_arrivals_40_revised'] == i, 'arrival_error_40']
    data1.append(y1)
    x.append([i])
    y1_mean.append(np.mean(y1))
    if np.std(y1) != 0:
        y1_std.append(np.std(y1))
    else:
        y1_std.append(np.std(data['arrival_error_40']))
y1_mean = np.array(y1_mean)
y1_std = np.array(y1_std)
arr_mean_40 = y1_mean.copy()
arr_std_40 = y1_std.copy()
model_arr_mean_40 = LinearRegression()
model_arr_mean_40.fit(x, arr_mean_40)
arr_mean_40_pred = model_arr_mean_40.predict(x)
model_arr_std_40 = LinearRegression()
model_arr_std_40.fit(x, arr_std_40)  
arr_std_40_pred = model_arr_std_40.predict(x)
plt.figure(figsize=(5.2, 3.5), dpi=1200)
plt.plot(sched_arr_40, y1_mean, 'b')
# plt.plot(sched_arr_40, y1_std, 'g')    
plt.plot(sched_arr_40, arr_mean_40_pred, 'r')
plt.xlabel('Scheduled arrivals (40 NM)')
plt.ylabel('Error')
plt.legend(['Mean CI','Regression CI'])
plt.plot(sched_arr_40, y1_mean - y1_std, '--b')
plt.plot(sched_arr_40, y1_mean + y1_std, '--b')
plt.fill_between(sched_arr_40, 
                 y1_mean - y1_std, 
                 y1_mean + y1_std,
                 color='b',
                 alpha=0.2)
plt.plot(sched_arr_40, arr_mean_40_pred - arr_std_40_pred, '--r')
plt.plot(sched_arr_40, arr_mean_40_pred + arr_std_40_pred, '--r')
plt.fill_between(sched_arr_40, 
                 arr_mean_40_pred - arr_std_40_pred, 
                 arr_mean_40_pred + arr_std_40_pred,
                 color='r',
                 alpha=0.2)  
x1 = sched_arr_40
y1 = data['arrival_error_40']
plt.boxplot(data1, positions=x1)
plt.yticks(ticks=np.array(range(np.sort(y1.unique()).min(), np.sort(y1.unique()).max(), 3)))
plt.savefig('arr_variation_40.pdf', format='pdf')
plt.show()

# Error in function of scheduled arrivals
y2_mean = []
y2_std = []
data2 = []
sched_dep = np.sort(data['scheduled_departures'].unique())
x = []
for i in np.sort(data['scheduled_departures'].unique()):
    y2 = data.loc[data['scheduled_departures'] == i, 'departure_error']
    data2.append(y2)
    x.append([i])
    y2_mean.append(np.mean(y2))
    if np.std(y2) != 0:
        y2_std.append(np.std(y2))
    else:
        y2_std.append(np.std(data['departure_error']))
y2_mean = np.array(y2_mean)
y2_std = np.array(y2_std)
dep_mean = y2_mean.copy()
dep_std = y2_std.copy()
model_dep_mean = LinearRegression()
model_dep_mean.fit(x, dep_mean)
dep_mean_pred = model_dep_mean.predict(x)
model_dep_std = LinearRegression()
model_dep_std.fit(x, dep_std)  
dep_std_pred = model_dep_std.predict(x)
plt.figure(dpi=1200)
plt.plot(sched_dep, y2_mean, 'b')    
# plt.plot(sched_dep, y2_std, 'g') 
plt.plot(sched_dep, dep_mean_pred, 'r')
plt.xlabel('Scheduled departures')
plt.ylabel('Error')
plt.legend(['Mean CI','Regression CI'])
plt.plot(sched_dep, y2_mean - y2_std, '--b')
plt.plot(sched_dep, y2_mean + y2_std, '--b')
plt.fill_between(sched_dep, 
                 y2_mean - y2_std, 
                 y2_mean + y2_std,
                 color='b',
                 alpha=0.2) 
plt.plot(sched_dep, dep_mean_pred - dep_std_pred, '--r')
plt.plot(sched_dep, dep_mean_pred + dep_std_pred, '--r')
plt.fill_between(sched_dep, 
                 dep_mean_pred - dep_std_pred, 
                 dep_mean_pred + dep_std_pred,
                 color='r',
                 alpha=0.2)     
x2 = np.sort(data['scheduled_departures'].unique())
y2 = data['departure_error']
plt.boxplot(data2, positions=x2)
plt.yticks(ticks=np.array(range(np.sort(y2.unique()).min(), np.sort(y2.unique()).max(), 3)))
plt.savefig('dep_variation.pdf', format='pdf')
plt.show()

# Plotting histograms and normal distributions (40 NM)
y1 = data['arrival_error_40']
plt.figure(dpi=1200)
plt.hist(y1, density=True, bins=np.sort(data['arrival_error_40'].unique()))
plt.xlabel('Error')
plt.title('Probability Distribution of Arrival Error (40 NM)')

# Plotting normal distribution
arr_mu_40 = np.average(data['arrival_error_40'])
arr_sigma_40 = np.std(data['arrival_error_40'])
x = np.linspace(arr_mu_40 - 5*arr_sigma_40, arr_mu_40 + 5*arr_sigma_40, 100)
plt.plot(x, stats.norm.pdf(x, arr_mu_40, arr_sigma_40))
plt.legend(['Normal Distribution','Histogram (40 NM)'])
plt.savefig('arr_dist_40.pdf', format='pdf')
plt.show()

# Plotting histograms
y2 = data['departure_error']
plt.figure(dpi=1200)
plt.hist(y2, density=True, bins=np.sort(data['departure_error'].unique()))
plt.xlabel('Error')
# plt.ylabel('Probability')
plt.title('Probability Distribution of Departure Error')

# Plotting normal distribution
dep_mu = np.average(data['departure_error'])
dep_sigma = np.std(data['departure_error'])
x = np.linspace(dep_mu - 5*dep_sigma, dep_mu + 5*dep_sigma, 100)
plt.plot(x, stats.norm.pdf(x, dep_mu, dep_sigma))
plt.legend(['Normal Distribution','Histogram'])
plt.savefig('dep_dist.pdf', format='pdf')
plt.show()

# Arrivals and departures columns
while taf_data.loc[0, 'time'] != data.loc[0, 'time']:
    taf_data.drop(0, axis='rows', inplace=True)
    taf_data.reset_index(drop=True, inplace=True)
    std_taf_data.drop(0, axis='rows', inplace=True)
    std_taf_data.reset_index(drop=True, inplace=True)
while taf_data.loc[taf_data.shape[0]-1, 'time'] != data.loc[data.shape[0]-1, 'time']:
    taf_data.drop(taf_data.shape[0]-1, axis='rows', inplace=True)
    taf_data.reset_index(drop=True, inplace=True)
    std_taf_data.drop(std_taf_data.shape[0]-1, axis='rows', inplace=True)
    std_taf_data.reset_index(drop=True, inplace=True)
taf_data['arrival_demand_40'] = std_taf_data['arrival_demand_40'] = data['scheduled_arrivals_40_revised']
taf_data['departure_demand'] = std_taf_data['departure_demand'] = data['scheduled_departures']

###############################################################################
# Encoding categorical features
###############################################################################

# Preprocessing data
print('Encoding runway configurations and weather condition...')
data['weather_condition'] = data['ifr'] + data['lifr']
data['10R_10L'] = np.zeros(len(data))
data['28L_28R'] = np.zeros(len(data))
for i in range(len(data)):
    if data.loc[i,'configuration'] == '10R|10L':
        data.loc[i,'configuration'] = 1 # '10R_10L'
        data.loc[i,'10R_10L'] = 1
    elif data.loc[i,'configuration'] == '28L|28R':
        data.loc[i,'configuration'] = 2 # '28L_28R'
        data.loc[i,'28L_28R'] = 1
    else:
        data.loc[i,'configuration'] = 0 # None
        
    if data.loc[i,'base_configuration'] == '10R|10L':
        data.loc[i,'base_configuration'] = 1 # '10R_10L'
    elif data.loc[i,'base_configuration'] == '28L|28R':
        data.loc[i,'base_configuration'] = 2 # '28L_28R'
    else:
        data.loc[i,'base_configuration'] = 0 # None
print('Done.\n')

# Defining wind states
print('Defining and encoding wind states...')
cw_max = 20 # Maxium crosswind (knots)
tw_max = 5 # Maximum tailwind (knots)
print('Crosswind limit of ' + str(cw_max) + ' knots.')
print('Tailwind limit of ' + str(tw_max) + ' knots.\n')
data['wind_state'] = np.zeros(len(data)) # Column of wind states
data['wind_state'] = pd.to_numeric(data['wind_state'], downcast='integer')
data['10R_10L_OK'] = np.zeros(len(data)) # 0 - not available, 1 - available
data['28L_28R_OK'] = np.zeros(len(data)) # 0 - not available, 1 - available
# 0 - Both available (0 is always for all runways available)
# 1 - 10R|10L available
# 2 - 28L|28R available
# 3 - Both unavailable
print('Case 0 - Both available.')
print('Case 1 - 10R|10L available.')
print('Case 2 - 28L|28R available.')
print('Case 3 - Both unavailable.\n')
for i in range(len(data)):
    if abs(data.loc[i,'crosswind_10']) > cw_max:
        data.loc[i,'wind_state'] = 3
        data.loc[i,'10R_10L_OK'] = 0
        data.loc[i,'28L_28R_OK'] = 0
    elif data.loc[i,'headwind_10'] < -tw_max:
        data.loc[i,'wind_state'] = 2
        data.loc[i,'10R_10L_OK'] = 0
        data.loc[i,'28L_28R_OK'] = 1
    elif data.loc[i,'headwind_10'] > tw_max:
        data.loc[i,'wind_state'] = 1
        data.loc[i,'10R_10L_OK'] = 1
        data.loc[i,'28L_28R_OK'] = 0
    else:
        data.loc[i,'wind_state'] = 0
        data.loc[i,'10R_10L_OK'] = 1
        data.loc[i,'28L_28R_OK'] = 1

for k in data['wind_state'].unique():
    print('Case ' + str(k) + ' has ' 
          + str(len(data.loc[data['wind_state'] == k,]))
          + ' occurences.')
print('')

###############################################################################
# Runway Configuration Capacity Envelopes (RCCEs)
###############################################################################

# Importing capacity envelopes
print('Generating capacity envelopes...')
points = {0: {}, 1: {}, 2: {}} # 0 - VMC, 1 - IMC, 2 - LIMC
legend = {0: {}, 1: {}, 2: {}}
folder = 'data//curves//'
files = [f for f in listdir(folder) if isfile(join(folder, f))]
for file in files:
    if file[:3] == 'VMC':
        wc = 0
        if file[4:11] == '28L_28R':
            p = 0.99
        else:
            p = 0.99
    else:
        if file[:3] == 'IMC':
            wc = 1
            if file[4:11] == '28L_28R':
                p = 0.95
            else:
                p = 0.95
        else:
            wc = 2
            if file[4:11] == '28L_28R':
                p = 0.9
            else:
                p = 0.9
    if file[-8:] == '{:.4f}'.format(p).replace('0.','') + '.csv':
        name = file.replace('_RCCE_' + '{:.4f}'.format(p).replace('0.','') + '.csv','')
        curve = pd.read_csv(folder + file)
        print(name)
        if name[-7:] == '10R_10L':
            rc = 1
        elif name[-7:] == '28L_28R':
            rc = 2
        points[wc][rc] = curve
        legend[wc][rc] = name
print('Done.\n')

# Plotting capacity envelopes
legends = []
print('Plotting VMC and IMC capacity envelopes...')
plt.figure(dpi=1200)
for wc in points:
    for rc in points[wc]:
        plt.plot(points[wc][rc]['x'],points[wc][rc]['y'])
        label = legend[wc][rc].replace('_10R_10L', ' 10R/10L').replace('_28L_28R', ' 28L/28R')
        legends.append(label)
plt.title('RCCEs', fontsize=16)
plt.legend(legends, fontsize=16, ncol=1, loc='center left',
           bbox_to_anchor=(1.02, 0.5), framealpha=0.9,
           borderaxespad=0.0, handlelength=1.2)
plt.xlabel('Arrivals', fontsize=16)
plt.ylabel('Departures', fontsize=16)
plt.tick_params(axis='both', labelsize=16)
plt.savefig('rcces_gru.pdf', format='pdf', bbox_inches='tight')
plt.show()
print('Done.\n')

###############################################################################
# Maximum features' values
###############################################################################

# Total number of time slots
N = len(data)

# Calculating accumulated arrivals and departures
print('Determining a list of days...')
accumulated_arrivals = [data.loc[0,'scheduled_arrivals_40_revised']]
accumulated_departures = [data.loc[0,'scheduled_departures']]
day = data.loc[0,'day']
days = [day]
for i in range(1,N):
    if data.loc[i,'day'] == day:
        accumulated_arrivals[-1] += data.loc[i,'scheduled_arrivals_40_revised']
        accumulated_departures[-1] += data.loc[i,'scheduled_departures']
    else:
        accumulated_arrivals.append(data.loc[i,'scheduled_arrivals_40_revised'])
        accumulated_departures.append(data.loc[i,'scheduled_departures'])
        day = data.loc[i,'day']
        days.append(day)
accumulated_info = pd.DataFrame(np.array([days, accumulated_arrivals, accumulated_departures], dtype=object).T, columns=['day', 'accumulated_arrivals', 'accumulated_departures'])
a_max = max(accumulated_info['accumulated_arrivals'])
d_max = max(accumulated_info['accumulated_departures'])
print('Maximum accumulated arrivals: ' + str(a_max))
print('Maximum accumulated departures: ' + str(d_max) + '\n')
# a_max = 50
# d_max = 50
print('Done.\n')

###############################################################################
# Set of actions
###############################################################################

print('Creating curve points for each weather condition and runway configuration...')
A = 1 # '10R_10L'
B = 2 # '28L_28R'
C = 0 # None
configs = [C, A, B] 
weather_conds = [0, 1, 2] # ['VMC', 'IMC', 'LIMC'] # VMC - 0, IMC - 1, LIMC - 2
wind_states = {0: [A,B], 1: [A], 2: [B], 3: [C]}
u = {}
v = {}
u_max = 0
v_max = 0
for wc in weather_conds:
    for rc in configs:
        if rc == C:
            u[wc, rc] = np.array([0])
            v[wc, rc] = np.array([0])
        else:
            u[wc, rc] = np.array([i for i in range(np.max(points[wc][rc]['x']) + 1)])
            v[wc, rc] = np.array(f(np.array(points[wc][rc]['x']),
                                   np.array(points[wc][rc]['y']),
                                   u[wc, rc]), dtype=np.int64)
        if np.max(u[wc, rc]) > u_max:
            u_max = np.max(u[wc, rc])
        if np.max(v[wc, rc]) > v_max:
            v_max = np.max(v[wc, rc])
print('Done.\n')

# All possible actions

print('\nCreating the set of possible actions...')

cm_possible_actions = [] # Possible actions list
for rc in configs:
    if rc == C:
        cm_possible_actions.append([rc,0,0])
    else:
        for arr in u[0, rc]:
            for dep in range(v[0, rc][arr] + 1):
                cm_possible_actions.append([rc, arr, dep])
                
# possible_actions = np.array(possible_actions)

print('Done.\n')

# Counting actions

print('Counting actions...')

max_state_action_counter = 0

for ws in wind_states:
    for wc in weather_conds:
        for a_i in range(u_max + 1):
            for d_i in range(v_max + 1):
                state_action_counter = 0
                for rc in wind_states[ws]:
                    u_values = u[wc, rc]  # Precompute u values
                    v_values = v[wc, rc]  # Precompute v values
                    arr_lim = min(a_i, u_values[-1]) # Minimum arrivals (discarding clearly suboptimal actions)
                    arr_min = min(u_values[-1] - list(v_values[::-1]).index(v_values[0]), arr_lim) # Maxium arrivals
                    for arr in range(arr_min, arr_lim): # For each possible arrival count
                        if d_i >= v_values[arr]: # If the departure capacity is lower than the departure queue
                            state_action_counter += 1
                    state_action_counter += 1
                    if state_action_counter > max_state_action_counter:
                        max_state_action_counter = state_action_counter
                        
print('Done.\n')

# Determining the set of actions

print('Creating set of valid actions...')

cm_keys = np.full((len(wind_states), 
                len(weather_conds), 
                u_max + 1,
                v_max + 1,
                max_state_action_counter),
                -1,
                dtype='int')

for ws in wind_states:
    for wc in weather_conds:
        for a_i in range(u_max + 1):
            for d_i in range(v_max + 1):
                state_action_counter = 0
                for rc in wind_states[ws]:
                    u_values = u[wc, rc]  # Precompute u values
                    v_values = v[wc, rc]  # Precompute v values
                    arr_lim = min(a_i, u_values[-1]) # Minimum arrivals (discarding clearly suboptimal actions)
                    arr_min = min(u_values[-1] - list(v_values[::-1]).index(v_values[0]), arr_lim) # Maxium arrivals
                    for arr in range(arr_min, arr_lim): # For each possible arrival count
                        if d_i >= v_values[arr]: # If the departure capacity is lower than the departure queue
                            dep = v_values[arr] # Departure service rates are given by the departure capacity
                            action = [rc, arr, dep]
                            cm_keys[ws, wc, a_i, d_i, state_action_counter] = cm_possible_actions.index(action)
                            state_action_counter += 1
                    dep = min(d_i, v_values[arr_lim]) # For the maximum arrival count, take the minimum between the departure queue and the capacity
                    action = [rc, arr_lim, dep]
                    cm_keys[ws, wc, a_i, d_i, state_action_counter] = cm_possible_actions.index(action)
                    state_action_counter += 1

cm_possible_actions = np.array(cm_possible_actions)

cm_valid_bool = np.full((len(wind_states), 
                len(weather_conds), 
                u_max + 1,
                v_max + 1,
                len(cm_possible_actions)),
                False,
                dtype='bool')

for ws in range(cm_keys.shape[0]):
    for wc in range(cm_keys.shape[1]):
        for a_i in range(cm_keys.shape[2]):
            for d_i in range(cm_keys.shape[3]):
                for state_action_counter in range(cm_keys.shape[4]):
                    idx = cm_keys[ws, wc, a_i, d_i, state_action_counter]
                    if idx != -1:
                        cm_valid_bool[ws, wc, a_i, d_i][idx] = True
        
print('All done.\n')

# All possible actions for strategic DCB

print('\nCreating the set of possible actions...')

g_max = 4

possible_actions = [[g_f] for g_f in range(0, g_max + 1)] # Possible actions list
                
print('Done.\n')

# Determining the set of actions

print('Creating set of all valid actions...')

keys = np.full((96,
                g_max + 1),
                -1,
                dtype='int')

for t in range(96):
    state_action_counter = 0
    for g_f in range(0, min(96 - t, g_max + 1)):
        action = [g_f]
        keys[t, state_action_counter] = possible_actions.index(action)
        state_action_counter += 1

valid_bool = np.full((96,
                      len(possible_actions)),
                      False,
                      dtype='bool')

for t in range(keys.shape[0]):
    for state_action_counter in range(keys.shape[1]):
        idx = keys[t, state_action_counter]
        if idx != -1:
            valid_bool[t][idx] = True
        
print('All done.\n')

possible_actions = np.array(possible_actions)

# Test days: 30 randomly selected days, held out from the weather probability estimates
test_days = ['2023-2-9', '2022-8-17', '2022-5-27', '2022-3-25', '2022-5-28', '2023-2-28', '2022-10-23', '2023-8-16', '2022-11-27', '2022-11-7', '2022-12-2', '2022-6-9', '2023-3-25', '2023-10-22', '2022-8-24', '2023-8-6', '2023-11-1', '2023-10-27', '2023-9-25', '2023-10-20', '2023-12-12', '2023-5-8', '2022-5-26', '2022-6-13', '2022-6-12', '2023-2-15', '2022-2-16', '2023-11-15', '2023-5-19', '2023-11-30'] # random.sample(test_days, k=min(30, len(test_days)))

###############################################################################
# Determining weather and wind uncertainty
###############################################################################

n_wc = len(weather_conds)
n_ws = len(wind_states)
p_wc_trans = np.zeros((n_wc, n_wc))
p_wc_error = np.zeros((n_wc, n_wc))
p_ws_trans = np.zeros((n_ws, n_ws))
p_ws_error = np.zeros((n_ws, n_ws))
prev_data = data.loc[:(data.shape[0]-2),:]
prev_data.reset_index(drop=True, inplace=True)
curr_data = data.loc[1:,]
curr_data.reset_index(drop=True, inplace=True)
test_mask = data['day'].isin(test_days).to_numpy()  # Test-day observations are not used in the estimates
for i in range(prev_data.shape[0]):
    # Transitions between consecutive 15-min slots only (the dataset has missing days)
    consecutive = curr_data.loc[i,'time'] - prev_data.loc[i,'time'] == datetime.timedelta(minutes=15)
    if consecutive and not test_mask[i] and not test_mask[i+1]:
        p_wc_trans[prev_data.loc[i,'weather_condition'], curr_data.loc[i,'weather_condition']] += 1
        p_ws_trans[prev_data.loc[i,'wind_state'], curr_data.loc[i,'wind_state']] += 1
for i in range(p_wc_trans.shape[0]):
    p_wc_trans[i] = p_wc_trans[i] / np.sum(p_wc_trans[i])
for i in range(p_ws_trans.shape[0]):
    p_ws_trans[i] = p_ws_trans[i] / np.sum(p_ws_trans[i])
for i in range(taf_data.shape[0]):
    if not test_mask[i]:
        p_wc_error[taf_data.loc[i,'weather_condition'], data.loc[i,'weather_condition']] += 1
        p_ws_error[taf_data.loc[i,'wind_state'], data.loc[i,'wind_state']] += 1
for i in range(p_wc_error.shape[0]):
    p_wc_error[i] = p_wc_error[i] / np.sum(p_wc_error[i])
for i in range(p_ws_error.shape[0]):
    p_ws_error[i] = p_ws_error[i] / np.sum(p_ws_error[i])

# plt.figure(dpi=2400)
plt.matshow(p_wc_trans)
plt.colorbar()
plt.title('Weather transition probability')
plt.savefig('wc_trans.pdf', format='pdf')
plt.show()

# plt.figure(dpi=2400)
plt.matshow(p_ws_trans)
plt.colorbar()
plt.title('Wind transition probability')
plt.savefig('ws_trans.pdf', format='pdf')
plt.show()

# plt.figure(dpi=2400)
plt.matshow(p_wc_error)
plt.colorbar()
plt.title('Weather confusion probability')
plt.savefig('wc_error.pdf', format='pdf')
plt.show()

# plt.figure(dpi=2400)
plt.matshow(p_ws_error)
plt.colorbar()
plt.title('Wind confusion probability')
plt.savefig('ws_error.pdf', format='pdf')
plt.show()

###############################################################################
# Filtering congested data
###############################################################################

print('Filtering congested data...')

# Observation low and high
cm_observation_low = [0,0,0,0,0]
cm_observation_high = [2,50,50,2,3]
# observation_low = [0,0,0,0,0,0,0,0,0,0]
# observation_high = [2,50,50,2,3,a_max,d_max,2,3,95]
observation_low = [0,0,0,0,0]
observation_high = [2,50,50,2,3]
congested_days = days.copy()

# # Environment for data filtering
# data_filter = cm.RunwayEnv(alpha = 2, delta = 0.3, 
#                            taf_data=taf_data, data=data, 
#                            training_days=days, 
#                            u=u, u_max=u_max, v=v, v_max=v_max,
#                            possible_actions=cm_possible_actions,
#                            valid_bool=cm_valid_bool,
#                            wind_states=wind_states, 
#                            observation_low=cm_observation_low, 
#                            observation_high=cm_observation_high)

# congested_days = [] # Initializing list of congested days

# for day in days: # For each day of the entire dataset

#     print(day)
    
#     # Compute single action following a greedy policy of a naive agent
#     state, info = data_filter.reset(day=day) # Test for the same day
#     test_reward = 0 # Initialize test reward
#     done = False # Not done
#     truncated = False # Not truncated
    
#     # While day has not ended
#     while not (done or truncated):
        
#         # Greedy policy of the naive agent
#         action = data_filter.base_action()
        
#         # Take the action and observe the state and reward
#         state, reward, done, truncated, info = data_filter.base_step(action)
        
#         # Updating return
#         test_reward += reward

#     # Store congested days
#     if test_reward <= -10:
#         congested_days.append(day)
    
# # Filtering
# data = data[data.day.isin(congested_days)].reset_index(drop=True)

# print('Total of ' + str(len(congested_days)) + ' congested days.\n')

###############################################################################
# Creating datasets
###############################################################################

# Cutting time window and splitting data
# print('Splitting data into training and test datasets...')
# n_days = len(congested_days) # Total number of days to be considered
# split_ratio = 0.7 # Percentage of the dataset for training
# n_days = min(n_days, len(congested_days))
# sampled_days = random.sample(congested_days,k=n_days)
# training_days = random.sample(sampled_days, k=int(split_ratio*n_days))
# test_days = list(np.array(sampled_days)[~np.isin(sampled_days,training_days)])
# test_days_2 = random.sample(test_days,k=int(0.01*n_days))
# training_days_2 = list(np.array(congested_days)[~np.isin(congested_days,test_days_2)])
# print('Dataset split into ' 
#       + str(len(training_days)) + ' days for training and '
#       + str(len(test_days)) + ' days for testing.')


# test_days = ['2023-9-10',
#  '2023-12-6',
#  '2022-9-19',
#  '2022-5-29',
#  '2023-6-1',
#  '2023-6-30',
#  '2023-1-29',
#  '2023-11-15',
#  '2022-8-2',
#  '2023-11-1',
#  '2023-1-30',
#  '2022-8-17',
#  '2023-12-7',
#  '2023-1-5',
#  '2022-3-18',
#  '2023-11-25',
#  '2023-3-9',
#  '2022-10-14',
#  '2023-3-7',
#  '2022-9-2',
#  '2023-10-7',
#  '2023-2-6',
#  '2022-5-23',
#  '2023-3-23',
#  '2023-5-19',
#  '2022-2-17',
#  '2023-5-3',
#  '2023-11-4',
#  '2022-1-16',
#  '2022-7-29',
#  '2023-2-1',
#  '2023-3-21',
#  '2023-8-8',
#  '2023-5-24',
#  '2022-8-20',
#  '2023-8-19',
#  '2023-4-28',
#  '2023-8-10',
#  '2022-8-1',
#  '2023-10-28',
#  '2022-10-4',
#  '2023-2-15',
#  '2022-11-24',
#  '2022-11-4',
#  '2022-10-29',
#  '2022-10-30',
#  '2022-11-20',
#  '2023-8-11',
#  '2022-12-7',
#  '2023-9-23',
#  '2023-6-29',
#  '2023-2-7',
#  '2022-2-28',
#  '2023-6-21',
#  '2022-9-13',
#  '2022-10-19',
#  '2023-9-18',
#  '2023-1-1',
#  '2022-6-12',
#  '2022-1-8',
#  '2023-7-9',
#  '2023-1-3',
#  '2022-12-2',
#  '2023-10-18',
#  '2022-12-22',
#  '2022-12-9',
#  '2023-2-10',
#  '2023-4-30',
#  '2022-11-19',
#  '2022-8-6',
#  '2023-8-3',
#  '2022-12-30',
#  '2022-5-10',
#  '2023-12-2',
#  '2023-7-17',
#  '2023-10-27',
#  '2022-11-12',
#  '2023-8-24',
#  '2023-7-21',
#  '2022-11-26',
#  '2023-11-23',
#  '2023-12-5',
#  '2023-4-12',
#  '2022-7-14',
#  '2022-5-27',
#  '2023-6-11',
#  '2023-5-12',
#  '2022-11-10',
#  '2022-7-9',
#  '2022-5-4',
#  '2022-3-22',
#  '2023-5-9',
#  '2022-10-8',
#  '2023-9-1',
#  '2023-10-8',
#  '2023-11-18',
#  '2023-12-13',
#  '2023-3-25',
#  '2022-10-17',
#  '2023-12-11',
#  '2022-8-24',
#  '2022-6-17',
#  '2022-9-26',
#  '2022-12-6',
#  '2022-11-1',
#  '2022-11-3',
#  '2023-10-15',
#  '2023-9-4',
#  '2022-11-16',
#  '2023-10-30',
#  '2022-3-30',
#  '2022-1-6',
#  '2023-8-16',
#  '2023-3-12',
#  '2022-2-16',
#  '2023-6-14',
#  '2022-8-3',
#  '2022-4-17',
#  '2022-11-7',
#  '2022-10-1',
#  '2023-3-31',
#  '2022-11-25',
#  '2023-9-16',
#  '2023-9-15',
#  '2023-1-18',
#  '2023-12-21',
#  '2023-1-12',
#  '2023-10-22',
#  '2023-6-17',
#  '2022-1-4',
#  '2023-10-20',
#  '2022-10-28',
#  '2022-6-4',
#  '2022-3-25',
#  '2023-4-8',
#  '2022-11-22',
#  '2022-4-30',
#  '2022-3-24',
#  '2022-2-26',
#  '2022-12-28',
#  '2022-4-25',
#  '2022-7-15',
#  '2023-11-30',
#  '2022-3-12',
#  '2022-1-28',
#  '2023-2-13',
#  '2023-5-23',
#  '2023-2-9',
#  '2022-6-9',
#  '2022-5-26',
#  '2022-4-23',
#  '2023-8-14',
#  '2022-11-27',
#  '2023-8-6',
#  '2022-5-16',
#  '2023-12-22',
#  '2022-2-11',
#  '2022-4-29',
#  '2022-5-19',
#  '2022-9-17',
#  '2022-7-24',
#  '2023-7-6',
#  '2022-10-23',
#  '2022-4-21',
#  '2023-9-25',
#  '2022-3-11',
#  '2022-10-21',
#  '2023-11-21',
#  '2022-7-2',
#  '2023-12-17',
#  '2023-1-24',
#  '2023-8-5',
#  '2022-5-3',
#  '2022-12-29',
#  '2023-4-26',
#  '2022-3-15',
#  '2022-5-28',
#  '2023-2-28',
#  '2022-6-15',
#  '2022-6-10',
#  '2022-4-14',
#  '2023-1-14',
#  '2022-1-24',
#  '2022-9-30',
#  '2022-12-10',
#  '2022-1-13',
#  '2023-12-12',
#  '2022-9-24',
#  '2023-3-14',
#  '2022-6-13',
#  '2023-6-8',
#  '2022-4-4',
#  '2023-3-13',
#  '2022-7-7',
#  '2022-1-2',
#  '2022-9-6',
#  '2022-8-11',
#  '2023-2-5',
#  '2023-11-8',
#  '2023-8-20',
#  '2023-5-25',
#  '2022-4-1',
#  '2023-5-30',
#  '2023-5-28',
#  '2023-9-22',
#  '2023-5-8',
#  '2022-9-27',
#  '2022-6-29',
#  '2023-11-5']

training_days = list(np.array(congested_days)[~np.isin(congested_days,test_days)])

###############################################################################
# Running Algorithm
###############################################################################

# %%
import importlib
importlib.reload(sdcb)

def compute_heuristic_baseline(env):
    """
    Compute baseline reward by applying the heuristic tactical action
    to the original flight schedule without any strategic delays using METAR data.
    """
    obs, _ = env.reset(mode='test', evaluate=True)
    terminated = False
    
    while not terminated:
        # Apply heuristic tactical action with no strategic delay (g_f = 0)
        obs, reward, terminated, truncated, info = env.step(0, mode='test', evaluate=True)
    
    # Return the accumulated objective (includes reset and step rewards)
    return env.objective

def compute_forecast_baseline(env):
    """
    Tactical cost of the original flight schedule (no strategic delay) on the forecast weather (TAF),
    the reference of the forecast check.
    """
    obs, _ = env.reset(mode='deploy', evaluate=True)
    terminated = False
    while not terminated:
        obs, reward, terminated, truncated, info = env.step(0, mode='deploy', evaluate=True)
    return env.objective

bresults = []
results = []
delayed_arr = []
all_delays = []  # Collect all imposed delays
daily_delays = {}  # Store delays per day
check_log = []  # Forecast check decisions per day

# Maskable PPO hyperparameters (Section III-E); the seed also seeds the environment's generator (np_random) at its first reset
PPO_KWARGS = dict(gamma=1, learning_rate=3e-4, batch_size=64, ent_coef=ENT_COEF, clip_range=0.2, seed=0, device='cpu')  # ENT_COEF: 0 in setting S1 (library default), 0.01 in S2
os.makedirs(MODEL_DIR, exist_ok=True)  # Trained models
os.makedirs(OUT_DIR, exist_ok=True)  # Result figures

for day in test_days:
    
    env = sdcb.RunwayEnv(alpha = 2, beta = 1, delta = 0.3, eta = 0.1,
                           data=data, taf_data=taf_data, arrival_data=arrival_data,
                           congested_days=congested_days, 
                           u=u, u_max=u_max, v=v, v_max=v_max,
                           possible_actions=possible_actions,
                           valid_bool=valid_bool,
                           wind_states=wind_states, 
                           observation_low=observation_low, 
                           observation_high=observation_high,
                           p_wc_trans=p_wc_trans, p_ws_trans=p_ws_trans,
                           p_wc_error=p_wc_error, p_ws_error=p_ws_error,
                           chosen_day=day, training_reward=TRAINING_REWARD)

    model = MaskablePPO("MlpPolicy", env, verbose=1, **PPO_KWARGS)
    file = MODEL_DIR + "/sdcb_ppo_fb_" + day + ".zip"
    file_path = Path(file)
    if (file_path).exists():
        print('Loading model...')
        prev_model = MaskablePPO.load(file, env=env) # Load model
        model.set_parameters(prev_model.get_parameters())
        print('Done.\n')
    else:
        print('Starting training process for day ' + day + '...')
        start = time.time()
        model.learn(500_000)
        end = time.time()
        print('Training finished in ' + str(round(end-start, 2)) + ' seconds.\n')
        print('Saving model...')
        model.save(MODEL_DIR + "/sdcb_ppo_fb_" + day + ".zip") # Save model
        print('Done.\n')
 
    print('Testing...')
    
    # Create baseline environment for heuristic comparison
    baseline_env = sdcb.RunwayEnv(alpha = 2, beta = 1, delta = 0.3, eta = 0.1,
                                 data=data, taf_data=taf_data, arrival_data=arrival_data,
                                 congested_days=congested_days, 
                                 u=u, u_max=u_max, v=v, v_max=v_max,
                                 possible_actions=possible_actions,
                                 valid_bool=valid_bool,
                                 wind_states=wind_states, 
                                 observation_low=observation_low, 
                                 observation_high=observation_high,
                                 p_wc_trans=p_wc_trans, p_ws_trans=p_ws_trans,
                                 p_wc_error=p_wc_error, p_ws_error=p_ws_error,
                                 chosen_day=day)
    
    # Compute baseline by applying heuristic to original flight schedule
    base_rew = compute_heuristic_baseline(baseline_env)
    
    deploy_env = sdcb.RunwayEnv(alpha = 2, beta = 1, delta = 0.3, eta = 0.1,
                               data=data, taf_data=taf_data, arrival_data=arrival_data,
                               congested_days=congested_days, 
                               u=u, u_max=u_max, v=v, v_max=v_max,
                               possible_actions=possible_actions,
                               valid_bool=valid_bool,
                               wind_states=wind_states, 
                               observation_low=observation_low, 
                               observation_high=observation_high,
                               p_wc_trans=p_wc_trans, p_ws_trans=p_ws_trans,
                               p_wc_error=p_wc_error, p_ws_error=p_ws_error,
                               chosen_day=day)
    
    # Compute single action using the model
    obs, _ = deploy_env.reset(mode='deploy', evaluate=True)
    terminated = False
    while not terminated:
        
        # Retrieve current action mask
        action_masks = get_action_masks(deploy_env)
        action_idx, _states = model.predict(obs, action_masks=action_masks, deterministic=True)
        obs, reward, terminated, truncated, info = deploy_env.step(action_idx, mode='deploy', evaluate=True)
    
    # Delays to deploy: the policy's choice or, with the forecast check, the original schedule when the delays do
    # not reduce the tactical cost of the day on the forecast (TAF) on which the policy chose them
    delays = list(deploy_env.imposed_delays)
    if FORECAST_CHECK:
        forecast_env = sdcb.RunwayEnv(alpha = 2, beta = 1, delta = 0.3, eta = 0.1,
                                     data=data, taf_data=taf_data, arrival_data=arrival_data,
                                     congested_days=congested_days,
                                     u=u, u_max=u_max, v=v, v_max=v_max,
                                     possible_actions=possible_actions,
                                     valid_bool=valid_bool,
                                     wind_states=wind_states,
                                     observation_low=observation_low,
                                     observation_high=observation_high,
                                     p_wc_trans=p_wc_trans, p_ws_trans=p_ws_trans,
                                     p_wc_error=p_wc_error, p_ws_error=p_ws_error,
                                     chosen_day=day)
        forecast_base = compute_forecast_baseline(forecast_env)
        kept = deploy_env.objective < forecast_base
        n_delayed = sum(g > 0 for g in delays)
        check_log.append((day, forecast_base, deploy_env.objective, kept, n_delayed))
        print('Forecast check: tactical cost on the forecast ' + str(round(forecast_base, 2)) + ' without the delays, '
              + str(round(deploy_env.objective, 2)) + ' with them -> '
              + ('no delays proposed' if n_delayed == 0 else 'delays kept' if kept else 'original schedule kept'))
        if not kept:
            delays = [0] * len(delays)
    
    # Plotting results 
    plt.show()
    plt.plot(deploy_env.arr_delayed)
    plt.title(day)
    plt.xlabel('Time steps')
    plt.ylabel('Delayed arrivals')
    plt.savefig(OUT_DIR + '/sdcb_ppo_policy_' + day + '.pdf', format='pdf')
    plt.show()
        
    # Compute single action using the model
    obs, info = env.reset(mode='test', evaluate=True)
    # model_rew = 0
    terminated = False
    counter = 0
    while not terminated:
        
        obs, reward, terminated, truncated, info = env.step(delays[counter], 
                                                            mode='test', 
                                                            evaluate=True)  
        counter += 1
    model_rew = env.objective
    
    print('Day: ' + day
          + ', strategic model reward: ' + str(round(model_rew,2))
          + ', baseline policy reward: ' + str(round(base_rew,2)))
    
    # Saving results
    bresults.append(base_rew)
    results.append(model_rew)
    
    # Collect delays for distribution analysis
    all_delays.extend(delays)
    daily_delays[day] = list(delays)
    
    print('Done.\n')

if FORECAST_CHECK:
    print('\n' + '=' * 60)
    print('FORECAST CHECK')
    print('=' * 60)
    proposed = [entry for entry in check_log if entry[4] > 0]
    print('Delays proposed on ' + str(len(proposed)) + ' of ' + str(len(check_log)) + ' days; kept on '
          + str(sum(1 for entry in proposed if entry[3])) + ', original schedule kept on the others.')
    for day_, cost_without, cost_with, kept, n_delayed in check_log:
        print('  ' + day_ + ': forecast cost ' + str(round(cost_without, 2)) + ' without delays, '
              + str(round(cost_with, 2)) + ' with ' + str(n_delayed) + ' delayed flights -> '
              + ('no delays proposed' if n_delayed == 0 else 'kept' if kept else 'rejected'))

# Plotting results 
plt.show()
plt.plot(results,'r')
plt.plot(bresults, 'b')
plt.legend(['SDCB','Baseline'])
plt.xlabel('Days')
plt.ylabel('Tactical Cost')
plt.savefig(OUT_DIR + '/sdcb_ppo_results.pdf', format='pdf')
plt.show()

# Plotting improvement
plt.figure(dpi=1200)
improv_list = np.array(results) - np.array(bresults)
plt.title('SDCB vs Baseline')
plt.hist(improv_list, bins=50)
plt.ylabel('Frequency')
plt.xlabel('Difference')
plt.savefig(OUT_DIR + '/sdcb_ppo_base_improvement.pdf', format='pdf')
plt.show()

# ============================================================================
# DELAY DISTRIBUTION ANALYSIS
# ============================================================================

print('='*80)
print('ALLOCATED DELAYS DISTRIBUTION ANALYSIS')
print('='*80 + '\n')

all_delays_arr = np.array(all_delays)

print(f'[DELAY STATISTICS]')
print('-'*80)
print(f'{"Total number of time slots":<30} {len(all_delays_arr):>20}')
print(f'{"Mean delay (hours)":<30} {all_delays_arr.mean():>20.4f}')
print(f'{"Median delay (hours)":<30} {np.median(all_delays_arr):>20.4f}')
print(f'{"Min delay (hours)":<30} {all_delays_arr.min():>20.4f}')
print(f'{"Max delay (hours)":<30} {all_delays_arr.max():>20.4f}')
print(f'{"Std Dev delay (hours)":<30} {all_delays_arr.std():>20.4f}')
print(f'{"% Time slots with delays":<30} {(np.sum(all_delays_arr > 0) / len(all_delays_arr) * 100):>19.2f}%')
print(f'{"% Time slots no delay":<30} {(np.sum(all_delays_arr == 0) / len(all_delays_arr) * 100):>19.2f}%')

# ============================================================================
# HISTOGRAM: Delay Distribution
# ============================================================================
plt.figure(figsize=(12, 6))
bins = np.linspace(all_delays_arr.min(), all_delays_arr.max(), 50)
plt.hist(all_delays_arr, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(all_delays_arr.mean(), color='r', linestyle='--', linewidth=2.5, label=f'Mean: {all_delays_arr.mean():.4f} h')
plt.axvline(np.median(all_delays_arr), color='g', linestyle='--', linewidth=2.5, label=f'Median: {np.median(all_delays_arr):.4f} h')
plt.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
plt.xlabel('Allocated Delay (hours)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of Allocated Delays Across All Time Slots', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(OUT_DIR + '/sdcb_delay_distribution_histogram.pdf', format='pdf', bbox_inches='tight')
plt.show()

# ============================================================================
# BOX PLOT: Delay Distribution
# ============================================================================
plt.figure(figsize=(10, 6))
bp = plt.boxplot(all_delays_arr, vert=True, patch_artist=True, tick_labels=['Allocated Delays'])
bp['boxes'][0].set_facecolor('lightblue')
bp['whiskers'][0].set_color('black')
bp['whiskers'][1].set_color('black')
bp['caps'][0].set_color('black')
bp['caps'][1].set_color('black')
plt.ylabel('Delay (hours)', fontsize=12)
plt.title('Box Plot of Allocated Delays', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR + '/sdcb_delay_boxplot.pdf', format='pdf', bbox_inches='tight')
plt.show()

# ============================================================================
# BAR PLOT: Delays by Day
# ============================================================================
plt.figure(figsize=(14, 6))
daily_means = [np.mean(daily_delays[day]) for day in test_days]
daily_stds = [np.std(daily_delays[day]) for day in test_days]
x_pos = np.arange(len(test_days))
plt.bar(x_pos, daily_means, yerr=daily_stds, capsize=5, color='steelblue', alpha=0.7, edgecolor='black')
plt.xlabel('Days', fontsize=12)
plt.ylabel('Mean Allocated Delay (hours)', fontsize=12)
plt.title('Mean Allocated Delays by Day (with Std Dev)', fontsize=14, fontweight='bold')
plt.xticks(x_pos, test_days, rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR + '/sdcb_delay_by_day.pdf', format='pdf', bbox_inches='tight')
plt.show()

# ============================================================================
# TIME SERIES: Cumulative Delays
# ============================================================================
plt.figure(figsize=(14, 6))
cumulative_delays = np.cumsum(all_delays_arr)
plt.plot(cumulative_delays, 'b-', linewidth=2, alpha=0.8)
plt.fill_between(range(len(cumulative_delays)), cumulative_delays, alpha=0.3, color='blue')
plt.xlabel('Time Slot Index', fontsize=12)
plt.ylabel('Cumulative Delay (hours)', fontsize=12)
plt.title('Cumulative Allocated Delays Over Time', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR + '/sdcb_cumulative_delays.pdf', format='pdf', bbox_inches='tight')
plt.show()

# ============================================================================
# HISTOGRAM: Non-zero delays only
# ============================================================================
non_zero_delays = all_delays_arr[all_delays_arr > 0]
if len(non_zero_delays) > 0:
    plt.figure(figsize=(12, 6))
    plt.hist(non_zero_delays, bins=40, color='coral', edgecolor='black', alpha=0.7)
    plt.axvline(non_zero_delays.mean(), color='r', linestyle='--', linewidth=2.5, label=f'Mean: {non_zero_delays.mean():.4f} h')
    plt.axvline(np.median(non_zero_delays), color='g', linestyle='--', linewidth=2.5, label=f'Median: {np.median(non_zero_delays):.4f} h')
    plt.xlabel('Allocated Delay (hours)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Non-Zero Allocated Delays', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(OUT_DIR + '/sdcb_delay_distribution_nonzero.pdf', format='pdf', bbox_inches='tight')
    plt.show()
    
    print(f'\n[NON-ZERO DELAY STATISTICS]')
    print('-'*80)
    print(f'{"Number of non-zero delays":<30} {len(non_zero_delays):>20}')
    print(f'{"% of all delays":<30} {(len(non_zero_delays) / len(all_delays_arr) * 100):>19.2f}%')
    print(f'{"Mean non-zero delay (hours)":<30} {non_zero_delays.mean():>20.4f}')
    print(f'{"Median non-zero delay (hours)":<30} {np.median(non_zero_delays):>20.4f}')
    print(f'{"Max non-zero delay (hours)":<30} {non_zero_delays.max():>20.4f}')
else:
    print(f'No non-zero delays found')

# ============================================================================
# DELAY COUNT DISTRIBUTION (0, 1, 2, 3, 4 time-slots)
# ============================================================================

print('\n[DELAY COUNT DISTRIBUTION]')
print('-'*80)

# Count flights in each delay category
delay_counts = {}
for i in range(5):  # 0 to 4 time-slots
    count = np.sum(all_delays_arr == i)
    delay_counts[i] = count
    percentage = (count / len(all_delays_arr)) * 100
    print(f'{"Flights with " + str(i) + " time-slot delay(s)":<30} {count:>20} ({percentage:>6.2f}%)')

# ============================================================================
# BAR PLOT: Delay Count Distribution
# ============================================================================
plt.figure(figsize=(10, 6))
delay_values = list(delay_counts.keys())
delay_frequencies = list(delay_counts.values())
colors = ['green' if i == 0 else 'steelblue' for i in delay_values]
bars = plt.bar([str(i) for i in delay_values], delay_frequencies, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}\n({height/len(all_delays_arr)*100:.1f}%)',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.xlabel('Number of Time-Slot Delays', fontsize=12)
plt.ylabel('Number of Flights', fontsize=12)
plt.title('Distribution of Flights by Delay Category', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR + '/sdcb_delay_count_distribution.pdf', format='pdf', bbox_inches='tight')
plt.show()

# ============================================================================
# PIE CHART: Delay Count Distribution
# ============================================================================
plt.figure(figsize=(10, 8))
labels = [f'{i} time-slot{"s" if i != 1 else ""}' for i in delay_values]
colors_pie = ['#90EE90', '#4472C4', '#ED7D31', '#A5A5A5', '#FFC000']
explode = (0.05, 0, 0, 0, 0)  # Explode the 0-delay slice
plt.pie(delay_frequencies, labels=labels, autopct='%1.1f%%', startangle=90, 
        colors=colors_pie[:len(delay_frequencies)], explode=explode, textprops={'fontsize': 11})
plt.title('Distribution of Delays Across All Time Slots', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_DIR + '/sdcb_delay_count_pie.pdf', format='pdf', bbox_inches='tight')
plt.show()

print('Done.\n')

# ============================================================================
# PLOTTING STATISTICS FOR ALL RESULTS (NO PERCENTILE FILTER)
# ============================================================================

print('='*80)
print('STATISTICS FOR ALL RESULTS (NO PERCENTILE FILTER)')
print('='*80 + '\n')

results_all = np.array(results)
bresults_all = np.array(bresults)
improv_all = results_all - bresults_all

print(f'{"Metric":<30} {"SDCB":>20} {"Baseline":>20}')
print('-'*80)
print(f'{"Min":<30} {results_all.min():>20.2f} {bresults_all.min():>20.2f}')
print(f'{"Max":<30} {results_all.max():>20.2f} {bresults_all.max():>20.2f}')
print(f'{"Mean":<30} {results_all.mean():>20.2f} {bresults_all.mean():>20.2f}')
print(f'{"Median":<30} {np.median(results_all):>20.2f} {np.median(bresults_all):>20.2f}')
print(f'{"Std Dev":<30} {results_all.std():>20.2f} {bresults_all.std():>20.2f}')

print(f'\nImprovement Metrics:')
print('-'*80)
print(f'{"Total Improvement":<30} {improv_all.sum():>20.2f}')
print(f'{"Mean Improvement":<30} {improv_all.mean():>20.2f}')
print(f'{"Median Improvement":<30} {np.median(improv_all):>20.2f}')
# Safe percentage improvement calculation
valid_idx_all = np.abs(bresults_all) > 0
if np.any(valid_idx_all):
    pct_improv_all_valid = (improv_all[valid_idx_all] / np.abs(bresults_all[valid_idx_all]) * 100)
    print(f'{"Mean % Improvement":<30} {pct_improv_all_valid.mean():>19.2f}%')
else:
    print(f'{"Mean % Improvement":<30} {"N/A (all zero baseline)":>19}%')

# Line plot for all results
plt.figure(figsize=(14, 6))
plt.plot(results_all, 'r-', marker='o', linewidth=2, markersize=6, label='SDCB', alpha=0.8)
plt.plot(bresults_all, 'b-', marker='s', linewidth=2, markersize=6, label='Baseline', alpha=0.8)
plt.legend(fontsize=12)
plt.xlabel('Days', fontsize=12)
plt.ylabel('Tactical Cost', fontsize=12)
plt.title('SDCB vs Baseline: All Results (No Filter)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR + '/sdcb_ppo_results_all_nofilter.pdf', format='pdf', bbox_inches='tight')
plt.show()

# Histogram of improvements for all results
plt.figure(figsize=(12, 6))
plt.hist(improv_all, bins=25, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(improv_all.mean(), color='r', linestyle='--', linewidth=2.5, label=f'Mean: {improv_all.mean():.2f}')
plt.axvline(np.median(improv_all), color='g', linestyle='--', linewidth=2.5, label=f'Median: {np.median(improv_all):.2f}')
plt.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
plt.xlabel('Improvement (SDCB - Baseline)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of Daily Improvements (All Results, No Filter)', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(OUT_DIR + '/sdcb_ppo_base_improvement_all_nofilter.pdf', format='pdf', bbox_inches='tight')
plt.show()

# Box plot comparison for all results
plt.figure(figsize=(10, 6))
bp = plt.boxplot([results_all, bresults_all], tick_labels=['SDCB', 'Baseline'], patch_artist=True, widths=0.6)
bp['boxes'][0].set_facecolor('lightcoral')
bp['boxes'][1].set_facecolor('lightblue')
plt.ylabel('Tactical Cost', fontsize=12)
plt.title('SDCB vs Baseline: Box Plot Comparison (All Results)', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR + '/sdcb_vs_baseline_boxplot_all_nofilter.pdf', format='pdf', bbox_inches='tight')
plt.show()

# Scatter plot for all results
plt.figure(figsize=(10, 6))
plt.scatter(bresults_all, results_all, alpha=0.6, s=120, edgecolors='darkblue', linewidth=1.5, color='coral')
min_val = min(bresults_all.min(), results_all.min())
max_val = max(bresults_all.max(), results_all.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2.5, label='Equal Performance')
plt.xlabel('Baseline Cost', fontsize=12)
plt.ylabel('SDCB Cost', fontsize=12)
plt.title('SDCB vs Baseline: Scatter Plot (All Results)', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR + '/sdcb_vs_baseline_scatter_all_nofilter.pdf', format='pdf', bbox_inches='tight')
plt.show()

# Cost reductions for all results, plotted as positive values when SDCB lowers
# the tactical cost relative to the baseline.
plt.figure(figsize=PAPER_BAR_FIGSIZE)
cost_reduction_all = bresults_all - results_all
sorted_indices = np.argsort(cost_reduction_all)[::-1]
cumulative_reduction_all = np.cumsum(cost_reduction_all[sorted_indices])
colors = ['green' if x > 0 else 'red' if x < 0 else 'lightgray' for x in cost_reduction_all[sorted_indices]]
plt.bar(range(len(sorted_indices)), cost_reduction_all[sorted_indices], color=colors, alpha=0.7, edgecolor='black')
plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
plt.xlabel('Days (sorted by cost reduction)', fontsize=PAPER_BAR_LABEL_FONTSIZE)
plt.ylabel('Cost Reduction', fontsize=PAPER_BAR_LABEL_FONTSIZE)
plt.title('Daily Cost Reductions', fontsize=PAPER_BAR_TITLE_FONTSIZE, fontweight='bold')
plt.tick_params(axis='both', labelsize=PAPER_BAR_TICK_FONTSIZE)
plt.grid(True, alpha=0.3, axis='y')
plt.subplots_adjust(left=0.18, bottom=0.18, top=0.88)
plt.tight_layout()
plt.savefig(OUT_DIR + '/sdcb_improvements_sorted_all_nofilter.pdf', format='pdf', bbox_inches='tight')
plt.show()

# Violin plot for all results
plt.figure(figsize=(10, 6))
parts = plt.violinplot([results_all, bresults_all], positions=[1, 2], showmeans=True, showmedians=True)
for pc in parts['bodies']:
    pc.set_facecolor('lightblue')
    pc.set_alpha(0.7)
plt.xticks([1, 2], ['SDCB', 'Baseline'], fontsize=PAPER_TICK_FONTSIZE)
plt.ylabel('Tactical Cost', fontsize=PAPER_LABEL_FONTSIZE)
plt.title('SDCB vs Baseline: Distribution Density (Violin Plot, All Results)', fontsize=PAPER_TITLE_FONTSIZE, fontweight='bold')
plt.tick_params(axis='y', labelsize=PAPER_TICK_FONTSIZE)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR + '/sdcb_vs_baseline_violin_all_nofilter.pdf', format='pdf', bbox_inches='tight')
plt.show()

print('\nDone.\n')

# Plotting results filtered by 90th percentile of baseline (worst 90% values)
percentile_90 = np.percentile(bresults, 90)
mask_90 = np.array(bresults) <= percentile_90
results_filtered = np.array(results)[mask_90]
bresults_filtered = np.array(bresults)[mask_90]

plt.figure()
plt.plot(results_filtered, 'r', marker='o', label='SDCB')
plt.plot(bresults_filtered, 'b', marker='s', label='Baseline')
plt.legend(['SDCB','Baseline'])
plt.xlabel('Days (90th percentile filtered)')
plt.ylabel('Tactical Cost')
plt.title(f'Results - Baseline <= 90th Percentile ({percentile_90:.2f})')
plt.savefig(OUT_DIR + '/sdcb_ppo_results_90p.pdf', format='pdf')
plt.show()

# Plotting improvement for 90th percentile filtered data
plt.figure(dpi=1200)
improv_list_90p = results_filtered - bresults_filtered
plt.title('SDCB vs Baseline (90th Percentile Filtered)')
plt.hist(improv_list_90p, bins=20)
plt.ylabel('Frequency')
plt.xlabel('Difference')
plt.axvline(np.mean(improv_list_90p), color='r', linestyle='--', label=f'Mean: {np.mean(improv_list_90p):.2f}')
plt.legend()
plt.savefig(OUT_DIR + '/sdcb_ppo_base_improvement_90p.pdf', format='pdf')
plt.show()

print(f'90th percentile of baseline: {percentile_90:.2f}')
print(f'Number of days with baseline <= 90th percentile: {len(results_filtered)}/{len(results)}')
print(f'Mean improvement (90th percentile): {np.mean(improv_list_90p):.2f}')
print(f'Mean improvement (90th percentile): {(np.mean(improv_list_90p) / np.mean(np.abs(bresults_filtered)) * 100):.2f}%')
print(f'Median improvement (90th percentile): {np.median(improv_list_90p):.2f}')
print('Done.\n')

# %%
###############################################################################
# Extended Metrics and Comparison Analysis
###############################################################################

print('\n' + '='*80)
print('COMPREHENSIVE SDCB vs BASELINE COMPARISON')
print('='*80 + '\n')

results_arr = np.array(results)
bresults_arr = np.array(bresults)

# ============================================================================
# 1. DESCRIPTIVE STATISTICS
# ============================================================================
print('[1] DESCRIPTIVE STATISTICS')
print('-'*80)
print(f'{"Metric":<30} {"SDCB":>20} {"Baseline":>20}')
print('-'*80)
print(f'{"Min":<30} {results_arr.min():>20.2f} {bresults_arr.min():>20.2f}')
print(f'{"Max":<30} {results_arr.max():>20.2f} {bresults_arr.max():>20.2f}')
print(f'{"Mean":<30} {results_arr.mean():>20.2f} {bresults_arr.mean():>20.2f}')
print(f'{"Median":<30} {np.median(results_arr):>20.2f} {np.median(bresults_arr):>20.2f}')
print(f'{"Std Dev":<30} {results_arr.std():>20.2f} {bresults_arr.std():>20.2f}')
print(f'{"25th Percentile":<30} {np.percentile(results_arr, 25):>20.2f} {np.percentile(bresults_arr, 25):>20.2f}')
print(f'{"75th Percentile":<30} {np.percentile(results_arr, 75):>20.2f} {np.percentile(bresults_arr, 75):>20.2f}')
print(f'{"IQR":<30} {np.percentile(results_arr, 75) - np.percentile(results_arr, 25):>20.2f} {np.percentile(bresults_arr, 75) - np.percentile(bresults_arr, 25):>20.2f}')

# ============================================================================
# 2. IMPROVEMENT METRICS
# ============================================================================
print(f'\n[2] IMPROVEMENT METRICS')
print('-'*80)
improv_arr = results_arr - bresults_arr
# Safe percentage improvement calculation
valid_idx_arr = np.abs(bresults_arr) > 0
improv_pct = np.full_like(improv_arr, np.nan, dtype=float)
improv_pct[valid_idx_arr] = (improv_arr[valid_idx_arr] / np.abs(bresults_arr[valid_idx_arr])) * 100

print(f'{"Total Improvement":<30} {improv_arr.sum():>20.2f}')
print(f'{"Mean Improvement":<30} {improv_arr.mean():>20.2f}')
print(f'{"Median Improvement":<30} {np.median(improv_arr):>20.2f}')
print(f'{"Min Improvement":<30} {improv_arr.min():>20.2f}')
print(f'{"Max Improvement":<30} {improv_arr.max():>20.2f}')
valid_pct = improv_pct[~np.isnan(improv_pct)]
if len(valid_pct) > 0:
    print(f'{"Mean % Improvement":<30} {np.nanmean(improv_pct):>19.2f}%')
    print(f'{"Median % Improvement":<30} {np.nanmedian(improv_pct):>19.2f}%')
else:
    print(f'{"Mean % Improvement":<30} {"N/A (all zero baseline)":>19}%')
    print(f'{"Median % Improvement":<30} {"N/A (all zero baseline)":>19}%')

# ============================================================================
# 3. WIN/LOSS ANALYSIS
# ============================================================================
print(f'\n[3] WIN/LOSS ANALYSIS')
print('-'*80)
wins = np.sum(results_arr > bresults_arr)
losses = np.sum(results_arr < bresults_arr)
ties = np.sum(results_arr == bresults_arr)
win_rate = (wins / len(results_arr)) * 100

print(f'{"SDCB Wins":<30} {wins:>20} ({win_rate:>6.2f}%)')
print(f'{"SDCB Losses":<30} {losses:>20} ({100-win_rate-ties/len(results_arr)*100:>6.2f}%)')
print(f'{"Ties":<30} {ties:>20} ({ties/len(results_arr)*100:>6.2f}%)')
print(f'{"Average Win Margin":<30} {improv_arr[results_arr > bresults_arr].mean() if wins > 0 else 0:>20.2f}')
print(f'{"Average Loss Margin":<30} {abs(improv_arr[results_arr < bresults_arr].mean()) if losses > 0 else 0:>20.2f}')

# ============================================================================
# 4. STATISTICAL TESTS
# ============================================================================
print(f'\n[4] STATISTICAL TESTS')
print('-'*80)
from scipy.stats import ttest_rel, wilcoxon

# Paired t-test
t_stat, t_pval = ttest_rel(results_arr, bresults_arr)
print(f'Paired t-test:')
print(f'  t-statistic: {t_stat:.4f}')
print(f'  p-value: {t_pval:.6f}')
print(f'  Significant: {"Yes" if t_pval < 0.05 else "No"} (alpha=0.05)')

# Wilcoxon signed-rank test
w_stat, w_pval = wilcoxon(results_arr, bresults_arr)
print(f'\nWilcoxon signed-rank test:')
print(f'  statistic: {w_stat:.4f}')
print(f'  p-value: {w_pval:.6f}')
print(f'  Significant: {"Yes" if w_pval < 0.05 else "No"} (alpha=0.05)')

# ============================================================================
# 5. BOX PLOT COMPARISON
# ============================================================================
print(f'\n[5] GENERATING VISUALIZATIONS...')
plt.figure(figsize=(10, 6))
bp = plt.boxplot([results_arr, bresults_arr], tick_labels=['SDCB', 'Baseline'], patch_artist=True)
bp['boxes'][0].set_facecolor('lightcoral')
bp['boxes'][1].set_facecolor('lightblue')
plt.ylabel('Tactical Cost')
plt.title('SDCB vs Baseline: Distribution Comparison')
plt.grid(axis='y', alpha=0.3)
plt.savefig(OUT_DIR + '/sdcb_vs_baseline_boxplot.pdf', format='pdf', bbox_inches='tight')
plt.show()

# ============================================================================
# 6. SCATTER PLOT
# ============================================================================
plt.figure(figsize=(10, 6))
plt.scatter(bresults_arr, results_arr, alpha=0.6, s=100, edgecolors='black', linewidth=1)
# Add diagonal line for reference
min_val = min(bresults_arr.min(), results_arr.min())
max_val = max(bresults_arr.max(), results_arr.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Equal Performance')
plt.xlabel('Baseline Cost')
plt.ylabel('SDCB Cost')
plt.title('SDCB vs Baseline: Scatter Plot')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(OUT_DIR + '/sdcb_vs_baseline_scatter.pdf', format='pdf', bbox_inches='tight')
plt.show()

# ============================================================================
# 7. CUMULATIVE IMPROVEMENT PLOT
# ============================================================================
plt.figure(figsize=(12, 6))
sorted_indices = np.argsort(improv_arr)[::-1]  # Sort by improvement (descending)
cumulative_improv = np.cumsum(improv_arr[sorted_indices])
plt.bar(range(len(sorted_indices)), improv_arr[sorted_indices], color=['green' if x > 0 else 'red' for x in improv_arr[sorted_indices]])
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
plt.xlabel('Days (sorted by improvement)', fontsize=PAPER_LABEL_FONTSIZE)
plt.ylabel('Improvement (SDCB - Baseline)', fontsize=PAPER_LABEL_FONTSIZE)
plt.title('Daily Improvements Sorted (Descending)', fontsize=PAPER_TITLE_FONTSIZE, fontweight='bold')
plt.tick_params(axis='both', labelsize=PAPER_TICK_FONTSIZE)
plt.grid(axis='y', alpha=0.3)
plt.savefig(OUT_DIR + '/sdcb_improvements_sorted.pdf', format='pdf', bbox_inches='tight')
plt.show()

# ============================================================================
# 8. VIOLIN PLOT
# ============================================================================
plt.figure(figsize=(10, 6))
parts = plt.violinplot([results_arr, bresults_arr], positions=[1, 2], showmeans=True, showmedians=True)
plt.xticks([1, 2], ['SDCB', 'Baseline'], fontsize=PAPER_FOCUS_TICK_FONTSIZE)
plt.ylabel('Tactical Cost', fontsize=PAPER_FOCUS_LABEL_FONTSIZE)
plt.title('SDCB vs Baseline: Distribution (Violin Plot)', fontsize=PAPER_FOCUS_TITLE_FONTSIZE, fontweight='bold')
plt.tick_params(axis='y', labelsize=PAPER_FOCUS_TICK_FONTSIZE)
plt.grid(axis='y', alpha=0.3)
plt.savefig(OUT_DIR + '/sdcb_vs_baseline_violin.pdf', format='pdf', bbox_inches='tight')
plt.show()

# ============================================================================
# 9. SUMMARY STATISTICS TABLE
# ============================================================================
plt.figure(figsize=(12, 6))
plt.axis('tight')
plt.axis('off')

summary_data = [
    ['Metric', 'SDCB', 'Baseline', 'Difference'],
    ['Mean', f'{results_arr.mean():.2f}', f'{bresults_arr.mean():.2f}', f'{results_arr.mean() - bresults_arr.mean():.2f}'],
    ['Median', f'{np.median(results_arr):.2f}', f'{np.median(bresults_arr):.2f}', f'{np.median(results_arr) - np.median(bresults_arr):.2f}'],
    ['Std Dev', f'{results_arr.std():.2f}', f'{bresults_arr.std():.2f}', f'{results_arr.std() - bresults_arr.std():.2f}'],
    ['Min', f'{results_arr.min():.2f}', f'{bresults_arr.min():.2f}', f'{results_arr.min() - bresults_arr.min():.2f}'],
    ['Max', f'{results_arr.max():.2f}', f'{bresults_arr.max():.2f}', f'{results_arr.max() - bresults_arr.max():.2f}'],
    ['Win Rate', f'{win_rate:.2f}%', f'{100-win_rate:.2f}%', '-'],
]

table = plt.table(cellText=summary_data, cellLoc='center', loc='center', 
                  colWidths=[0.25, 0.25, 0.25, 0.25])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Color header row
for i in range(4):
    table[(0, i)].set_facecolor('#40466e')
    table[(0, i)].set_text_props(weight='bold', color='white')

plt.title('Summary Statistics: SDCB vs Baseline', fontsize=14, weight='bold', pad=20)
plt.savefig(OUT_DIR + '/sdcb_vs_baseline_summary_table.pdf', format='pdf', bbox_inches='tight')
plt.show()

print('Done.\n')

# ============================================================================
# PERCENTAGE IMPROVEMENT VISUALIZATIONS
# ============================================================================
print('\n' + '='*80)
print('PERCENTAGE IMPROVEMENT ANALYSIS')
print('='*80 + '\n')

# Calculate percentage improvements (safe division handling zero baseline values)
valid_baseline_idx = np.abs(bresults_arr) > 0
improv_pct = np.full_like(improv_arr, np.nan, dtype=float)
improv_pct[valid_baseline_idx] = (improv_arr[valid_baseline_idx] / np.abs(bresults_arr[valid_baseline_idx])) * 100
pct_reduction = np.full_like(improv_arr, np.nan, dtype=float)
pct_reduction[valid_baseline_idx] = ((bresults_arr[valid_baseline_idx] - results_arr[valid_baseline_idx]) / np.abs(bresults_arr[valid_baseline_idx])) * 100

# Filter out NaN values for statistics
improv_pct_valid = improv_pct[~np.isnan(improv_pct)]
pct_reduction_valid = pct_reduction[~np.isnan(pct_reduction)]
num_zero_baseline = np.sum(~valid_baseline_idx)

print(f'[PERCENTAGE IMPROVEMENT METRICS]')
print('-'*80)
if len(improv_pct_valid) > 0:
    print(f'{"Mean % Improvement":<30} {np.nanmean(improv_pct):>19.2f}%')
    print(f'{"Median % Improvement":<30} {np.nanmedian(improv_pct):>19.2f}%')
    print(f'{"Min % Improvement":<30} {np.nanmin(improv_pct):>19.2f}%')
    print(f'{"Max % Improvement":<30} {np.nanmax(improv_pct):>19.2f}%')
    print(f'{"Std Dev % Improvement":<30} {np.nanstd(improv_pct):>19.2f}%')
    print(f'{"% Days with Improvement":<30} {(np.nansum(improv_pct > 0) / len(improv_pct_valid) * 100):>19.2f}%')
    print(f'{"Valid days (non-zero baseline)":<30} {len(improv_pct_valid):>20}')
    print(f'{"Days excluded (zero baseline)":<30} {num_zero_baseline:>20}')
else:
    print('No valid baseline values (all zero) - cannot calculate percentage improvement')

# ============================================================================
# HISTOGRAM: Percentage Improvements
# ============================================================================
plt.figure(figsize=(12, 6))
plt.hist(improv_pct, bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
plt.axvline(improv_pct.mean(), color='r', linestyle='--', linewidth=2.5, label=f'Mean: {improv_pct.mean():.2f}%')
plt.axvline(np.median(improv_pct), color='b', linestyle='--', linewidth=2.5, label=f'Median: {np.median(improv_pct):.2f}%')
plt.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
plt.xlabel('Percentage Improvement (%)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of Percentage Improvements (SDCB vs Baseline)', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(OUT_DIR + '/sdcb_pct_improvement_histogram.pdf', format='pdf', bbox_inches='tight')
plt.show()

# ============================================================================
# LINE PLOT: Percentage Improvements Over Days
# ============================================================================
plt.figure(figsize=(14, 6))
plt.plot(improv_pct, 'g-', marker='o', linewidth=2, markersize=6, alpha=0.8)
plt.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
plt.axhline(y=improv_pct.mean(), color='r', linestyle='--', linewidth=2, label=f'Mean: {improv_pct.mean():.2f}%', alpha=0.8)
plt.xlabel('Days', fontsize=12)
plt.ylabel('Percentage Improvement (%)', fontsize=12)
plt.title('Daily Percentage Improvements (SDCB vs Baseline)', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR + '/sdcb_pct_improvement_timeline.pdf', format='pdf', bbox_inches='tight')
plt.show()

# ============================================================================
# BAR PLOT: Percentage Improvements (Sorted)
# ============================================================================
plt.figure(figsize=PAPER_BAR_FIGSIZE)
pct_reduction_plot = np.nan_to_num(pct_reduction, nan=0.0)
sorted_indices_pct = np.argsort(pct_reduction_plot)[::-1]  # Sort by percentage reduction (descending)
colors_pct = ['green' if x > 0 else 'red' if x < 0 else 'lightgray' for x in pct_reduction_plot[sorted_indices_pct]]
plt.bar(range(len(sorted_indices_pct)), pct_reduction_plot[sorted_indices_pct], color=colors_pct, alpha=0.7, edgecolor='black')
plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
plt.xlabel('Days (sorted by % reduction)', fontsize=PAPER_BAR_LABEL_FONTSIZE)
plt.ylabel('Cost Reduction (%)', fontsize=PAPER_BAR_LABEL_FONTSIZE)
plt.title('Daily Percentage Cost Reductions', fontsize=PAPER_BAR_TITLE_FONTSIZE, fontweight='bold')
plt.tick_params(axis='both', labelsize=PAPER_BAR_TICK_FONTSIZE)
plt.grid(True, alpha=0.3, axis='y')
plt.subplots_adjust(left=0.18, bottom=0.18, top=0.88)
plt.tight_layout()
plt.savefig(OUT_DIR + '/sdcb_pct_improvements_sorted.pdf', format='pdf', bbox_inches='tight')
plt.show()

# ============================================================================
# BOX PLOT: Percentage Improvement Distribution
# ============================================================================
if len(improv_pct_valid) > 0:
    plt.figure(figsize=(10, 6))
    bp_pct = plt.boxplot(improv_pct_valid, vert=True, patch_artist=True, tick_labels=['% Improvement'])
    bp_pct['boxes'][0].set_facecolor('lightgreen')
    plt.ylabel('Percentage Improvement (%)', fontsize=12)
    plt.title('Distribution of Percentage Improvements', fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Zero improvement')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR + '/sdcb_pct_improvement_boxplot.pdf', format='pdf', bbox_inches='tight')
    plt.show()
else:
    print('Skipping box plot - no valid percentage improvements to plot')

# ============================================================================
# SCATTER: Baseline vs Percentage Improvement
# ============================================================================
plt.figure(figsize=(10, 6))
scatter = plt.scatter(bresults_arr, pct_reduction, c=pct_reduction, cmap='RdYlGn', s=150, 
                     edgecolors='black', linewidth=1.5, alpha=0.8)
plt.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
plt.xlabel('Baseline Cost', fontsize=PAPER_FOCUS_LABEL_FONTSIZE)
plt.ylabel('Cost Reduction (%)', fontsize=PAPER_FOCUS_LABEL_FONTSIZE)
plt.title('Baseline Performance vs Cost Reduction', fontsize=PAPER_FOCUS_TITLE_FONTSIZE, fontweight='bold')
cbar = plt.colorbar(scatter)
cbar.set_label('Cost Reduction (%)', fontsize=PAPER_FOCUS_LABEL_FONTSIZE)
cbar.ax.tick_params(labelsize=PAPER_FOCUS_TICK_FONTSIZE)
plt.tick_params(axis='both', labelsize=PAPER_FOCUS_TICK_FONTSIZE)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR + '/sdcb_baseline_vs_pct_improvement.pdf', format='pdf', bbox_inches='tight')
plt.show()

# ============================================================================
# CUMULATIVE PERCENTAGE IMPROVEMENT
# ============================================================================
plt.figure(figsize=(14, 6))
sorted_idx_cumsum = np.argsort(improv_pct)[::-1]
cumulative_pct_improv = np.cumsum(improv_pct[sorted_idx_cumsum])
plt.plot(cumulative_pct_improv, 'g-', linewidth=2.5, marker='o', markersize=5, alpha=0.8)
plt.fill_between(range(len(cumulative_pct_improv)), cumulative_pct_improv, alpha=0.3, color='green')
plt.xlabel('Days (sorted by % improvement)', fontsize=12)
plt.ylabel('Cumulative Percentage Improvement (%)', fontsize=12)
plt.title('Cumulative Percentage Improvement Over Days', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR + '/sdcb_cumulative_pct_improvement.pdf', format='pdf', bbox_inches='tight')
plt.show()

# ============================================================================
# DUAL-AXIS: Absolute vs Percentage Improvement
# ============================================================================
plt.figure(figsize=(14, 6))
ax1 = plt.gca()
ax2 = ax1.twinx()

line1 = ax1.plot(range(len(improv_arr)), improv_arr, 'b-', marker='s', linewidth=2, markersize=5, label='Absolute Improvement', alpha=0.8)
line2 = ax2.plot(range(len(improv_pct)), improv_pct, 'g-', marker='o', linewidth=2, markersize=5, label='Percentage Improvement', alpha=0.8)

ax1.set_xlabel('Days', fontsize=12)
ax1.set_ylabel('Absolute Improvement (Reward)', fontsize=12, color='b')
ax2.set_ylabel('Percentage Improvement (%)', fontsize=12, color='g')
ax1.tick_params(axis='y', labelcolor='b')
ax2.tick_params(axis='y', labelcolor='g')
ax1.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
ax1.grid(True, alpha=0.3)

# Combine legends
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, fontsize=11, loc='upper left')

plt.title('Absolute vs Percentage Improvement (Side-by-side Comparison)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_DIR + '/sdcb_absolute_vs_pct_improvement.pdf', format='pdf', bbox_inches='tight')
plt.show()

print('Done generating percentage improvement plots.\n')

# %%
# ============================================================================
# TESTING WITH STOCHASTIC WEATHER/WIND
# ============================================================================
print('\n' + '='*80)
print('TESTING WITH STOCHASTIC WEATHER/WIND')
print('='*80 + '\n')

# Each day is replayed on a weather/wind scenario drawn with the training sampler
# (Eqs. 17-18). Baseline and SDCB share the same scenario; the SDCB delays are the
# ones chosen on TAF (deploy), as in the deterministic test. Costs are in EUR.
results_stochastic = []
bresults_stochastic = []

print('Running stochastic evaluation on subset of days...')
stochastic_days = test_days[:min(5, len(test_days))]  # Test on first 5 days to save time

for k, day in enumerate(stochastic_days):
    print(f'  Evaluating day {day} on a sampled weather scenario...')
    scenario_seed = 1000 + k  # Same weather scenario for the baseline and the SDCB replay
    
    # Baseline with stochastic transitions
    baseline_env_stoch = sdcb.RunwayEnv(alpha = 2, beta = 1, delta = 0.3, eta = 0.1,
                                       data=data, taf_data=taf_data, arrival_data=arrival_data,
                                       congested_days=congested_days, 
                                       u=u, u_max=u_max, v=v, v_max=v_max,
                                       possible_actions=possible_actions,
                                       valid_bool=valid_bool,
                                       wind_states=wind_states, 
                                       observation_low=observation_low, 
                                       observation_high=observation_high,
                                       p_wc_trans=p_wc_trans, p_ws_trans=p_ws_trans,
                                       p_wc_error=p_wc_error, p_ws_error=p_ws_error,
                                       chosen_day=day)
    
    obs, _ = baseline_env_stoch.reset(seed=scenario_seed, mode='train', evaluate=True)
    terminated = False
    while not terminated:
        obs, reward, terminated, truncated, info = baseline_env_stoch.step(0, mode='train', evaluate=True)
    base_rew_stoch = baseline_env_stoch.objective
    
    # SDCB with stochastic transitions
    env = sdcb.RunwayEnv(alpha = 2, beta = 1, delta = 0.3, eta = 0.1,
                        data=data, taf_data=taf_data, arrival_data=arrival_data,
                        congested_days=congested_days, 
                        u=u, u_max=u_max, v=v, v_max=v_max,
                        possible_actions=possible_actions,
                        valid_bool=valid_bool,
                        wind_states=wind_states, 
                        observation_low=observation_low, 
                        observation_high=observation_high,
                        p_wc_trans=p_wc_trans, p_ws_trans=p_ws_trans,
                        p_wc_error=p_wc_error, p_ws_error=p_ws_error,
                        chosen_day=day)
    
    # Load the trained model
    model = MaskablePPO("MlpPolicy", env, verbose=0, **PPO_KWARGS)
    file = MODEL_DIR + "/sdcb_ppo_fb_" + day + ".zip"
    if Path(file).exists():
        prev_model = MaskablePPO.load(file, env=env)
        model.set_parameters(prev_model.get_parameters())
        
        deploy_env = sdcb.RunwayEnv(alpha = 2, beta = 1, delta = 0.3, eta = 0.1,
                                   data=data, taf_data=taf_data, arrival_data=arrival_data,
                                   congested_days=congested_days, 
                                   u=u, u_max=u_max, v=v, v_max=v_max,
                                   possible_actions=possible_actions,
                                   valid_bool=valid_bool,
                                   wind_states=wind_states, 
                                   observation_low=observation_low, 
                                   observation_high=observation_high,
                                   p_wc_trans=p_wc_trans, p_ws_trans=p_ws_trans,
                                   p_wc_error=p_wc_error, p_ws_error=p_ws_error,
                                   chosen_day=day)
        
        obs, _ = deploy_env.reset(mode='deploy', evaluate=True)
        terminated = False
        while not terminated:
            action_masks = get_action_masks(deploy_env)
            action_idx, _states = model.predict(obs, action_masks=action_masks, deterministic=True)
            obs, reward, terminated, truncated, info = deploy_env.step(action_idx, mode='deploy', evaluate=True)
        
        # Compute SDCB with deploy environment delays
        obs, _ = env.reset(seed=scenario_seed, mode='train', evaluate=True)
        terminated = False
        counter = 0
        while not terminated:
            obs, reward, terminated, truncated, info = env.step(deploy_env.imposed_delays[counter], 
                                                                mode='train', 
                                                                evaluate=True)
            counter += 1
        model_rew_stoch = env.objective
        
        bresults_stochastic.append(base_rew_stoch)
        results_stochastic.append(model_rew_stoch)

if len(results_stochastic) > 0:
    print('\nStochastic Evaluation Results:')
    print('-'*80)
    
    results_stoch_arr = np.array(results_stochastic)
    bresults_stoch_arr = np.array(bresults_stochastic)
    improv_stoch = results_stoch_arr - bresults_stoch_arr
    
    print(f'{"Metric":<30} {"SDCB":>20} {"Baseline":>20}')
    print('-'*80)
    print(f'{"Mean":<30} {results_stoch_arr.mean():>20.2f} {bresults_stoch_arr.mean():>20.2f}')
    print(f'{"Std Dev":<30} {results_stoch_arr.std():>20.2f} {bresults_stoch_arr.std():>20.2f}')
    print(f'{"Mean Improvement":<30} {improv_stoch.mean():>20.2f}')
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    x_pos = np.arange(len(results_stochastic))
    width = 0.35
    plt.bar(x_pos - width/2, results_stochastic, width, label='SDCB (stochastic)', alpha=0.8)
    plt.bar(x_pos + width/2, bresults_stochastic, width, label='Baseline (stochastic)', alpha=0.8)
    plt.xlabel('Test Days')
    plt.ylabel('Tactical Cost')
    plt.title('SDCB vs Baseline: Stochastic Weather/Wind Evaluation')
    plt.legend()
    plt.xticks(x_pos, [f'Day {i+1}' for i in range(len(results_stochastic))])
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(OUT_DIR + '/sdcb_vs_baseline_stochastic.pdf', format='pdf', bbox_inches='tight')
    plt.show()
    
    # Comparison: observed weather (METAR) vs sampled weather scenario
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(results[:len(results_stochastic)], 'o-', linewidth=2, markersize=8, label='Deterministic')
    plt.plot(results_stochastic, 's--', linewidth=2, markersize=8, label='Stochastic')
    plt.xlabel('Test Days')
    plt.ylabel('SDCB Cost')
    plt.title('SDCB: Deterministic vs Stochastic')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(bresults[:len(bresults_stochastic)], 'o-', linewidth=2, markersize=8, label='Deterministic')
    plt.plot(bresults_stochastic, 's--', linewidth=2, markersize=8, label='Stochastic')
    plt.xlabel('Test Days')
    plt.ylabel('Baseline Cost')
    plt.title('Baseline: Deterministic vs Stochastic')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUT_DIR + '/deterministic_vs_stochastic_comparison.pdf', format='pdf', bbox_inches='tight')
    plt.show()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print('\n' + '='*80)
print('FINAL SUMMARY')
print('='*80)
print(f'\nSDCB achieved an average improvement of {improv_arr.mean():.2f} over the baseline')
print(f'with a win rate of {win_rate:.2f}% across {len(results_arr)} test days.')
print(f'\nThe improvement is {"statistically significant" if t_pval < 0.05 else "not statistically significant"} (p={t_pval:.6f})')
print('='*80 + '\n')
# %%
