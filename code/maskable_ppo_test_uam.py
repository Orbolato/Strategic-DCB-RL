
# %%
from sb3_contrib import MaskablePPO
import improved_gym_sdcboptimizer_specific_uam as sdcb
# from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks
from uam_capacity_curves import generate_uam_curves, export_curves_to_dicts
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
_parser = argparse.ArgumentParser(description='Strategic DCB experiment, case 2: UAM')
_parser.add_argument('--setting', choices=['S1', 'S2'], default='S1',
                     help='S1 (default): entropy coefficient 0 and linear training reward, the settings of the '
                          'original experiments; S2: entropy coefficient 0.01 and the quadratic evaluation cost as '
                          'training reward, the settings stated in the paper')
_parser.add_argument('--forecast-check', action='store_true',
                     help="keep each day's ground delays only if they reduce the tactical cost of the day on the "
                          'forecast (TAF); otherwise keep the original schedule')
_args, _ = _parser.parse_known_args()
SETTING, FORECAST_CHECK = _args.setting, _args.forecast_check
TRAINING_REWARD = {'S1': 'linear', 'S2': 'quadratic'}[SETTING]
ENT_COEF = {'S1': 0.0, 'S2': 0.01}[SETTING]
MODEL_DIR = {'S1': 'Results_UAM', 'S2': 'Results_UAM_S2'}[SETTING]  # trained models of the setting
OUT_DIR = MODEL_DIR + ('_check' if FORECAST_CHECK else '')  # result figures of this run
print(f'Training setting {SETTING} (training reward: {TRAINING_REWARD}, entropy coefficient: {ENT_COEF}), '
      f'forecast check: {"on" if FORECAST_CHECK else "off"}; models in {MODEL_DIR}/, figures in {OUT_DIR}/')

###############################################################################
# Data preprocessing
###############################################################################

# Importing data preprocessed in R
print('\nLoading METAR and traffic data...')
data = pd.read_csv('data_uam.csv')
data['time'] = pd.to_datetime(data['time'], format='mixed')
print('Data loaded.\n')

# Importing data preprocessed in R
print('\nLoading flight data...')
arrival_data = pd.read_csv('arrival_data_uam.csv')
arrival_data['slot_time'] = pd.to_datetime(arrival_data['slot_time'], format='mixed')  # Arrival slot (matches day_slot and data_uam demand)
arrival_data['scheduled_departure_time'] = pd.to_datetime(arrival_data['scheduled_departure_time'], format='mixed')
arrival_data['scheduled_arrival_time'] = pd.to_datetime(arrival_data['scheduled_arrival_time'], format='mixed')
arrival_data['scheduled_arrival_3_time'] = pd.to_datetime(arrival_data['scheduled_arrival_3_time'], format='mixed')
print('Data loaded.\n')

# Sort arrival_data by scheduled_arrival_3_time to ensure flights are processed in chronological order
arrival_data = arrival_data.sort_values('scheduled_arrival_3_time').reset_index(drop=True)
print('Arrival data sorted by scheduled_arrival_3_time.\n')

# Recalculate day column from the arrival slot, the slot used for the flight's day_slot and the demand counts
print('Recalculating day assignments based on arrival slots...')
arrival_data['date'] = arrival_data['slot_time'].dt.date
for row in range(len(arrival_data)):
    if arrival_data.loc[row, 'slot_time'].hour < 6:
        arrival_data.loc[row, 'date'] -= datetime.timedelta(days=1)

# Create day column in format 'YYYY-M-D'
arrival_data['day'] = ''
for row in range(len(arrival_data)):
    year = arrival_data.loc[row, 'date'].year
    month = arrival_data.loc[row, 'date'].month
    day = arrival_data.loc[row, 'date'].day
    arrival_data.loc[row, 'day'] = str(year) + '-' + str(month) + '-' + str(day)
print('Day assignments recalculated based on arrival slots.\n')

# Importing TAF data preprocessed in R
print('\nLoading TAF data...')
taf_data = pd.read_csv('new_taf_list_uam.csv')
taf_data['time'] = pd.to_datetime(taf_data['time'], format='mixed')
print('Data loaded.\n')

print('Removing partial periods...')
while(data.loc[0,'time'].hour != 6):
    data.drop(0, axis='rows', inplace=True)
    data.reset_index(drop=True, inplace=True)
while(data.loc[len(data)-1,'time'].hour != 5):
    data.drop(len(data)-1, axis='rows', inplace=True)
    data.reset_index(drop=True, inplace=True)
print('Done.\n')

print('Determining periods...')
data['date'] = data['time'].dt.date

# Adjust dates for times before 06:00 (they belong to previous operational day)
mask_before_6am = data['time'].dt.hour < 6
data.loc[mask_before_6am, 'date'] = data.loc[mask_before_6am, 'date'] - pd.Timedelta(days=1)

# Vectorized day computation
data['day'] = data['date'].apply(lambda x: f"{x.year}-{x.month}-{x.day}")
print('Done.\n')

# Arrivals and departures columns
while taf_data.loc[0, 'time'] != data.loc[0, 'time']:
    taf_data.drop(0, axis='rows', inplace=True)
    taf_data.reset_index(drop=True, inplace=True)
while taf_data.loc[taf_data.shape[0]-1, 'time'] != data.loc[data.shape[0]-1, 'time']:
    taf_data.drop(taf_data.shape[0]-1, axis='rows', inplace=True)
    taf_data.reset_index(drop=True, inplace=True)
taf_data['arrival_demand_3'] = data['scheduled_arrivals_3']
taf_data['departure_demand'] = data['scheduled_departures']

###############################################################################
# Encoding categorical features
###############################################################################

# Preprocessing data
print('Encoding runway configurations and weather condition...')
data['weather_condition'] = data['ifr'] + data['lifr']
data['A'] = np.zeros(len(data))
data['B'] = np.zeros(len(data))
for i in range(len(data)):
    if data.loc[i,'configuration'] == 'A':
        data.loc[i,'configuration'] = 1 # 'A'
        data.loc[i,'A'] = 1
    elif data.loc[i,'configuration'] == 'B':
        data.loc[i,'configuration'] = 2 # 'B'
        data.loc[i,'B'] = 1
    else:
        data.loc[i,'configuration'] = 0 # None
print('Done.\n')

# Defining wind states
print('Defining and encoding wind states...') # Inverted with respect to runway conditions
cw_max = 20 # Maxium crosswind (knots)
tw_max = 5 # Maximum tailwind (knots)
print('Crosswind limit of ' + str(cw_max) + ' knots.')
print('Tailwind limit of ' + str(tw_max) + ' knots.\n')
data['wind_state'] = np.zeros(len(data)) # Column of wind states
data['wind_state'] = pd.to_numeric(data['wind_state'], downcast='integer')
data['A_OK'] = np.zeros(len(data)) # 0 - not available, 1 - available
data['B_OK'] = np.zeros(len(data)) # 0 - not available, 1 - available
# 0 - Both available (0 is always for all runways available)
# 1 - A available
# 2 - B available
# 3 - Both unavailable
print('Case 0 - Both available.')
print('Case 1 - A available.')
print('Case 2 - B available.')
print('Case 3 - Both unavailable.\n')
for i in range(len(data)):
    if abs(data.loc[i,'headwind_10']) > cw_max:
        data.loc[i,'wind_state'] = 3
        data.loc[i,'A_OK'] = 0
        data.loc[i,'B_OK'] = 0
    elif data.loc[i,'crosswind_10'] < -tw_max:
        data.loc[i,'wind_state'] = 2
        data.loc[i,'A_OK'] = 0
        data.loc[i,'B_OK'] = 1
    elif data.loc[i,'crosswind_10'] > tw_max:
        data.loc[i,'wind_state'] = 1
        data.loc[i,'A_OK'] = 1
        data.loc[i,'B_OK'] = 0
    else:
        data.loc[i,'wind_state'] = 0
        data.loc[i,'A_OK'] = 1
        data.loc[i,'B_OK'] = 1

for k in data['wind_state'].unique():
    print('Case ' + str(k) + ' has '
          + str(len(data.loc[data['wind_state'] == k,]))
          + ' occurences.')
print('')

# Forecast (TAF) wind states, encoded with the same vertiport rule
taf_data['wind_state'] = np.select([taf_data['headwind_10'].abs() > cw_max,
                                    taf_data['crosswind_10'] < -tw_max,
                                    taf_data['crosswind_10'] > tw_max],
                                   [3, 2, 1], default=0)

###############################################################################
# Maximum features' values
###############################################################################

# Total number of time slots
N = len(data)

# Calculating accumulated arrivals and departures
print('Determining a list of days...')
accumulated_arrivals = [data.loc[0,'scheduled_arrivals_3']]
accumulated_departures = [data.loc[0,'scheduled_departures']]
day = data.loc[0,'day']
days = [day]
for i in range(1,N):
    if data.loc[i,'day'] == day:
        accumulated_arrivals[-1] += data.loc[i,'scheduled_arrivals_3']
        accumulated_departures[-1] += data.loc[i,'scheduled_departures']
    else:
        accumulated_arrivals.append(data.loc[i,'scheduled_arrivals_3'])
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

print('Generating UAM capacity curves...')
A = 1 # Side A: optimized for arrivals
B = 2 # Side B: optimized for departures
C = 0 # None
configs = [C, A, B] 
weather_conds = [0, 1, 2] # ['VMC', 'IMC', 'LIMC'] # VMC - 0, IMC - 1, LIMC - 2
wind_states = {0: [A,B], 1: [A], 2: [B], 3: [C]}

# Generate UAM capacity curves
uam_curves = generate_uam_curves()
u_uam, v_min_uam, v_max_uam = export_curves_to_dicts(uam_curves)

# Map UAM curves to u, v_min, v_max using [wc, rc] indexing
# UAM curves use side_id (1=A, 2=B) which maps to rc (1=A, 2=B)
u = {}
v_min = {}  # Minimum departures for each arrival (lower bound)
v_max = {}  # Maximum departures for each arrival (upper bound)
u_max = 0
v_max_global = 0

for wc in weather_conds:
    print(f'  Weather condition {wc}...', end=' ')
    
    # Create curves for this weather condition
    for rc in configs:
        if rc == C:
            # No runway configuration - no operations
            u[wc, rc] = np.array([0])
            v_min[wc, rc] = np.array([0])
            v_max[wc, rc] = np.array([0])
        else:
            # Get curves from UAM (rc=1 for Side A, rc=2 for Side B)
            u[wc, rc] = u_uam[wc][rc].copy()
            v_min[wc, rc] = v_min_uam[wc][rc].copy()
            v_max[wc, rc] = v_max_uam[wc][rc].copy()
        
        # Track max values
        if len(u[wc, rc]) > 0 and np.max(u[wc, rc]) > u_max:
            u_max = np.max(u[wc, rc])
        if len(v_max[wc, rc]) > 0 and np.max(v_max[wc, rc]) > v_max_global:
            v_max_global = np.max(v_max[wc, rc])
    
    print(f'u_max={u_max}, v_max_global={v_max_global}')

print('Done.\n')

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
observation_low = [0,0,0,0,0]
observation_high = [2,50,50,2,3]
congested_days = days.copy()

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
PPO_KWARGS = dict(gamma=1, learning_rate=3e-4, batch_size=64, ent_coef=ENT_COEF, clip_range=0.2, seed=0, device='cpu')  # ENT_COEF: 0 in setting S1 (library default, as in the original experiments), 0.01 in S2
os.makedirs(MODEL_DIR, exist_ok=True)  # Trained models
os.makedirs(OUT_DIR, exist_ok=True)  # Result figures

for day in test_days:
    
    env = sdcb.RunwayEnv(alpha = 2, beta = 1, delta = 0.3, eta = 0.1,
                           data=data, taf_data=taf_data, arrival_data=arrival_data,
                           congested_days=congested_days, 
                           u=u, u_max=u_max, v_min=v_min, v_max=v_max,
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
        model.learn(300_000)
        end = time.time()
        print('Training finished in ' + str(round(end-start, 2)) + ' seconds.\n')
        print('Saving model...')
        model.save(MODEL_DIR + "/sdcb_ppo_fb_" + day + ".zip") # Save model
        print('Done.\n')
 
    print('Testing...')
    
    # Create baseline environment for heuristic comparison (after first env instantiation, look for the second one in the loop)
    baseline_env = sdcb.RunwayEnv(alpha = 2, beta = 1, delta = 0.3, eta = 0.1,
                                 data=data, taf_data=taf_data, arrival_data=arrival_data,
                                 congested_days=congested_days, 
                                 u=u, u_max=u_max, v_min=v_min, v_max=v_max,
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
                               u=u, u_max=u_max, v_min=v_min, v_max=v_max,
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
                                     u=u, u_max=u_max, v_min=v_min, v_max=v_max,
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
if len(improv_pct_valid) > 0:
    plt.figure(figsize=(12, 6))
    plt.hist(improv_pct_valid, bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
    plt.axvline(np.nanmean(improv_pct), color='r', linestyle='--', linewidth=2.5, label=f'Mean: {np.nanmean(improv_pct):.2f}%')
    plt.axvline(np.nanmedian(improv_pct), color='b', linestyle='--', linewidth=2.5, label=f'Median: {np.nanmedian(improv_pct):.2f}%')
    plt.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    plt.xlabel('Percentage Improvement (%)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Percentage Improvements (SDCB vs Baseline)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(OUT_DIR + '/sdcb_pct_improvement_histogram.pdf', format='pdf', bbox_inches='tight')
    plt.show()
else:
    print('Skipping histogram - no valid percentage improvements to plot')

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
                                       u=u, u_max=u_max, v_min=v_min, v_max=v_max,
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
                        u=u, u_max=u_max, v_min=v_min, v_max=v_max,
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
                                   u=u, u_max=u_max, v_min=v_min, v_max=v_max,
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
