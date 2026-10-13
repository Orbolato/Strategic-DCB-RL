"""
Extract Flight Durations from Helicopter Logs

This script analyzes flight logs and extracts the average flight duration
for each origin-destination pair by calculating the time between
first altitude=0 (takeoff) and last altitude=0 (landing).

Log format: simt, id, lat, lon, distflown, alt, cas, tas, gs

Output: flight_durations.csv
Columns: origin, destination, duration_minutes, num_flights, source
"""

import pandas as pd
import glob
import re
from pathlib import Path

print('=' * 80)
print('Extracting Flight Durations from Helicopter Logs')
print('=' * 80)

# Path to logs
log_dir = 'profile_implementation/SBGR_DEST/Logs'

# Find all log files
log_files = sorted(glob.glob(f'{log_dir}/STATELOG_*.log'))

print(f'\nFound {len(log_files)} log files\n')

flight_durations = {}

for log_file in log_files:
    # Extract route from filename: STATELOG_SBMT-SBGR_*.log
    filename = Path(log_file).stem
    
    # Remove STATELOG_ prefix
    if filename.startswith('STATELOG_'):
        filename = filename[9:]  # Remove "STATELOG_"
    
    # Find the last underscore followed by date pattern (YYYYMMDD)
    # Working backwards: parts after date are time (_HH-MM-SS), then date
    # Pattern: Origin-Destination_YYYYMMDD_HH-MM-SS
    import re
    match = re.match(r'(.+?)_\d{8}_\d{2}-\d{2}-\d{2}', filename)
    
    if match:
        route = match.group(1)  # This is "ORIGIN-DESTINATION"
        # Split by the last hyphen to separate origin and destination
        # But be careful since VP-001 has hyphens too
        # Strategy: split by hyphen and rejoin - origin will have no more hyphens before SBGR/SBSP/SBMT/VP
        parts = route.split('-')
        
        # Find where the destination starts
        # Destination is always at the end, usually SBGR
        # For VP, the last part is part of the destination (VP-001, VP-002, VP-003)
        # Strategy: Always take the last part after splitting by dash as destination if it's city-like (3-4 letters)
        parts = route.split('-')
        
        # Last part should be destination
        if len(parts) >= 2:
            destination = parts[-1]
            origin = '-'.join(parts[:-1])  # Everything except the last part joined by '-'
        else:
            origin = 'UNKNOWN'
            destination = 'UNKNOWN'
        
        if origin != 'UNKNOWN' and destination != 'UNKNOWN':
            route_key = f"{origin}-{destination}"
            
            if route_key not in flight_durations:
                flight_durations[route_key] = {
                    'origin': origin,
                    'destination': destination,
                    'durations': [],
                    'flights': 0
                }
            
            # Parse the log file
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                
                # Skip header lines (starting with #)
                data_lines = [line.strip() for line in lines if not line.startswith('#') and line.strip()]
                
                # Parse CSV data: simt, id, lat, lon, distflown, alt, cas, tas, gs
                # alt is the 6th column (index 5)
                altitude_zero_times = []
                
                for line in data_lines:
                    try:
                        parts = line.split(',')
                        if len(parts) >= 6:
                            simt = float(parts[0])  # Simulation time (seconds)
                            alt = float(parts[5])   # Altitude
                            
                            # Check if altitude is 0 or very close to 0 (within 0.01 meters)
                            if abs(alt) < 0.01:
                                altitude_zero_times.append(simt)
                    except (ValueError, IndexError):
                        continue
                
                # If we found altitude~0 entries, calculate duration
                if len(altitude_zero_times) >= 2:
                    first_time = altitude_zero_times[0]
                    last_time = altitude_zero_times[-1]
                    duration_seconds = last_time - first_time
                    duration_minutes = duration_seconds / 60.0
                    
                    flight_durations[route_key]['durations'].append(duration_minutes)
                    flight_durations[route_key]['flights'] += 1
                    
                    print(f'{Path(log_file).name}')
                    print(f'  Route: {route_key}')
                    print(f'  Duration: {duration_minutes:.2f} minutes ({duration_seconds:.1f} seconds)')
                    
            except Exception as e:
                print(f'  Error processing {log_file}: {e}')

print('\n' + '=' * 80)
print('Summary')
print('=' * 80 + '\n')

# Create results dataframe
results = []
for route, data in sorted(flight_durations.items()):
    if len(data['durations']) > 0:
        avg_duration = sum(data['durations']) / len(data['durations'])
        results.append({
            'origin': data['origin'],
            'destination': data['destination'],
            'route': route,
            'duration_minutes': round(avg_duration, 2),
            'num_flights': len(data['durations']),
            'source': 'Helicopter Logs'
        })
        
        print(f'{route}:')
        print(f'  Average Duration: {avg_duration:.2f} minutes')
        print(f'  Number of Flights: {len(data["durations"])}')
        print(f'  Min: {min(data["durations"]):.2f}, Max: {max(data["durations"]):.2f}')
        print()

# Save to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('flight_durations.csv', index=False)

print(f'\nSaved: flight_durations.csv ({len(results_df)} routes)')
print('\nContents:')
print(results_df.to_string(index=False))
