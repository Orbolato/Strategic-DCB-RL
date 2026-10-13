import csv
import math
import os
from collections import defaultdict

def calculate_heading(lat1, lon1, lat2, lon2):
    """
    Calculate heading (bearing) from point 1 to point 2 in degrees.
    Returns heading in degrees (0-360), where 0 is North, 90 is East, etc.
    """
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    dlon = lon2_rad - lon1_rad
    
    x = math.sin(dlon) * math.cos(lat2_rad)
    y = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon)
    
    heading = math.atan2(x, y)
    heading_deg = math.degrees(heading)
    heading_deg = (heading_deg + 360) % 360
    
    return heading_deg

def calculate_distance_nm(lat1, lon1, lat2, lon2):
    """
    Calculate distance between two points in nautical miles using Haversine formula.
    """
    R = 3440.065  # Earth radius in nautical miles
    
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c

def read_routes(csv_file):
    """
    Read CSV file and group waypoints by route.
    Returns dictionary with route as key and list of waypoints as value.
    """
    routes = defaultdict(list)
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            route = row['route']
            waypoint = {
                'id': int(row['id']),
                'name': row['name'],
                'lat': float(row['lat']),
                'lon': float(row['lon']),
                'type': row['type']
            }
            routes[route].append(waypoint)
    
    # Sort waypoints by id for each route
    for route in routes:
        routes[route].sort(key=lambda x: x['id'])
    
    return routes

def generate_scn_content(
    route_name,
    waypoints,
    origin_distance_nm=1.0,
    dest_distance_nm=3.0,
    dest_distance_nm_short=1.0,
    alt_50=50,
    alt_300=300,
    alt_1500=1500,
    vs_climb_descent=500,
    speed_climb_descent=48,
    speed_cruise=96,
    delay_before_deletion=5.0
):
    """
    Generate SCN file content for a single route using manual altitude/speed control.
    
    Uses ATDIST (distance-triggered) commands instead of waypoint-based autopilot.
    
    For routes shorter than 10 NM, uses terminal altitude profile (300 ft, 48 kts).
    For routes 10 NM or longer, uses full climb/cruise/descent profile.
    
    Args:
        route_name: Name of the route
        waypoints: List of waypoint dictionaries
        origin_distance_nm: Distance in NM from origin to trigger cruise climb (default: 1.0)
        dest_distance_nm: Distance in NM from destination to trigger descent for long routes (default: 3.0)
        dest_distance_nm_short: Distance in NM from destination to trigger descent for short routes (default: 1.0)
        alt_50: Altitude for initial climb and final descent (default: 50 ft)
        alt_300: Altitude for departure/arrival terminal area (default: 300 ft)
        alt_1500: Altitude for cruise flight (default: 1500 ft)
        vs_climb_descent: Vertical speed for all climbs/descents (default: 500 ft/min)
        speed_climb_descent: Speed during climb/descent phases (default: 48 kts)
        speed_cruise: Speed during cruise phase (default: 96 kts)
        delay_before_deletion: Delay in seconds before aircraft deletion (default: 5.0)
    """
    if len(waypoints) < 2:
        return None
    
    # Extract origin and destination
    origin = waypoints[0]
    destination = waypoints[-1]
    
    # Calculate total route distance
    total_distance = 0.0
    for i in range(len(waypoints) - 1):
        segment_dist = calculate_distance_nm(
            waypoints[i]['lat'], waypoints[i]['lon'],
            waypoints[i+1]['lat'], waypoints[i+1]['lon']
        )
        total_distance += segment_dist
    
    # Determine if route is short (< 10 NM)
    is_short_route = total_distance < 10.0
    
    # Calculate heading from origin to next waypoint
    first_nav_wp = waypoints[1] if len(waypoints) > 2 else destination
    heading = calculate_heading(origin['lat'], origin['lon'], first_nav_wp['lat'], first_nav_wp['lon'])
    
    # Create aircraft ID from route name
    aircraft_id = f"EVTOL_{route_name.replace('-', '_')}"
    
    # Track waypoint name counts to handle duplicates
    waypoint_name_map = {}
    waypoint_count = {}
    
    for idx, wp in enumerate(waypoints):
        wp_name = wp['name']
        if wp_name not in waypoint_count:
            waypoint_count[wp_name] = 0
        else:
            waypoint_count[wp_name] += 1
        
        # Create unique name for duplicate waypoints
        if waypoint_count[wp_name] > 0:
            unique_name = f"{wp_name}_{waypoint_count[wp_name]}"
        else:
            unique_name = wp_name
        
        waypoint_name_map[idx] = unique_name
    
    # Start building SCN content
    scn_lines = []
    
    # Initial setup commands
    scn_lines.append("00:00:00.00>TRAILS ON")
    scn_lines.append("00:00:00.00>RESO OFF")
    mid_lat = (origin['lat'] + destination['lat']) / 2
    mid_lon = (origin['lon'] + destination['lon']) / 2
    scn_lines.append(f"00:00:00.00>PAN {mid_lat:.6f} {mid_lon:.6f}")
    scn_lines.append("")
    
    # Logging and metrics section
    scn_lines.append("# Log")
    scn_lines.append("00:00:00.00> CRELOG STATELOG 0.1 State log")
    scn_lines.append("00:00:00.00> STATELOG ADD traf.id traf.lat traf.lon traf.distflown traf.alt traf.cas traf.tas traf.gs")
    scn_lines.append("00:00:00.00> STATELOG ON")
    scn_lines.append("00:00:00.00> PLUGIN LOAD UAMMETRICS")
    scn_lines.append("00:00:00.00> UAMMETRICS START")
    scn_lines.append("")
    
    # Define waypoints - include all non-airport waypoints (origin, intermediate, destination)
    scn_lines.append("# Define waypoints")
    
    # Define origin if it's not an airport
    if origin['type'] != "Airport":
        origin_shortname_defwpt = waypoint_name_map[0]
        scn_lines.append(f"00:00:00.00>DEFWPT {origin_shortname_defwpt} {origin['lat']} {origin['lon']} FIX")
    
    # Define intermediate waypoints
    for idx in range(1, len(waypoints) - 1):
        wp = waypoints[idx]
        unique_name = waypoint_name_map[idx]
        scn_lines.append(f"00:00:00.00>DEFWPT {unique_name} {wp['lat']} {wp['lon']} FIX")
    
    # Define destination if it's not an airport
    if destination['type'] != "Airport":
        dest_shortname_defwpt = waypoint_name_map[len(waypoints) - 1]
        scn_lines.append(f"00:00:00.00>DEFWPT {dest_shortname_defwpt} {destination['lat']} {destination['lon']} FIX")
    
    scn_lines.append("")
    
    # Get origin and destination short names for ATDIST commands
    origin_shortname = waypoint_name_map[0]
    dest_shortname = waypoint_name_map[len(waypoints) - 1]
    
    # Create aircraft at origin
    scn_lines.append(f"00:00:00.00>CRE {aircraft_id} EVTOL {origin_shortname} {heading:.10f} 0 0")
    scn_lines.append(f"00:00:00.00>ORIG {aircraft_id} {origin_shortname}")
    scn_lines.append(f"# 00:00:00.00>DEST {aircraft_id} {dest_shortname}")
    scn_lines.append("")
    
    # Climb profile
    scn_lines.append("# Climb profile")
    scn_lines.append(f"00:00:00.00>ALT {aircraft_id} {alt_50} {vs_climb_descent}")
    scn_lines.append(f"00:00:00.00>ATALT {aircraft_id} {alt_50} SPD {aircraft_id} {speed_climb_descent}")
    scn_lines.append(f"00:00:00.00>ATALT {aircraft_id} {alt_50} ALT {aircraft_id} {alt_300} {vs_climb_descent}")
    
    # For long routes (>= 10 NM): climb to cruise altitude and set cruise speed
    if not is_short_route:
        scn_lines.append(f"00:00:00.00>ATDIST {aircraft_id} {origin_shortname} {origin_distance_nm} SPD {aircraft_id} {speed_cruise}")
        scn_lines.append(f"00:00:00.00>ATDIST {aircraft_id} {origin_shortname} {origin_distance_nm} ALT {aircraft_id} {alt_1500} {vs_climb_descent}")
    
    scn_lines.append("")
    
    # Add waypoints
    scn_lines.append("# Add waypoints")
    for idx in range(1, len(waypoints) - 1):
        wp_name = waypoint_name_map[idx]
        scn_lines.append(f"00:00:00.00>ADDWPT {aircraft_id} {wp_name}")
    
    scn_lines.append("")
    
    # Descent profile
    scn_lines.append("# Descent profile")
    
    # Choose distance parameter based on route profile
    current_dest_distance_nm = dest_distance_nm_short if is_short_route else dest_distance_nm
    
    # For long routes, descend from cruise altitude; for short routes, stay at terminal altitude
    if not is_short_route:
        scn_lines.append(f"00:00:00.00>ATDIST {aircraft_id} {dest_shortname} {current_dest_distance_nm} SPD {aircraft_id} {speed_climb_descent}")
        scn_lines.append(f"00:00:00.00>ATDIST {aircraft_id} {dest_shortname} {current_dest_distance_nm} ALT {aircraft_id} {alt_300} {vs_climb_descent}")
        scn_lines.append(f"00:00:00.00>ATDIST {aircraft_id} {dest_shortname} {current_dest_distance_nm} ATALT {aircraft_id} {alt_300} VNAV {aircraft_id} ON")
    else:
        # For short routes: enable VNAV at destination distance
        scn_lines.append(f"00:00:00.00>ATDIST {aircraft_id} {dest_shortname} {current_dest_distance_nm} ATALT {aircraft_id} {alt_300} VNAV {aircraft_id} ON")
    
    # Add destination waypoint after last intermediate waypoint
    if len(waypoints) > 2:
        last_wp_name = waypoint_name_map[len(waypoints) - 2]
        scn_lines.append(f"00:00:00.00>AFTER {aircraft_id} {last_wp_name} ADDWPT {dest_shortname}")
    
    scn_lines.append(f"00:00:00.00>{aircraft_id} AT {dest_shortname} ALT {alt_50} {vs_climb_descent}")
    scn_lines.append(f"00:00:00.00>{aircraft_id} AT {dest_shortname} SPD 0")
    scn_lines.append(f"00:00:00.00>ATDIST {aircraft_id} {dest_shortname} {current_dest_distance_nm} ATSPD {aircraft_id} 0 ALT {aircraft_id} 0 {vs_climb_descent}")
    scn_lines.append(f"00:00:00.00>ATDIST {aircraft_id} {dest_shortname} {current_dest_distance_nm} ATALT {aircraft_id} 0 DELAY {delay_before_deletion} DEL {aircraft_id}")
    
    return "\n".join(scn_lines)

def routes_to_scn(
    csv_file,
    output_dir=None,
    origin_distance_nm=1.0,
    dest_distance_nm=3.0,
    dest_distance_nm_short=1.0,
    alt_50=50,
    alt_300=300,
    alt_1500=1500,
    vs_climb_descent=500,
    speed_climb_descent=48,
    speed_cruise=96,
    delay_before_deletion=5.0
):
    """
    Convert routes CSV to SCN files with customizable flight parameters.
    
    Args:
        csv_file: Path to the CSV file with routes
        output_dir: Output directory for SCN files (default: same as CSV file)
        origin_distance_nm: Distance in NM from origin to trigger cruise (default: 1.0)
        dest_distance_nm: Distance in NM from destination to trigger descent for long routes (default: 3.0)
        dest_distance_nm_short: Distance in NM from destination to trigger descent for short routes (default: 1.0)
        alt_50: Altitude for initial climb and final descent (default: 50 ft)
        alt_300: Altitude for departure/arrival terminal area (default: 300 ft)
        alt_1500: Altitude for cruise flight (default: 1500 ft)
        vs_climb_descent: Vertical speed for climbs/descents (default: 500 ft/min)
        speed_climb_descent: Speed during climb/descent (default: 48 kts)
        speed_cruise: Speed during cruise (default: 96 kts)
        delay_before_deletion: Delay before aircraft deletion (default: 5.0 s)
    """
    if output_dir is None:
        output_dir = os.path.dirname(csv_file)
    
    # Read routes from CSV
    routes = read_routes(csv_file)
    
    print(f"Found {len(routes)} routes")
    print(f"Flight Parameters:")
    print(f"  Origin distance: {origin_distance_nm} NM")
    print(f"  Destination distance (long routes): {dest_distance_nm} NM")
    print(f"  Destination distance (short routes): {dest_distance_nm_short} NM")
    print(f"  Altitude sequence: {alt_50} → {alt_300} → {alt_1500} → {alt_300} → {alt_50} ft")
    print(f"  Vertical speed: {vs_climb_descent} ft/min")
    print(f"  Climb/descent speed: {speed_climb_descent} kts")
    print(f"  Cruise speed: {speed_cruise} kts")
    print()
    
    # Generate SCN file for each route
    for route_name, waypoints in sorted(routes.items()):
        scn_content = generate_scn_content(
            route_name,
            waypoints,
            origin_distance_nm,
            dest_distance_nm,
            dest_distance_nm_short,
            alt_50,
            alt_300,
            alt_1500,
            vs_climb_descent,
            speed_climb_descent,
            speed_cruise,
            delay_before_deletion
        )
        
        if scn_content:
            output_file = os.path.join(output_dir, f"{route_name}.scn")
            with open(output_file, 'w') as f:
                f.write(scn_content)
            print(f"Generated: {route_name}.scn")
        else:
            print(f"Skipped: {route_name} (insufficient waypoints)")

if __name__ == "__main__":
    # ==================== CONFIGURATION PARAMETERS ====================
    # Flight parameters for eVTOL UAM scenarios (all customizable)
    ORIGIN_DISTANCE_NM = 1.0          # Distance from origin to trigger cruise climb (NM)
    DEST_DISTANCE_NM = 3.0            # Distance from destination to trigger descent (long routes) (NM)
    DEST_DISTANCE_NM_SHORT = 1.0      # Distance from destination to trigger descent (short routes) (NM)
    
    ALT_50 = 50                       # Initial climb and final descent altitude (ft)
    ALT_300 = 300                     # Departure/arrival terminal area altitude (ft)
    ALT_1500 = 1500                   # Cruise altitude (ft)
    
    VS_CLIMB_DESCENT = 500            # Vertical speed for all climbs/descents (ft/min)
    SPEED_CLIMB_DESCENT = 48          # Speed during climb/descent phases (kts)
    SPEED_CRUISE = 96                 # Speed during cruise phase (kts)
    
    DELAY_BEFORE_DELETION = 5.0       # Delay before aircraft deletion after landing (seconds)
    # ===================================================================
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Input CSV file
    csv_file = os.path.join(script_dir, "routes_uam_detailed_rmsp.csv")
    
    # Convert routes to SCN files with customizable parameters
    routes_to_scn(
        csv_file,
        origin_distance_nm=ORIGIN_DISTANCE_NM,
        dest_distance_nm=DEST_DISTANCE_NM,
        dest_distance_nm_short=DEST_DISTANCE_NM_SHORT,
        alt_50=ALT_50,
        alt_300=ALT_300,
        alt_1500=ALT_1500,
        vs_climb_descent=VS_CLIMB_DESCENT,
        speed_climb_descent=SPEED_CLIMB_DESCENT,
        speed_cruise=SPEED_CRUISE,
        delay_before_deletion=DELAY_BEFORE_DELETION
    )
    
    print("\nDone! All SCN files have been generated with manual control approach.")
