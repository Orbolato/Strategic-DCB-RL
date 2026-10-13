"""
UAM Metrics Plugin for BlueSky

Tracks safety and operational metrics for Urban Air Mobility operations:

Safety Metrics:
1. LoWC (Loss of Well Clear): < 500m horizontal separation
2. NMAC (Near Mid-Air Collision): < 150m horizontal AND < 30m vertical separation
3. MAC (Mid-Air Collision): < 10m horizontal AND < 30m vertical separation

Operational Metrics:
- Aircraft lifecycle events (CREATE/DELETE)
- Per-aircraft flight times and 3D flight distances
- Airborne delays (via post-processing)

All safety metrics normalized per flight hour.
"""

import os
import json
import numpy as np
import bluesky as bs
from bluesky import stack, traf, sim
from bluesky.core import Entity, timed_function
from bluesky.tools import geo
from bluesky.tools.aero import nm

# Safety metric thresholds
LOWC_HORIZ_THRESHOLD_M = 500.0  # Loss of Well Clear horizontal [m]
NMAC_HORIZ_THRESHOLD_M = 150.0  # NMAC horizontal [m]
NMAC_VERT_THRESHOLD_M = 30.0  # NMAC vertical [m]
MAC_HORIZ_THRESHOLD_M = 10.0  # Mid-Air Collision horizontal [m]

# Global plugin instance
safety_metrics = None


def init_plugin():
    """Plugin initialization function."""
    global safety_metrics
    safety_metrics = SafetyMetrics()

    config = {"plugin_name": "UAMMETRICS", "plugin_type": "sim"}

    return config, {}


class SafetyMetrics(Entity):
    """UAM metrics collection entity - tracks safety and operational metrics."""

    def __init__(self):
        super().__init__()

        # Plugin state
        self.active = False

        # Event tracking - unique pairs that have ever violated
        self.lowc_pairs_all = set()
        self.nmac_pairs_all = set()
        self.mac_pairs_all = set()

        # Previous state for edge detection
        self.prev_lowc = set()
        self.prev_nmac = set()
        self.prev_mac = set()

        # Accumulated totals for deleted aircraft
        self.total_flight_time_sec = 0.0
        self.total_flight_distance_nm = 0.0

        # Per-aircraft flight times for completed flights (for post-processing)
        self.aircraft_flight_times = {}  # {aircraft_id: flight_time_seconds}
        self.aircraft_flight_distances = {}  # {aircraft_id: flight_distance_nm}

        # CR activation tracking - count total activations by type
        self.cr_activations = {
            "heading": 0,
            "speed": 0,
            "altitude": 0,
            "vertical_speed": 0,
        }

        # Output files
        self.event_logfile = None
        self.operational_logfile = None
        self.output_dir = "output/metrics"

        # Per-aircraft tracking arrays (auto-resized by BlueSky)
        with self.settrafarrays():
            self.flight_time = np.array([], dtype=float)  # [s]
            self.flight_distance = np.array([], dtype=float)  # [nm] - 3D distance
            self.prev_lat = np.array([], dtype=float)  # [deg]
            self.prev_lon = np.array([], dtype=float)  # [deg]
            self.prev_alt = np.array([], dtype=float)  # [m] - for 3D distance calculation
            self.create_time = np.array([], dtype=float)  # [s] - spawn timestamp
            # Previous CR activation states for edge detection
            self.prev_hdg_active = np.array([], dtype=bool)
            self.prev_tas_active = np.array([], dtype=bool)
            self.prev_alt_active = np.array([], dtype=bool)
            self.prev_vs_active = np.array([], dtype=bool)

    def create(self, n=1):
        """Called when aircraft are created."""
        super().create(n)
        # Initialize new aircraft states
        self.flight_time[-n:] = 0.0
        self.flight_distance[-n:] = 0.0
        self.prev_lat[-n:] = traf.lat[-n:]
        self.prev_lon[-n:] = traf.lon[-n:]
        self.prev_alt[-n:] = traf.alt[-n:]
        self.create_time[-n:] = sim.simt  # Store creation timestamp
        # Initialize CR activation tracking
        self.prev_hdg_active[-n:] = False
        self.prev_tas_active[-n:] = False
        self.prev_alt_active[-n:] = False
        self.prev_vs_active[-n:] = False

        # Log CREATE events to operational log file
        if self.active and self.operational_logfile:
            for i in range(n):
                acid = traf.id[-(n - i)]
                self.operational_logfile.write(f"{sim.simt:.2f},CREATE,{acid},\n")
            self.operational_logfile.flush()

    def delete(self, idx):
        """Called when aircraft are deleted."""
        # Log DELETE events and accumulate totals before deletion
        if isinstance(idx, (list, np.ndarray)):
            for i in idx:
                flight_time = self.flight_time[i]
                flight_distance = self.flight_distance[i]
                acid = traf.id[i]
                self.total_flight_time_sec += flight_time
                self.total_flight_distance_nm += flight_distance

                # Store per-aircraft flight time and distance for post-processing
                self.aircraft_flight_times[acid] = flight_time
                self.aircraft_flight_distances[acid] = flight_distance

                # Log DELETE event to operational log file
                if self.active and self.operational_logfile:
                    self.operational_logfile.write(
                        f"{sim.simt:.2f},DELETE,{acid},{flight_time:.2f}\n"
                    )
        else:
            flight_time = self.flight_time[idx]
            flight_distance = self.flight_distance[idx]
            acid = traf.id[idx]
            self.total_flight_time_sec += flight_time
            self.total_flight_distance_nm += flight_distance

            # Store per-aircraft flight time and distance for post-processing
            self.aircraft_flight_times[acid] = flight_time
            self.aircraft_flight_distances[acid] = flight_distance

            # Log DELETE event to operational log file
            if self.active and self.operational_logfile:
                self.operational_logfile.write(
                    f"{sim.simt:.2f},DELETE,{acid},{flight_time:.2f}\n"
                )

        if self.active and self.operational_logfile:
            self.operational_logfile.flush()

        super().delete(idx)

    @timed_function(dt=1.0)  # Update every 1 second
    def update(self, dt):
        """Periodic update function."""
        if not self.active or traf.ntraf < 1:
            return

        # Accumulate flight time for all active aircraft
        # dt is the actual elapsed simulation time since last update
        self.flight_time += dt

        # Calculate distance traveled since last update
        for i in range(traf.ntraf):
            if self.prev_lat[i] != 0.0 and self.prev_lon[i] != 0.0:
                # Calculate horizontal distance traveled [nm]
                dist_horiz_nm = geo.kwikdist(
                    self.prev_lat[i], self.prev_lon[i], traf.lat[i], traf.lon[i]
                )
                # Calculate vertical distance [nm] - convert from meters
                # 1 meter ≈ 0.000189394 nm
                alt_change_m = abs(traf.alt[i] - self.prev_alt[i])
                alt_change_nm = alt_change_m * 0.000189394
                # Calculate 3D distance
                dist_3d_nm = np.sqrt(dist_horiz_nm**2 + alt_change_nm**2)
                self.flight_distance[i] += dist_3d_nm

            # Update previous position and altitude
            self.prev_lat[i] = traf.lat[i]
            self.prev_lon[i] = traf.lon[i]
            self.prev_alt[i] = traf.alt[i]

        # Track CR activation events (edge detection: False → True transitions)
        # Access CR module through bs.traf.cr (BlueSky's conflict resolution module)
        if hasattr(bs.traf, "cr") and bs.traf.cr is not None:
            # Get current CR activation states from properties
            hdg_active = (
                bs.traf.cr.hdgactive
                if hasattr(bs.traf.cr, "hdgactive")
                else np.zeros(traf.ntraf, dtype=bool)
            )
            tas_active = (
                bs.traf.cr.tasactive
                if hasattr(bs.traf.cr, "tasactive")
                else np.zeros(traf.ntraf, dtype=bool)
            )
            alt_active = (
                bs.traf.cr.altactive
                if hasattr(bs.traf.cr, "altactive")
                else np.zeros(traf.ntraf, dtype=bool)
            )
            vs_active = (
                bs.traf.cr.vsactive
                if hasattr(bs.traf.cr, "vsactive")
                else np.zeros(traf.ntraf, dtype=bool)
            )

            # Detect new activations (False → True transitions)
            new_hdg_activations = np.logical_and(hdg_active, ~self.prev_hdg_active)
            new_tas_activations = np.logical_and(tas_active, ~self.prev_tas_active)
            new_alt_activations = np.logical_and(alt_active, ~self.prev_alt_active)
            new_vs_activations = np.logical_and(vs_active, ~self.prev_vs_active)

            # Log new activations to operational events file
            if self.operational_logfile:
                for i in range(traf.ntraf):
                    if new_hdg_activations[i]:
                        self.cr_activations["heading"] += 1
                        self.operational_logfile.write(
                            f"{sim.simt:.2f},CR_HEADING,{traf.id[i]},\n"
                        )
                    if new_tas_activations[i]:
                        self.cr_activations["speed"] += 1
                        self.operational_logfile.write(
                            f"{sim.simt:.2f},CR_SPEED,{traf.id[i]},\n"
                        )
                    if new_alt_activations[i]:
                        self.cr_activations["altitude"] += 1
                        self.operational_logfile.write(
                            f"{sim.simt:.2f},CR_ALTITUDE,{traf.id[i]},\n"
                        )
                    if new_vs_activations[i]:
                        self.cr_activations["vertical_speed"] += 1
                        self.operational_logfile.write(
                            f"{sim.simt:.2f},CR_VSPEED,{traf.id[i]},\n"
                        )

                # Flush if any activations occurred
                if (
                    new_hdg_activations.any()
                    or new_tas_activations.any()
                    or new_alt_activations.any()
                    or new_vs_activations.any()
                ):
                    self.operational_logfile.flush()

            # Update previous states
            self.prev_hdg_active[:] = hdg_active
            self.prev_tas_active[:] = tas_active
            self.prev_alt_active[:] = alt_active
            self.prev_vs_active[:] = vs_active

        # Skip conflict detection if less than 2 aircraft
        if traf.ntraf < 2:
            return

        # Calculate all pairwise distances
        qdr, dist_nm = geo.kwikqdrdist_matrix(
            np.asmatrix(traf.lat),
            np.asmatrix(traf.lon),
            np.asmatrix(traf.lat),
            np.asmatrix(traf.lon),
        )

        # Convert to meters for comparison
        dist_m = np.asarray(dist_nm) * nm

        # Calculate vertical separation matrix [m]
        alt_matrix = traf.alt.reshape((traf.ntraf, 1))
        vert_sep_m = np.abs(alt_matrix - alt_matrix.T)

        # Create identity matrix to exclude self-pairs
        I = np.eye(traf.ntraf, dtype=bool)  # noqa

        # Detect violations with proper logic:
        # LoWC: horizontal < 500m (any vertical)
        lowc_matrix = (dist_m < LOWC_HORIZ_THRESHOLD_M) & ~I

        # NMAC: horizontal < 150m AND vertical < 30m (both conditions required)
        nmac_matrix = (
            (dist_m < NMAC_HORIZ_THRESHOLD_M)
            & (vert_sep_m < NMAC_VERT_THRESHOLD_M)
            & ~I
        )

        # MAC: horizontal < 10m AND vertical < 30m (same vertical threshold as NMAC)
        mac_matrix = (
            (dist_m < MAC_HORIZ_THRESHOLD_M) & (vert_sep_m < NMAC_VERT_THRESHOLD_M) & ~I
        )

        # Extract unique pairs (use frozenset to avoid counting (A,B) and (B,A) separately)
        lowc_current = {
            frozenset([traf.id[i], traf.id[j]])
            for i, j in zip(*np.where(lowc_matrix))
            if i < j
        }
        nmac_current = {
            frozenset([traf.id[i], traf.id[j]])
            for i, j in zip(*np.where(nmac_matrix))
            if i < j
        }
        mac_current = {
            frozenset([traf.id[i], traf.id[j]])
            for i, j in zip(*np.where(mac_matrix))
            if i < j
        }

        # Detect new events (transitions into violation state)
        new_lowc = lowc_current - self.prev_lowc
        new_nmac = nmac_current - self.prev_nmac
        new_mac = mac_current - self.prev_mac

        # Log new events
        for pair in new_lowc:
            self.lowc_pairs_all.add(pair)
            self._log_event("LOWC", pair, dist_m, vert_sep_m)

        for pair in new_nmac:
            self.nmac_pairs_all.add(pair)
            self._log_event("NMAC", pair, dist_m, vert_sep_m)

        for pair in new_mac:
            self.mac_pairs_all.add(pair)
            self._log_event("MAC", pair, dist_m, vert_sep_m)

        # Update previous state
        self.prev_lowc = lowc_current
        self.prev_nmac = nmac_current
        self.prev_mac = mac_current

    def _log_event(self, event_type, pair, dist_m, vert_sep_m):
        """Log a safety event to CSV file."""
        if not self.event_logfile:
            return

        # Get aircraft IDs and indices
        acid1, acid2 = list(pair)
        try:
            idx1 = traf.id.index(acid1)
            idx2 = traf.id.index(acid2)
        except ValueError:
            # Aircraft may have been deleted
            return

        # Get separation values
        horiz_m = dist_m[idx1, idx2]
        vert_m = vert_sep_m[idx1, idx2]

        # Write to CSV
        self.event_logfile.write(
            f"{sim.simt:.2f},{event_type},{acid1},{acid2},"
            f"{horiz_m:.2f},{vert_m:.2f}\n"
        )
        self.event_logfile.flush()

    def _calculate_total_flight_hours(self):
        """Calculate total flight hours including active and deleted aircraft."""
        active_flight_time = np.sum(self.flight_time)
        total_time_sec = self.total_flight_time_sec + active_flight_time
        return total_time_sec / 3600.0

    def _export_metrics(self):
        """Export metrics to JSON file."""
        total_flight_hours = self._calculate_total_flight_hours()

        # Avoid division by zero
        if total_flight_hours == 0.0:
            return False, "No flight time recorded"

        # Calculate normalized metrics
        metrics = {
            "scenario_info": {
                "simulation_time_sec": sim.simt,
                "total_flight_hours": total_flight_hours,
                "total_flight_distance_nm": self.total_flight_distance_nm
                + np.sum(self.flight_distance),
            },
            "safety_metrics": {
                "lowc_per_flight_hour": len(self.lowc_pairs_all) / total_flight_hours,
                "nmac_per_flight_hour": len(self.nmac_pairs_all) / total_flight_hours,
                "mac_per_flight_hour": len(self.mac_pairs_all) / total_flight_hours,
            },
            "event_counts": {
                "total_lowc_events": len(self.lowc_pairs_all),
                "total_nmac_events": len(self.nmac_pairs_all),
                "total_mac_events": len(self.mac_pairs_all),
            },
            "cr_activations": {
                "heading_alerts": self.cr_activations["heading"],
                "speed_alerts": self.cr_activations["speed"],
                "altitude_alerts": self.cr_activations["altitude"],
                "vertical_speed_alerts": self.cr_activations["vertical_speed"],
                "total_alerts": sum(self.cr_activations.values()),
            },
            "thresholds": {
                "lowc_horizontal_m": LOWC_HORIZ_THRESHOLD_M,
                "nmac_horizontal_m": NMAC_HORIZ_THRESHOLD_M,
                "nmac_vertical_m": NMAC_VERT_THRESHOLD_M,
                "mac_horizontal_m": MAC_HORIZ_THRESHOLD_M,
                "mac_vertical_m": NMAC_VERT_THRESHOLD_M,
            },
            "aircraft_flight_times": self.aircraft_flight_times,
            "aircraft_flight_distances_3d_nm": self.aircraft_flight_distances,
        }

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Write metrics to JSON file using scenario name
        scenname = stack.get_scenname() or "unknown_scenario"
        filename = os.path.join(self.output_dir, f"{scenname}_safety_metrics.json")

        with open(filename, "w") as f:
            json.dump(metrics, f, indent=2)

        return True, f"Metrics exported to {filename}"

    def reset(self):
        """Called when simulation is reset."""
        super().reset()

        # Reset all counters
        self.lowc_pairs_all.clear()
        self.nmac_pairs_all.clear()
        self.mac_pairs_all.clear()
        self.prev_lowc.clear()
        self.prev_nmac.clear()
        self.prev_mac.clear()

        self.total_flight_time_sec = 0.0
        self.total_flight_distance_nm = 0.0
        self.aircraft_flight_times.clear()
        self.aircraft_flight_distances.clear()

        # Reset CR activation counters
        self.cr_activations = {
            "heading": 0,
            "speed": 0,
            "altitude": 0,
            "vertical_speed": 0,
        }

        # Close log files if open
        if self.event_logfile:
            self.event_logfile.close()
            self.event_logfile = None

        if self.operational_logfile:
            self.operational_logfile.close()
            self.operational_logfile = None

        self.active = False

    @stack.command(name="UAMMETRICS")
    def command(self, cmd: "txt" = "STATUS"):  # noqa
        """Stack command handler for UAM metrics."""
        cmd = cmd.upper()

        if cmd == "START":
            if self.active:
                return False, "UAM metrics already active"

            self.active = True

            # Create output directory if it doesn't exist
            os.makedirs(self.output_dir, exist_ok=True)

            # Open event log files using scenario name
            scenname = stack.get_scenname() or "unknown_scenario"

            # Safety events log
            safety_log_filename = os.path.join(
                self.output_dir, f"{scenname}_safety_events.csv"
            )
            self.event_logfile = open(safety_log_filename, "w")
            self.event_logfile.write(
                "Time,EventType,Aircraft1,Aircraft2,Horiz_m,Vert_m\n"
            )

            # Operational events log (CREATE/DELETE)
            operational_log_filename = os.path.join(
                self.output_dir, f"{scenname}_operational_events.csv"
            )
            self.operational_logfile = open(operational_log_filename, "w")
            self.operational_logfile.write("Time,EventType,AircraftID,FlightTime_sec\n")

            return (
                True,
                f"Safety metrics started. Logging to {safety_log_filename} and {operational_log_filename}",
            )

        elif cmd == "STOP":
            if not self.active:
                return False, "Safety metrics not active"

            self.active = False

            # Close log files
            if self.event_logfile:
                self.event_logfile.close()
                self.event_logfile = None

            if self.operational_logfile:
                self.operational_logfile.close()
                self.operational_logfile = None

            # Export metrics
            success, msg = self._export_metrics()

            return success, f"UAM metrics stopped. {msg}"

        elif cmd == "STATUS":
            if not self.active:
                return True, "UAM metrics: INACTIVE"

            status_msg = (
                f"UAM metrics: ACTIVE\n"
                f"LoWC events: {len(self.lowc_pairs_all)}\n"
                f"NMAC events: {len(self.nmac_pairs_all)}\n"
                f"MAC events: {len(self.mac_pairs_all)}"
            )
            return True, status_msg

        elif cmd == "RESET":
            self.reset()
            return True, "UAM metrics reset"

        elif cmd == "EXPORT":
            if not self.active:
                return False, "UAM metrics not active. Use START first."

            success, msg = self._export_metrics()
            return success, msg

        else:
            return False, f"Unknown command: {cmd}. Use START/STOP/STATUS/RESET/EXPORT"
