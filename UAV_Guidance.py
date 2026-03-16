# Guidance system: waypoint navigation, cross-track erorr path following, and tangent
# law loitering. Outputs target altitude, airspeed, and heading to the flight control
# system (FCS) each timestep.

import numpy as np

class GuidanceSystem:
    """
    Two-mode guidance state: WAYPOINT and LOITER
    
    In WAYPOINT mode, the aircraft follows straight-line paths between waypoints using a
    cross-track error guidance. Once all waypoints are met, it switches to LOITER mode
    and orbits indefinitely around the final waypoint using a tangent law guidance.
    """
    def __init__(self, waypoints, loiter_center, loiter_radius):
        """
        waypoints: List of tuples [(x, y, z), (x, y, z), ...] in NED frame (ft)
        loiter_center: Tuple (x, y, z) in NED
        loiter_radius: radius in ft
        """
        self.waypoints = waypoints
        self.current_wp_index = 0
        self.loiter_center = loiter_center
        self.loiter_radius = loiter_radius
        
        # State machine: 'WAYPOINT' or 'LOITER'
        self.mode = 'WAYPOINT'
        
        # Waypoint capture radius - considered "reached" when within this distance of the waypoint (ft)
        self.wp_radius = 500.0  # [ft]

        # Previous waypoint for path following calculations (initialized to None, will be set to first waypoint on first call)
        self.prev_wp = None

    def get_commands(self, state):
        """
        Returns target_alt [ft], target_speed [ft/s], target_heading [rad]
        """
        x, y, z = state[10], state[11], state[12]
        
        if self.prev_wp is None:
            self.prev_wp = (x, y, z)

        if self.mode == 'WAYPOINT':
            target_wp = self.waypoints[self.current_wp_index]
            is_last_wp = (self.current_wp_index == len(self.waypoints) - 1)
            
            # The last waypoint transitions to LOITER once we're near the orbit circle, not just
            # near the center point so that the entry is smoother.
            if is_last_wp:
                cx, cy = self.loiter_center[0], self.loiter_center[1]
                dist_to_loiter_center = np.sqrt((cx - x)**2 + (cy - y)**2)
                should_advance = abs(dist_to_loiter_center - self.loiter_radius) < self.wp_radius
            else:
                dist_to_wp = np.sqrt((target_wp[0] - x)**2 + (target_wp[1] - y)**2)
                should_advance = dist_to_wp < self.wp_radius
            
            if should_advance:
                print(f"Reached Waypoint {self.current_wp_index + 1}!")
                self.prev_wp = self.waypoints[self.current_wp_index]
                self.current_wp_index += 1

                if self.current_wp_index >= len(self.waypoints):
                    print("All waypoints reached. Switching to LOITER mode.")
                    self.mode = 'LOITER'
                    return self._compute_loiter(x, y, z)

                target_wp = self.waypoints[self.current_wp_index]

            # Cross-track error guidance (NLGL style, Park et al. 2004)
            # Rather than point directly at the waypoint, we calculate and ideal path line between
            # the previous and current waypoint. We then compute the signed lateral derivation from it
            # and use an arctan correction to blend back into the path. This results in smoother,
            # more efficient trajectories than pure pursuit.
            Ax, Ay, Az = self.prev_wp
            Bx, By, Bz = target_wp
            dx = Bx - Ax
            dy = By - Ay
            path_len = np.sqrt(dx**2 + dy**2)
            
            if path_len < 1e-6:
                # Fallback to pure pursuit if path length is zero (edge case)
                target_heading = np.arctan2(By - y, Bx - x)
            else:
                # Angle of ideal path line
                path_angle = np.arctan2(dy, dx)
                
                # Signed cross-track error (positive if to the left of the path, negative if to the right)
                e_y = -np.sin(path_angle) * (x - Ax) + np.cos(path_angle) * (y - Ay)
                
                # Apply Course Correction where the gain (k_path) defines how aggressively to steer back to the path.
                k_path = 0.0009
                correction = np.clip(np.arctan(k_path * e_y), -np.pi/6, np.pi/6) # Limit correction to +/- 30 degrees
                
                # If to the right (positive e_y), subtract from path angle to steer left.
                target_heading = path_angle - correction
            
            target_alt = -target_wp[2]
            target_speed = 222.01 # Fast cruise for point-to-point [ft/s]
                        
            return target_alt, target_speed, target_heading
            
        elif self.mode == 'LOITER':
            return self._compute_loiter(x, y, z)

    def _compute_loiter(self, x, y, z):
        """
        Tangent guidance law for circular loiter
        
        The desired heading is tangent to the circle defined by the loiter center and radius. Preferable 
        to a moving waypoint because it results in smoother trajectories and more consistent speed.
        """
        cx, cy, cz = self.loiter_center
        dx = x - cx
        dy = y - cy
        d = np.sqrt(dx**2 + dy**2)
        
        angle_from_center = np.arctan2(dy, dx)
        tangent_angle = angle_from_center - (np.pi / 2)
        
        # k_loiter sets how aggresively to steer back to the radius
        k_loiter = 0.012
        radial_error = d - self.loiter_radius
        correction = np.clip(np.arctan(k_loiter * radial_error), -np.pi/4, np.pi/4)    # We want to approach the tangent, not the center
        
        target_heading = tangent_angle - correction
        target_alt = -cz
        
        # Slightly lower speed for smoother loitering (still works at 222 ft/s, but looks nicer at 180 ft/s)
        target_speed = 180 # [ft/s]
        
        return target_alt, target_speed, target_heading