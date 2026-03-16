# Cascaded PID controller for fixed-wing UAV
# Longitudinal: altitude outer loop -> pitch inner loop + speed loop
# Lateral: heading outer loop -> roll inner loop + yaw damper for sideslip

import numpy as np

class PID:
    """
    PID controller with a filtered derivative and two anti-windup strategies
    
    Anti-windup strategy is selectable based on the presence of the Tt parameter:
    - Tt is not None: Back-calculation used on inner loops (pitch, roll, speed)
    - Tt is None: conditional integration used on the altitude outer loop
    """

    def __init__(self, kp, ki, kd, limit_min=-np.inf, limit_max=np.inf, N=10, Tt=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.limit_min = limit_min
        self.limit_max = limit_max
        self.N = N # Derivative filter coefficient 

        self.integral = 0
        self.prev_error = None
        self.d_filtered = 0        

        
        # Auto-calculate Tt if not provided
        # Full PID: Tt = sqrt(Ti * Td), PI only: Tt = Ti
        if ki != 0:
            if Tt is not None:
                self.Tt = Tt
            else:
                Ti = abs(kp / ki)
                if kd != 0:
                    Td = abs(kd / kp)
                    self.Tt = np.sqrt(Ti * Td)
                else:
                    self.Tt = Ti
        else:
            self.Tt = None  # No integrator = no anti-windup

        
    def update(self, error, dt):
        # Proportional
        P = self.kp * error
        
        # Derivative with low-pass filtering
        # The raw de/dt amplifies high-frequency noise, so we apply a first order low-pass filter.
        if self.prev_error is None:
            self.prev_error = error
            D = 0
        else:
            raw_derivative = (error - self.prev_error) / dt if dt > 0 else 0
            H = (self.N * dt) / (1 + self.N * dt)  # Transfer function of the low-pass filter
            self.d_filtered += H * (self.kd * raw_derivative - self.d_filtered)  # Filtered derivative
            D = self.d_filtered
        self.prev_error = error
        
        if self.Tt is not None:
            # Back-calculation anti-windup (inner loops)
            self.integral += error * dt
            output_unsat = P + (self.ki * self.integral) + D
            output = np.clip(output_unsat, self.limit_min, self.limit_max)
            if self.ki != 0:
                self.integral += (dt / self.Tt) * (output - output_unsat) / self.ki  # Anti-windup back-calculation
                # Divide by ki to keep the correction sign-correct for negative gains
        else:
            # Conditional integration (altitude outer loop)
            output_unsat = P + (self.ki * self.integral) + D
            saturated_high = output_unsat >= self.limit_max
            saturated_low = output_unsat <= self.limit_min

            integral_step = error * self.ki # Use error * ki to catch the sign correctly when ki is negative
            winding_up = (saturated_high and integral_step > 0) or (saturated_low and integral_step < 0)
            if not winding_up:
                self.integral += error * dt  # Only integrate when not saturating futher
            output = np.clip(P + (self.ki * self.integral) + D, self.limit_min, self.limit_max)
        return output

class FlightControlSystem:
    """
    Cascaded autopilot with trim feedforward for a fixed-wing UAV
    
    Each channel has an outer loop that determines the desired attitude command and an inner
    loop that drives the control surfaces to achieve it. Trim values from the UAV model are added
    as feedforward so the integrators only need to cover deviations from the nominal flight conditions.
    """

    def __init__(self, dt, trim_values):
        self.dt = dt

        # Store trim values for feedforward
        self.de_trim = trim_values['de']
        self.th_trim = trim_values['th']
        self.alpha_trim = trim_values['alpha']

        # Longitudinal (Pitch & Altitude)
        # Inner Loop: Pitch Attitude -> Elevator (negative gains because positive pitch error requires negative elevator deflection)
        self.pid_pitch = PID(kp=-2.5, ki=-0.8, kd=-1.5, limit_min=np.deg2rad(-30), limit_max=np.deg2rad(17))
        
        # Outer Loop: Altitude -> Desired Pitch (Tt=None for conditional integration anti-windup)
        self.pid_alt = PID(kp=0.08, ki=0.002, kd=0.05, limit_min=np.deg2rad(-20), limit_max=np.deg2rad(20), Tt=None)
        
        # Speed Loop: Airspeed -> Throttle
        self.pid_speed = PID(kp=0.25, ki=0.05, kd=0.05, limit_min=0, limit_max=1.0)
        

        # Lateral (Roll & Heading)
        # Inner Loop: Roll Attitude -> Aileron
        self.pid_roll = PID(kp=-3.5, ki=-0.2, kd=-1.2, limit_min=np.deg2rad(-25), limit_max=np.deg2rad(25))
        
        # Outer Loop: Heading -> Desired Roll
        self.pid_heading = PID(kp=4.0, ki=0.0, kd=0.7, limit_min=np.deg2rad(-50), limit_max=np.deg2rad(50))

        # Yaw Damper (Rudder) PID to minimize Sideslip (v velocity)
        self.pid_yaw_damper = PID(kp=-2.0, ki=0.0, kd=-0.5, limit_min=np.deg2rad(-15), limit_max=np.deg2rad(15))

    def compute_controls(self, state, target_alt, target_speed, target_heading):
        u, v, w = state[0], state[1], state[2]
        q0, q1, q2, q3 = state[3], state[4], state[5], state[6]
        x, y, z = state[10], state[11], state[12]
        
        # Altitude (z is positive DOWN, so alt = -z)
        alt = -z
        
        # Airspeed
        Va = np.sqrt(u**2 + v**2 + w**2)
        
        # Attitude (Quaternion to Euler)
        theta = np.arcsin(2*(q0*q2 - q3*q1))    # Pitch (Theta)
        phi = np.arctan2(2*(q0*q1 + q2*q3), 1 - 2*(q1**2 + q2**2))  # Roll (Phi)
        psi = np.arctan2(2*(q0*q3 + q1*q2), 1 - 2*(q2**2 + q3**2))  # Yaw (Psi)
        
        # Longitudinal Control
        # Altitude outer loop outputs desired pitch angle, alpha_trim offsets theta_cmd so the 
        # inner loop commands pitch relative to the trim attitude rather than zero.
        theta_cmd = self.alpha_trim + self.pid_alt.update(target_alt - alt, self.dt)
        de = self.de_trim + self.pid_pitch.update(theta_cmd - theta, self.dt)
        throttle = self.th_trim + self.pid_speed.update(target_speed - Va, self.dt)
        throttle = np.clip(throttle, 0, 1) # Safety clamp

        # Lateral Control
        # Wrap heading error to -pi to pi for correct behavior across the 0/360 boundary
        psi_error = (target_heading - psi + np.pi) % (2 * np.pi) - np.pi
        phi_cmd = self.pid_heading.update(psi_error, self.dt)
        da = self.pid_roll.update(phi_cmd - phi, self.dt)
        dr = self.pid_yaw_damper.update(v, self.dt)
        
        return [de, da, dr, throttle]