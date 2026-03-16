# Numerical trim solver for steady, level, symmetric flight
# Solves for the alpha, dy, throttle comibnation that yields zero
# translational and pitch acceleration at a given velocity and altitude.

import numpy as np
from scipy.optimize import minimize
from UAV_Model import UAV
from UAV_EOM import equations_of_motion

def _trim_cost_function(trim_params, uav, V_trim, altitude):
    """
    Returns the sume of squared accelerations for a given trim state.
    The optimizer will drive this towards zero to find the trim condition.
    """
    
    alpha, de, throttle = trim_params
    
    # Steady level flight assumptions:
    # beta = 0, p = q = r = 0, phi = 0, gamma = 0, theta = alpha, psi = 0
    u = V_trim * np.cos(alpha)
    v = 0
    w = V_trim * np.sin(alpha)
    
    # Pitch quaternion (pure pitch)
    q0 = np.cos(alpha / 2)
    q1 = 0
    q2 = np.sin(alpha / 2)
    q3 = 0
    
    # State Vector construction
    state = np.array([
        u, v, w,            # Velocities
        q0, q1, q2, q3,     # Orientation
        0, 0, 0,            # Rates (p, q, r)
        0, 0, -altitude         # Position (NED: 0, 0, -1000 ft altitude)
    ])
    
    # Controls vector [elevator, aileron, rudder, throttle]
    controls = [de, 0, 0, throttle]
    
    # Derivatives
    d_state = equations_of_motion(0, state, uav, controls)
    
    # Calculate cost: we want dot_u, dot_w, and dot_q to be zero.
    dot_u = d_state[0]
    dot_w = d_state[2]
    dot_q = d_state[8]
    cost = dot_u**2 + dot_w**2 + dot_q**2
    
    return cost

def compute_trim_state(uav, V_trim, altitude=1000):
    """
    Solves for trim (alpha, de, throttle) at the given velocity and altitude.
    Altitude is needed to compute air density for the aerodynamic forces.

    Returns: alpha [rad], de [rad], throttle [0-1]
    """

    print(f"Solving trim for V={V_trim} ft/s...")

    # Initial guess for [alpha, de, throttle]
    x0 = [np.deg2rad(5), np.deg2rad(-5), 0.5]
    bounds = [
        (np.deg2rad(-10), np.deg2rad(20)), # Alpha
        (np.deg2rad(-30), np.deg2rad(30)), # Elevator
        (0.0, 1.0)                         # Throttle
    ]
    
    results = minimize(
        _trim_cost_function,
        x0,
        args=(uav, V_trim, altitude),
        method='SLSQP',
        bounds=bounds,
        options={'ftol': 1e-10, 'disp': False}
    )
    
    if not results.success:
        raise ValueError(f"Trim Solver Failed: {results.message}")
        
    alpha_trim, de_trim, th_trim = results.x
    return alpha_trim, de_trim, th_trim


# Diagnostic
if __name__ == "__main__":
    my_uav = UAV()
    V_trim = 222.01 # ft/s
    
    # Check for how trim changes with altitude (air density)
    for alt, V in [(1000, 222.01), (10000, 222.01), (13000, 222.01)]:
        alpha_trim, de_trim, th_trim = compute_trim_state(my_uav, V, altitude=alt)

        print("\n--- Trim Results ---")
        print(f"Angle of Attack (alpha): {np.degrees(alpha_trim):.4f} deg")
        print(f"Elevator Angle (de): {np.degrees(de_trim):.4f} deg")
        print(f"Throttle: {th_trim*100:.2f} %")

        
        # Residual check
        from UAV_EOM import equations_of_motion as eom
        u = V_trim * np.cos(alpha_trim)
        w = V_trim * np.sin(alpha_trim)
        q0 = np.cos(alpha_trim / 2)
        q2 = np.sin(alpha_trim / 2)
        state = np.array([u, 0, w, q0, 0, q2, 0, 0, 0, 0, 0, 0, -alt])
        controls = [de_trim, 0, 0, th_trim]
        derivatives = eom(0, state, my_uav, controls)
        print("\n--- Trim Residuals (should be near zero) ---")
        print(f"du: {derivatives[0]:.2e}")
        print(f"dw: {derivatives[2]:.2e}")
        print(f"dq: {derivatives[8]:.2e}")