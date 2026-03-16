# Top-level simulation: ties together the UAV model, EOM, guidance, control, wind,
# and EKF. Runs a waypoint navigation and loiter mission at 20 Hz using RK4 
# integration, then plots the results.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from UAV_Model import UAV
from UAV_EOM import equations_of_motion
from UAV_Control import FlightControlSystem
from UAV_Guidance import GuidanceSystem
from UAV_Trim import compute_trim_state
from UAV_Animate import animate_trajectory
from UAV_Wind import DrydenWindModel
from UAV_EKF import UAVKalmanFilter

def run_simulation():
    # Setup
    my_uav = UAV()
    V_target = 222.01   # Cruise airspeed [ft/s]
    initial_altitude = 7000 # Starting altitude [ft]

    # Trim at starting altitude with ISA atmosphere
    alpha_trim, de_trim, th_trim = compute_trim_state(my_uav, V_target, altitude=initial_altitude)
    
    # Build initial state consistent with tri solution
    u_init = V_target * np.cos(alpha_trim)
    w_init = V_target * np.sin(alpha_trim)
    q0, q2 = np.cos(alpha_trim/2), np.sin(alpha_trim/2)
    
    initial_state = np.array([
        u_init, 0, w_init,
        q0, 0, q2, 0,
        0, 0, 0,
        0, 0, -7000
    ])
    
    dt = 0.05
    trim_settings = {'de': de_trim, 'th': th_trim, 'alpha': alpha_trim}
    fcs = FlightControlSystem(dt=dt, trim_values=trim_settings)
    
    # EKF state estimator (IMU + GPS) 1 Hz GPS update
    ekf = UAVKalmanFilter(initial_state, dt=dt, gps_hz=1.0)

    # [North(x), East(y), Down(z)]
    waypoints = [
        (1500, 5300, -9000),
        (2600, 14000, -10000),
        (-700, 31500, -12500),
        (0, 35000, -13000)
    ]
    loiter_center = waypoints[-1] # Loiter around the last waypoint
    loiter_radius = 1500.0 # [ft]
    guidance = GuidanceSystem(waypoints, loiter_center, loiter_radius)

    # Wind Model: mean + Dryden turbulance (MIL-HDBK-1797)
    mean_wind_ned_vector = np.array([-15.0, -15.0, 0.0]) # ~30 mph from NE -> SW in ft/s
    dryden = DrydenWindModel(dt=dt, sigma=4.5, L=1750.0, seed=42) # Moderate turbulence

    # Altitude rate limiter: realistic climb rate to prevent the controller from panicking at large errors
    alt_cmd_rate_limit = 115 # [ft/s] (~6900 ft/min)
    alt_cmd_current = -initial_state[12]

    # Simulation loop
    t_span = 500.0 # Time span for the simulation in seconds
    steps = int(t_span / dt)
    time_log, state_log = np.zeros(steps), np.zeros((steps, 13))
    control_log = np.zeros((steps, 4)) # de, da, dr, throttle
    ekf_log = np.zeros((steps, 13)) # EKF state estimates

    current_state = np.copy(initial_state)
    
    print("Starting Waypoint Navigation & Loiter Simulation...")
    for i in range(steps):
        t = i * dt
        
        # Guidance
        target_alt, target_speed, target_heading = guidance.get_commands(current_state)
        
        # Rate-limit altitude command
        alt_error_raw = target_alt - alt_cmd_current
        alt_step = np.clip(alt_error_raw, -alt_cmd_rate_limit * dt, alt_cmd_rate_limit * dt)
        alt_cmd_current += alt_step

        # Control
        controls = fcs.compute_controls(current_state, alt_cmd_current, target_speed, target_heading)
        
        # Wind
        Va = np.sqrt(current_state[0]**2 + current_state[1]**2 + current_state[2]**2)
        wind_ned = mean_wind_ned_vector + dryden.update(Va)

        # RK4 integration of the 6-DOF EOM
        k1 = equations_of_motion(t, current_state, my_uav, controls, wind_ned)
        k2 = equations_of_motion(t + dt/2, current_state + k1*dt/2, my_uav, controls, wind_ned)
        k3 = equations_of_motion(t + dt/2, current_state + k2*dt/2, my_uav, controls, wind_ned)
        k4 = equations_of_motion(t + dt, current_state + k3*dt, my_uav, controls, wind_ned)
        current_state = current_state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        current_state[3:7] /= np.linalg.norm(current_state[3:7]) # Normalize quat

        # Step EKF with IMU measurements (simulate IMU noise)
        ekf.step(current_state, controls)

        time_log[i] = t
        state_log[i, :] = current_state
        control_log[i, :] = controls
        ekf_log[i, :] = ekf.get_estimated_state()
    
    print("Simulation Complete.")
    plot_3d_trajectory(state_log, ekf_log, waypoints, loiter_center, loiter_radius)
    plot_ekf_errors(time_log, state_log, ekf_log)
    plot_control_inputs(time_log, control_log)
    # Uncomment below to generate animation GIF (slow, take a few minutes)
    # animate_trajectory(state_log, time_log, waypoints, loiter_center, loiter_radius, ekf_log)

def plot_3d_trajectory(state_log, ekf_log, waypoints, l_center, l_radius):
    """
    True trajectory (blue) overlaid with EKF estimates (orange)
    """
    x, y, z = state_log[:, 10], state_log[:, 11], state_log[:, 12]
    alt = -z
    ex, ey, ez = ekf_log[:, 10], ekf_log[:, 11], ekf_log[:, 12]
    e_alt = -ez
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # UAV path
    ax.plot(y, x, alt, label='True UAV Trajectory', color='blue', linewidth=2)
    
    # EKF estimates
    ax.plot(ey, ex, e_alt, label='EKF Estimated Trajectory', color='orange', linewidth=1, linestyle='--', alpha=0.7)
    
    # Start point
    ax.scatter(state_log[0, 11], state_log[0, 10], -state_log[0, 12], color='green', s=100, marker='o', label='Start')
    
    # Waypoints
    for i, wp in enumerate(waypoints):
        ax.scatter(wp[1], wp[0], -wp[2], color='red', s=100, marker='x', label=f'WP' if i==0 else "")
        
    # Loiter circle
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_y = l_center[1] + l_radius * np.cos(theta)
    circle_x = l_center[0] + l_radius * np.sin(theta)
    circle_z = np.full_like(theta, -l_center[2])
    ax.plot(circle_y, circle_x, circle_z, 'r--', alpha=0.5, label='Loiter Pattern')
    
    # Labels
    ax.set_xlabel('East (Y) [ft]')
    ax.set_ylabel('North (X) [ft]')
    ax.set_zlabel('Altitude [ft]')
    ax.set_title('3D UAV Navigation: True vs EKF Estimated Trajectory')
    ax.legend()
    ax.set_aspect('equal')   
    plt.show()

def plot_ekf_errors(time_log, state_log, ekf_log):
    """
    Per-axis and 3D position error, showing EKF convergence against GPS noise
    """
    pos_error = np.linalg.norm(state_log[:, 10:13] - ekf_log[:, 10:13], axis=1)
    err_x = state_log[:, 10] - ekf_log[:, 10]
    err_y = state_log[:, 11] - ekf_log[:, 11]
    err_z = state_log[:, 12] - ekf_log[:, 12]
 
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
 
    axes[0].plot(time_log, err_x, color='steelblue',   linewidth=0.9, label='North error')
    axes[0].axhline(0, color='k', linewidth=0.5, linestyle='--')
    axes[0].set_ylabel('North error [ft]')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
 
    axes[1].plot(time_log, err_y, color='darkorange',  linewidth=0.9, label='East error')
    axes[1].axhline(0, color='k', linewidth=0.5, linestyle='--')
    axes[1].set_ylabel('East error [ft]')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
 
    axes[2].plot(time_log, -err_z, color='forestgreen', linewidth=0.9, label='Altitude error')
    axes[2].axhline(0, color='k', linewidth=0.5, linestyle='--')
    axes[2].set_ylabel('Altitude error [ft]')
    axes[2].legend(); axes[2].grid(True, alpha=0.3)
 
    axes[3].plot(time_log, pos_error, color='firebrick', linewidth=0.9, label='3D position error (norm)')
    axes[3].set_ylabel('3D error [ft]')
    axes[3].set_xlabel('Time [s]')
    axes[3].legend(); axes[3].grid(True, alpha=0.3)
 
    rms = np.sqrt(np.mean(pos_error**2))
    fig.suptitle(f'EKF Position Estimation Error  (RMS = {rms:.1f} ft  |  GPS σ = {UAVKalmanFilter.SIGMA_GPS_H:.0f} ft horiz)')
    plt.tight_layout()
    plt.show()

def plot_control_inputs(time_log, control_log):
    """
    Control surface deflections and throttle over the final 10 seconds of flight.
    """
    t2, t1 = time_log[-1], max(0, time_log[-1] - 10) # Last 10 seconds of flight
    indices = (time_log >= t1) & (time_log <= t2)
    
    de, da, dr, th = control_log[indices, 0], control_log[indices, 1], control_log[indices, 2], control_log[indices, 3]
    time_subset = time_log[indices]
    
    plt.figure(figsize=(12, 8))
    plt.subplot(4, 1, 1)
    plt.plot(time_subset, np.degrees(de), label='Elevator (de)', color='blue')
    plt.ylabel('Elevator (deg)')
    plt.legend()
    plt.grid()

    plt.subplot(4, 1, 2)
    plt.plot(time_subset, np.degrees(da), label='Aileron (da)', color='orange')
    plt.ylabel('Aileron (deg)')
    plt.legend()
    plt.grid()

    plt.subplot(4, 1, 3)
    plt.plot(time_subset, np.degrees(dr), label='Rudder (dr)', color='green')
    plt.ylabel('Rudder (deg)')
    plt.legend()
    plt.grid()

    plt.subplot(4, 1, 4)
    plt.plot(time_subset, th * 100, label='Throttle (%)', color='red')
    plt.ylabel('Throttle (%)')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.grid()

    plt.suptitle(f'Control Inputs Over Time (t={t1}s to t={t2}s)')
    plt.show()

if __name__ == "__main__":
    run_simulation()