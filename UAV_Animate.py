# Generates an animated 3D GIF of the UAV mission from simulation output.
# Shows the true trajectory alongside the EKF estimated trajectory so the
# estimator performance is visible frame-by-frame.
# This is a standalone utility — call animate_trajectory() from UAV_Sim.py
# or run it separately if you've saved state_log and ekf_log to disk.
# Requires Pillow: pip install pillow

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def animate_trajectory(state_log, time_log, waypoints, l_center, l_radius,
                       ekf_log=None, filename="uav_mission_profile.gif",
                       frame_skip=10, fps=20):
    """
    Renders a 3D animated GIF with a growing trajectory tail and moving UAV marker.
    If ekf_log is provided, the EKF estimated path is drawn alongside the true path.

    frame_skip controls time compression — every frame_skip-th data point becomes
    one animation frame. At dt=0.05s and frame_skip=10, each frame covers 0.5s of
    flight. Higher values = smaller file and faster render.
    """
    print("Generating animation GIF — this takes a few minutes...")

    x,  y,  z  = state_log[:, 10], state_log[:, 11], state_log[:, 12]
    alt = -z

    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection='3d')

    # Static scene elements
    ax.scatter(state_log[0, 11], state_log[0, 10], -state_log[0, 12],
               color='green', s=100, marker='o', label='Start')

    for i, wp in enumerate(waypoints):
        ax.scatter(wp[1], wp[0], -wp[2], color='red', s=100, marker='x',
                   label='Waypoints' if i == 0 else '')

    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(l_center[1] + l_radius * np.cos(theta),
            l_center[0] + l_radius * np.sin(theta),
            np.full(100, -l_center[2]),
            'r--', alpha=0.3, label='Loiter Orbit')

    ax.set_xlabel('East [ft]')
    ax.set_ylabel('North [ft]')
    ax.set_zlabel('Altitude [ft]')
    ax.set_title('UAV Navigation and Loiter — True vs EKF Estimated')
    ax.set_xlim([y.min() - 500, y.max() + 500])
    ax.set_ylim([x.min() - 500, x.max() + 500])
    ax.set_zlim([alt.min() - 200, alt.max() + 500])
    ax.set_aspect('equal')

    # Dynamic elements — true trajectory
    true_line, = ax.plot([], [], [], 'b-',  linewidth=2,   label='True Path')
    true_dot,  = ax.plot([], [], [], 'ko',  markersize=6)

    # Dynamic elements — EKF estimate (only if ekf_log provided)
    if ekf_log is not None:
        ex, ey, ez = ekf_log[:, 10], ekf_log[:, 11], ekf_log[:, 12]
        e_alt = -ez
        ekf_line, = ax.plot([], [], [], color='orange', linewidth=1,
                            linestyle='--', alpha=0.8, label='EKF Estimate')
        ekf_dot,  = ax.plot([], [], [], 'o', color='orange', markersize=5)

    ax.legend()

    def update(idx):
        # Update true path
        true_line.set_data(y[:idx], x[:idx])
        true_line.set_3d_properties(alt[:idx])
        true_dot.set_data([y[idx]], [x[idx]])
        true_dot.set_3d_properties([alt[idx]])

        if ekf_log is not None:
            ekf_line.set_data(ey[:idx], ex[:idx])
            ekf_line.set_3d_properties(e_alt[:idx])
            ekf_dot.set_data([ey[idx]], [ex[idx]])
            ekf_dot.set_3d_properties([e_alt[idx]])
            return true_line, true_dot, ekf_line, ekf_dot

        return true_line, true_dot

    frames = range(0, len(time_log), frame_skip)
    anim   = animation.FuncAnimation(fig, update, frames=frames,
                                     interval=50, blit=False)

    try:
        anim.save(filename, writer='pillow', fps=fps)
        print(f"Saved: {filename}")
    except Exception as e:
        print(f"Failed to save animation: {e}")
        print("Make sure Pillow is installed: pip install pillow")