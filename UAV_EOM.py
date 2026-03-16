# 6-DOF rigid-body equations of motion for the NED frame.
# Attitude is represented as a quaternion to avoid singularities/gimbal lock.

import numpy as np
from scipy.spatial.transform import Rotation as R

def equations_of_motion(t, state, uav, controls, wind_NED=np.zeros(3)):
    """
    Returns the state derivatives d_state/dt for RK4 integration.

    State Vector Definition
    [0-2]   u, v, w         : Body frame velocities [ft/s]
    [3-6]   q0, q1, q2, q3  : Quaternion (Scalar First: w, x, y, z) 
    [7-9]   p, q, r         : Body angular rates [rad/s]
    [10-12] x, y, z         : NED Position [ft] (z positive down)
    """

    # Current state
    u, v, w = state[0], state[1], state[2]
    q0, q1, q2, q3 = state[3], state[4], state[5], state[6]
    p, q, r = state[7], state[8], state[9]
    
    # Normalize each step to avoid accumulated numerical drift in the quaternion
    q_norm = np.sqrt(q0**2 + q1**2 + q2**2 + q3**2)
    q0, q1, q2, q3 = q0/q_norm, q1/q_norm, q2/q_norm, q3/q_norm

    # Rotate the forces on body to the body frame
    r_quat = R.from_quat([q1, q2, q3, q0])

    # Rotate wind from NED to body frame for aerodynamic calculations
    wind_body = r_quat.inv().apply(wind_NED)

    F_aero_thrust, M_aero_thrust = uav.calculate_forces_moments(state, controls, wind_body)
    
    # Gravity acts in the NED frame -z direction, so rotate it into the body frame
    g_vec_body = r_quat.inv().apply(np.array([0, 0, uav.grav]))
    F_gravity = uav.mass * g_vec_body
    
    # Total forces and moments
    F_total = F_aero_thrust + F_gravity
    M_total = M_aero_thrust
    
    vel_body = np.array([u, v, w])
    omega_body = np.array([p, q, r])
    
    # Translational EOM
    accel_linear = (F_total / uav.mass) - np.cross(omega_body, vel_body)
    
    # Rotational EOM
    J_omega = uav.J @ omega_body
    cross_term = np.cross(omega_body, J_omega)
    accel_angular = uav.J_inv @ (M_total - cross_term)
    
    # Kinematics (Quaternion Derivative)
    # Matrix form for q_dot = 0.5 * Q(omega) * q
    dq0 = 0.5 * (-q1*p - q2*q - q3*r)
    dq1 = 0.5 * (q0*p - q3*q + q2*r)
    dq2 = 0.5 * (q3*p + q0*q - q1*r)
    dq3 = 0.5 * (-q2*p + q1*q + q0*r)
    
    # Navigation: rotate body velocities into NED frame for position derivatives
    vel_inertial = r_quat.apply(vel_body)
    
    # Pack Derivatives
    d_state = np.concatenate([
        accel_linear,           # du, dv, dw
        [dq0, dq1, dq2, dq3],   # dq0, dq1, dq2, dq3
        accel_angular,          # dp, dq, dr
        vel_inertial            # dx, dy, dz
    ])
    
    return d_state