# Extended Kalman Filter for UAV navigation (20 Hz)
# GPS correction at 1 Hz. Loosely-coupled INS/GPS architecture

import numpy as np

class UAVKalmanFilter:
    """
    10-state EKF: NED position (3), NED velocity (3), attitude quaternion (4)

    The IMU drives the prediction step, accelerometer and gyro readings integrate
    the state forward between GPS fixes. The GPS provides a direct position and
    velocity correction when a fix arrives. Angular rates are treated as known
    inputs from the gyro rather than estimated states.

    State vector:
        x = [px, py, pz,        NED position [ft]
             vx, vy, vz,        NED velocity [ft/s]
             q0, q1, q2, q3]    body-to-NED quaternion, scalar-first
 
    Process model (IMU-driven):
        p_dot = v
        v_dot = R(q) * f_imu + g_ned
        q_dot = 0.5 * Omega(omega_imu) * q
 
    GPS measurement model:
        z = H * x + noise,  H selects [px, py, pz, vx, vy, vz] from state
 
    Sensor noise values are representative of a MEMS IMU and civilian GPS.
    """

    # Sensor noise (1-sigma)
    SIGMA_ACCEL   = 0.5     # accelerometer [ft/s^2]
    SIGMA_GYRO    = 0.005   # gyroscope [rad/s]
    SIGMA_GPS_H   = 16.4    # GPS horizontal position [ft] (~5 m)
    SIGMA_GPS_V   = 32.8    # GPS vertical position [ft] (~10 m)
    SIGMA_GPS_VEL = 0.98    # GPS velocity [ft/s] (~0.3 m/s)

    def __init__(self, initial_state, dt, gps_hz=1.0, g=32.174):
        """
        initial_state : sim-format state vector (13,) [u,v,w, q, p,q,r, x,y,z]
        dt: simulation timestep [s]
        gps_hz: GPS update rate [Hz]
        g: gravitational acceleration [ft/s^2]
        """
        self.dt = dt
        self.g = g
        self.gps_step = int(round(1.0 / (gps_hz * dt)))  # steps between GPS updates
        self.step_count = 0

        # Convert sim state → EKF state
        # Sim:  [u,v,w, q0,q1,q2,q3, p,q,r, x,y,z]
        # EKF:  [px,py,pz, vx,vy,vz, q0,q1,q2,q3]
        u, v, w = initial_state[0:3]
        q0, q1, q2, q3 = initial_state[3:7]
        x, y, z = initial_state[10:13]

        # Rotate body velocities to NED for initial velocity estimate
        v_ned = self._quat_to_rotmat(np.array([q0,q1,q2,q3])) @ np.array([u, v, w])

        self.x = np.array([x, y, z,
                           v_ned[0], v_ned[1], v_ned[2],
                           q0, q1, q2, q3])

        # Initial covariance: uncertainty in initial conditions
        p_pos  = 50.0**2    # [ft^2]
        p_vel  = 2.0**2     # [(ft/s)^2]
        p_att  = 0.01**2    # [rad^2]
        self.P = np.diag([p_pos, p_pos, p_pos,
                          p_vel, p_vel, p_vel,
                          p_att, p_att, p_att, p_att])

        # Process noise Q: Uncertainty injected by IMU noise each step
        sa2 = self.SIGMA_ACCEL**2 * dt
        sg2 = (self.SIGMA_GYRO**2 * dt) / 4   # /4 because quaternion has 4 components
        self.Q = np.diag([0, 0, 0,
                          sa2, sa2, sa2,
                          sg2, sg2, sg2, sg2])

        # Measurement noise R: GPS position and velocity uncertainties
        sp2 = self.SIGMA_GPS_H**2
        sv2 = self.SIGMA_GPS_V**2
        sv_vel2 = self.SIGMA_GPS_VEL**2
        self.R_gps = np.diag([sp2, sp2, sv2,
                               sv_vel2, sv_vel2, sv_vel2])

        # H selects [px, py, pz, vx, vy, vz] directly from the state vector
        self.H = np.zeros((6, 10))
        self.H[0:3, 0:3] = np.eye(3)    # position rows
        self.H[3:6, 3:6] = np.eye(3)    # velocity rows

        # Storage for plotting
        self.state_log = []

    def step(self, true_state, controls):
        """
        Advance one timestep: simulate IMU, predict, and GPS update if due.
        """
        # Simulate IMU measurements (true + noise)
        accel_imu, gyro_imu = self._simulate_imu(true_state)

        # Predict
        self._predict(accel_imu, gyro_imu)

        # GPS update at 1 Hz
        if self.step_count % self.gps_step == 0:
            self._update_gps(self._simulate_gps(true_state))

        self.step_count += 1
        self.state_log.append(self.x.copy())
        return self._to_sim_state(true_state)

    def get_estimated_state(self):
        """
        Return the current EKF estimate in sim state format (13,). Angular rates are zero.
        """
        return self._to_sim_state_from_x()

    def _predict(self, accel_imu, gyro_imu):
        """
        Propagate state and covariance using IMU measurements.

        State propagation uses the nonlinear process model directly.
        Covariance propagation uses the linearised Jacobian F.
        """
        dt = self.dt
        g  = self.g
        px, py, pz = self.x[0:3]
        vx, vy, vz = self.x[3:6]
        q = self.x[6:10]

        R = self._quat_to_rotmat(q)
        p_rate, q_rate, r_rate = gyro_imu

        # Propagate position
        px_new = px + vx * dt
        py_new = py + vy * dt
        pz_new = pz + vz * dt

        # Propagate velocity
        a_ned = R @ accel_imu + np.array([0, 0, g])
        vx_new = vx + a_ned[0] * dt
        vy_new = vy + a_ned[1] * dt
        vz_new = vz + a_ned[2] * dt

        # Propagate attitude quaternion
        Omega = self._omega_matrix(p_rate, q_rate, r_rate)
        q_new = q + 0.5 * (Omega @ q) * dt
        q_new /= np.linalg.norm(q_new)     # Re-normalise

        self.x = np.array([px_new, py_new, pz_new,
                           vx_new, vy_new, vz_new,
                           q_new[0], q_new[1], q_new[2], q_new[3]])

        # Jacobian F
        F = np.zeros((10, 10))
        F[0:3, 3:6]   = np.eye(3)                   # d(p_dot)/dv
        F[3:6, 6:10]  = self._dRa_dq(q, accel_imu)  # d(v_dot)/dq
        F[6:10, 6:10] = 0.5 * Omega                 # d(q_dot)/dq

        # Discrete state transition: Phi ~ I + F*dt
        Phi = np.eye(10) + F * dt

        # Covariance propagation
        self.P = Phi @ self.P @ Phi.T + self.Q

    def _update_gps(self, z_gps):
        """
        Standard EKF measurement update.
 
        Uses the Joseph form of the covariance update (P = (I-KH)P(I-KH)^T + KRK^T)
        instead of the simpler P = (I-KH)P because the Joseph form stays symmetric
        and positive-definite even with numerical round-off errors.
        """
        H = self.H
        y = z_gps - H @ self.x              # Innovation
        S = H @ self.P @ H.T + self.R_gps   # Innovation covariance
        K = self.P @ H.T @ np.linalg.inv(S) # Kalman gain

        self.x = self.x + K @ y # State update
        I_KH = np.eye(10) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R_gps @ K.T  # Joseph form

        # Re-normalise quaternion after update
        self.x[6:10] /= np.linalg.norm(self.x[6:10])

    def _simulate_imu(self, true_state):
        """
        Generate noisy IMU readings from the true sim state.

        The accelerometer measures specific force in the body frame:
            f = a_body - g_body  =  R^T * (a_ned - g_ned)
        Since the sim gives body velocities, we approximate body acceleration
        as the NED acceleration rotated to body frame, minus gravity in body frame.
        For the EKF, we use the full body-frame velocity u,v,w and the attitude
        to reconstruct what an IMU would measure, then add noise.

        In a real system the IMU output is used directly.
        """
        q               = true_state[3:7]
        p_r, q_r, r_r   = true_state[7:10]

        # Gravity in body frame
        R_n2b = self._quat_to_rotmat(q).T
        g_body = R_n2b @ np.array([0, 0, self.g])
        f_body_true = -g_body

        accel_meas = f_body_true + np.random.randn(3) * self.SIGMA_ACCEL
        gyro_meas  = np.array([p_r, q_r, r_r]) + np.random.randn(3) * self.SIGMA_GYRO

        return accel_meas, gyro_meas

    def _simulate_gps(self, true_state):
        """
        Synthesise a noisy GPS fix from the true state.
        Horizontal and vertical noise are different, matching typical GPS specs.
        """
        x_ned = true_state[10:13]
        u, v, w = true_state[0:3]
        q = true_state[3:7]

        R_b2n = self._quat_to_rotmat(q)
        v_ned = R_b2n @ np.array([u, v, w])

        # Add GPS noise (horizontal and vertical noise differ)
        pos_noise = np.array([
            np.random.randn() * self.SIGMA_GPS_H,
            np.random.randn() * self.SIGMA_GPS_H,
            np.random.randn() * self.SIGMA_GPS_V
        ])
        vel_noise = np.random.randn(3) * self.SIGMA_GPS_VEL

        return np.concatenate([x_ned + pos_noise, v_ned + vel_noise])

    def _to_sim_state(self, true_state):
        """
        Return EKF-estimated state in sim format (13,).
        Angular rates p,q,r come from the true state (not estimated by EKF).
        """
        q   = self.x[6:10]
        R   = self._quat_to_rotmat(q)
        v_ned = self.x[3:6]
        v_body = R.T @ v_ned

        out = true_state.copy()
        out[0:3]   = v_body         # estimated body velocities
        out[3:7]   = q              # estimated attitude
        out[10:13] = self.x[0:3]    # estimated position
        return out

    def _to_sim_state_from_x(self):
        """
        Convert EKF internal state to the 13-element sim state format.
        """
        q = self.x[6:10]
        R = self._quat_to_rotmat(q)
        v_ned = self.x[3:6]
        v_body = R.T @ v_ned

        out = np.zeros(13)
        out[0:3] = v_body
        out[3:7] = q
        out[10:13] = self.x[0:3]
        return out

    # Math helper functions

    @staticmethod
    def _quat_to_rotmat(q):
        """
        Body-to-NED rotation matrix from scalar-first quaternion.
        """
        q0, q1, q2, q3 = q
        return np.array([
            [q0**2+q1**2-q2**2-q3**2, 2*(q1*q2-q0*q3),         2*(q1*q3+q0*q2)        ],
            [2*(q1*q2+q0*q3),         q0**2-q1**2+q2**2-q3**2,  2*(q2*q3-q0*q1)        ],
            [2*(q1*q3-q0*q2),         2*(q2*q3+q0*q1),          q0**2-q1**2-q2**2+q3**2]
        ])

    @staticmethod
    def _omega_matrix(p, q, r):
        """
        Skew-symmetric Omega matrix for quaternion kinematics: q_dot = 0.5*Omega*q.
        """
        return np.array([
            [ 0, -p, -q, -r],
            [ p,  0,  r, -q],
            [ q, -r,  0,  p],
            [ r,  q, -p,  0]
        ])

    @staticmethod
    def _dRa_dq(q, a):
        """
        Analytical Jacobian of R(q)*a with respect to quaternion q.
        Returns 3x4 matrix.

        Derived by differentiating each element of R(q)*a w.r.t. each quaternion
        component.
        """
        q0, q1, q2, q3 = q
        ax, ay, az = a

        return np.array([
            # d/dq0           d/dq1           d/dq2           d/dq3
            [ 2*q0*ax - 2*q3*ay + 2*q2*az,
              2*q1*ax + 2*q2*ay + 2*q3*az,
             -2*q2*ax + 2*q1*ay + 2*q0*az,
             -2*q3*ax - 2*q0*ay + 2*q1*az ],  # row 0 (x-component)

            [ 2*q3*ax + 2*q0*ay - 2*q1*az,
              2*q2*ax - 2*q1*ay - 2*q0*az,
              2*q1*ax + 2*q2*ay + 2*q3*az,
              2*q0*ax - 2*q3*ay + 2*q2*az ],  # row 1 (y-component)

            [-2*q2*ax + 2*q1*ay + 2*q0*az,
              2*q3*ax + 2*q0*ay - 2*q1*az,
             -2*q0*ax + 2*q3*ay - 2*q2*az,
              2*q1*ax + 2*q2*ay + 2*q3*az ]   # row 2 (z-component)
        ])