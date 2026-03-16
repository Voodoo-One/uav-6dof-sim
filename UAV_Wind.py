# Dryden atmospheric turbulance model for altitudes above 2000 ft.
# Implements the high-altitude isotropic turbulance model from MIL-HBDK-1797

import numpy as np

class DrydenWindModel:
    """
    Generates correlated wind gusts by filtering white noise through filters that
    reproduce the Dryden power spectral density (PSD).

    Unlike with simple random noise, there is temporal correlation so gusts ramp up and decay
    over time.

    Transfer functions (MIL-HDBK-1797):
        Longitudinal (u):  H_u(s) = sigma*sqrt((2L/(πV)))  *  1/(1 + tau*s)                     [1st order]
        Lateral (v):       H_v(s) = sigma*sqrt((L/(πV)))   *  (1 + sqrt(3*τ*s))/(1 + tau*s)^2   [2nd order]
        Vertical (w):      H_w(s) = sigma*sqrt((L/(πV)))   *  (1 + sqrt(3*τ*s))/(1 + tau*s)^2   [2nd order]

    Where tau = L/V is the turbulence time constant and L is the spatial scale length. The lateral and
    vertical filters are 2nd order due to having greater energy than the longitudinal axis.

    Second-order filters 
        A = [[0,  1],  [-1/tau^2,  -2/tau]]
        B = [0,  sqrt(π)]^tau
        C = [K/tau^2,  K*sqrt(3/tau)] where K = sigma*sqrt((tau/π))

    High-altitude parameters (h > 2,000 ft, MIL-HDBK-1797):
        L_u = L_v = L_w = 1,750 ft   (turbulence scale length, spatially isotropic)
        sigma_u = sigma_v = sigma_w = 4.5 ft/s   (moderate intensity, representative of W20 ~ 30 ft/s)

    All outputs are in the NED (North-East-Down) inertial frame. The EOM already
    rotates NED wind into the body frame, so no additional transformation is needed
    here.
    """

    def __init__(self, dt, sigma=4.5, L=1750.0, seed=None):
        """
        dt = simulation timestep [s]
        sigma = turbulance intensity [ft/s] (4.5 = moderate)
        L = spacial scale length [ft] (1750 ft for h>2000 ft)
        seed = option RNG seed for reproducible runs
        """
        self.dt = dt
        self.sigma = sigma
        self.L = L

        if seed is not None:
            np.random.seed(seed)

       # One scalar for first-order, two-element vector for second-order.
        self.x_u = 0.0
        self.x_v = np.zeros(2)
        self.x_w = np.zeros(2)

    def reset(self):
        """
        Reset all filter states to zero (call between simulation runs).
        """
        self.x_u = 0.0
        self.x_v = np.zeros(2)
        self.x_w = np.zeros(2)

    def update(self, V_a):
        """
        Advance the turbulence model one timestep and return NED gust components.

        tau is computed each step from the current airspeed.
        """
        dt = self.dt
        V  = max(V_a, 1.0)  # Guard against near-zero airspeed at init
        sigma  = self.sigma
        L  = self.L
        tau  = L / V    # Turbulence time constant [s]

        # Longitudinal gust (first-order)
        # Euler-Maruyama descretisation gives the update below
        a_u = 1.0 - dt / tau
        b_u = sigma * np.sqrt(2.0 * dt / tau)
        self.x_u = a_u * self.x_u + b_u * np.random.randn()
        u_g = self.x_u

        # Lateral gust (second-order)
        # K is chosen such that the steady-state output variance equals sigma^2
        K_v        = sigma * np.sqrt(tau / np.pi)
        A_v        = np.array([[0.0,       1.0    ], [-1.0/tau**2, -2.0/tau]])
        C_v        = np.array([K_v / tau**2, K_v * np.sqrt(3.0) / tau])
        noise_v    = np.sqrt(np.pi * dt) * np.random.randn()
        self.x_v   = self.x_v + dt * (A_v @ self.x_v) + np.array([0.0, noise_v])
        v_g        = C_v @ self.x_v

        # Vertical gust (same as lateral)
        C_w        = C_v.copy()
        noise_w    = np.sqrt(np.pi * dt) * np.random.randn()
        self.x_w   = self.x_w + dt * (A_v @ self.x_w) + np.array([0.0, noise_w])
        w_g        = C_w @ self.x_w

        return np.array([u_g, v_g, w_g])


# Diagnostic test to verify the DrydenWindModel implementation
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dt = 0.05
    t_end = 300.0
    V_a = 222.01
    sigma = 4.5
    model = DrydenWindModel(dt=dt, sigma=sigma, seed=42)
    steps = int(t_end / dt)
    time = np.arange(steps) * dt
    gusts = np.array([model.update(V_a) for _ in range(steps)])

    print("--- Dryden turbulence statistics ---")
    labels = ["u_g (longitudinal)", "v_g (lateral)", "w_g (vertical)"]
    for i, lbl in enumerate(labels):
        rms = np.std(gusts[:, i])
        print(f"  {lbl:25s}:  rms = {rms:.3f} ft/s  (target σ = {sigma:.1f} ft/s)")

    # Time-series plot
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    colors = ["steelblue", "darkorange", "forestgreen"]
    for i, (ax, lbl, col) in enumerate(zip(axes, labels, colors)):
        ax.plot(time, gusts[:, i], color=col, linewidth=0.8)
        ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
        ax.set_ylabel(f"{lbl}\n[ft/s]", fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle(f"Dryden Turbulence — Moderate (σ={sigma} ft/s, L={model.L} ft, MIL-HDBK-1797)")
    plt.tight_layout()
    plt.show()