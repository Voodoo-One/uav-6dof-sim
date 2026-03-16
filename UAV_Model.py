# UAV aerodynamic model, inertia properties, and force/moment calculations
# Aerodynmic coefficients are are a mix of legacy wind tunnel values and XFLR5 vortext-lattice
# results scaled to math the aircraft's geometry.

import numpy as np
from scipy.spatial.transform import Rotation as R

class UAV:
    def __init__(self):
        # Physical properties
        self.weight = 320.572     # [lbsf]
        self.grav = 32.17405      # [ft/s^2]
        self.mass = self.weight / self.grav # [slugs]
        
        # Inertia Properties:
        # Since I don't have measured inertia values for this ariframe, I scaled legacy values from 
        # the AAI RQ-2 Pioneer by mass and geometric ratios.
        scale_mass = self.weight / 420.0
        scale_width = (11.551 / 16.90)**2 
        scale_length = 0.6**2   # Estimated conservative length ratio

        # Inertia tensor [slug-ft^2]
        self.Ixx = 34.832 * scale_mass * scale_width
        self.Iyy = 67.08 * scale_mass * scale_length
        self.Izz = 82.22 * scale_mass * scale_length
        self.Ixz = -4.902 * scale_mass * scale_length
        self.J = np.array([
            [self.Ixx, 0,        self.Ixz],
            [0,        self.Iyy, 0],
            [self.Ixz, 0,        self.Izz]
        ])
        self.J_inv = np.linalg.inv(self.J) # Pre-calculate inverse for efficiency

        # Aerodynamic reference values
        self.bw = 11.551    # Span [ft]
        self.cbar = 0.963   # Mean chord [ft]
        self.S = 11.118     # Wing area [ft^2]
        self.AR = 12        # Aspect ratio

        # Parasite drag and induced drag factor with Oswald efficiency
        self.CD0 = 0.016
        self.e = 0.706
        self.k_drag = 1 / (np.pi * self.AR * self.e)

        # Sea level standard density for thrust scaling
        self.rho_sea = 0.002379 # slugs/ft^3
        self.T_sea = 518.67 # Rankine

    @staticmethod
    def isa_atmosphere(altitude):
        """
        ISA troposphere model for 0-36000 ft

        Derived from the hydrostatic equation and ideal gas law, with a standard lapse rate of 
        0.00356616 R/ft.

        Returns desnsity [slugs/ft^3] and temperature [Rankine]
        """
        T_sea = 518.67          # Sea level standard temperature in Rankine
        lapse_rate = 0.00356616 # Temperature lapse rate in Rankine/ft
        rho_sea = 0.002379      # Sea level standard density in slugs/ft^3
        h_tropopause = 36000    # Tropopause altitude in ft

        h = np.clip(altitude, 0, h_tropopause)
        T = T_sea - lapse_rate * h
        rho = rho_sea * (T / T_sea)**(4.2561) # Exponent = (g / (R * lapse_rate)) = 4.2561 for air

        return rho, T

    def get_aero_coeffs(self, alpha, beta, de, da, dr, p, q, r, Vt):
        """
        Returns aerodynamic coefficients [CL, CD, Cm, CY, Cl, Cn] based on a standard stability-derivative
        model. Coefficients tagged "legacy" are from the original wind tunnel data and remaining values
        were computed from the vortex-lattice model using the actual geometry of the aircraft.
        """

        # Avoid divide by zero if Vt is very small (e.g. initialization)
        if Vt < 0.1: Vt = 0.1
        
        # Non-dimensional rates
        c_2V = self.cbar / (2 * Vt)
        b_2V = self.bw / (2 * Vt)

        # Longitudinal coefficients
        CL0, CL_a, CL_adot, CL_q, CL_de = 0.385, 5.7613, 2.42, 11.25231, 0.401  # Legacy CL0, CL_adot, and CL_de
        CD_de = 0.0180  # Legacy CD_a and CD_de
        Cm0, Cm_a, Cm_adot, Cm_q, Cm_de = 0.194, -1.45606, -11.0, -35.52731, -1.76  # Legacy Cm0, Cm_adot, and Cm_de

        alpha_dot = q # Approximation: alpha_dot ~ q for small angles and flat flight

        CL = CL0 + (CL_a * alpha) + (CL_de * de) + (CL_q * q * c_2V) + (CL_adot * alpha_dot * c_2V)
        CD = self.CD0 + (self.k_drag * CL**2) + (CD_de * abs(de))
        Cm = Cm0 + (Cm_a * alpha) + (Cm_de * de) + (Cm_q * q * c_2V) + (Cm_adot * alpha_dot * c_2V)

        # Lateral coefficients
        CY_beta, CY_dr, CY_p, CY_r = -0.22549, 0.191, -0.02967,0.19663   # Legacy CY_dr, added CY_p and CY_r from XFLR5
        Cl_beta, Cl_p, Cl_r, Cl_da, Cl_dr = -0.01784, -0.60144, 0.18314, -0.161, -0.00229   # Legacy Cl_da and Cl_dr
        Cn_beta, Cn_p, Cn_r, Cn_da, Cn_dr = 0.09024, -0.09088, -0.07541, 0.0200, -0.0917    # Legacy Cn_da and Cn_dr

        CY = (CY_beta * beta) + (CY_dr * dr) + (CY_p * p * b_2V) + (CY_r * r * b_2V)
        Cl = (Cl_beta * beta) + (Cl_da * da) + (Cl_dr * dr) + (Cl_p * p * b_2V) + (Cl_r * r * b_2V)
        Cn = (Cn_beta * beta) + (Cn_da * da) + (Cn_dr * dr) + (Cn_p * p * b_2V) + (Cn_r * r * b_2V)

        return np.array([CL, CD, Cm, CY, Cl, Cn])

    def calculate_forces_moments(self, state, controls, wind_body_frame = np.zeros(3)):
        """
        Calculates body frame forces [Fx, Fy, Fz] and moments [L, M, N] in lbf and ft-lbf
        
        Wind is passed in already rotated into the body frame by EOM. The airspeed components
        are computed relative the air mass so that aerodynamic forces correctly reflect the true
        AoA in a crosswind.
        """
        
        u, v, w = state[0], state[1], state[2]
        p, q, r = state[7], state[8], state[9]
        z = state[12]
        de, da, dr, throttle = controls

        # Density at current altitude from ISA Atmosphere
        altitude = -z # z is positive down
        rho, _ = self.isa_atmosphere(altitude) # Update density based on altitude for force calculations

        # Airspeed relative to the air mass
        u_air = u - wind_body_frame[0]
        v_air = v - wind_body_frame[1]
        w_air = w - wind_body_frame[2]
        Vt = max(np.sqrt(u_air**2 + v_air**2 + w_air**2), 0.1)

        alpha = np.arctan2(w_air, u_air)
        beta  = np.arcsin(v_air / Vt)   # Valid for small sideslip angles

        q_bar = 0.5 * rho * Vt**2

        coeffs = self.get_aero_coeffs(alpha, beta, de, da, dr, p, q, r, Vt)
        CL, CD, Cm, CY, Cl, Cn = coeffs
        
        # Transforming aerodynamic forces into body frame
        Fx_aero = q_bar * self.S * (-CD * np.cos(alpha) + CL * np.sin(alpha))
        Fy_aero = q_bar * self.S * CY
        Fz_aero = q_bar * self.S * (-CD * np.sin(alpha) - CL * np.cos(alpha))

        L_aero = q_bar * self.S * self.bw * Cl
        M_aero = q_bar * self.S * self.cbar * Cm
        N_aero = q_bar * self.S * self.bw * Cn

        # Thrust model
        # Peak values are estimated from engine power (220 hp) with a propeller efficiency of ~0.8
        # at cruise speed, scaled by the ratio of current air density to sea level density to reflect
        # the reduced thrust at higher altitudes.
        max_thrust = 410    # Approximated based on 220 hp engine, cruise speed, and propeller efficiency ~0.8, converted to lbs
        Fx_thrust = max_thrust * throttle * (rho / self.rho_sea)  # Assuming throttle is 0-1 percentage
        
        # Total body forces
        Fx = Fx_aero + Fx_thrust
        Fy = Fy_aero
        Fz = Fz_aero

        return np.array([Fx, Fy, Fz]), np.array([L_aero, M_aero, N_aero])