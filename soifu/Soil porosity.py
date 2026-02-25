import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


# ============================================================================
# PARAMETER DEFINITIONS
# Based on Meurer et al. (2020) Table 4 and Text [cite: 535, 536]
# ============================================================================

class SoilParameters:
    def __init__(self, bioturbation_scenario='high', root_scenario='high'):
        # --- Fixed Parameters ---
        self.phi = 0.4  # Total porosity [cm3/cm3]
        self.phi_min = 0.3  # Minimum (textural) porosity [cm3/cm3]
        self.ft_mic = 0.8  # Micropore fraction of textural porosity
        self.gamma_s = 2.7  # Particle density [g/cm3]
        self.gamma_r = 1.2  # Root density [g/cm3]
        self.fr_c = 0.2  # Fraction of coarse roots
        self.fcasts_mic = 0.8  # Fraction of micropores in casts
        self.epsilon_casts = 0.6  # Void ratio of casts

        # --- Scenario Varied Parameters ---
        # Bioturbation rate (tau_s) [year^-1]
        if bioturbation_scenario == 'high':
            self.tau_s = 0.12
        else:
            self.tau_s = 0.012

        # Root production (Rg) [g/cm3/year]
        # Paper notes: Rg = Br * tau_r. We use Rg directly in equations.
        if root_scenario == 'high':
            self.Rg = 0.0012
        else:
            self.Rg = 0.00012

        # Root turnover rate [year^-1] (Assumed 1.0 based on annual cycle)
        self.tau_r = 1.0

        # --- Derived Parameters ---
        # Bulk density: gamma_b = gamma_s * (1 - phi) [Eq 32, cite: 553]
        # Note: Assumes constant total porosity for the parameterization
        self.gamma_b = self.gamma_s * (1.0 - self.phi)

        # Root biomass: Br = Rg / tau_r
        self.Br = self.Rg / self.tau_r

        # Textural Porosities (Constant) [cite: 364, 365]
        self.phi_t_mic = self.ft_mic * self.phi_min
        self.phi_t_mes = (1.0 - self.ft_mic) * self.phi_min


# ============================================================================
# ODE SYSTEM (Equations 22a-c)
# ============================================================================

def soil_structure_odes(y, t, params):
    """
    Computes derivatives for Structural Porosities.
    y = [phi_mac, phi_s_mes, phi_s_mic]
    """
    phi_mac = y[0]  # Macroporosity (all structural)
    phi_s_mes = y[1]  # Structural Mesoporosity
    phi_s_mic = y[2]  # Structural Microporosity

    # Calculate current Matrix Porosity (Textural + Structural)
    # phi_mat = phi_min + phi_s_mes + phi_s_mic
    phi_mat = params.phi_min + phi_s_mes + phi_s_mic

    # Calculate Void Ratio of the Matrix (epsilon)
    # epsilon = phi_mat / (1 - phi)
    epsilon = phi_mat / (1.0 - params.phi)

    # Calculate fraction of structural matrix pores
    # Used to partition effects between meso and micro classes
    sum_s = phi_s_mes + phi_s_mic
    if sum_s > 1e-12:
        frac_mes = phi_s_mes / sum_s
        frac_mic = phi_s_mic / sum_s
    else:
        # Fallback to prevent division by zero (though biologically unlikely)
        frac_mes = 0.0
        frac_mic = 0.0

    # Common terms
    # Root term factor: (Br * tau_r / gamma_r)
    # Since Rg = Br * tau_r, this is (Rg / gamma_r)
    term_roots = (params.Rg / params.gamma_r)

    # Earthworm term factor: (gamma_b * tau_s / gamma_s)
    term_worms = (params.gamma_b * params.tau_s / params.gamma_s)

    # --- Eq 22a: Macroporosity ---
    # Growth of coarse roots creates macro, casting fills macro
    dphi_mac = (params.fr_c * term_roots) + \
               ((epsilon - params.epsilon_casts) * term_worms)

    # --- Eq 22b: Structural Mesoporosity ---
    # Root decay creates meso (fine roots), growth compresses it
    # Casting creates meso, ingestion destroys it
    dphi_s_mes = ((1.0 - params.fr_c - frac_mes) * term_roots) + \
                 (((1.0 - params.fcasts_mic) * params.epsilon_casts - frac_mes * epsilon) * term_worms)

    # --- Eq 22c: Structural Microporosity ---
    # Root growth compresses micro (negative term)
    # Casting creates micro, ingestion destroys it
    dphi_s_mic = (-frac_mic * term_roots) + \
                 ((params.fcasts_mic * params.epsilon_casts - frac_mic * epsilon) * term_worms)

    return [dphi_mac, dphi_s_mes, dphi_s_mic]


# ============================================================================
# SIMULATION & PLOTTING
# ============================================================================

def run_scenarios_figure_5():
    """
    Reproduces Figure 5 from Meurer et al. (2020)
    """
    # 1. Define Initial Conditions
    # Text says: "0.32 and 0.08 for phi_mic and phi_mes" [cite: 518]
    # We must convert these Total Porosities to Structural components for the ODE.

    # Temporary params to calculate textural baseline
    temp_p = SoilParameters()

    # Initial Total values
    init_phi_mic_total = 0.32
    init_phi_mes_total = 0.08
    init_phi_mac_total = 0.0

    # Calculate Initial Structural values: Total - Textural
    init_phi_s_mic = init_phi_mic_total - temp_p.phi_t_mic  # 0.32 - 0.24 = 0.08
    init_phi_s_mes = init_phi_mes_total - temp_p.phi_t_mes  # 0.08 - 0.06 = 0.02

    y0 = [init_phi_mac_total, init_phi_s_mes, init_phi_s_mic]

    # 2. Define Time
    years = 100
    t = np.linspace(0, years, years * 10 + 1)  # 10 steps per year

    # 3. Define Scenarios
    scenarios = [
        # (Label, Bioturbation, Roots, Color, LineStyle)
        ("High bioturb, High roots", 'high', 'high', 'blue', '-'),
        ("High bioturb, Low roots", 'high', 'low', 'magenta', '-'),
        ("Low bioturb, High roots", 'low', 'high', 'cyan', '-'),
        ("Low bioturb, Low roots", 'low', 'low', 'black', '-')
    ]

    # 4. Initialize Plot
    fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

    # 5. Run Simulations
    for label, bio_scen, root_scen, color, ls in scenarios:
        params = SoilParameters(bio_scen, root_scen)

        # Solve ODE
        solution = odeint(soil_structure_odes, y0, t, args=(params,))

        # Extract Structural results
        phi_mac = solution[:, 0]
        phi_s_mes = solution[:, 1]
        phi_s_mic = solution[:, 2]

        # Calculate Total Porosities for plotting [cite: 367-369]
        # Total = Textural (constant) + Structural (dynamic)
        phi_total_mes = params.phi_t_mes + phi_s_mes
        phi_total_mic = params.phi_t_mic + phi_s_mic

        # Convert to percentage
        pct_mac = phi_mac * 100
        pct_mes = phi_total_mes * 100
        pct_mic = phi_total_mic * 100

        # Plotting
        axes[0].plot(t, pct_mac, color=color, linestyle=ls, linewidth=1.5, label=label)
        axes[1].plot(t, pct_mes, color=color, linestyle=ls, linewidth=1.5)
        axes[2].plot(t, pct_mic, color=color, linestyle=ls, linewidth=1.5)

    # 6. Formatting

    # (a) Macroporosity
    axes[0].set_ylabel('Macroporosity (%)', fontsize=12)
    axes[0].text(0.02, 0.9, '(a)', transform=axes[0].transAxes, fontsize=14, fontweight='bold')
    axes[0].set_ylim(0, 5.0)
    axes[0].set_yticks([0, 2.5, 5.0])
    # Custom legend to match paper style approx
    axes[0].legend(loc='upper left', bbox_to_anchor=(0.1, 0.8), frameon=False, fontsize=9)

    # (b) Mesoporosity
    axes[1].set_ylabel('Mesoporosity (%)', fontsize=12)
    axes[1].text(0.02, 0.9, '(b)', transform=axes[1].transAxes, fontsize=14, fontweight='bold')
    axes[1].set_ylim(0, 10)
    axes[1].set_yticks([0, 5, 10])

    # (c) Microporosity
    axes[2].set_ylabel('Microporosity (%)', fontsize=12)
    axes[2].set_xlabel('Years', fontsize=12)
    axes[2].text(0.02, 0.9, '(c)', transform=axes[2].transAxes, fontsize=14, fontweight='bold')
    axes[2].set_ylim(0, 40)
    axes[2].set_yticks([0, 20, 40])

    # Grid and cleanup
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=10)
        # Add light horizontal lines for reference if desired, but paper is clean

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_scenarios_figure_5()