from dataclasses import dataclass
import numpy as np
from typing import Optional


@dataclass
class TCParameters:
    # Structure parameters
    a_soma: float = 0.1  # ws structure (μS)
    a_dend: float = 2.0  # wd structure (μS)
    a_tree: float = 0.1  # wsd structure (μS)
    b: float = -3.0  # Vd structure (μS)

    # Capacitances
    C: float = 2.0  # Somatic capacitance (nF)
    Cd: float = 2.0  # M0's capacitance (nF)
    Csd: float = 5.0  # Si's capacitance (nF)

    # Voltage and calcium parameters
    e: float = 0.1  # Steepness parameter for calcium influx in soma (mV/mV)
    delta_CaV: float = 5.16  # Calcium rise speed in dendrites (mV)
    delta_eps_z: float = 8.0  # Maximum subtraction to z's decay rate (a.u.)
    delta_V: float = 1.0  # Depolarization slope Vd AdEx (mV)
    delta_Vsd: float = 5.0  # Depolarization slope Vsd,i (mV)

    # Current-related parameters
    d_res: float = 2.0  # Maximum increment rate for KCa related currents (nA/ms)
    d_z: float = 40.0  # Update on Kv influence after somatic spike (nA)

    # Voltage parameters
    EL: float = -61.0  # Resting membrane voltage (mV)
    eps: float = 1.0  # wz's decay rate (a.u.)
    eps_z0: float = 10.0  # z's initial decay rate (a.u.)

    # Coupling constants
    g0: float = 1.0  # Coupling constant from Si towards M0 (μS)
    g_i: float = 7.0  # Coupling constant from M0 towards Si (i=1,2,3) (μS)
    gamma1: float = (
        25.0  # Plasticity and scaling parameter PF input into Si (i=1,2,3) (a.u.)
    )
    g_SD: float = 6.0  # Coupling constant from M0 towards soma (μS)
    g_DS: float = 3.0  # Coupling constant from soma towards M0 (μS)
    g_L: float = 0.1  # Leak conductance (μS)

    # Currents and dynamics
    Iint: float = 1.2 * 91.0  # Nav-associated intrinsic current (nA)
    k: float = (
        5e-4  # Slope for quasilinear region of a dynamics (low calcium levels) (1/nA)
    )
    K_CF: float = 1.0  # CF-input current at which (vCaCF) is at half its capacity (nA)
    K_PF: float = 0.5  # PF-input current at which (vCaPF) is at half its capacity (nA)
    l: float = 0.1  # Speed of z's decay rate reduction after surpassing zlim (1/nA)
    m: float = 0.2  # Slope of activation of KCa channels (1/nM)

    # Calcium and scaling parameters
    alpha_max: float = 0.09  # Scaling of [Ca2+] for wz (nA/nM)
    vCa_PF: float = 1.5  # Maximum calcium influx rate from PF input (nM/ms)
    vCa_CF: float = 80.0  # Maximum calcium influx rate from CF input (nM/ms)
    n: float = 0.2  # Slope of a activation at high calcium levels (1/nM)
    nz: float = 0.9  # Update coefficient for z after dendritic spike (a.u.)
    eta_R: float = 0.1  # Update coefficient for dres after dendritic spike (a.u.)
    P: float = 2.0  # Power coefficient for gKCa influence on dres (a.u.)
    Ca_th: float = 50.0  # Calcium levels for KCa channel activation (nM)

    # Spiking and threshold parameters
    DSF0: float = 5.0  # M0's easiness to spike
    DSF2: float = 1.85  # S2's easiness to spike
    DSF3: float = 2.9  # S3's easiness to spike

    # Time constants
    tau_s2: float = 150.0  # S3's refractory constant after spike (ms)
    tau_s3: float = 75.0  # S2's refractory constant after spike (ms)
    tau_Ca: float = 100.0  # Calcium levels decay constant (ms)
    tau_R: float = 75.0  # Potassium-related currents decay constant (ms)
    tau_w: float = 1.0  # Adaptation variable's decay constant (ms)

    # Voltage thresholds and reset values
    Vth_soma: float = -5.0  # Somatic spiking threshold (mV)
    Vth_dend: float = -35.0  # M0's spiking threshold (mV)
    Vth_tree: float = -20.0  # Dendritic tree's spiking threshold (mV)
    Vreset_soma: float = -55.0  # Somatic reset voltage (mV)
    Vreset_tree2: float = -50.0  # S2's reset voltage (mV)
    Vreset_tree3: float = -55.0  # S3's reset voltage (mV)
    Vh_h: float = -45.0  # M0's threshold for exponential depolarization (mV)
    Vh_2: float = -42.5  # S2's threshold for exponential depolarization (mV)
    Vh_3: float = -42.5  # S3's threshold for exponential depolarization (mV)
    Vh_d: float = (
        -40.0
    )  # Voltage threshold in the dendrites for Cav channel activation (mV)
    Vh_s: float = -42.5  # Voltage threshold in the soma for Cav channel activation (mV)

    # Initial and reset values
    w0: float = 4.0  # Initial parameter for wz nullcline positioning (nA)
    w_reset: float = 15.0  # wz' reset value (nA)
    wd_reset: float = 15.0  # wd' reset value (nA)

    z_lim: float = 50.0  # z threshold


class PurkinjeCellTC:
    def __init__(self, params: TCParameters = TCParameters()):
        self.p = params

        # Initialize state variables
        # Voltages
        self.Vs = self.p.Vreset_soma  # Soma
        self.Vd = self.p.Vreset_soma  # Main dendrite (M0)
        self.Vsd = np.array([self.p.Vreset_tree2] * 3)  # Dendritic trees (S1-S3)

        # Adaptation variables
        self.ws = 0.0  # Soma
        self.wd = 0.0  # Main dendrite
        self.wsd = np.zeros(3)  # Dendritic trees

        # Calcium dynamics
        self.Ca_PF = 0.0
        self.Ca_CF = 0.0

        # Slow dynamics
        self.z = 0.0
        self.d_res = 0.0

        # Spike tracking
        self.last_spike_time = -np.inf
        self.last_dend_spike_times = np.array([-np.inf] * 3)

    def compute_derivatives(
        self,
        t: float,
        Iinh: float = 0.0,
        Iinj: float = 0.0,
        IPF: Optional[np.ndarray] = None,
        ICF: float = 0.0,
    ):
        """Compute all differential equations for one time step"""
        if IPF is None:
            IPF = np.zeros(3)

        # Calcium dynamics
        dCa_PF = self._compute_vCa_PF(IPF) * np.sum(
            [np.exp((self.Vsd[i] - self.p.Vth_tree) / 5.16) for i in range(3)]
        ) - (self.Ca_PF / self.p.tau_Ca)

        dCa_CF = self._compute_vCa_CF(ICF) / (
            1 + np.exp(-0.1 * (self.Vs - self.p.Vth_soma))
        ) - (self.Ca_CF / self.p.tau_Ca)

        # Compute total calcium and KCa activation
        Ca_total = self.Ca_CF + self.Ca_PF
        g_KCa = 1 / (1 + np.exp(-self.p.m * (Ca_total - self.p.Ca_th)))

        # Slow dynamics
        eps_z = self.p.eps_z0 - self.p.delta_eps_z / (
            1 + np.exp(-self.p.l * (self.z + self.d_res - self.p.z_lim))
        )
        dz = -eps_z * self.z / self.p.tau_R
        d_dres = (g_KCa**self.p.P) * self.p.d_res - self.d_res / self.p.tau_R

        # Soma dynamics
        dVs = (1 / self.p.C) * (
            (self.Vs - self.p.EL) ** 2
            + self.p.b * (self.Vs - self.p.EL) * self.ws
            - self.ws**2
            + (self.p.Iint + Iinh + Iinj)
            - (self.z + self.d_res)
            + self.p.g_SD * (self.Vd - self.Vs)
        )

        dws = (
            self.p.eps
            * (
                self.p.a_soma * (self.Vs - self.p.EL)
                - self.ws
                + self.p.w0
                - self._compute_alpha(Ca_total)
            )
            / self.p.tau_w
        )

        # Main dendrite (M0) dynamics
        dVd = (1 / self.p.Cd) * (
            self.p.g_DS * (self.Vs - self.Vd)
            + np.sum([self.p.g_i * (self.Vsd[i] - self.Vd) for i in range(3)])
            + self.p.DSF0
            * self.p.delta_V
            * np.exp((self.Vd - self.p.Vth_dend) / self.p.delta_V)
            - self.wd
        )

        dwd = (self.p.a_dend * (self.Vd - self.p.EL) - self.wd) / self.p.tau_w

        # Dendritic trees dynamics
        dVsd = np.zeros(3)
        dwsd = np.zeros(3)

        DSFi_0 = np.array([0, self.p.DSF2, self.p.DSF3])  # S1 is passive
        for i in range(3):
            tau = self.p.tau_s2 if i == 1 else self.p.tau_s3
            DSF = DSFi_0[i] * (
                1 - 0.9 * np.exp(-(t - self.last_dend_spike_times[i]) / tau)
            )

            dVsd[i] = (1 / self.p.Csd) * (
                self.p.g0 * (self.Vd - self.Vsd[i])
                + DSF
                * self.p.delta_Vsd
                * np.exp((self.Vsd[i] - self.p.Vth_tree) / self.p.delta_Vsd)
                - self.wsd[i]
                + self.p.gamma1 * IPF[i]
            )

            dwsd[i] = (
                self.p.a_tree * (self.Vsd[i] - self.p.EL) - self.wsd[i]
            ) / self.p.tau_w

        return dVs, dws, dVd, dwd, dVsd, dwsd, dCa_PF, dCa_CF, dz, d_dres

    def _compute_vCa_PF(self, IPF):
        """Compute calcium influx rate from PF input"""
        I_total = np.sum(IPF) * self.p.gamma1
        return self.p.vCa_PF * I_total / (self.p.K_PF + I_total)

    def _compute_vCa_CF(self, ICF):
        """Compute calcium influx rate from CF input"""
        return self.p.vCa_CF * ICF / (self.p.K_CF + ICF)

    def _compute_alpha(self, Ca):
        """Compute alpha parameter for adaptation dynamics"""
        if Ca < self.p.Ca_th:
            return self.p.k * (self.p.Ca_th - Ca) + self.p.alpha_max / (
                1 + np.exp(-self.p.n * (Ca - self.p.Ca_th))
            )
        else:
            return self.p.alpha_max / (1 + np.exp(-self.p.n * (Ca - self.p.Ca_th)))

    def update(
        self,
        dt: float,
        t: float,
        Iinh: float = 0.0,
        Iinj: float = 0.0,
        IPF: Optional[np.ndarray] = None,
        ICF: float = 0.0,
    ):
        """Update all state variables using Euler integration"""
        # Get derivatives
        dVs, dws, dVd, dwd, dVsd, dwsd, dCa_PF, dCa_CF, dz, d_dres = (
            self.compute_derivatives(t, Iinh, Iinj, IPF, ICF)
        )

        # Update state variables
        self.Vs += dt * dVs
        self.ws += dt * dws
        self.Vd += dt * dVd
        self.wd += dt * dwd
        self.Vsd += dt * dVsd
        self.wsd += dt * dwsd
        self.Ca_PF += dt * dCa_PF
        self.Ca_CF += dt * dCa_CF
        self.z += dt * dz
        self.d_res += dt * d_dres

        # Handle spikes
        self._handle_spikes(t)

    def _handle_spikes(self, t: float):
        """Handle spike events and resets"""
        # Somatic spike
        if self.Vs >= self.p.Vth_soma:
            self.Vs = self.p.Vreset_soma
            self.ws = 15.0  # Reset adaptation
            self.z += self.p.d_z
            self.last_spike_time = t

        # Dendritic spikes
        for i in range(3):
            if self.Vsd[i] >= self.p.Vth_tree:
                self.Vsd[i] = self.p.Vreset_tree2 if i == 1 else self.p.Vreset_tree3
                self.wsd[i] = 15.0  # Reset adaptation
                self.z += self.p.nz * self.p.d_z  # Smaller effect than somatic spike
                self.d_res += self.p.eta_R * self.p.d_z
                self.last_dend_spike_times[i] = t
