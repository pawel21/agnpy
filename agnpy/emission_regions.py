import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.coordinates import Distance
from .spectra import PowerLaw, BrokenPowerLaw, SmoothlyBrokenPowerLaw


MEC2 = (const.m_e * const.c * const.c).cgs


__all__ = ["Blob"]


class Blob:
    """Simple spherical emission region.

    **Note:** all these quantities are defined in the comoving frame so they are actually
    primed quantities, when referring the notation in [DermerMenon2009]_.

    Parameters
    ----------
    R_b : :class:`~astropy.units.Quantity`
        radius of the blob
    z : float
        redshift of the source
    delta_D : float
        Doppler factor of the relativistic outflow
    Gamma : float
        Lorentz factor of the relativistic outflow
    B : :class:`~astropy.units.Quantity`
        magnetic field in the blob (Gauss)
    spectrum_norm : :class:`~astropy.units.Quantity`
        normalization of the electron spectra, can be, following 
        the notation in [DermerMenon2009]_:

            - :math:`n_{e,\,tot}`: total electrons density, in :math:`\mathrm{cm}^{-3}`
            - :math:`u_e` : total electrons energy density, in :math:`\mathrm{erg}\,\mathrm{cm}^{-3}`
            - :math:`W_e` : total energy in electrons, in :math:`\mathrm{erg}`
    
    spectrum_dict : dictionary
        dictionary containing type and spectral shape information, e.g.:

        .. code-block:: python

            spectrum_dict = {
                "type": "PowerLaw", 
                "parameters": {
                    "p": 2.8, 
                    "gamma_min": 1e2, 
                    "gamma_max": 1e7
                }
            }
            
    gamma_size : int
        size of the array of electrons Lorentz factors
    """

    def __init__(
        self, R_b, z, delta_D, Gamma, B, spectrum_norm, spectrum_dict, gamma_size=200
    ):
        self.R_b = R_b.to("cm")
        self.z = z
        self.d_L = Distance(z=self.z).cgs
        self.V_b = 4 / 3 * np.pi * np.power(self.R_b, 3)
        self.delta_D = delta_D
        self.Gamma = Gamma
        self.Beta = np.sqrt(1 - 1 / np.power(self.Gamma, 2))
        # viewing angle
        self.mu_s = (1 - 1 / (self.Gamma * self.delta_D)) / self.Beta
        self.B = B.to("G")
        self.spectrum_norm = spectrum_norm
        self.spectrum_dict = spectrum_dict
        # size of the electron Lorentz factor grid
        self.gamma_size = gamma_size
        self.gamma_min = self.spectrum_dict["parameters"]["gamma_min"]
        self.gamma_max = self.spectrum_dict["parameters"]["gamma_max"]
        # grid of Lorentz factor for the integration in the blob comoving frame
        self.gamma = np.logspace(
            np.log10(self.gamma_min), np.log10(self.gamma_max), self.gamma_size
        )
        # grid of Lorentz factors for integration in the external frame
        self.gamma_to_integrate = np.logspace(1, 9, self.gamma_size)
        # assign the spectral density
        if spectrum_dict["type"] == "PowerLaw":
            _model = PowerLaw
        if spectrum_dict["type"] == "BrokenPowerLaw":
            _model = BrokenPowerLaw
        if spectrum_dict["type"] == "SmoothlyBrokenPowerLaw":
            _model = SmoothlyBrokenPowerLaw
        if spectrum_norm.unit == u.Unit("cm-3"):
            self.n_e = _model.from_normalised_density(
                spectrum_norm, **spectrum_dict["parameters"]
            )
        if spectrum_norm.unit == u.Unit("erg cm-3"):
            self.n_e = _model.from_normalised_u_e(u_e, **spectrum_dict["parameters"])
        if spectrum_norm.unit == u.Unit("erg"):
            u_e = (spectrum_norm / self.V_b).to("erg cm-3")
            self.n_e = _model.from_normalised_u_e(u_e, **spectrum_dict["parameters"])

    def __str__(self):
        """printable summary of the blob"""
        summary = (
            "* spherical emission region\n"
            + f" - R_b (radius of the blob): {self.R_b:.2e}\n"
            + f" - V_b (volume of the blob): {self.V_b:.2e}\n"
            + f" - z (source redshift): {self.z:.2f}\n"
            + f" - d_L (source luminosity distance):{self.d_L:.2e}\n"
            + f" - delta_D (blob Doppler factor): {self.delta_D:.2e}\n"
            + f" - Gamma (blob Lorentz factor): {self.delta_D:.2e}\n"
            + f" - Beta (blob relativistic velocity): {self.Beta:.2e}\n"
            + f" - mu_s (cosine of the jet viewing angle): {self.mu_s:.2e}\n"
            + f" - B (magnetic field tangled to the jet): {self.B:.2e}\n"
            + str(self.n_e)
        )
        return summary

    def set_gamma_size(self, gamma_size):
        """change size of the array of electrons Lorentz factors"""
        self.gamma_size = gamma_size
        self.gamma = np.logspace(
            np.log10(self.gamma_min), np.log10(self.gamma_max), self.gamma_size
        )
        self.gamma_to_integrate = np.logspace(1, 9, self.gamma_size)

    def N_e(self, gamma):
        """number of electrons as a function of the Lorentz factor, 
        :math:`N_e(\gamma') = V_b\,n_e(\gamma')`"""
        return self.V_b * self.n_e(gamma)

    @property
    def n_e_tot(self):
        """total electrons density

        .. math::
            n_{e,\,tot} = \int^{\gamma'_{max}}_{\gamma'_{min}} d\gamma' \, n_e(\gamma')
        """
        return np.trapz(self.n_e(self.gamma), self.gamma)

    @property
    def N_e_tot(self):
        """total electrons number

        .. math::
            N_{e,\,tot} = \int^{\gamma'_{max}}_{\gamma'_{min}} d\gamma' \, N_e(\gamma')
        """
        return np.trapz(self.N_e(self.gamma), self.gamma)

    @property
    def u_e(self):
        """total electrons energy density

        .. math::
            u_{e} = m_e\,c^2\,\int^{\gamma'_{max}}_{\gamma'_{min}} d\gamma' \,  \gamma' \, n_e(\gamma')
        """
        return MEC2 * np.trapz(self.gamma * self.n_e(self.gamma), self.gamma)

    @property
    def W_e(self):
        """total energy in non-thermal electrons

        .. math::
            W_{e} = m_e\,c^2\,\int^{\gamma'_{max}}_{\gamma'_{min}} d\gamma' \,  \gamma' \, N_e(\gamma')
        """
        return MEC2 * np.trapz(self.gamma * self.N_e(self.gamma), self.gamma)

    @property
    def P_jet_e(self):
        """jet power in electrons

        .. math::
            P_{jet,\,e} = 2 \pi R_b^2 \\beta \Gamma^2 c u_e
        """
        prefactor = (
            2
            * np.pi
            * np.power(self.R_b, 2)
            * self.Beta
            * np.power(self.Gamma, 2)
            * const.c
        )
        return (prefactor * self.u_e).to("erg s-1")

    @property
    def P_jet_B(self):
        """jet power in magnetic field

        .. math::
            P_{jet,\,B} = 2 \pi R_b^2 \\beta \Gamma^2 c \\frac{B^2}{8\pi}
        """
        prefactor = (
            2
            * np.pi
            * np.power(self.R_b, 2)
            * self.Beta
            * np.power(self.Gamma, 2)
            * const.c
        )
        U_B = np.power(self.B.value, 2) / (8 * np.pi) * u.Unit("erg cm-3")
        return (prefactor * U_B).to("erg s-1")
