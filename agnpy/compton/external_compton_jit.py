# module containing the External Compton radiative process
import numpy as np
from astropy.constants import c, sigma_T, G
from ..utils.math import (
    trapz_loglog,
    log,
    axes_reshaper,
    gamma_to_integrate,
    mu_to_integrate,
    phi_to_integrate,
)
from ..utils.conversion import nu_to_epsilon_prime, r_to_R_g_units
from ..utils.geometry import x_re_shell, mu_star_shell, x_re_ring
from ..targets import (
    CMB,
    PointSourceBehindJet,
    SSDisk,
    SphericalShellBLR,
    RingDustTorus,
)

from .kernels import isotropic_kernel, compton_kernel
from .kernels_jit import isotropic_kernel_jit, compton_kernel_jit

__all__ = ["ExternalComptonJit"]


class ExternalComptonJit:
    """class for External Compton radiation computation

    Parameters
    ----------
    blob : :class:`~agnpy.emission_regions.Blob`
        emission region and electron distribution hitting the photon target
    target : :class:`~agnpy.targets`
        class describing the target photon field
    r : :class:`~astropy.units.Quantity`
        distance of the blob from the Black Hole (i.e. from the target photons)
    """

    def __init__(self, blob, target, r=None, integrator=np.trapz):
        self.blob = blob
        # we integrate on a larger grid to account for the transformation
        # of the electron density in the reference frame of the BH
        self.gamma = self.blob.gamma_to_integrate
        self.target = target
        self.r = r
        self.integrator = integrator
        self.set_mu()
        self.set_phi()

    def set_mu(self, mu_size=100):
        self.mu_size = mu_size
        if isinstance(self.target, SSDisk):
            # in case of hte disk the mu interval does not go from -1 to 1
            r_tilde = (self.r / self.target.R_g).to_value("")
            self.mu = self.target.evaluate_mu_from_r_tilde(
                self.target.R_in_tilde, self.target.R_out_tilde, r_tilde
            )
        else:
            self.mu = np.linspace(-1, 1, self.mu_size)

    def set_phi(self, phi_size=50):
        self.phi_size = phi_size
        self.phi = np.linspace(0, 2 * np.pi, self.phi_size)

    @staticmethod
    def evaluate_sed_flux_dt(
        nu,
        z,
        d_L,
        delta_D,
        mu_s,
        R_b,
        L_disk,
        xi_dt,
        epsilon_dt,
        R_dt,
        r,
        n_e,
        *args,
        integrator=np.trapz,
        gamma=gamma_to_integrate,
        phi=phi_to_integrate
    ):
        r"""Evaluates the flux SED,
        :math:`\nu F_{\nu} \, [\mathrm{erg}\,\mathrm{cm}^{-2}\,\mathrm{s}^{-1}]`,
        for External Compton on a monochromatic isotropic target photon field
        for a general set of model parameters

        Parameters
        ----------
        nu : :class:`~astropy.units.Quantity`
            array of frequencies, in Hz, to compute the sed
            **note** these are observed frequencies (observer frame)
        z : float
            redshift of the source
        d_L : :class:`~astropy.units.Quantity`
            luminosity distance of the source
        delta_D: float
            Doppler factor of the relativistic outflow
        mu_s : float
            cosine of the angle between the blob motion and the jet axis
        L_disk : :class:`~astropy.units.Quantity`
            Luminosity of the disk whose radiation is being reprocessed by the BLR
        xi_dt : float
            fraction of the disk radiation reprocessed by the disk
        epsilon_dt : string
            peak (dimensionless) energy of the black body radiated by the torus
        R_dt : :class:`~astropy.units.Quantity`
            radius of the ting-like torus
        r : :class:`~astropy.units.Quantity`
            distance between the Broad Line Region and the blob
        n_e : :class:`~agnpy.spectra.ElectronDistribution`
            electron energy distribution
        *args
            parameters of the electron energy distribution (k_e, p, ...)
        ssa : bool
            whether to consider or not the self-absorption, default false
        integrator : func
            which function to use for integration, default `numpy.trapz`
        gamma : :class:`~numpy.ndarray`
            array of Lorentz factor over which to integrate the electron
            distribution
        mu, phi : :class:`~numpy.ndarray`
            arrays of cosine of zenith and azimuth angles to integrate over

        **Note** arguments after *args are keyword-only arguments

        Returns
        -------
        :class:`~astropy.units.Quantity`
            array of the SED values corresponding to each frequency
        """
        # conversions
        epsilon_s = nu_to_epsilon_prime(nu, z)
        # multidimensional integration
        _gamma, _phi, _epsilon_s = axes_reshaper(gamma, phi, epsilon_s)
        V_b = 4 / 3 * np.pi * np.power(R_b, 3)
        N_e = V_b * n_e.evaluate(_gamma / delta_D, *args)
        x_re = x_re_ring(R_dt, r)
        mu = (r / x_re).to_value("")
        kernel = compton_kernel_jit(_gamma, _epsilon_s.value, epsilon_dt, mu_s, mu, _phi)
        integrand = N_e / np.power(_gamma, 2) * kernel
        integral_gamma = integrator(integrand, gamma, axis=0)
        integral_phi = np.trapz(integral_gamma, phi, axis=0)
        prefactor_num = (
            3 * sigma_T * xi_dt * L_disk * np.power(epsilon_s, 2) * np.power(delta_D, 3)
        )
        prefactor_denom = (
            np.power(2, 8)
            * np.power(np.pi, 3)
            * np.power(d_L, 2)
            * np.power(x_re, 2)
            * np.power(epsilon_dt, 2)
        )
        return (prefactor_num / prefactor_denom * integral_phi).to("erg cm-2 s-1")

    def sed_flux_dt(self, nu):
        """evaluates the flux SED for External Compton on a ring dust torus"""
        return self.evaluate_sed_flux_dt(
            nu,
            self.blob.z,
            self.blob.d_L,
            self.blob.delta_D,
            self.blob.mu_s,
            self.blob.R_b,
            self.target.L_disk,
            self.target.xi_dt,
            self.target.epsilon_dt,
            self.target.R_dt,
            self.r,
            self.blob.n_e,
            *self.blob.n_e.parameters,
            integrator=self.integrator,
            gamma=self.gamma,
            phi=self.phi
        )

    def sed_flux(self, nu):
        """SEDs for external Compton"""
        if isinstance(self.target, RingDustTorus):
            return self.sed_flux_dt(nu)


    # function to test kernel
    @staticmethod
    def calc_kernel_params(nu,
            z,
            d_L,
            delta_D,
            mu_s,
            R_b,
            L_disk,
            xi_dt,
            epsilon_dt,
            R_dt,
            r,
            n_e,
            *args,
            integrator=np.trapz,
            gamma=gamma_to_integrate,
            phi=phi_to_integrate
        ):
        # conversions
        epsilon_s = nu_to_epsilon_prime(nu, z)
        # multidimensional integration
        _gamma, _phi, _epsilon_s = axes_reshaper(gamma, phi, epsilon_s)
        V_b = 4 / 3 * np.pi * np.power(R_b, 3)
        N_e = V_b * n_e.evaluate(_gamma / delta_D, *args)
        x_re = x_re_ring(R_dt, r)
        mu = (r / x_re).to_value("")
        # compton_kernel(_gamma, _epsilon_s, epsilon_dt, mu_s, mu, _phi)
        return _gamma, _epsilon_s, epsilon_dt, mu_s, mu, _phi

    def get_kernel_params(self, nu):
        return self.calc_kernel_params(
            nu,
            self.blob.z,
            self.blob.d_L,
            self.blob.delta_D,
            self.blob.mu_s,
            self.blob.R_b,
            self.target.L_disk,
            self.target.xi_dt,
            self.target.epsilon_dt,
            self.target.R_dt,
            self.r,
            self.blob.n_e,
            *self.blob.n_e.parameters,
            integrator=self.integrator,
            gamma=self.gamma,
            phi=self.phi
        )
