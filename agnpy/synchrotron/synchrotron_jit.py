from astropy.constants import e, h, c, m_e, G
from agnpy.utils.conversion import nu_to_epsilon_prime, B_to_cgs, lambda_c
from agnpy.utils.math import axes_reshaper, gamma_to_integrate
from numba import float64, jit, vectorize

e = e.gauss
gamma_to_integrate = np.logspace(1, 8, 1000)

def evaluate_sed_flux_jit(
    nu,
    z,
    d_L,
    delta_D,
    B,
    R_b,
    *args,
    ssa=False,
    integrator=np.trapz,
    gamma=gamma_to_integrate,
):
    # conversions
    epsilon = nu_to_epsilon_prime(nu, z, delta_D)
    B_cgs = B_to_cgs(B)
    # reshape for multidimensional integration
    _gamma, _epsilon = axes_reshaper(gamma, epsilon)
    V_b = 4 / 3 * np.pi * np.power(R_b, 3)
    n_e = LogParabola # new line
    N_e = V_b * n_e.evaluate(_gamma, *args)
    x = np.float64(calc_x(B_cgs, _epsilon, _gamma))
    # fold the electron distribution with the synchrotron power
    # N_e * single electron synch power
    integrand=fun_to_integrand(x, B_cgs, _epsilon, _gamma, N_e)
    # add unit, numba not handle units
    integrand *= (u.Fr**3*u.g**0.5)/(u.cm**0.5*u.J*u.s**2)
    emissivity = integrator(integrand, gamma, axis=0)
    prefactor = np.power(delta_D, 4) / (4 * np.pi * np.power(d_L, 2))
    sed = (prefactor * epsilon * emissivity).to("erg cm-2 s-1")

    return sed

def calc_x(B_cgs, epsilon, gamma):
    """ratio of the frequency to the critical synchrotron frequency from
    Eq. 7.34 in [DermerMenon2009]_, argument of R(x),
    note B has to be in cgs Gauss units"""
    x = (
        4
        * np.pi
        * epsilon
        * np.power(m_e, 2)
        * np.power(c, 3)
        / (3 * e * B_cgs * h * np.power(gamma, 2))
    )
    return x.to_value("")

@jit(nopython=True)
def fun_to_integrand(x, B_cgs, epsilon, gamma, N_e):
    P = single_electron_synch_power(x, B_cgs, epsilon, gamma)
    return N_e*P

@jit(nopython=True)
def single_electron_synch_power(x, B_cgs, epsilon, gamma):
    """angle-averaged synchrotron power for a single electron,
    to be folded with the electron distribution
    """
    prefactor = np.sqrt(3) * np.power(e, 3) * B_cgs / h
    return prefactor * R(x)

@vectorize([float64(float64)])
def R(x):
    """Eq. 7.45 in [Dermer2009]_, angle-averaged integrand of the radiated power, the
    approximation of this function, given in Eq. D7 of [Aharonian2010]_, is used.
    """
    term_1_num = 1.808 * np.power(x, 1 / 3)
    term_1_denom = np.sqrt(1 + 3.4 * np.power(x, 2 / 3))
    term_2_num = 1 + 2.21 * np.power(x, 2 / 3) + 0.347 * np.power(x, 4 / 3)
    term_2_denom = 1 + 1.353 * np.power(x, 2 / 3) + 0.217 * np.power(x, 4 / 3)
    return term_1_num / term_1_denom * term_2_num / term_2_denom * np.exp(-x)
