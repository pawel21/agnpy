"""profile and test synchrotron and synchrotron self Comton radiation"""
import sys

sys.path.append("../")
import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.coordinates import Distance
import matplotlib.pyplot as plt
from agnpy.emission_regions import Blob
from agnpy.synchrotron import Synchrotron
from agnpy.compton import SynchrotronSelfCompton, ExternalCompton
from agnpy.targets import SSDisk, SphericalShellBLR, RingDustTorus

from agnpy.synchrotron import SynchrotronJit
from agnpy.compton import SynchrotronSelfComptonJit
from agnpy.compton import ExternalComptonJit

# to profile
import cProfile, pstats
import timeit


# functions to profile and time
def profile(command, label):
    """function to profile a given command"""
    print(f"->{command} profiling section...")
    cProfile.run(command, f"Profile_{label}.prof")
    prof = pstats.Stats(f"Profile_{label}.prof")
    # prof.strip_dirs().sort_stats("time").print_stats(10)
    prof.sort_stats('cumulative').print_stats(5)

def timing(command, number):
    """function to time a given command, returns time in seconds"""
    return timeit.timeit(command, globals=globals(), number=number)

    # define the blob
spectrum_norm = 6e42 * u.erg
parameters = {"p1": 2, "p2": 3.5, "gamma_b": 1e4, "gamma_min": 20, "gamma_max": 5e7}
spectrum_dict = {"type": "BrokenPowerLaw", "parameters": parameters}
R_b = 1e16 * u.cm
B = 0.56 * u.G
z = 1
delta_D = 40
Gamma = 40
blob = Blob(R_b, z, delta_D, Gamma, B, spectrum_norm, spectrum_dict)
print("\nblob definition:")
print(blob)

# disk parameters
M_sun = const.M_sun.cgs
M_BH = 1.2 * 1e9 * M_sun
R_g = ((const.G * M_BH) / (const.c * const.c)).cgs
L_disk = 2 * 1e46 * u.Unit("erg s-1")
eta = 1 / 12
R_in = 6 * R_g
R_out = 200 * R_g
disk = SSDisk(M_BH, L_disk, eta, R_in, R_out)
print("\ndisk definition:")
print(disk)

# dust torus definition
T_dt = 1e3 * u.K
epsilon_dt = 2.7 * ((const.k_B * T_dt) / (const.m_e * const.c * const.c)).decompose()
csi_dt = 0.1
dt = RingDustTorus(L_disk, csi_dt, epsilon_dt*u.K)
print("\ntorus definition:")
print(dt)

ec_dt = ExternalCompton(blob, dt, r=1e17 * u.cm)
ec_jit_dt = ExternalComptonJit(blob, dt, r=1e17 * u.cm)
nu = np.logspace(15, 30) * u.Hz

ec_dt_sed_command = "ec_dt.sed_flux(nu)"
ec_jit_dt_sed_command = "ec_jit_dt.sed_flux(nu)"

n = 100
print("\nprofiling sed computation external compton on disk:")
profile(ec_dt_sed_command , "ec_disk_sed")
time_ec_disk = timing(ec_dt_sed_command , n)
time_ec_disk /= n
print(f"time: {time_ec_disk:.2e} s")

n = 100
print("\nprofiling sed computation external compton on disk jit:")
profile(ec_jit_dt_sed_command , "ec_disk_sed")
time_ec_jit_disk = timing(ec_jit_dt_sed_command , n)
time_ec_jit_disk /= n
print(f"time: {time_ec_jit_disk:.2e} s")
