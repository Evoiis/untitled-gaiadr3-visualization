from galpy.orbit import Orbit
from galpy.potential import MWPotential2014
import numpy as np

import time
import math
import os


KPC_TO_PC = 1000.0    # 1 kpc = 1000 pc
# km/s to pc/Gyr, 3600s * 24hours * 365days * 1 billion years / 3.086e+13 km per parsec
KMS_TO_PCGYR = 1021.9

def convert_inputs(x_kpc, y_kpc, z_kpc, vx_kms, vy_kms, vz_kms):
    """
    Convert galpy output units to network input units.

    Args:
        x_kpc, y_kpc, z_kpc:    galactocentric position in kpc  (from o.x(), o.y(), o.z())
        vx_kms, vy_kms, vz_kms: galactocentric velocity in km/s (from o.vx(), o.vy(), o.vz())

    Returns:
        x, y, z    in parsecs
        vx, vy, vz in pc/Gyr
    """
    x  = x_kpc  * KPC_TO_PC
    y  = y_kpc  * KPC_TO_PC
    z  = z_kpc  * KPC_TO_PC
    vx = vx_kms * KMS_TO_PCGYR
    vy = vy_kms * KMS_TO_PCGYR
    vz = vz_kms * KMS_TO_PCGYR
    return x, y, z, vx, vy, vz

def integrate_orbit(orbit: Orbit, time_range):

    orbit.integrate(time_range, MWPotential2014, method="dop853_c")

    return orbit.helioX(time_range), orbit.helioY(time_range), orbit.helioZ(time_range)

def generate_training_set(n_stars, n_timesteps=100):
    start = time.time()
    ra               = np.random.uniform(0, 360, n_stars)
    dec              = np.degrees(np.arcsin(np.random.uniform(-1, 1, n_stars)))
    parallax         = np.exp(np.random.uniform(np.log(0.275), np.log(800), n_stars))
    radial_velocity  = np.random.uniform(-500, 500, n_stars)
    times            = np.linspace(-1, 1, n_timesteps)
    
    dist_kpc = 1 / parallax
    max_pm   = 500 / (4.74 * dist_kpc)  # max pm for 500 km/s
    pmra     = np.random.uniform(-1, 1, n_stars) * max_pm
    pmdec    = np.random.uniform(-1, 1, n_stars) * max_pm

    vxvv = np.column_stack([ra, dec, dist_kpc, pmra, pmdec, radial_velocity])
    
    print(f"Init orbit, t_diff:{time.time() - start}")
    orbit = Orbit(vxvv, radec=True, ro=8., vo=220.)

    times = np.append(times, 0)
    times = np.sort(times)

    print(f"Extract inputs for MLP, t_diff:{time.time() - start}")
    x0, y0, z0, vx0, vy0, vz0 = convert_inputs(
        orbit.x(), orbit.y(), orbit.z(),
        orbit.vx(), orbit.vy(), orbit.vz()
    )

    print(f"Integrate Orbits, t_diff:{time.time() - start}")
    x_out, y_out, z_out = integrate_orbit(orbit, times)
    x_out = x_out.flatten() * KPC_TO_PC
    y_out = y_out.flatten() * KPC_TO_PC
    z_out = z_out.flatten() * KPC_TO_PC

    print(f"Build result, t_diff:{time.time() - start}")
    x0  = np.repeat(x0,  len(times))
    y0  = np.repeat(y0,  len(times))
    z0  = np.repeat(z0,  len(times))
    vx0 = np.repeat(vx0, len(times))
    vy0 = np.repeat(vy0, len(times))
    vz0 = np.repeat(vz0, len(times))
    t_col = np.tile(times, n_stars)

    return np.column_stack([x0, y0, z0, vx0, vy0, vz0, t_col, x_out, y_out, z_out])

def write_data(n_stars, batch_size, data_folder):
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)

    if len(os.listdir(data_folder)) > 0:
        print(f"Detected files in {data_folder}, not generating data to prevent overwrite.")
        return
    
    for i in range(math.ceil(n_stars / batch_size)):
        print(f"Generating part {i}")
        data = generate_training_set(batch_size)
    
        np.save(f"{data_folder}/orbit_train_part{i:04d}.npy", data)

def main():
    write_data(1000000, 10000, "training_data_3")
    write_data(100000, 10000, "validation_data_3")
    write_data(100000, 10000, "test_data_3")
    

if __name__ == "__main__":
    main()
