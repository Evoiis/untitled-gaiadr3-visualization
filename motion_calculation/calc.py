import numpy as np
import pandas as pd
from galpy.orbit import Orbit
from galpy.potential import MWPotential2014
import time
import math
from sys import getsizeof

def integrate_orbits(
    ra, dec, parallax, pmra, pmdec, radial_velocity,
    t_start_gyr,
    t_end_gyr,
    n_timesteps=100,    
):
    
    now = time.time()
    dist_kpc = 1 / parallax
    # vxvv: [RA(deg), Dec(deg), dist(kpc), pmRA(mas/yr), pmDec(mas/yr), Vlos(km/s)]
    vxvv = np.column_stack([ra, dec, dist_kpc, pmra, pmdec, radial_velocity])

    o = Orbit(vxvv, radec=True, ro=8., vo=220.)
    # print(f"A: Curr: {time.time() - now}")

    # list of times at which to output (0 has to be in this!)
    time_range  = np.linspace(t_start_gyr, t_end_gyr, n_timesteps)
    # print(f"{time_range=}")

    o.integrate(time_range,  MWPotential2014, method='dop853_c')
    # print(f"B: Curr: {time.time() - now}")
    times_gyr = o.t

    x = o.helioX(times_gyr)
    y = o.helioY(times_gyr)
    z = o.helioZ(times_gyr)
    # print(f"C: Curr: {time.time() - now}")
    
    # print(f"D: Curr: {time.time() - now}")

    return times_gyr, x, y, z, o


def speed_test(index, ra, dec, parallax, pmra, pmdec, radial_velocity, output_string, t_start, t_end, n_timesteps):    
    # print("Integrating orbits...")
    start = time.time()
    times_gyr, x, y, z, o = integrate_orbits(
        ra, dec, parallax, pmra, pmdec, radial_velocity,
        t_start_gyr=t_start,
        t_end_gyr=t_end + t_start,
        n_timesteps=n_timesteps,
    )
    time_taken = time.time() - start
    # print(f"{n_timesteps=}")
    # print(f"Done, time taken: {time_taken}")
    # print(output_string)
    # print(f"Trajectory shape: {x.shape}")  # expect (n_stars, 199)
    # print(f"{3*getsizeof(x)=}")
    # print(f"{times_gyr=}")

    df = pd.DataFrame()
    df["x"] = x.tolist()
    df["y"] = y.tolist()
    df["z"] = z.tolist()


    df.to_csv(f"./data/timesteps{n_timesteps}_tstart{t_start}_tend{t_end}_{index}.csv")
    # print(f"{df["x"][0]=}, {df["y"][0]=}, {df["z"][0]=}")
    # print("\n\n")


    return time_taken


def benchmark(n_timesteps, t_start, t_end_mul=1):
    n_stars = 50000
    n_samples = 15

    ra               = np.random.uniform(0, 360, n_stars)
    dec              = np.random.uniform(-90, 90, n_stars)
    parallax         = np.random.uniform(0.3, 770, n_stars)
    pmra             = np.random.uniform(-10, 10, n_stars)
    pmdec            = np.random.uniform(-10, 10, n_stars)
    radial_velocity  = np.random.uniform(-50, 50, n_stars)

    result_strings = []
    t_end = 0.000000001 * 2
    
    while t_end < 0.1:
        times = []
        for n in range(n_samples):
            times.append(speed_test(n, ra, dec, parallax, pmra, pmdec, radial_velocity, f"{n_stars=}, {t_start=}, {t_end=}", t_start, t_end, n_timesteps))

        result_strings.append(f"Average time_taken({t_end=}): {sum(times) / n_samples}")
        t_end *= 10
    
    for s in result_strings:
        print(s)

def main():

    t_start = 0
    # print("16")
    # benchmark(16, t_start)
    # print("8")
    # benchmark(8, t_start)
    # print("4")
    # benchmark(4, t_start)
    # print("2")
    # benchmark(2, t_start)
    print("2")
    benchmark(2, t_start, 2)

if __name__ == "__main__":
    main()
