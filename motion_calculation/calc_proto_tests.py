# def validate_orbits(o, x, y, z, times_gyr, ra, dec, dist_kpc):
#     print("=== Orbit Validation ===\n")

#     # 1. Check t=0 matches input positions
#     print("1. Checking t=0 matches input heliocentric positions...")
#     from astropy.coordinates import SkyCoord
#     import astropy.units as u
#     from astropy.coordinates import Galactocentric

#     galcen_frame = Galactocentric(
#         galcen_distance=8.0 * u.kpc,
#         z_sun=25 * u.pc
#     )

#     coords = SkyCoord(
#         ra=ra * u.deg,
#         dec=dec * u.deg,
#         distance=dist_kpc * u.kpc,
#         frame='icrs'
#     ).transform_to(galcen_frame)

#     # then subtract Sun to get heliocentric
#     gc_x = coords.x.to(u.kpc).value
#     gc_y = coords.y.to(u.kpc).value
#     gc_z = coords.z.to(u.kpc).value

#     positions_t0 = get_position_at_time(x, y, z, times_gyr, 0.0)

#     # positions_t0 is heliocentric, add Sun back to get galactocentric
#     sun_x = 8.0
#     expected_x = positions_t0[:, 0] + sun_x
#     expected_y = positions_t0[:, 1]  # Sun offset is 0 in y
#     expected_z = positions_t0[:, 2]  # Sun offset is 0 in z

#     match = np.allclose(-expected_x, gc_x, atol=0.1) and \
#             np.allclose(expected_y, gc_y, atol=0.1) and \
#             np.allclose(expected_z, gc_z, atol=0.1)
    
#     # Debug: print raw values side by side
#     print("galpy heliocentric (x,y,z) at t=0:")
#     print(positions_t0[:3])

#     print("\nastropy galactocentric (x,y,z):")
#     print(np.column_stack([gc_x[:3], gc_y[:3], gc_z[:3]]))

#     print("\ngalpy galactocentric (positions_t0 + sun):")
#     print(np.column_stack([positions_t0[:3, 0] + 8.0, positions_t0[:3, 1], positions_t0[:3, 2]]))

#     print("\ninput ra/dec/dist:")
#     print(np.column_stack([ra[:3], dec[:3], dist_kpc[:3]]))

#     if not match:
#         diff = np.sqrt(
#             (expected_x - gc_x)**2 +
#             (expected_y - gc_y)**2 +
#             (expected_z - gc_z)**2
#         )
#         print(f"   Max position error: {diff.max():.4f} kpc")

#     # 2. Energy conservation
#     print("\n2. Checking energy conservation...")
#     E = o.E(times_gyr)  # (n_stars, n_timesteps)
#     E_std = np.std(E, axis=1)
#     E_mean = np.mean(np.abs(E), axis=1)
#     E_rel = E_std / E_mean  # relative variation
#     print(f"   Max absolute energy std:  {E_std.max():.6f}")
#     print(f"   Max relative energy drift: {E_rel.max():.6f}")
#     print(f"   Energy conserved (rel < 1e-4): {np.all(E_rel < 1e-4)}")

#     # 3. Positions stay within the galaxy
#     print("\n3. Checking positions stay within galaxy bounds...")
#     r = np.sqrt(x**2 + y**2 + z**2)
#     print(f"   Min distance from galactic center: {r.min():.2f} kpc")
#     print(f"   Max distance from galactic center: {r.max():.2f} kpc")
#     print(f"   All within 30 kpc: {np.all(r < 30.0)}")
#     outliers = np.sum(r > 30.0)
#     if outliers > 0:
#         print(f"   WARNING: {outliers} position(s) exceeded 30 kpc")

#     # 4. Reversibility — t=0 from both directions should match
#     print("\n4. Checking reversibility at t=0...")
#     t0_idx = np.searchsorted(times_gyr, 0.0)

#     # Get positions just before and just after t=0
#     pos_before = np.stack([x[:, t0_idx-1], y[:, t0_idx-1], z[:, t0_idx-1]], axis=1)
#     pos_at     = np.stack([x[:, t0_idx],   y[:, t0_idx],   z[:, t0_idx]],   axis=1)
#     pos_after  = np.stack([x[:, t0_idx+1], y[:, t0_idx+1], z[:, t0_idx+1]], axis=1)

#     # t=0 should be between before and after, not a discontinuous jump
#     jump_before = np.linalg.norm(pos_at - pos_before, axis=1)
#     jump_after  = np.linalg.norm(pos_after - pos_at,  axis=1)

#     print(f"   Max position jump before t=0: {jump_before.max():.4f} kpc")
#     print(f"   Max position jump after  t=0: {jump_after.max():.4f} kpc")
#     print(f"   Continuous at t=0: {np.all(jump_before < 0.01) and np.all(jump_after < 0.01)}")
    
#     ratio = jump_before / (jump_after + 1e-10)
#     continuous = np.all(ratio > 0.5) and np.all(ratio < 2.0)
#     print(f"   Jump ratio before/after t=0: min={ratio.min():.3f} max={ratio.max():.3f}")
#     print(f"   Continuous at t=0 (ratio 0.5-2.0): {continuous}")

#     print("\n=== Validation Complete ===")
