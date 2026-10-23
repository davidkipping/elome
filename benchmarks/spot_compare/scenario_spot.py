"""The spot-crossing scenario every code in the comparison evaluates.

A planet of radius R_P (stellar radii) crosses the disc along the chord
y = Y_CHORD, at sky positions given directly (no orbit, so no orbital
conventions enter). The star is seen equator-on, aligned and not
rotating during the crossing: a transit lasts hours, a rotation days.
Limb darkening I(mu) = 1 - w (1 - mu^2), w = W_LD: Elómë's hybrid2 with
its pole weight at 0, the quadratic law (2w, -w) for the others.

Spots are (longitude, latitude, angular radius [deg], contrast), the
contrast the spot's intensity over the photosphere's. At rotation phase 0
on an equator-on star, longitude 0 faces the observer and a spot at
(lon, lat) is at sky (cos(lat) sin(lon), sin(lat)) in all three codes
(checked by an occultor scan; spotter with obl=None, see README).

Compared: relative flux, each code's flux over its own unocculted flux,
which removes the codes' different normalisations.
"""
import math

import numpy as np

W_LD = 0.4
U_QUAD = (2.0 * W_LD, -W_LD)
R_P = 0.1
Y_CHORD = 0.2
X_SPAN = 1.15
OOT = (5.0, 0.0)                          # an unocculted sky position
LAT_CHORD = math.degrees(math.asin(Y_CHORD))

CASES = {
    # one spot on the chord, larger than the planet
    "chord": [(10.0, LAT_CHORD, 8.0, 0.5)],
    # two overlapping spots on the chord (their deficits add)
    "overlap": [(5.0, LAT_CHORD, 8.0, 0.5), (14.0, LAT_CHORD + 3.0, 5.0, 0.6)],
    # a spot near the limb, foreshortened, on the chord
    "limb": [(55.0, LAT_CHORD, 8.0, 0.5)],
}

N_PREC = 241                              # precision grid
N_SPEED = 1000                            # one light curve, for the timings

JAX_YDEG = (10, 15, 20, 25, 30)           # jaxoplanet: spherical-harmonic degree
SPOTTER_NSIDE = (32, 64, 128, 256)        # spotter: HEALPix resolution


def grid(n):
    """(x, y) sky positions of the planet along the chord."""
    x = np.linspace(-X_SPAN, X_SPAN, n)
    return x, np.full_like(x, Y_CHORD)


def spots_rad(case):
    """The case's spots with angles in radians: (lon, lat, alpha, c)."""
    return [(math.radians(lo), math.radians(la), math.radians(al), c)
            for lo, la, al, c in CASES[case]]
