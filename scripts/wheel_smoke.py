"""Offline end-to-end check of an installed wcc-sim wheel (CI wheel-smoke job).

Five synthetic stars on a 256 x 256 subarray at each focus level, then a FITS
round trip. No network: the catalog is given, so there is no Gaia query.
"""
import os
import tempfile

import numpy as np
from astropy.io import fits
from astropy.table import Table

import wcc_etc
import wcc_sim
from wcc_sim import simulate_field

print(f"wcc_sim {wcc_sim.__version__}, wcc_etc {wcc_etc.__version__}")
assert not os.path.exists(os.path.join(os.getcwd(), "src", "wcc_sim")), "run outside the checkout"

RA0, DEC0 = 150.0, 2.0
d = 1.5 / 3600.0
catalog = Table(
    {
        "source_id": np.arange(5, dtype=np.int64),
        "ra": [RA0, RA0 + d, RA0 - d, RA0, RA0 + d / 2],
        "dec": [DEC0, DEC0 + d / 2, DEC0 - d / 2, DEC0 + d, DEC0 - d],
        "phot_g_mean_mag": [12.0, 15.0, 15.0, 20.5, 17.0],
        "phot_bp_mean_mag": [12.4, 16.2, 14.9, 21.0, np.nan],
        "phot_rp_mean_mag": [11.6, 14.0, 15.0, 20.2, 16.4],
    }
)
for focus in (0, 1, 2):
    field = simulate_field(
        RA0, DEC0, sensorfilter="zwo:r", catalog=catalog, shape=(256, 256),
        stamp_npix=129, exptime=90.0, focus=focus, seed=1,
    )
    assert field.image_adu.shape == (256, 256)
    assert field.saturation_mask.any(), "the G=12 star must saturate"
    assert field.params["scatter_file"].startswith("wcc_etc scatter map")
    print(f"focus {focus}: max {field.image_adu.max():.0f} ADU, "
          f"{int(field.saturation_mask.sum())} saturated px")

path = os.path.join(tempfile.mkdtemp(), "smoke.fits")
field.write(path)
with fits.open(path) as hdul:
    names = [h.name for h in hdul]
    assert names == ["SCI", "SATMASK", "CAT", "CLEAN"], names
    assert len(hdul["CAT"].data) == hdul["SCI"].header["NSRC"]
print("PASS: wheel smoke")
