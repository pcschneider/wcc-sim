"""Fetch PHANGS-HST mosaics (Lee et al. 2022) as ImageComponents.

The HLSP at https://archive.stsci.edu/hlsps/phangs-hst/ holds drizzled,
flux-calibrated mosaics (0.04"/px, e-/s, PHOTFLAM/PHOTPLAM in the header) of
38 nearby spirals in F275W, F336W, F435W or F438W, F555W and F814W. Files are
400 to 580 MB per band and are downloaded on first use into `cache_dir`
(default ~/.cache/wcc_sim/phangs).

    comp = phangs_component("ngc1300", "r")      # F555W mosaic as an ImageComponent
    simulate_field(ra, dec, sensorfilter="zwo:r", extended_sources=[comp], ...)
"""

import os
import re
import shutil
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

PHANGS_URL = "https://archive.stsci.edu/hlsps/phangs-hst/"

#: WCC band letter -> acceptable HST filters (ACS BVI or WFC3 equivalents).
HST_FILTERS = {"g": ("f435w", "f438w"), "r": ("f555w",), "i": ("f814w",)}

_SCI_RE = re.compile(r'hlsp_phangs-hst_hst_[\w-]+_(\w+)_(f\d{3}w)_v\d+_exp-drc-sci\.fits')


def _cache_dir(cache_dir=None):
    d = Path(cache_dir or os.environ.get("WCC_SIM_PHANGS_CACHE")
             or Path.home() / ".cache" / "wcc_sim" / "phangs")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _galaxy_key(galaxy):
    key = re.sub(r"[\s_-]", "", str(galaxy)).lower()
    if not re.fullmatch(r"[a-z0-9]+", key):
        raise ValueError(f"not a galaxy name: {galaxy!r}")
    return key


def _listing(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read().decode("utf-8", "replace")


def list_phangs_galaxies():
    """Galaxy keys with a full mosaic on the archive (e.g. 'ngc1300')."""
    names = re.findall(r'href="([a-z0-9]+)mosaic/"', _listing(PHANGS_URL))
    return sorted(set(names))


def list_phangs_files(galaxy):
    """{hst_filter: filename} of drizzled science mosaics for `galaxy`.

    Galaxies with a single pointing live in `<galaxy>/`, multi-pointing ones
    in `<galaxy>mosaic/`; the latter is preferred when both exist.
    """
    key = _galaxy_key(galaxy)
    for sub in (f"{key}mosaic/", f"{key}/"):
        try:
            html = _listing(PHANGS_URL + sub)
        except urllib.error.HTTPError:
            continue
        files = {m.group(2): m.group(0) for m in _SCI_RE.finditer(html)}
        if files:
            return {k: sub + v for k, v in files.items()}
    raise ValueError(f"no PHANGS-HST science mosaic found for {galaxy!r}")


def fetch_phangs(galaxy, band, cache_dir=None):
    """Download (once) the PHANGS mosaic for WCC `band` ('g', 'r' or 'i');
    returns the local path. 400 to 580 MB per file."""
    files = list_phangs_files(galaxy)
    try:
        hst = next(f for f in HST_FILTERS[band] if f in files)
    except StopIteration:
        raise ValueError(
            f"{galaxy}: no {HST_FILTERS[band]} mosaic (have {sorted(files)})"
        ) from None
    dest = _cache_dir(cache_dir) / Path(files[hst]).name
    if not dest.exists():
        tmp = dest.with_suffix(".part")
        with urllib.request.urlopen(PHANGS_URL + files[hst], timeout=60) as r, \
                open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        tmp.rename(dest)
    return dest


def ab_zeropoint(header):
    """AB zero point of an HST product in e-/s from PHOTFLAM/PHOTPLAM."""
    return (-2.5 * np.log10(header["PHOTFLAM"])
            - 5.0 * np.log10(header["PHOTPLAM"]) - 2.408)


def load_hst_image(path):
    """An HST drizzled science image (e-/s) as an ImageComponent."""
    from .extended import ImageComponent

    with fits.open(path, memmap=False) as hdul:   # one copy, not two
        hdu = hdul[0] if hdul[0].data is not None else hdul[1]
        data = np.asarray(hdu.data, dtype=np.float32)
        header = hdu.header
    return ImageComponent(data=data, wcs=WCS(header), zp_ab=float(ab_zeropoint(header)))


def phangs_component(galaxy, band, cache_dir=None):
    """fetch_phangs + load_hst_image."""
    return load_hst_image(fetch_phangs(galaxy, band, cache_dir=cache_dir))
