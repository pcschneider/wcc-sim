"""Monte-Carlo cross-check of the wcc-etc SNR model against wcc-sim + wcc-phot.

The three packages make one chain: ``wcc_etc`` predicts a signal-to-noise
ratio from a closed-form aperture equation, ``wcc_sim`` renders a noisy frame
from the same instrument model, and ``wcc_phot`` measures a star on that
frame. Nothing in the test suites of the three packages checks that the
prediction and the measurement agree, and they are computed by different code
with different aperture conventions.

This module closes that loop empirically. It renders one isolated star, draws
many noise realisations of it through ``wcc_sim``'s own noise stage, measures
each realisation, and compares the *mean* and the *scatter* of the measured
flux against what ``wcc_etc`` said they would be. The scatter across
realisations is the only truth reference available for the noise model --
every other check compares one approximation against another.

Three quantities are returned per configuration:

``etc_aperture``
    The frame measured with the aperture convention ``wcc_etc`` itself uses
    (``psfsim._radial_cumulative``: whole pixels, selected by pixel-centre
    distance) with the background subtracted at its known value. This is the
    like-for-like test of the model, and it is the one that should pass.

``pipeline``
    The frame measured by ``wcc_phot.apphot.aperture_photometry_frame`` --
    exact-area photutils weights, background estimated from a sigma-clipped
    annulus, and the annulus penalty in the error budget. This is what a user
    actually gets, and it is *not* expected to match the ETC: the differences
    are real and are what this module quantifies.

``pipeline_err``
    ``wcc_phot``'s own analytic error bar, checked against its own empirical
    scatter.

Notes
-----
The scene render is deterministic, so it is done once per configuration and
only the noise stage is repeated. The noise model is per-pixel independent,
so the frame is cropped to a window around the star; the crop is exact, not
an approximation.

The fractional uncertainty on a standard deviation from ``n`` samples is
``1/sqrt(2(n-1))``, reported as ``sd_err``. Do not read a noise ratio to
better than that.
"""

from __future__ import annotations

import numpy as np
from astropy.table import Table
from photutils.aperture import CircularAperture

__all__ = ["crosscheck", "ANNULUS_VARIANCE_PENALTY"]

#: wcc_phot's default annulus is r_in = 1.5 r_ap, r_out = 2.5 r_ap, so its
#: area is 4x the aperture's and the background-estimate term inflates the
#: variance by 1 + 1/4. wcc_etc assumes the background is known exactly and
#: carries no equivalent term.
ANNULUS_VARIANCE_PENALTY = 1.25


def _one_star_catalog(ra, dec, gmag, spt):
    return Table(
        {
            "ra": [float(ra)],
            "dec": [float(dec)],
            "phot_g_mean_mag": [float(gmag)],
            "phot_bp_mean_mag": [np.nan],
            "phot_rp_mean_mag": [np.nan],
            "spt": [str(spt)],
        }
    )


def _whole_pixel_weights(shape, xc, yc, r_pix):
    """The wcc_etc aperture: pixels whose centre lies within r_pix."""
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    return (np.hypot(xx - xc, yy - yc) <= r_pix).astype(float)


def _summarise(samples, predicted):
    mean = float(np.mean(samples))
    sd = float(np.std(samples, ddof=1))
    out = {"signal_e": mean, "noise_e": sd, "snr": mean / sd if sd > 0 else np.inf}
    if predicted is not None:
        ps, pn = predicted["signal_e"], predicted["noise_e"]
        out["signal_ratio"] = mean / ps if ps else np.nan
        out["noise_ratio"] = sd / pn if pn else np.nan
        out["snr_ratio"] = out["snr"] / predicted["snr"] if predicted["snr"] else np.nan
    return out


def crosscheck(
    spt="G5V",
    sensorfilter="zwo:r",
    gmag=22.0,
    exptime=60.0,
    n_reads=1,
    focus=0,
    r_aper_mas=70.0,
    nseed=400,
    seed0=0,
    ra=150.0,
    dec=2.0,
    npix=None,
    wings=False,
    scatter=False,
):
    """Compare the wcc-etc prediction against measured wcc-sim realisations.

    Parameters
    ----------
    spt, gmag : str, float
        Pickles spectral type and Gaia G magnitude of the star. ``wcc_sim``
        normalises in Gaia G, so the ETC scene is built the same way and the
        two share a count rate by construction.
    sensorfilter, focus : str, int
        Focal-plane position (``'zwo:r'``, ``'qcmos:bb'``, ...) and focus
        state (0, 1 or 2 waves).
    exptime, n_reads : float, int
        Exposure time in seconds and number of coadded frames.
    r_aper_mas : float
        Requested aperture radius. The radius actually used is the one
        ``wcc_etc`` snaps to, and it is applied to every measurement here so
        the comparison is at a common radius.
    nseed : int
        Noise realisations. The uncertainty on a measured standard deviation
        is ``1/sqrt(2(nseed-1))`` -- 3.5% at 400, 2.2% at 1000.
    wings, scatter : bool
        Off by default: the ETC's aperture model has no wing or halo term, so
        leaving them on would compare against something the ETC never
        claimed to predict. Turn them on to measure that difference.

    Returns
    -------
    dict
        ``predicted`` (the ETC), ``etc_aperture``, ``pipeline``,
        ``pipeline_err``, and ``meta``.
    """
    from wcc_phot.apphot import aperture_photometry_frame
    from wcc_phot.geometry import geometry_from_r_ap
    from wcc_phot.io import Frame
    from wcc_sim import simulate_field
    from wcc_sim.detectors import get_geometry, make_base_simulation
    from wcc_sim.render import add_noise_and_digitize
    from wcc_sim.starflux import make_star_simulation, sky_and_dark_rates

    etc = make_star_simulation(spt, sensorfilter, mag=gmag)
    pred = etc.get_snr(exptime, r_aper_mas=r_aper_mas, n_reads=n_reads, warn=False)
    base = make_base_simulation(sensorfilter)
    plate_scale = get_geometry(sensorfilter, sim=etc).plate_scale_mas
    r_pix = pred["r_aper_mas"] / plate_scale
    sky, dark = sky_and_dark_rates(base)

    geom = geometry_from_r_ap(r_pix)  # annulus at the wcc_phot default, 1.5-2.5 r_ap
    if npix is None:
        # room for the full annulus plus a margin; odd, so the star sits on a
        # pixel centre exactly as the ETC's odd render grid assumes.
        npix = int(2 * np.ceil(geom.r_out) + 11) | 1
    shape = (npix, npix)
    centre = (npix - 1) / 2.0

    field = simulate_field(
        ra,
        dec,
        sensorfilter=sensorfilter,
        focus=focus,
        exptime=exptime,
        n_reads=n_reads,
        catalog=_one_star_catalog(ra, dec, gmag, spt),
        shape=shape,
        add_noise=False,
        wings=wings,
        scatter=scatter,
        write_clean=False,
    )
    per_pix_bkg = (sky + dark) * exptime
    source_e = np.asarray(field.image_clean, dtype=np.float32) - np.float32(per_pix_bkg)

    w_whole = _whole_pixel_weights(shape, centre, centre, r_pix)
    bkg_whole = float(w_whole.sum()) * per_pix_bkg

    meta = {key: field.params[key] for key in
            ("sky_e_s", "dark_e_s", "exptime", "n_reads", "read_noise")}

    etc_ap = np.empty(nseed)
    pipe = np.empty(nseed)
    pipe_err = np.empty(nseed)
    for k in range(nseed):
        noisy = add_noise_and_digitize(
            source_e, base, exptime, n_reads,
            np.random.default_rng(seed0 + k), add_noise=True,
        )
        image_e = np.asarray(noisy["image_e"], dtype=float)
        etc_ap[k] = float((image_e * w_whole).sum()) - bkg_whole
        frame = Frame(
            image_e=image_e,
            satmask=np.asarray(noisy["satmask"], dtype=bool),
            catalog=field.catalog,
            wcs=field.wcs,
            meta=meta,
        )
        flux, err, _bkg, _flags = aperture_photometry_frame(
            frame, centre, centre, geom
        )
        pipe[k] = float(flux[0])
        pipe_err[k] = float(err[0])

    predicted = {
        "signal_e": float(pred["signal_e"]),
        "noise_e": float(pred["noise_e"]),
        "snr": float(pred["snr"]),
        "n_pix": int(pred["n_pix"]),
        "enclosed_fraction": float(pred["enclosed_fraction"]),
        "r_aper_mas": float(pred["r_aper_mas"]),
        "r_aper_pix": r_pix,
        "saturated": bool(pred["saturated"]),
    }
    pipe_summary = _summarise(pipe, predicted)
    pipe_summary["mean_analytic_err"] = float(np.mean(pipe_err))
    # does wcc_phot's own error bar match wcc_phot's own scatter?
    pipe_summary["err_over_scatter"] = (
        pipe_summary["mean_analytic_err"] / pipe_summary["noise_e"]
    )
    return {
        "predicted": predicted,
        "etc_aperture": _summarise(etc_ap, predicted),
        "pipeline": pipe_summary,
        "meta": {
            "nseed": nseed,
            "sd_err": 1.0 / np.sqrt(2 * (nseed - 1)),
            "npix": npix,
            "aperture_area_px": float(np.pi * r_pix**2),
            "n_pix_whole": int(w_whole.sum()),
            "sum_w2_exact": float(
                (CircularAperture([(centre, centre)], r=r_pix)
                 .to_mask(method="exact")[0].to_image(shape) ** 2).sum()
            ),
            "spt": spt, "sensorfilter": sensorfilter, "gmag": gmag,
            "exptime": exptime, "n_reads": n_reads, "focus": focus,
        },
    }
