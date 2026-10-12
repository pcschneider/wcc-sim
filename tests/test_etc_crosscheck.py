"""End-to-end: does the wcc-etc SNR prediction match a measured wcc-sim frame?

wcc_etc predicts SNR from a closed-form aperture equation; wcc_sim renders a
noisy frame from the same instrument model; wcc_phot measures a star on it.
Each package tests itself, but nothing tested that the prediction and the
measurement agree. These tests do, using the scatter across many noise
realisations as the truth reference for the noise model.

Two distinct claims are asserted, and only the first is a claim about
correctness:

1. Measured with the aperture convention wcc_etc itself uses, and with the
   background subtracted at its known value, the ETC's signal and noise
   reproduce the Monte Carlo. This is the model being right.

2. Measured by wcc_phot as a user would, the answer differs, in a direction
   and by an amount that is pinned here. The ETC is optimistic: it uses a
   different aperture convention, it does not model background estimated
   from an annulus, and it carries no annulus penalty in the variance. These
   are documented ETC assumptions (see its Table 7), not bugs -- but they are
   unquantified in the ETC, and if any of them changes these numbers move.

The noise assertions are loose on purpose: the fractional uncertainty on a
standard deviation from n samples is 1/sqrt(2(n-1)), which is 3.5% at the
n=400 used here. Tightening them means raising nseed, which costs runtime.
"""

import numpy as np
import pytest

from wcc_phot.validation import ANNULUS_VARIANCE_PENALTY, crosscheck

pytestmark = pytest.mark.slow

NSEED = 400
SD_TOL = 4.0 / np.sqrt(2 * (NSEED - 1))  # ~14%, a 4-sigma band on the MC scatter


@pytest.fixture(scope="module")
def faint():
    """A background-dominated case: this is where the noise model has to work."""
    return crosscheck(gmag=25.0, exptime=300.0, nseed=NSEED)


@pytest.fixture(scope="module")
def bright():
    """A source-dominated case."""
    return crosscheck(gmag=20.0, exptime=60.0, nseed=NSEED)


@pytest.mark.parametrize("case", ["bright", "faint"])
def test_etc_signal_matches_measured_frame(case, request):
    """Mean measured flux reproduces eta * R_src * t to well under a percent."""
    r = request.getfixturevalue(case)
    assert r["etc_aperture"]["signal_ratio"] == pytest.approx(1.0, abs=0.01)


@pytest.mark.parametrize("case", ["bright", "faint"])
def test_etc_noise_matches_measured_scatter(case, request):
    """Predicted sigma reproduces the scatter across noise realisations.

    This is the only empirical check of the noise model anywhere in the three
    packages: everything else compares one closed form against another.
    """
    r = request.getfixturevalue(case)
    assert r["etc_aperture"]["noise_ratio"] == pytest.approx(1.0, abs=SD_TOL)


def test_etc_aperture_convention_is_whole_pixel(bright):
    """The ETC counts whole pixels; a circle of the same radius has less area.

    If this fails, psfsim._radial_cumulative has changed convention and the
    exact-area comparison in test_pipeline_is_not_the_etc needs revisiting.
    """
    meta = bright["meta"]
    assert meta["n_pix_whole"] == bright["predicted"]["n_pix"]
    assert meta["n_pix_whole"] > meta["aperture_area_px"] > meta["sum_w2_exact"]


def test_pipeline_error_bars_are_honest(faint):
    """wcc_phot's analytic error bar matches wcc_phot's own empirical scatter."""
    assert faint["pipeline"]["err_over_scatter"] == pytest.approx(1.0, abs=SD_TOL)


def test_pipeline_is_not_the_etc(faint):
    """The ETC is optimistic against the delivered pipeline, and by how much.

    Pinned so that a change in either package's aperture or background
    handling shows up as a test failure rather than as a silent drift between
    what a proposer is promised and what they measure.
    """
    pipe = faint["pipeline"]
    # exact-area aperture + flux in the background annulus both lose signal
    assert pipe["signal_ratio"] < 1.0
    assert pipe["signal_ratio"] == pytest.approx(0.98, abs=0.03)
    # and the annulus penalty inflates the noise the ETC never carried
    assert pipe["noise_ratio"] > 1.0 - SD_TOL
    assert pipe["snr_ratio"] < 1.0


def test_annulus_penalty_matches_default_geometry():
    """wcc_phot's default annulus (1.5-2.5 r_ap) is 4x the aperture area."""
    from wcc_phot.geometry import geometry_from_r_ap

    g = geometry_from_r_ap(4.0)
    ap_area = np.pi * g.r_ap**2
    ann_area = np.pi * (g.r_out**2 - g.r_in**2)
    assert 1.0 + ap_area / ann_area == pytest.approx(ANNULUS_VARIANCE_PENALTY, rel=1e-12)
