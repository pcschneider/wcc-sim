"""Offline checks of wcc_sim.phangs (no archive access)."""
import numpy as np
import pytest


def test_ab_zeropoint_hst_f814w():
    from wcc_sim.phangs import ab_zeropoint

    # ACS/WFC F814W PHANGS header values; the published AB zero point is 25.94
    header = {"PHOTFLAM": 6.96152778750000e-20, "PHOTPLAM": 8047.29225}
    assert ab_zeropoint(header) == pytest.approx(25.94, abs=0.02)


def test_fetch_phangs_downloads_once(tmp_path, monkeypatch):
    import io

    from wcc_sim import phangs

    monkeypatch.setattr(phangs, "list_phangs_files",
                        lambda galaxy: {"f555w": "ngc1300mosaic/x_f555w_v1_exp-drc-sci.fits"})
    calls = []

    def fake_urlopen(url, timeout=None):
        calls.append(url)
        return io.BytesIO(b"FITS")

    monkeypatch.setattr(phangs.urllib.request, "urlopen", fake_urlopen)
    dest = phangs.fetch_phangs("NGC 1300", "r", cache_dir=tmp_path)
    assert dest == tmp_path / "x_f555w_v1_exp-drc-sci.fits"
    assert dest.read_bytes() == b"FITS" and not dest.with_suffix(".part").exists()
    assert phangs.fetch_phangs("ngc1300", "r", cache_dir=tmp_path) == dest
    assert len(calls) == 1
    with pytest.raises(ValueError, match="no .* mosaic"):
        phangs.fetch_phangs("ngc1300", "g", cache_dir=tmp_path)
    with pytest.raises(ValueError, match="galaxy name"):
        phangs._galaxy_key("../etc")


def test_filename_pattern_and_galaxy_key():
    from wcc_sim.phangs import _SCI_RE, _galaxy_key

    for name, instr in [
        ("hlsp_phangs-hst_hst_acs-wfc_ngc1300mosaic_f435w_v1_exp-drc-sci.fits", "acs-wfc"),
        ("hlsp_phangs-hst_hst_wfc3-uvis_ngc4321mosaic_f438w_v1_exp-drc-sci.fits", "wfc3-uvis"),
    ]:
        m = _SCI_RE.search(name)
        assert m and m.group(2) in ("f435w", "f438w")
    assert _SCI_RE.search("hlsp_phangs-hst_hst_acs-wfc_ngc1300mosaic-j93s01i8q_f555w_v1_flc.fits") is None
    assert _galaxy_key("NGC 1300") == _galaxy_key("ngc-1300") == "ngc1300"


def test_load_hst_image_roundtrip(tmp_path):
    from astropy.io import fits
    from astropy.wcs import WCS

    from wcc_sim.phangs import load_hst_image

    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [49.9, -19.4]
    w.wcs.crpix = [5.5, 5.5]
    w.wcs.cdelt = [-0.04 / 3600, 0.04 / 3600]
    h = w.to_header()
    h["PHOTFLAM"] = 6.96152778750000e-20
    h["PHOTPLAM"] = 8047.29225
    path = tmp_path / "sci.fits"
    fits.PrimaryHDU(np.ones((10, 10), dtype=np.float32), header=h).writeto(path)
    comp = load_hst_image(path)
    assert comp.data.shape == (10, 10)
    assert comp.zp_ab == pytest.approx(25.94, abs=0.02)
    assert comp.wcs.wcs.crval[0] == pytest.approx(49.9)
