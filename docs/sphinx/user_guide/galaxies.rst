Galaxies and other extended sources
===================================

Stars are points; everything else is an *extended source*. wcc-sim draws
extended light in three ways, all through one mechanism and all added to a
frame with ``simulate_field(extended_sources=[...])``:

.. list-table::
   :header-rows: 1
   :widths: 26 34 40

   * - Component
     - What it describes
     - Flux is set by
   * - :class:`~wcc_sim.extended.SersicComponent`
     - One analytic elliptical Sersic profile (disk ``n=1``, bulge ``n=4``).
     - A Gaia-G magnitude and a stellar template, or a surface brightness
       at the effective radius.
   * - :class:`~wcc_sim.extended.GalSimComponent`
     - Any `GalSim <https://galsim-developers.github.io/GalSim/>`_ profile:
       bulge-plus-disk sums, sheared or rotated profiles, ``RealGalaxy``
       stamps.
     - A Gaia-G magnitude and a stellar template; the profile's own
       ``flux`` is ignored.
   * - :class:`~wcc_sim.extended.ImageComponent`
     - A calibrated image with a WCS, such as an HST drizzled mosaic of a
       real galaxy.
     - The image's own pixel values and AB zero point, so each WCC band
       carries the galaxy's measured brightness in that band.

The first needs nothing beyond the base install. The other two need the
``galaxies`` extra (``pip install wcc-sim[galaxies]``, which brings in
``galsim`` and ``reproject``); neither package is imported until such a
component is rendered, so the base package does not depend on them.

How extended light enters a frame
---------------------------------

Point sources are rendered on an 11x oversampled grid with sub-pixel
placement (:doc:`psf`). Doing that for a galaxy that covers millions of
pixels would need terabytes of memory, so extended sources take a different
route, in :mod:`wcc_sim.extended`:

1. **Render the unconvolved profile on the native detector grid** in e-/s
   per pixel. Each component type has its own renderer
   (``render_component_profile`` dispatches on the class); what they share
   is the convention that a pixel holds the surface brightness *sampled at
   the pixel centre* times the pixel area (for ``ImageComponent``, bilinear
   interpolation of the input), with no pixel-response convolution yet.
2. **Group components by** ``(template, ebv)`` and sum the profiles of each
   group. Grouping exists for ``chromatic=True`` (:doc:`psf`), where the PSF
   kernel depends on the spectrum; otherwise every group gets the same
   kernel.
3. **Convolve each group once by FFT** with the PSF binned to native
   resolution (``render.bin_oversampled``). That kernel is the oversampled
   PSF summed into detector pixels, so the pixel response enters here,
   exactly once. The kernel is normalised to unit sum and, when the wing
   model is active, scaled by ``1 / wing.flux_norm(half)`` so an extended
   source follows the same stamp-truncation convention as a star drawn
   with :func:`~wcc_sim.render.add_star` (see :doc:`scatter`). With the
   default 129-pixel in-focus stamp this holds back under 1% of the flux
   (about 2.4% for a 33-pixel stamp); it is a documented convention, not a
   loss.
4. **Add to the star image** before the noise stage, scaled by the exposure
   time. From here on a galaxy is just more electrons: it gets Poisson
   noise, contributes to saturation, and appears in ``CLEAN`` like
   everything else.

No wing halo is drawn for extended light. For a smooth profile the halo
would redistribute flux by less than the noise, so the renormalisation in
step 3 is all that is needed to keep the integrated flux consistent with the
point-source path.

Analytic Sersic profiles
------------------------

.. code-block:: python

   from wcc_sim import simulate_field, SersicComponent

   disk = SersicComponent(ra=210.80, dec=54.35, n=1.0, r_eff_arcsec=120.0,
                          ellip=0.2, pa_deg=40.0, total_mag=8.3, template="G2V")
   field = simulate_field(210.80, 54.35, extended_sources=[disk])

The profile is astropy's ``Sersic2D`` evaluated at pixel centres in row
chunks, with the cuspy centre re-evaluated on a 9x subgrid and averaged.
``total_mag`` is an integrated Gaia-G Vega magnitude and ``template`` a
Pickles spectral type, so the count rate follows the same chain as a star of
that magnitude and type (:doc:`fluxes`); ``ebv`` reddens it. The amplitude
is solved analytically from the Sersic total-flux integral
(:func:`~wcc_sim.extended.sersic_total_over_amplitude`). Alternatively
``sb_mag_arcsec2`` sets the surface brightness at the effective radius.
``pa_deg`` is the major-axis angle counter-clockwise from the detector +x
axis and ``ellip`` is ``1 - b/a`` with the major axis kept at ``r_eff``.

GalSim profiles
---------------

.. code-block:: python

   import galsim
   from wcc_sim import simulate_field, GalSimComponent, empty_catalog

   bulge = galsim.Sersic(n=4, half_light_radius=0.3, flux=0.3)
   disk = galsim.Exponential(half_light_radius=1.2, flux=0.7).shear(
       q=0.45, beta=30 * galsim.degrees)
   galaxy = GalSimComponent(ra=150.1, dec=2.2, profile=galsim.Add([bulge, disk]),
                            total_mag=14.0, template="G2V")
   field = simulate_field(150.1, 2.2, sensorfilter="zwo:r", shape=(512, 512),
                          catalog=empty_catalog(), extended_sources=[galaxy])

``profile`` is any ``galsim.GSObject`` with sizes in arcsec. GalSim is used
only to *describe* the light distribution: wcc-sim calls ``withFlux`` with
the rate implied by ``total_mag`` and ``template``, so the ``flux`` values
above only set the bulge-to-disk ratio. The profile is drawn with
``drawImage(method="no_pixel")``, which samples surface brightness at pixel
centres times pixel area, the same convention as the Sersic path, directly
onto the part of the frame inside its ``getGoodImageSize`` box, so a
profile larger than the frame costs no more than the frame. The central
cusp is re-drawn on the same 9x subgrid as the Sersic path and summed;
without that a compact ``n=4`` profile would lose about 20% of its flux to
pixel-centre sampling. Profiles smaller than a pixel belong in the
point-source catalog.

Two conventions differ from the Sersic path and are easy to trip over:

- GalSim's ``shear(q=...)`` is area-preserving (major axis grows by
  ``1/sqrt(q)`` while the minor shrinks), whereas ``SersicComponent.ellip``
  keeps the major axis at ``r_eff``. To reproduce a Sersic component
  exactly, compress one axis instead:
  ``galsim.Sersic(n, half_light_radius=r).transform(1, 0, 0, 1 - ellip).rotate(pa)``.
- Position angles agree: both are counter-clockwise from +x.

Real galaxies from calibrated images
------------------------------------

:class:`~wcc_sim.extended.ImageComponent` turns any calibrated image with a
WCS into an extended source. It exists for full-frame simulations of real
galaxies at a resolution close to WCC's, which in practice means Hubble
mosaics: the IMX455 pixel is 16.87 mas and the WCC PSF about 50 mas FWHM,
and no ground-based survey resolves that.

.. code-block:: python

   from wcc_sim import simulate_field, empty_catalog
   from wcc_sim.phangs import phangs_component

   comp = phangs_component("ngc1300", "r")            # F555W mosaic, cached locally
   field = simulate_field(49.9208, -19.4111, sensorfilter="zwo:r", exptime=600,
                          catalog=empty_catalog(), extended_sources=[comp],
                          output="ngc1300_r.fits")

The component holds the image array, its WCS and an AB zero point
``zp_ab`` such that a pixel of value ``v`` has AB magnitude
``zp_ab - 2.5 log10(v)``. For an HST product in e-/s that is
``-2.5 log10(PHOTFLAM) - 5 log10(PHOTPLAM) - 2.408``
(:func:`~wcc_sim.phangs.ab_zeropoint`). Rendering, in
``render_image_profile``:

1. **Cut the input to the frame footprint** with ``Cutout2D`` (a square of
   1.05x the frame diagonal around the pointing), so a 12000x12000 mosaic is
   not interpolated whole. A frame entirely outside the image renders
   nothing.
2. **Reproject onto the frame WCS** with ``reproject_interp`` (bilinear).
   Interpolation preserves surface brightness per input pixel, so the result
   is multiplied by the pixel-area ratio (WCC pixel over input pixel, about
   0.18 for 0.04" HST pixels). The test pins the reprojected flux to the
   input flux at 2%.
3. **Convert to WCC electrons.** Each pixel is treated as an AB-flat
   source of magnitude ``zp_ab - 2.5 log10(v)``:
   :func:`~wcc_sim.starflux.rate_for_ab` pushes a flat-F_nu spectrum through
   the ETC once per sensorfilter and scales it. An AB-flat spectrum has the
   same AB magnitude in every bandpass, which is what makes a single zero
   point sufficient; the image's own band is taken to be the WCC band it is
   simulated in (see the caveats below).
4. NaNs outside the mosaic footprint and negative sky-subtraction residuals
   render as zero, so the Poisson stage never sees a negative expectation.

The frame then continues through the common convolution and noise steps.
Because the HST image already contains its foreground stars, pass
``catalog=empty_catalog()`` (:func:`wcc_sim.empty_catalog`): the Gaia stars
at that position would otherwise be drawn a second time.

PHANGS-HST mosaics on demand
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:mod:`wcc_sim.phangs` wraps the PHANGS-HST high-level science products
(Lee et al. 2022, `archive.stsci.edu/hlsps/phangs-hst
<https://archive.stsci.edu/hlsps/phangs-hst/>`_): drizzled, flux-calibrated
mosaics at 0.04"/px of 38 nearby spirals in F275W, F336W, F435W or F438W,
F555W and F814W.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Function
     - Does
   * - ``list_phangs_galaxies()``
     - Galaxy keys that have a full mosaic on the archive (``"ngc1300"``,
       ``"ngc4321"``, ...).
   * - ``list_phangs_files(galaxy)``
     - ``{hst_filter: path}`` of the science mosaics for one galaxy; the
       multi-pointing ``<galaxy>mosaic/`` directory is preferred over a
       single pointing.
   * - ``fetch_phangs(galaxy, band)``
     - Downloads the mosaic for WCC band ``"g"``, ``"r"`` or ``"i"`` once
       into the cache and returns its path. 400 to 580 MB per band.
   * - ``load_hst_image(path)``
     - Any HST drizzled science image as an ``ImageComponent`` (data,
       WCS and ``zp_ab`` from the header).
   * - ``phangs_component(galaxy, band)``
     - The two steps above in one call.

WCC bands map to HST filters as g → F435W (ACS) or F438W (WFC3/UVIS),
r → F555W, i → F814W. The cache directory is ``~/.cache/wcc_sim/phangs``
unless ``cache_dir=`` or the ``WCC_SIM_PHANGS_CACHE`` environment variable
says otherwise; a partial download is kept as ``.part`` and never mistaken
for a finished file.

A full colour demo
^^^^^^^^^^^^^^^^^^

One full IMX455 frame per band, then a Lupton asinh composite:

.. code-block:: python

   import numpy as np
   from astropy.visualization import make_lupton_rgb
   from wcc_sim import simulate_field, empty_catalog
   from wcc_sim.phangs import phangs_component

   ra, dec = 49.9208, -19.4111                      # NGC 1300
   frames = {}
   for band in "gri":
       comp = phangs_component("ngc1300", band)
       f = simulate_field(ra, dec, sensorfilter=f"zwo:{band}", exptime=600,
                          catalog=empty_catalog(), extended_sources=[comp],
                          seed=11, output=f"ngc1300_{band}.fits")
       frames[band] = f.image_adu - np.median(f.image_adu)

   rgb = make_lupton_rgb(frames["i"], frames["r"], frames["g"],
                         Q=8, stretch=np.percentile(frames["r"], 99.5))

Each full frame takes 30 to 40 s on a laptop and the FITS product is about
550 MB. NGC 1300 and NGC 4321 both fill the 2.7' x 1.8' frame; in 600 s a
21 mag/arcsec² disk gives only about 10 (i) to 50 (g) e- per WCC pixel, so
bin the display or stack for a pretty picture.
The tutorial notebook :doc:`../notebooks/11_galaxies_phangs` runs this for
NGC 1300 and NGC 4321 with the frames and composites shown.

Caveats to read the frames with
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Resolution.** The HST PSF (about 0.08" to 0.10" FWHM for ACS) is in
  the input and is convolved again with the WCC PSF, so the frame's
  effective resolution is about 0.1": roughly twice as coarse as WCC
  alone. It is not deconvolved, because the target PSF is narrower than the
  source PSF and deconvolution would only amplify noise.
- **Bands are mapped, not transformed.** F435W to g, F555W to r and F814W
  to i differ in pivot wavelength by 50 to 100 nm; colours are HST colours
  relabelled. A per-pixel SED interpolation across the three HST bands
  would refine this.
- **Whatever is in the mosaic is in the frame**: foreground stars (some
  saturated in HST and now saturating again in WCC), sky-subtraction
  residuals, drizzle edges. Negative pixels are clipped to zero.
- **Flux bookkeeping.** ``image_clean`` includes sky and dark, about 10 e-
  per pixel in a 600 s ``zwo:r`` frame, so compare a galaxy's electrons
  against a frame simulated without it rather than against the raw sum.

Checking the implementation
---------------------------

The tests in ``tests/test_extended.py`` and ``tests/test_phangs.py`` pin
the conventions above: a GalSim ``Sersic(n=1)`` reproduces the analytic
``SersicComponent`` in total flux and in off-cusp pixels; a flat patch of
known AB surface brightness on a coarser WCS lands in the frame with the
right electrons across the pixel-area ratio; and the AB zero point of a
real F814W header comes out at the published 25.94. The GalSim and
reproject tests skip when the packages are not installed.
