Scattered light
===============

Beyond the diffraction wings, every star deposits a faint, very extended
halo of light scattered by the optics. ``wcc-sim`` models it from a FRED
stray-light ray trace of the telescope (the same run that ``wcc-etc`` ships
and combines with a diffraction core in ``wcc_etc.scatter_psf``), reduced to
an azimuthally averaged radial profile that is packaged with the simulator
in ``wcc_sim/data/psfs/``.

How the halo enters the image
-----------------------------

The total PSF of a star is

.. math::

   \mathrm{total} = (1 - f_\mathrm{scat})\,\mathrm{core} + f_\mathrm{scat}\,\mathrm{halo},

where :math:`f_\mathrm{scat}` is the instrument-wide scattered fraction, a
property of the optics rather than of any stamp. The core (Airy or Huygens
defocus image plus the analytic wing) is rendered on an oversampled stamp;
the halo is added as a second, additive term by the wing machinery in
:mod:`wcc_sim.render`. It reaches about 37,000 detector pixels from the
star, past the chip diagonal, so a bright star in one corner still deposits
a correct halo in the opposite corner.

The profile keeps the ray trace's absolute surface brightness and is
truncated at its modelled reach rather than renormalised to unit energy:
the small part of the scattered power that lands outside the modelled disc
is off any detector and is deliberately not deposited.

Why a radial profile
--------------------

The ray-trace grid has cells of about 100 detector pixels, and the map is
azimuthally symmetric to a few per cent at every radius. A one-dimensional
profile therefore reproduces the halo to within its own symmetry scatter at
a tiny fraction of the cost of a full-plane two-dimensional kernel, and it
keeps the sub-pixel star placement of the oversampled core.

Normalisation
-------------

The ray trace quotes the scattered fraction two ways: per watt launched at
the aperture (``P_FULL``) and per watt reaching the image plane
(``F_FOCAL``). They differ by the mirror throughput. A unit-sum PSF counts
detected photons, so the image-plane fraction is the one that belongs in the
equation above; :func:`wcc_sim.scatter.scatter_halo` documents the default
it applies, and ``scatter_fraction=`` on :func:`wcc_sim.simulate_field`
overrides it. The FITS header records the fraction used (``SCATFRAC``) and
the map it came from (``SCATFILE``).

Limits
------

The ray trace is monochromatic at 450 nm and the halo is applied with the
same shape and fraction in every band. Surface scatter falls with
wavelength, so the halo is over-predicted in the red bands; the profile
metadata records ``WAVELEN`` so this can be revisited. ``wings=False``
switches the halo off together with the diffraction wings, and
``params["scatter"]`` reports what actually ran.
