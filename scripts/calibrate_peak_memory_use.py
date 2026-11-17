#!/usr/bin/env python
"""Calibrate the peak memory of wcc-sim's image resampling.

Measures, per configuration in fresh subprocess, the peak memory added by
SkyImageComponent.rate_map (wcc_sim/imagesource.py; image -> e-/s, then
resample_to_subgrid in wcc_sim/resample.py with reproject_exact) and by the
following render_subgrid (wcc_sim/render.py, PSF convolution of the sub-grid).
It then fits

    peak [bytes] = 8 * (a * n_sub + b * n_in) + c

    n_sub : sub-pixels = footprint area [detector px] * k**2
    n_in  : input-image pixels
    k     : sub-grid factor (sub-pixels per detector pixel per axis)
    a, b  : float64 arrays alive at the peak per sub-pixel / per input pixel
    c     : constant overhead [bytes]

Memory is measured two ways:
  rss  : Linux peak resident set size (VmHWM after resetting it via
         /proc/self/clear_refs) minus the resident size just before the call;
         this is what can exhaust RAM.
  tm   : tracemalloc peak (allocated Python/numpy memory, incl. untouched pages).

Configurations run smallest first; a configuration whose predicted peak (from
the running fit, x 1.3) exceeds --budget-gb is skipped.

Linux only: reads /proc/self/status and /proc/self/clear_refs.

Usage
-----
    python scripts/calibrate_peak_memory_use.py                       # full grid -> memory_calibration/
    python scripts/calibrate_peak_memory_use.py --budget-gb 4 --quick # small subset
    python scripts/calibrate_peak_memory_use.py --fit-only            # refit an existing results.jsonl
    python scripts/calibrate_peak_memory_use.py --resume --defocus    # add the defocused-PSF grid (focus 1, 2)
    python scripts/calibrate_peak_memory_use.py --resume --stamps     # add stamps 33, 65, 513, 1025 px (focus 0)
    python scripts/calibrate_peak_memory_use.py --out DIR             # output directory (default memory_calibration)
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

RA0, DEC0 = 291.0, 44.5
SF = "zwo:r"


# ----------------------------------------------------------------------------- worker

def _status_kb(key):
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith(key + ":"):
                return int(line.split()[1])
    raise KeyError(key)


def _reset_hwm():
    with open("/proc/self/clear_refs", "w") as fh:
        fh.write("5")


def _image_wcs_header(ny, nx, scale_mas):
    from astropy.wcs import WCS
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [RA0, DEC0]
    w.wcs.crpix = [(nx + 1) / 2, (ny + 1) / 2]
    s = scale_mas / 3.6e6
    w.wcs.cd = [[-s, 0.0], [0.0, s]]
    h = w.to_header()
    h["BUNIT"] = "Jy/pixel"
    return w, h


def worker(cfg):
    """Run one configuration; print one JSON line with the measurements."""
    import gc
    import tracemalloc

    from wcc_sim.detectors import detector_frame
    from wcc_sim.fitsimage import ImageInput
    from wcc_sim.imagesource import SkyImageComponent
    from wcc_sim.observation import WCCObservation
    from wcc_sim.render import render_subgrid

    ny, nx = cfg["n_in"]
    _, h = _image_wcs_header(ny, nx, cfg["scale_mas"])
    data = np.full((ny, nx), 1e-4)
    comp = SkyImageComponent(ImageInput.from_fits((data, h)))
    del data
    shape = None if cfg["shape"] is None else tuple(cfg["shape"])
    frame = detector_frame(SF, (RA0, DEC0), pa=cfg.get("pa", 0.0), shape=shape)
    obs = WCCObservation(comp, frame, subgrid=cfg["k"], max_memory_gb=None,
                         focus=cfg.get("focus"), stamp_npix=cfg.get("stamp"))
    estimate = obs.memory_estimate()[0]
    box = comp.footprint(frame.wcs, frame.shape)

    # PSF first, so its (cached) memory is not part of the render measurement
    from wcc_sim.resample import choose_subgrid
    k_used = choose_subgrid(comp.image.pixel_scale_mas, frame.plate_scale_mas,
                            override=cfg["k"])
    psf = obs.psf(k_used)

    # --- stage 1: rate_map (rate image + resample_to_subgrid)
    gc.collect()
    _reset_hwm()
    rss0 = _status_kb("VmRSS")
    tracemalloc.start()
    t0 = time.perf_counter()
    sub = comp.rate_map(frame.wcs, frame.shape, frame.plate_scale_mas, frame.sensorfilter,
                        k=obs.subgrid)
    t1 = time.perf_counter()
    tm_peak1 = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    hwm1 = _status_kb("VmHWM")
    rss1 = _status_kb("VmRSS")

    # --- stage 2: render_subgrid (sub kept, as in WCCObservation._rendered)
    gc.collect()
    _reset_hwm()
    rss2 = _status_kb("VmRSS")
    tracemalloc.start()
    t2 = time.perf_counter()
    cut = render_subgrid(sub, *psf)
    t3 = time.perf_counter()
    tm_peak2 = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    hwm2 = _status_kb("VmHWM")

    y0, y1, x0, x1 = box
    out = dict(cfg)
    out.update(
        k_used=sub.k, focus_used=obs.focus, stamp=obs.stamp_npix,
        box=list(box), n_det=(y1 - y0) * (x1 - x0),
        n_sub=int(sub.data.size), n_in=ny * nx,
        rss_resample=(hwm1 - rss0) * 1024, tm_resample=tm_peak1,
        retained_resample=(rss1 - rss0) * 1024, sub_bytes=int(sub.data.nbytes),
        rss_render=(hwm2 - rss2) * 1024, tm_render=tm_peak2,
        cut_bytes=int(cut.data.nbytes), rss_base=rss0 * 1024,
        t_resample=t1 - t0, t_render=t3 - t2, estimate=int(estimate),
    )
    print("RESULT " + json.dumps(out), flush=True)


# ----------------------------------------------------------------------------- driver

def configs(quick=False, defocus=False, stamps=False):
    """Grid of configurations. Detector plate scale ~16.87 mas (ZWO).

    defocus=True: the defocused-PSF grid only (focus 1 and 2, 257 px stamp), to
    compare the render_subgrid peak with the in-focus runs (focus 0, 129 px stamp).
    """
    c = []

    def add(group, n_in, scale, shape, k, **kw):
        c.append(dict(group=group, n_in=list(n_in), scale_mas=scale,
                      shape=None if shape is None else list(shape), k=k, **kw))

    if stamps:
        # PSF stamp size [px] at focus 0 (Airy PSF, any odd stamp): footprint at k = 1
        # and one k = 3 case per stamp; padded area x 12 float64 x 1.5 stays below ~7 GB
        for st in (33, 65, 513, 1025):
            for s in (1001, 2001, 3001, 4001, 5001):
                if st == 1025 and s > 4001:
                    continue
                add("S_D_footprint", (int(s * 16.87 / 50) + 20,) * 2, 50.0, (s, s), 1,
                    focus=0, stamp=st)
            add("S_B_k_clipped", (1500, 1500), 20.0, (1201, 1201), 3, focus=0, stamp=st)
        return c
    if defocus:
        for s in (1001, 2001, 3001, 4001, 5001, 6001):             # as D, focus 2
            add("F_D_footprint", (int(s * 16.87 / 50) + 20,) * 2, 50.0, (s, s), 1, focus=2)
        for s in (2001, 4001):                                       # focus 1 vs 2
            add("F_D_footprint", (int(s * 16.87 / 50) + 20,) * 2, 50.0, (s, s), 1, focus=1)
        for k in (3, 5, 7):                                          # as A, focus 2
            add("F_A_k_small_in", (500, 500), 20.0, (701, 701), k, focus=2)
        for k in (1, 3):                                             # as B, focus 2
            add("F_B_k_clipped", (1500, 1500), 20.0, (1201, 1201), k, focus=2)
        add("F_review_full", (2200, 3300), 50.0, None, None, focus=2)  # gummiks' case
        return c
    # A: n_sub varies via k; small input fully on a subarray (n_in = 0.25 M fixed)
    for k in (1, 3, 5, 7, 9, 11):
        add("A_k_small_in", (500, 500), 20.0, (701, 701), k)
    # B: n_sub varies via k; input clipped by the subarray (n_in = 2.25 M fixed)
    for k in (1, 3, 5, 7):
        add("B_k_clipped", (1500, 1500), 20.0, (1201, 1201), k)
    # C: n_in varies; tiny subarray, k = 1 (n_sub = 40 k fixed)
    for n in (1000, 2000, 3000, 4000, 6000, 8000, 10000):
        add("C_n_in", (n, n), 2.0, (201, 201), 1)
    # D: n_sub varies via footprint at k = 1; coarse input covering the subarray
    for s in (1001, 2001, 3001, 4001, 5001, 6001):
        add("D_footprint", (int(s * 16.87 / 50) + 20,) * 2, 50.0, (s, s), 1)
    # E: realistic defaults (k from the pixel-scale ratio)
    add("E_review_full", (2200, 3300), 50.0, None, None)          # gummiks' case
    add("E_review_full", (2200, 3300), 50.0, None, None, rep=2)   # reproducibility
    add("E_hst_sub", (4000, 4000), 5.58, (1001, 1001), None)      # k = 5
    add("E_hst_sub", (3000, 3000), 5.58, (2001, 2001), None)      # k = 5, image on frame
    add("E_mosaic_big", (8000, 8000), 50.0, (2001, 2001), None)   # mosaic >> subarray
    add("E_fine_full", (6400, 9600), 17.0, None, None)            # input ~ detector scale
    add("E_pa30", (2200, 3300), 50.0, (3001, 3001), None, pa=30.0)
    if quick:
        keep = {("A_k_small_in", 1), ("A_k_small_in", 5), ("C_n_in", 1)}
        c = [x for x in c if (x["group"], x["k"]) in keep][:6]
    return c


def predict_sizes(cfg):
    """(n_sub, n_in) from the geometry alone (no image data)."""
    from wcc_sim.detectors import detector_frame
    from wcc_sim.resample import choose_subgrid, footprint
    ny, nx = cfg["n_in"]
    w, _ = _image_wcs_header(ny, nx, cfg["scale_mas"])
    shape = None if cfg["shape"] is None else tuple(cfg["shape"])
    frame = detector_frame(SF, (RA0, DEC0), pa=cfg.get("pa", 0.0), shape=shape)
    box = footprint(w, (ny, nx), frame.wcs, frame.shape)
    k = choose_subgrid(cfg["scale_mas"], frame.plate_scale_mas, override=cfg["k"])
    n_det = 0 if box is None else (box[1] - box[0]) * (box[3] - box[2])
    return n_det * k * k, ny * nx


def fit(rows, key="rss_resample", const=True):
    """Least-squares (a, b, c) of key = 8 (a n_sub + b n_in) + c."""
    A = np.array([[8.0 * r["n_sub"], 8.0 * r["n_in"]] + ([1.0] if const else []) for r in rows])
    y = np.array([r[key] for r in rows], dtype=float)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return (*coef, 0.0) if not const else tuple(coef)


def predict(coef, n_sub, n_in):
    a, b, c = coef
    return 8.0 * (a * n_sub + b * n_in) + c


def _run_key(d):
    """Identity of a configuration (cfg, or result row where n_in is the pixel count)."""
    n_in = d["n_in"] if isinstance(d["n_in"], int) else d["n_in"][0] * d["n_in"][1]
    return (d["group"], n_in, d["scale_mas"], str(d["shape"]), d["k"], d.get("pa"), d.get("rep"),
            d.get("focus"), d.get("stamp"))


def run_grid(args):
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / "results.jsonl"
    rows = [json.loads(line) for line in out.read_text().splitlines()] if (args.resume and out.exists()) else []
    done = {_run_key(r) for r in rows}
    todo = []
    for cfg in configs(args.quick, args.defocus, args.stamps):
        n_sub, n_in = predict_sizes(cfg)
        todo.append((cfg, n_sub, n_in))
    todo.sort(key=lambda t: 8 * 14 * (t[1] + t[2]))
    budget = args.budget_gb * 1e9
    mode = "a" if args.resume else "w"
    with out.open(mode) as fh:
        for cfg, n_sub, n_in in todo:
            if _run_key(cfg) in done:
                continue
            coef = (14.0, 14.0, 3e8) if len(rows) < 6 else tuple(
                max(x, y) for x, y in zip(fit(rows), (0, 0, 0)))
            pred = 1.3 * max(predict(coef, n_sub, n_in),
                             predict(fit(rows, "rss_render") if len(rows) >= 6 else coef,
                                     n_sub, n_in))
            label = f"{cfg['group']:15s} n_in={cfg['n_in']} shape={cfg['shape']} k={cfg['k']}"
            if pred > budget:
                print(f"SKIP {label}: predicted {pred / 1e9:.1f} GB > budget", flush=True)
                continue
            t0 = time.time()
            p = subprocess.run([sys.executable, __file__, "--worker", json.dumps(cfg)],
                               capture_output=True, text=True)
            res = [line for line in p.stdout.splitlines() if line.startswith("RESULT ")]
            if p.returncode != 0 or not res:
                print(f"FAIL {label}: rc={p.returncode}\n{p.stderr[-2000:]}", flush=True)
                continue
            r = json.loads(res[-1][7:])
            rows.append(r)
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            print(f"DONE {label}: n_sub={r['n_sub'] / 1e6:.2f} M n_in={r['n_in'] / 1e6:.2f} M "
                  f"k={r['k_used']} resample rss={r['rss_resample'] / 1e9:.3f} GB "
                  f"tm={r['tm_resample'] / 1e9:.3f} GB render rss={r['rss_render'] / 1e9:.3f} GB "
                  f"estimate={r['estimate'] / 1e9:.3f} GB ({r['t_resample']:.0f}+{r['t_render']:.0f} s,"
                  f" wall {time.time() - t0:.0f} s)", flush=True)
    return rows


def report(rows, out_dir, candidate=None):
    """Fit summary and per-run table; candidate (a, b) defaults to the constants in
    wcc_sim.resample, the render check uses render.render_peak_bytes (with
    RENDER_FLOATS_PER_PADDED_PIXEL)."""
    from wcc_sim.render import RENDER_FLOATS_PER_PADDED_PIXEL, render_peak_bytes
    from wcc_sim.resample import PEAK_FLOATS_PER_INPUT_PIXEL, PEAK_FLOATS_PER_SUBPIXEL

    if candidate is None:
        candidate = (PEAK_FLOATS_PER_SUBPIXEL, PEAK_FLOATS_PER_INPUT_PIXEL)
    uniq = {}
    for r in rows:                      # drop repeated configurations (first run kept)
        uniq.setdefault(_run_key(r), r)
    rows = list(uniq.values())
    big = [r for r in rows if r["n_sub"] + r["n_in"] >= 2e6]   # constant overhead negligible
    lines = [f"# Resampling memory calibration ({len(rows)} runs, {len(big)} with "
             "n_sub + n_in >= 2 M used for the fits)", ""]
    for key in ("rss_resample", "tm_resample", "rss_render", "tm_render"):
        for const in (True, False):
            a, b, c = fit(big, key, const)
            res = [r[key] / predict((a, b, c), r["n_sub"], r["n_in"]) for r in big]
            lines.append(f"{key:13s} const={const!s:5s}: a = {a:6.2f} floats/sub-px, "
                         f"b = {b:6.2f} floats/input-px, c = {c / 1e6:7.1f} MB; "
                         f"measured/fit = {min(res):.3f} .. {max(res):.3f}")
    a, b = candidate
    for name, sel in (("n_sub + n_in >= 2 M", big), ("all runs", rows)):
        res = [r["rss_resample"] / predict((a, b, 0.0), r["n_sub"], r["n_in"]) for r in sel]
        lines.append(f"candidate a = {a}, b = {b} ({name}): measured RSS / estimate = "
                     f"{min(res):.3f} .. {max(res):.3f}")
        est_ren = {id(r): render_peak_bytes(tuple(r["box"]), r["k_used"], r.get("stamp", 129))
                   for r in sel}
        res = [(r["rss_render"] + r["sub_bytes"]) / est_ren[id(r)]
               for r in sel if r["k_used"] == 1]          # calibration basis of the coefficient
        lines.append(f"render, {RENDER_FLOATS_PER_PADDED_PIXEL} floats/padded px, k = 1 ({name}): "
                     f"measured RSS (incl. kept sub-grid) / estimate = "
                     f"{min(res):.3f} .. {max(res):.3f}")
        res = [max(r["rss_resample"], r["rss_render"] + r["sub_bytes"])
               / max(predict((a, b, 0.0), r["n_sub"], r["n_in"]), est_ren[id(r)]) for r in sel]
        lines.append(f"guard, max(resampling, render) ({name}): measured RSS / estimate = "
                     f"{min(res):.3f} .. {max(res):.3f}")
    lines += ["", "Per run, all estimates from the current code (a, b, render_peak_bytes): "
              "rss = measured RSS peak of resampling [GB], fl/px = rss / 8 / (n_sub + n_in), "
              "render = measured RSS peak of rendering [GB], rend+sub = render plus the "
              "sub-grid kept during rendering [GB], est = max(resampling, render estimate) [GB], "
              "res/est = rss / resampling estimate, ren/est = rend+sub / render estimate, "
              "guard = max(rss, rend+sub) / est", ""]
    hdr = (f"{'group':15s} {'n_in':>11s} {'shape':>12s} {'k':>2s} {'f':>1s} {'stamp':>5s} {'n_sub[M]':>8s} "
           f"{'n_in[M]':>7s} {'rss[GB]':>7s} {'tm[GB]':>7s} {'fl/px':>6s} {'render[GB]':>10s} "
           f"{'rend+sub':>8s} {'est[GB]':>7s} {'res/est':>7s} {'ren/est':>7s} {'guard':>6s} "
           f"{'t[s]':>5s}")
    lines.append(hdr)
    for r in sorted(rows, key=lambda r: (r["group"], r["n_sub"] + r["n_in"])):
        n = r["n_sub"] + r["n_in"]
        rs = r["rss_render"] + r["sub_bytes"]
        est_res = predict((a, b, 0.0), r["n_sub"], r["n_in"])
        est_ren = render_peak_bytes(tuple(r["box"]), r["k_used"], r.get("stamp", 129))
        est = max(est_res, est_ren)
        guard = max(r["rss_resample"], rs) / est
        lines.append(
            f"{r['group']:15s} {str(r['n_in']):>11s} {str(r['shape']):>12s} {r['k_used']:2d} "
            f"{r.get('focus_used', 0):1d} {r.get('stamp', 129):5d} "
            f"{r['n_sub'] / 1e6:8.2f} {r['n_in'] / 1e6:7.2f} {r['rss_resample'] / 1e9:7.3f} "
            f"{r['tm_resample'] / 1e9:7.3f} {r['rss_resample'] / 8 / n:6.2f} "
            f"{r['rss_render'] / 1e9:10.3f} {rs / 1e9:8.3f} {est / 1e9:7.3f} "
            f"{r['rss_resample'] / est_res:7.3f} {rs / est_ren:7.3f} {guard:6.3f} "
            f"{r['t_resample'] + r['t_render']:5.0f}")
    text = "\n".join(lines)
    (out_dir / "summary.txt").write_text(text + "\n")
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("Usage\n-----\n")[1])
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    ap.add_argument("--budget-gb", type=float, default=7.0,
                    help="skip configurations predicted above this peak [GB] (default 7)")
    ap.add_argument("--quick", action="store_true", help="a few small configurations only")
    ap.add_argument("--stamps", action="store_true",
                    help="PSF stamp grid only (33, 65, 513, 1025 px at focus 0); use with --resume")
    ap.add_argument("--defocus", action="store_true",
                    help="defocused-PSF grid only (focus 1 and 2); use with --resume")
    ap.add_argument("--resume", action="store_true", help="append to an existing results.jsonl")
    ap.add_argument("--fit-only", action="store_true", help="refit OUT/results.jsonl")
    ap.add_argument("--out", type=Path, default=Path("memory_calibration"),
                    help="output directory (default: memory_calibration in the current directory)")
    args = ap.parse_args()
    if not sys.platform.startswith("linux"):
        sys.exit("calibrate_peak_memory_use.py needs Linux (/proc/self/status, /proc/self/clear_refs)")
    if args.worker:
        worker(json.loads(args.worker))
        return
    if args.fit_only:
        rows = [json.loads(line) for line in (args.out / "results.jsonl").read_text().splitlines()]
    else:
        rows = run_grid(args)
    report(rows, args.out)


if __name__ == "__main__":
    main()
