#!/usr/bin/env python

import pcal
import pickle
import os
import glob
from astropy.io import fits
import util_efs
import numpy as np
from astropy.table import Table
import badnight

basepath = '/global/homes/s/schlafly/desiproject/users/schlafly/pcal'


# ccdfn = [
#     '/global/cfs/cdirs/cosmo/work/legacysurvey/dr11/survey-ccds-decam-delve-b.fits',
#     '/global/cfs/cdirs/cosmo/work/legacysurvey/dr11/survey-ccds-decam-delve.fits',
#     '/global/cfs/cdirs/cosmo/work/legacysurvey/dr11/survey-ccds-decam-dr10-dedup.fits',
#     '/global/cfs/cdirs/cosmo/work/legacysurvey/dr11/survey-ccds-decam-dr11-a.fits',
#     '/global/cfs/cdirs/cosmo/work/legacysurvey/dr11/survey-ccds-decam-dr11-b.fits',
#     '/global/cfs/cdirs/cosmo/work/legacysurvey/dr11/survey-ccds-decam-dr11-c.fits',
# ]

ccdfn = ['/global/cfs/cdirs/cosmo/work/legacysurvey/dr11/survey-ccds-decam-dr11-merged.fits']

photomdirname = ['/global/cfs/cdirs/cosmo/work/legacysurvey/dr11/zpt/decam/',
'/global/cfs/cdirs/cosmo/work/legacysurvey/dr10/attic/TO-ARCHIVE/zpt/decam/',
                 '/global/cfs/cdirs/cosmo/work/legacysurvey/dr9-archive/zpts-photom/decam/']

filtercolors = dict(g='blue', r='green', i='orange', z='red')


def combine_stats_with_ccds(imstatsd, ccds):
    imstatsa = np.concatenate([imstatsd[x] for x in imstatsd])
    _, uind = np.unique(ccds['expnum'])
    ccds = ccds[uind]
    mi, mc = util_efs.match(imstatsa['expnum'], ccds['expnum'])
    tab = Table()
    for field in ccds.dtype.names:
        tab[field] = ccds[field][mc]
    for field in imstatsa.dtype.names:
        tab[field] = imstatsa[field][mi]
    return tab


def plot_night(imstats, ccds, nightnum, bnd=None):
    from matplotlib import pyplot as p
    p.clf()
    mi, mc = util_efs.match(imstats['expnum'], ccds['expnum'])
    nightnuma = pcal.mjdnum(ccds['mjd_obs'])
    m = nightnuma[mc] == nightnum
    xpts = (ccds['mjd_obs'] - nightnum) * 24
    p.errorbar(xpts[mc[m]], imstats['median'][mi[m]],
               yerr=np.clip(imstats['sd'][mi[m]], 0, np.inf),
               fmt='_', capsize=3)
    p.title(nightnum)
    p.ylim(-0.1, 0.1)
    p.grid()
    p.twinx()
    p.ylim(0, 1)
    p.gca().yaxis.set_ticks([])
    badfrac = imstats['badfrac']
    badfrac[imstats['bmask'] != 0] = 1
    for f in 'griz':
        m2 = ccds['filter'][mc] == f
        p.plot(xpts[mc[m & m2]], badfrac[mi[m & m2]], 'x',
               color=filtercolors[f])
    p.twinx()
    p.ylim(1, 2.5)
    for f in 'griz':
        m2 = ccds['filter'][mc] == f
        p.plot(xpts[mc[m & m2]], ccds['airmass'][mc[m & m2]], '+',
               color='gray')
    return BadNightPlot(nightnum, p.gca(), badfile=bnd)


class QA:
    def __init__(self, stats, ccdsu, badfile):
        ms, mc = util_efs.match(stats['expnum'], ccdsu['expnum'])
        self.nightlist = np.unique(pcal.mjdnum(ccdsu['mjd_obs'][mc]))
        self.stats = stats
        self.ccdsu = ccdsu
        self.ind = 0
        self.connected = False
        self.left = None
        self.leftline = None
        self.right = None
        self.rightline = None
        self.badfile = badfile
        self.show_night()
        # get the relevant nights

    def next(self):
        if self.ind + 1 >= len(self.nightlist):
            raise StopIteration
        self.ind += 1
        self.show_night()

    def prev(self):
        if self.ind - 1 < 0:
            raise StopIteration
        self.ind -= 1
        self.show_night()

    def show_night(self, night=None):
        from matplotlib import pyplot as p
        self.left = None
        self.right = None
        if night is None:
            night = self.nightlist[self.ind]
        else:
            m = np.flatnonzero(self.nightlist == night)
            if len(m) == 0:
                raise ValueError(f'could not find night {night}')
            if len(m) > 1:
                raise ValueError(f'multiply nights matching {night}')
            self.ind = m[0]
        self.bnp = plot_night(self.stats, self.ccdsu, night, self.badfile)
        if not self.connected:
            p.gcf().canvas.mpl_connect('key_press_event', self)
            self.connected = True
        p.draw()

    def __call__(self, event):
        from matplotlib import pyplot as p
        if event.key == 'n':
            self.next()
        if event.key == 'p':
            self.prev()
        if event.key == 'b':
            bn = badnight.BadNight(filename=self.badfile)
            night = self.nightlist[self.ind]
            if bn.isbad(night):
                bn.remove_badnight(night)
            else:
                bn.add_badnight(night)
            if self.bnp is None:
                raise ValueError('self.bnp is None')
            else:
                self.bnp.draw()
            p.draw()
        if event.key == 'a':
            if event.inaxes != self.bnp.ax:
                return
            if self.left is not None:
                self.leftline.remove()
            self.left = event.xdata
            self.leftline = self.bnp.ax.axvline(self.left)
            self.bnp.ax.figure.canvas.draw()
        if event.key == 'o':
            if event.inaxes != self.bnp.ax:
                return
            if self.right is not None:
                self.rightline.remove()
            self.right = event.xdata
            self.rightline = self.bnp.ax.axvline(self.right)
            self.bnp.ax.figure.canvas.draw()
        if event.key == ' ':
            if self.right is None or self.left is None:
                print('Left and right edges of bad region not set.')
                return
            bn = badnight.BadNight(filename=self.badfile)
            night = self.nightlist[self.ind]
            bn.add_badnight(night, time=[self.left/24., self.right/24.])
            if self.bnp is None:
                raise ValueError('self.bnp is None')
            else:
                self.bnp.draw()
        if event.key == 't':
            if event.inaxes != self.bnp.ax:
                return
            print(event.x, event.y, event.inaxes, event.xdata, event.ydata)


class BadNightPlot:
    def __init__(self, night, ax, badfile):
        self.night = night
        self.ax = ax
        self.rec = [ ]
        self.badfile = badfile
        self.draw()
    def clear(self):
        if len(self.rec) != 0:
            for x in self.rec:
                # self.ax.patches.remove(x)
                x.remove()
            self.rec = [ ]
            self.ax.figure.canvas.draw()
    def draw(self):
        bn = badnight.BadNight(filename=self.badfile)
        self.clear()
        if bn.isbad(self.night):
            ylim = self.ax.get_ylim()
            xlim = self.ax.get_xlim()
            for f,l in bn.badtimes(self.night):
                for off in [-1, 0, 1]:
                    fh = (f+off) * 24 ; lh = (l+off) * 24
                    trec = self.ax.fill([fh, lh, lh, fh],
                                        [ylim[0], ylim[0], ylim[1], ylim[1]],
                                        edgecolor='black', hatch='/',
                                        fill=False)
                self.rec += trec
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)
        self.ax.figure.canvas.draw()


def gather_photomfn():
    imfn = []
    for ccdfn0 in ccdfn:
        ccds = fits.getdata(ccdfn0)
        imfn0 = np.unique(ccds['image_filename'])
        imfn.append(imfn0)
    imfn = np.concatenate(imfn)
    matches = []
    photomfn = []
    missing = []
    for fn in imfn:
        exists = 0
        for dn in photomdirname:
            testfn = os.path.join(
                dn[:-6], fn.replace('.fits.fz', '-photom.fits'))
            if os.path.exists(testfn):
                exists += 1
                if exists <= 1:
                    photomfn.append(testfn)
                else:
                    print(f'skipping {testfn}, duplicate')
        if exists == 0:
            missing.append(fn)
            print(f'cannot find {fn}')
        matches.append((fn, exists))
    print('All have one match?', np.all([x[1] == 1 for x in matches]))
    return photomfn, missing, matches


def flat_output(basename, ccds, plotfn=None):
    # we need to:
    # gather the zero points
    # gather the flat field offsets
    # make a nice file with all of them in it
    # expnum, ccdnum, zp, other stats?
    # we should also apply an absolute zero point

    statsa = []
    zpc = np.zeros(len(ccds), dtype='f4')
    zpc[:] = np.nan
    keepccd = np.zeros(len(ccds), dtype=bool)
    keepccd[:] = False
    ccd_offsets = dict()
    ccdfilt = ccds['filter'].astype('U1')
    flatvecs = dict()
    ccdalt, ccdaz = util_efs.rdllmjd2altaz(
        ccds['ra'], ccds['dec'], -30.16966100, -70.80652500, ccds['mjd_obs'],
        precess=False)
    ccds['airmass'] = util_efs.alt2airmass(ccdalt)

    for f in 'griz':
        res, stats0, paramdat, _ = pickle.load(open(basename.format(f), 'rb'))
        statsa.append(stats0)
        m = np.flatnonzero(ccdfilt == f)
        mjdnum = pcal.mjdnum(ccds['mjd_obs'][m])
        mn = np.flatnonzero(paramdat['descr'] == 'nightly a-term')
        mp, mm = util_efs.match(paramdat['idx'][mn], mjdnum)
        aterm = res.x[mn[mp]]
        kterm = res.x[paramdat['descr'] == 'k-term']
        if np.any(paramdat['descr'] == 'nightly k-term'):
            mkn = np.flatnonzero(paramdat['descr'] == 'nightly k-term')
            mpk, mmk = util_efs.match(paramdat['idx'][mkn], mjdnum)
            kterm = kterm + res.x[mkn[mpk]]
        zp0 = aterm - (ccds['airmass'][m[mm]] - 1.1) * kterm
        zpc[m[mm]] = zp0
        npccd = 16*8
        for flattype in ['f-term', 'color-flat']:
            mf = paramdat['descr'] == flattype
            if not np.any(mf):
                continue
            if flattype not in flatvecs:
                flatvecs[flattype] = dict()
            fpvec = res.x[mf]
            ccd_offsets0 = list([np.nan])
            for i in range(1, 63):
                tccd = fpvec[npccd*i:npccd*(i+1)]
                m = tccd != 0
                if np.sum(m) == 0:
                    med = 0
                else:
                    med = np.median(tccd[m])
                ccd_offsets0.append(med)
            ccd_offsets[f] = np.array(ccd_offsets0)
            flatvecs[flattype][f] = fpvec

    stats = np.concatenate(statsa)
    ms, mc = util_efs.match(stats['expnum'], ccds['expnum'])
    m = stats['sd'][ms] > -900
    zpc[mc[m]] -= stats['median'][ms[m]]

    ccdnum = pcal.ccdnametonum(ccds['ccdname'].astype('U3'))
    for f in 'griz':
        m = ccdfilt == f
        zpc[m] += ccd_offsets[f][ccdnum[m]]
        mdec = ccds['dec'] > -28
        absoff = np.nanmedian(zpc[m & mdec] - ccds['ccdzpt'][m & mdec])
        zpc[m] -= absoff
    # this is all one-number-per-exposure, except for the constant flat field

    m = ((stats['sd'][ms] > -900) &
         (stats['sd_chip'][ms, ccdnum[mc]] > 0) &
         (stats['n_chip'][ms, ccdnum[mc]] > 5) &
         (ccdnum[mc] == 31))  # apply per-chip zero-point only to S7
    zpc[mc[m]] -= stats['median_chip'][ms[m], ccdnum[mc[m]]]
    print('not applying median chip offsets')
    badchip = (stats['n_chip'] >= 1) & (
        (stats['sd_chip'] > 0.04) | (np.abs(stats['median_chip']) > 0.2))
    badchip[:, 31] |= stats['n_chip'][:, 31] < 4
    # S7 always need a zero point; if we can't get one, mark as bad.

    median_chip = stats['median_chip'].copy()
    median_chip[stats['n_chip'] < 3] = np.nan
    median_chip[:, 31] = np.nan  # ignore S7
    betweenchipscatter = np.nanstd(median_chip, axis=1)

    from numpy.lib.recfunctions import rec_append_fields
    keep = ((stats['sd'] < 0.04) & (stats['sd'] > 0) &
            np.abs(stats['median'] < 0.2) & (stats['n'] > 700) &
            (betweenchipscatter < 0.04) & (stats['badfrac'] < 0.15))
    stats = rec_append_fields(
        stats, ['zpt', 'keep'],
        [np.zeros(len(stats), dtype='f4'), np.zeros(len(stats), dtype='f4')])
    stats['zpt'][ms] = zpc[mc]
    stats['keep'][ms] = keep[ms]

    keepccd[mc] = keep[ms] & ~badchip[ms, ccdnum[mc]]
    ccds = rec_append_fields(ccds, ['pcalzpt', 'pcalkeep'], [zpc, keepccd])
    if plotfn is not None:
        make_plots(ccds, stats, flatvecs, plotfn)
    return ccds, stats, flatvecs


def make_plots(ccds, stats, flatvecs, plotfn=None, filters='griz'):
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib import pyplot as p
    import pcal
    if plotfn is not None:
        pdf = PdfPages(plotfn)
    from astropy.stats import mad_std
    for flattype in flatvecs:
        flatvec = flatvecs[flattype]
        for f in 'griz':
            if plotfn is None:
                p.figure()
            fpim = pcal.fpim(flatvec[f])
            p.clf()
            p.imshow(fpim, vmin=-0.02, vmax=0.02, origin='lower', cmap='binary')
            p.title(f)
            mu = np.median(fpim[fpim != 0])
            sigma = mad_std(fpim[fpim != 0])
            p.text(5, 8, rf'$\mu={mu:6.3f}$')
            p.text(5, 4, rf'$\sigma={sigma:6.3f}$')
            p.colorbar()
            if plotfn is not None:
                pdf.savefig()
    filt = ccds['filter'].astype('U1')
    ccdname = ccds['ccdname'].astype('U3')
    for f in filters:
        if plotfn is None:
            p.figure()
        p.clf()
        m = ((filt == f) & (ccds['pcalkeep'] != 0) & (ccds['zpt'] > 10) &
             (ccds['ccdzpt'] > 10) & (ccds['ccdzpt'] < 30))
        m1 = m & (ccdname == 'N4') & (ccds['dec'] > -28)
        sig1 = mad_std(ccds['pcalzpt'][m1] - ccds['zpt'][m1])
        m2 = m & (ccds['dec'] > -28)
        med2 = np.median(ccds['pcalzpt'][m2] - ccds['ccdzpt'][m2])
        m3 = (m  # & (np.random.uniform(size=len(filt)) > 0.99)
             & (ccdname == 'N4'))
        p.scatter(ccds['ra'][m3], ccds['dec'][m3],
                  c=(ccds['pcalzpt'][m3] - ccds['ccdzpt'][m3]),
                  vmin=-0.05, vmax=0.05,
                  rasterized=True, cmap='bwr_r')
        m4 = m & (ccds['dec'] < -30)
        med4 = np.median(ccds['pcalzpt'][m4] - ccds['ccdzpt'][m4])
        p.colorbar()
        p.xlim(0, 360)
        p.title(f)
        p.text(150, -60, rf'$\sigma = {sig1:6.3f}$ (dec > -28, N4, keep)')
        p.text(150, -65, rf'$\mu = {med2:6.3f}$ (dec > -28, keep))')
        p.text(150, -70, rf'$\mu = {med4:6.3f}$ (dec < -30, keep))')
        if plotfn is not None:
            pdf.savefig()
    if plotfn is not None:
        pdf.close()



if __name__ == '__main__':
    import argparse
    parse = argparse.ArgumentParser(
        description='Run pcal')
    parse.add_argument('data', help='name of object file to analyze')
    parse.add_argument('bandpass', help='bandpass to analyze')

    parse.add_argument('output', help='output file to write')
    parse.add_argument('--badnight', default=None, help='bad night file')
    args = parse.parse_args()
    obj0 = pickle.load(open(args.data, 'rb'))
    obj = obj0[args.bandpass]
    del obj0
    import gc
    gc.collect()
    bnfun = None
    if args.badnight is not None:
        import badnight
        bn = badnight.BadNight(args.badnight, mjd_fun=pcal.mjdnum)
        bnfun = bn.checkbad
    res = pcal.solve_dr11(obj, optiter=50, clipiter=5,
                          nproc=64, badfun=bnfun)
    pickle.dump(res, open(args.output, 'wb'))
