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


ccdfn = [
    '/global/cfs/cdirs/cosmo/work/legacysurvey/dr11/survey-ccds-decam-delve-b.kd.fits',
    '/global/cfs/cdirs/cosmo/work/legacysurvey/dr11/survey-ccds-decam-delve.kd.fits',
    '/global/cfs/cdirs/cosmo/work/legacysurvey/dr11/survey-ccds-decam-dr10-dedup.kd.fits',
    '/global/cfs/cdirs/cosmo/work/legacysurvey/dr11/survey-ccds-decam-dr11-a.kd.fits',
    '/global/cfs/cdirs/cosmo/work/legacysurvey/dr11/survey-ccds-decam-dr11-b.kd.fits',
    '/global/cfs/cdirs/cosmo/work/legacysurvey/dr11/survey-ccds-decam-dr11-c.kd.fits',
]

photomdirname = ['/global/cfs/cdirs/cosmo/work/legacysurvey/dr10/attic/TO-ARCHIVE/zpt/decam/',
                 '/global/cfs/cdirs/cosmo/work/legacysurvey/dr11/zpt/decam/',
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
    for fn in imfn:
        exists = 0
        for dn in photomdirname:
            testfn = os.path.join(
                dn[:-6], fn.replace('.fits.fz', '-photom.fits'))
            if os.path.exists(testfn):
                exists += 1
                photomfn.append(testfn)
        if exists == 0:
            import pdb
            pdb.set_trace()
        matches.append((fn, exists))
    print('All have one match?', np.all([x[1] == 1 for x in matches]))
    return photomfn


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
                          nproc=100, badfun=bnfun)
    pickle.dump(res, open(args.output, 'wb'))
