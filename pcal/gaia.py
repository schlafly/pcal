import os
import multiprocessing
import numpy as np
from astropy.io import fits
import healpy
import util_efs
from astropy.stats import mad_std

gaiaxpnside = 32
gaiaxpdir = (
    '/global/cfs/cdirs/desicollab/users/rongpu/data/gaia_dr3/'
    'xp_synthetic_decam_photometry')


coeffs_bprp = {'g': np.array([0.0416867638, -0.2958055870, 0.4922869977, -0.4032130276,
        0.1459382610, -0.0190536024]),
 'r': np.array([-0.0529926024, 0.0361338856, -0.0188637440, -0.0004627745,
        0.0025104619, -0.0004741510]),
 'i': np.array([-0.0873044994, 0.1770173852, -0.1933434841, 0.0996162932,
        -0.0245789844, 0.0023553514]),
 'z': np.array([-0.0431176311, 0.0586222512, -0.0112472116, -0.0188292696,
        0.0109515264, -0.0016498615])}

coeffs_g = {'g': np.array([-0.6417454609, 0.9144340164, -0.1939379255, 0.0165146864,
        -0.0006337951, 0.0000091121]),
 'r': np.array([-17.4056692603, 5.5389908307, -0.7004043218, 0.0440353839,
        -0.0013759207, 0.0000170644]),
 'i': np.array([112.4088645838, -34.6121088481, 4.2534703382, -0.2607416920,
        0.0079739763, -0.0000973534]),
 'z': np.array([50.9842611342, -15.8466786120, 1.9770712887, -0.1236950359,
        0.0038797140, -0.0000488000])}


def correct_photometry(obj, filt):
    gcorr = np.polyval(coeffs_g[filt][::-1], obj['PHOT_G_MEAN_MAG'])
    bprpcorr = np.polyval(coeffs_bprp[filt][::-1],
                          obj['PHOT_BP_MEAN_MAG'] - obj['PHOT_RP_MEAN_MAG'])
    return gcorr + bprpcorr


def gaia_offset(obj, filt):
    if filt not in 'griz':
        return None
    pix = healpy.ang2pix(gaiaxpnside, obj['ra'], obj['dec'], nest=True, lonlat=True)
    pix = np.unique(pix)
    gaiagrz = []
    gaiaedr3 = []
    m = (obj['bitmask'] == 0) & (obj['dpsfmag'] < 0.2)
    obj = obj[m]
    for p in pix:
        fname = f'XpSyntheticDECam_{p:05d}'
        if not os.path.exists(os.path.join(gaiaxpdir, fname + '.fits')):
            print(f'could not find {fname}')
            continue
        grz = fits.getdata(
            os.path.join(gaiaxpdir, fname + '.fits'))
        gobj = fits.getdata(
            os.path.join(gaiaxpdir, 'Gaia_EDR3', fname + '_Gaia_EDR3.fits'))
        assert len(grz) == len(gobj)
        assert np.all(grz['source_id'] == gobj['source_id'])
        gaiagrz.append(grz)
        gaiaedr3.append(gobj)
    gaiagrz = np.concatenate(gaiagrz)
    gaiaedr3 = np.concatenate(gaiaedr3)
    bprp = gaiaedr3['PHOT_BP_MEAN_MAG'] - gaiaedr3['PHOT_RP_MEAN_MAG']
    m = (bprp > 0.6) & (bprp < 2.5)
    gaiag = gaiaedr3['PHOT_G_MEAN_MAG']
    m = m & (gaiag > 13.5) & (gaiag < 17.65)
    gaiagrz = gaiagrz[m]
    gaiaedr3 = gaiaedr3[m]

    mo, mg, dog = util_efs.match_radec(
        obj['ra'], obj['dec'], gaiaedr3['RA'], gaiaedr3['DEC'],
        nneighbor=1, rad=1/60/60)
    correction = correct_photometry(gaiaedr3, filt)
    nobj = len(mo)
    if nobj > 0:
        maggaia = 22.5-2.5*np.log10(gaiagrz[f'flux_{filt}']) + correction
        magls = obj['instpsfmag']
        dmag = maggaia[mg] - magls[mo]
        med = np.nanmedian(dmag)
        mean = np.nanmean(dmag)
        std = np.nanstd(dmag)
        madstd = mad_std(dmag, ignore_nan=True)
    else:
        med = np.nan
        mean = np.nan
        madstd = np.nan
        std = np.nan
    return dict(mean=mean, median=med, std=std, madstd=madstd, nobj=nobj)


def worker(filename):
    hdr = fits.getheader(filename, 1)
    filt = hdr['filter'][0]
    obj = fits.getdata(filename)
    res = np.zeros(1, dtype=[
        ('filename', 'U200'), ('expnum', 'i4'),
        ('mean', 'f4'), ('median', 'f4'), ('std', 'f4'), ('madstd', 'f4'),
        ('nobj', 'f4')])
    off = gaia_offset(obj, filt)
    if off is None:
        return None
    for field in off:
        res[field] = off[field]
    res['filename'] = filename
    res['expnum'] = hdr['expnum']
    return res


def run_photomfn(photomfn, nproc=1):
    if nproc > 1:
        from multiprocessing import pool
        p = pool.Pool(nproc)
        out = p.map(worker, photomfn)
    else:
        import tqdm
        out = [worker(p) for p in tqdm.tqdm(photomfn)]
    return out


def package_output(gaiazp, ccds):
    from numpy.lib.recfunctions import rec_append_fields
    gaiazp = np.concatenate([x for x in gaiazp if x is not None])
    mg, mc = util_efs.match(gaiazp['expnum'], ccds['expnum'])
    filt = np.zeros(len(gaiazp), dtype='U1')
    filt[mg] = ccds['filter'][mc]
    good = np.zeros(len(gaiazp), dtype='bool')
    medians = {k: np.nanmedian(gaiazp['median'][filt == k]) for k in 'griz'}
    good = (gaiazp['madstd'] < 0.05)
    for f in 'griz':
        m = filt == f
        good[m] = (
            good[m] &
            (gaiazp['median'][m] > np.nanmedian(gaiazp['median'][m]) - 0.5))
    gaiazp = rec_append_fields(gaiazp, ['filter', 'good'], [filt, good])
    return gaiazp
