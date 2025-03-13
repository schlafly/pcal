# photometric calibration

# what do we need?
# MJD
# x, y
# instmag, uncertainty
# source id (from Gaia)
# Gaia color / mag?  (BP - RP, G?)
# expnum, ccdnum, filter


# need to look up conventions for gain and exposure time to make sure
# that we end up with the right magnitudes
# i.e., instmag = -2.5*log10(flux * gain / exposure time)


from astropy.io import fits
import numpy as np
import util_efs
# import jax
# from jax import numpy as jnp
import numba
import healpy
import tqdm
from astropy import stats
import multiprocessing
import multiprocessing.shared_memory

global_saved_data = dict()

badmask = dict(
    image_median = 1,
    image_scatter = 2,
    image_noobs = 4,
    image_manual = 8,
    image_manybad = 16,
    image_chip = 32,
    star_resid = 64,
    star_noobs = 128,
    obs_manual = 256,
)
image_masks = (badmask['image_median'] | badmask['image_scatter'] |
               badmask['image_noobs'] | badmask['image_manual'] |
               badmask['image_manybad'] | badmask['image_chip'])
star_masks = badmask['star_resid'] | badmask['star_noobs']


catdtype = [('objid', 'i8'),
            ('instmag', 'f4'),
            ('dmag', 'f4'),
            ('phot_g_mean_mag', 'f4'),
            ('bp_rp', 'f4'),
            ('x', 'f4'),
            ('y', 'f4'),
            ('filter', 'U1'),
            ('ccdnum', 'i4'),
            ('expnum', 'i4'),
            ('mjd_obs', 'f8'),
            ('airmass', 'f4'),
            ('weight', 'f4'),
            ('idx', 'i4'),
            ('fidx', 'i4'),
            ('avginstmag', 'f4'),
            ('fwhm', 'f4'),
            ('healpix', 'i4'),
            ('ra', 'f8'),
            ('dec', 'f8'),
            ('bmask', 'i4'),
            ]

ccdnamenumdict = {'S1': 25, 'S2': 26, 'S3': 27, 'S4': 28,
                  'S5': 29, 'S6': 30, 'S7': 31,
                  'S8': 19, 'S9': 20, 'S10': 21, 'S11': 22, 'S12': 23,
                  'S13': 24,
                  'S14': 13, 'S15': 14, 'S16': 15, 'S17': 16, 'S18': 17,
                  'S19': 18,
                  'S20': 8, 'S21': 9, 'S22': 10, 'S23': 11, 'S24': 12,
                  'S25': 4, 'S26': 5, 'S27': 6, 'S28': 7,
                  'S29': 1, 'S30': 2, 'S31': 3,
                  'N1': 32, 'N2': 33, 'N3': 34, 'N4': 35,
                  'N5': 36, 'N6': 37, 'N7': 38,
                  'N8': 39, 'N9': 40, 'N10': 41, 'N11': 42, 'N12': 43,
                  'N13': 44,
                  'N14': 45, 'N15': 46, 'N16': 47, 'N17': 48, 'N18': 49,
                  'N19': 50,
                  'N20': 51, 'N21': 52, 'N22': 53, 'N23': 54, 'N24': 55,
                  'N25': 56, 'N26': 57, 'N27': 58, 'N28': 59,
                  'N29': 60, 'N30': 61, 'N31': 62,
                  '': 9999
                  }


def ccdnametonum(ccdname):
    return util_efs.convert_val(ccdname, ccdnamenumdict, default=-1)


def xyccdtofidx(x, y, ccd):
    npix = 256
    cx = 2048 // npix
    cy = 4096 // npix
    return (ccd*cx*cy + (np.clip(y, 0, 4096) // npix).astype('i4')*cx +
            (np.clip(x, 0, 2048) // npix).astype('i4'))


xyccdtofidx.nidx = 8 * 16


def select_objects(obj, ccdnum=None):
    bp_rp = obj['phot_bp_mean_mag'] - obj['phot_rp_mean_mag']
    keep = ((obj['dpsfmag'] < 0.2) & (obj['dpsfmag'] > 0) &
            (obj['gaia_sourceid'] > 0) & (obj['bitmask'] == 0) &
            (obj['phot_g_mean_mag'] > 17) &
            (obj['phot_g_mean_mag'] < 19) &
            (bp_rp > 0.5) & (bp_rp < 3))
    if ccdnum is not None:
        keep &= ((ccdnum != 31) | (obj['x_fit'] <= 1023))
    return keep


def read_file(fname):
    nside = 64
    res, hdr = fits.getdata(fname, header=True)
    ccdnum = ccdnametonum(res['ccdname'])
    good = select_objects(res, ccdnum=ccdnum)
    res = res[good]
    out = np.zeros(len(res), dtype=catdtype)
    copy_fields = ['phot_g_mean_mag', 'filter', 'expnum']
    for field in copy_fields:
        out[field] = res[field]
    rename_fields = dict(gaia_sourceid='objid', instpsfmag='instmag',
                         dpsfmag='dmag')
    for field in rename_fields:
        out[rename_fields[field]] = res[field]
    out['bp_rp'] = res['phot_bp_mean_mag'] - res['phot_rp_mean_mag']
    out['x'] = res['x_fit']
    out['y'] = res['y_fit']
    out['ccdnum'] = ccdnum[good]
    out['mjd_obs'] = hdr['MJD-OBS']
    exptime = hdr['EXPTIME']
    alt, az = util_efs.rdllmjd2altaz(
        res['ra'], res['dec'], -30.16966100, -70.80652500, hdr['MJD-OBS'])
    out['airmass'] = util_efs.alt2airmass(alt)
    out['dmag'] = np.hypot(out['dmag'], 0.01)
    out['instmag'] = out['instmag']
    # + 2.5*np.log10(exptime) ; not needed for LS mags
    out['fwhm'] = hdr['fwhm']
    out['fidx'] = xyccdtofidx(out['x'], out['y'], out['ccdnum'])
    out['healpix'] = healpy.ang2pix(nside, res['ra_gaia'], res['dec_gaia'],
                                    lonlat=True)
    out['ra'] = res['ra_gaia']
    out['dec'] = res['dec_gaia']
    assert len(out) == len(np.unique(out['objid']))
    assert len(np.unique(out['expnum'])) <= 1
    return out


def gather_data(files, nproc=1):
    res = dict()

    # this is really silly, but it apparently helps to
    # group this section.  Otherwise I get random locks
    # I don't understand.  So this groups things into groups
    # of 1000 files, and accumulates those into res individually.

    files_all = files
    chunk = 1000

    if nproc > 1:
        from concurrent.futures import ProcessPoolExecutor
        executor = ProcessPoolExecutor(max_workers=nproc)

    for files in [files_all[i:i + chunk]
                  for i in range(0, len(files_all), chunk)]:
        if nproc <= 1:
            iterator = (read_file(f) for f in tqdm.tqdm(files))
        else:
            iterator = executor.map(read_file, files)

        for out in iterator:
            for hp in np.unique(out['healpix']):
                for filt in np.unique(out['filter']):
                    outd = res.get(filt, dict())
                    res[filt] = outd
                    outd[hp] = (outd.get(hp, list())
                                + [out[out['healpix'] == hp]])
                # should make this loop more efficient

    if nproc > 1:
        executor.shutdown()



    for filt in res:
        for hp in tqdm.tqdm(res[filt]):  # this loop should be threaded too.
            res[filt][hp] = np.concatenate(res[filt][hp])
            res[filt][hp].sort(order='objid')
            compute_weights_and_averages(res[filt][hp])
            neff = compute_neffobs(res[filt][hp]['weight'],
                                   res[filt][hp]['objid'])
            res[filt][hp] = res[filt][hp][neff > 0]
            uids = [(x['objid'], x['expnum']) for x in res[filt][hp]]
            if len(uids) != len(set(uids)):
                print('Warning, some duplicate detections in '
                      f'{filt}/{hp}')
    return res


def compute_weights_and_averages(res):
    if isinstance(res, dict):
        for i in tqdm.tqdm(res):
            compute_weights_and_averages(res[i])
        return

    assert np.all(np.diff(res['objid']) >= 0)
    weight = compute_weights(res)
    res['weight'] = weight
    res['avginstmag'] = compute_average(
        res['instmag'], weight, res['objid'])


def repartition_data_onefilter(obj, nbatch=1000):
    nobsall = []
    for hp in obj:
        tobj = obj[hp]
        uobjid = np.unique(tobj['objid'])
        oidx = np.searchsorted(uobjid, tobj['objid'])
        nobs = np.zeros(len(uobjid), dtype='i4')
        np.add.at(nobs, oidx, 1)
        nobsall.append(nobs)
    startnobsperbatch = np.cumsum(np.array([np.sum(x) for x in nobsall]))
    startnobs2perbatch = np.cumsum(np.array([np.sum(x**2) for x in nobsall]))
    nobs = np.concatenate([x for x in nobsall])
    nobs2 = np.concatenate([x**2 for x in nobsall])
    cumnobs = np.cumsum(nobs.astype('i8'))
    cumnobs2 = np.cumsum(nobs2.astype('i8'))
    cumnobs2perbatch = [
        np.cumsum(nobs.astype('i8')**2) + start
        for nobs, start
        in zip(nobsall, np.concatenate([[0], startnobs2perbatch[:-1]]))]
    cumnobsperbatch = [
        np.concatenate([[0], np.cumsum(nobs)[:-1]]) + start
        for nobs, start
        in zip(nobsall, np.concatenate([[0], startnobsperbatch[:-1]]))]
    starts2 = (np.arange(nbatch) / nbatch * cumnobs2[-1]).astype('i8')
    ends2 = np.concatenate([starts2[1:], [cumnobs2[-1]]])

    group = list()
    groups = list()
    ipart = 0
    lastindex = 0

    # loop over healpix
    # we want to add each healpix to a new group
    # including all observations of each object,
    # subject to trying to get each group to have the
    # same number of sum ndet^2
    # specifically, we want to include all the objects
    # up to the one with cumnobs2[batch] <= ends2[ipart]
    for i, hp in enumerate(obj):
        if len(cumnobs2perbatch[i]) == 0:
            continue
        offset = cumnobsperbatch[i][0]
        # loop over batches
        while ((ipart < len(ends2)) and
               (ends2[ipart] <= cumnobs2perbatch[i][-1])):
            # objindex _object_ within obj[hp]
            m = np.flatnonzero(cumnobs2perbatch[i] <= ends2[ipart])
            if len(m) > 0:
                objindex = np.max(m)
                index = np.sum(nobsall[i][:objindex + 1])
                group.append(obj[hp][lastindex - offset:index])
            else:
                index = 0
            groups.append(group)
            group = list()
            lastindex = index + offset
            ipart += 1
        group.append(obj[hp][lastindex - offset:])
        lastindex += len(obj[hp]) - (lastindex - offset)

    # sanity checks:
    # same number of rows
    # no objects in one group that are also in another
    assert cumnobs[-1] == np.sum([np.sum([len(x) for x in g]) for g in groups])
    flatgroups = sum(groups, [])
    for i in range(len(flatgroups) - 1):
        uobj1 = np.unique(flatgroups[i]['objid'])
        uobj2 = np.unique(flatgroups[i+1]['objid'])
        assert (len(uobj1) + len(uobj2) ==
                len(set(np.concatenate([uobj1, uobj2]))))

    out = dict()
    for i in range(len(groups)):
        # the groups come from different healpix and need not obey any
        # rules about ordering; sort these so that the objid are ascending
        out[i] = np.concatenate(groups[i])
        out[i].sort(order='objid')
    return out


def repartition_data(obj, nbatch=1000):
    out = dict()
    for filt in obj:
        out[filt] = repartition_data_onefilter(obj[filt])
    return out


def compute_weights(obj):
    # assumes objid are sorted
    u = util_efs.unique(obj['objid'])
    ndet = np.zeros(len(u), dtype='i4')
    ndet[0] = u[0]+1
    ndet[1:] = u[1:]-u[0:-1]
    oind = np.repeat(np.arange(len(u), dtype='i4'), ndet)
    wtsum = np.zeros(len(u), dtype='f4')
    weight = obj['dmag']**-2 * (obj['bmask'] == 0)
    wtsum = util_efs.add_arr_at_ind(wtsum, weight, oind)
    wtsum = wtsum + (wtsum == 0)
    weight = weight / wtsum[oind]
    return weight


def compute_average(val, weight, objid):
    # assumes objid are sorted
    u = util_efs.unique(objid)
    ndet = np.zeros(len(u), dtype='i4')
    ndet[0] = u[0]+1
    ndet[1:] = u[1:]-u[0:-1]
    oind = np.repeat(np.arange(len(u), dtype='i4'), ndet)
    avg = np.zeros(len(u), dtype='f4')
    avg = util_efs.add_arr_at_ind(avg, weight*val, oind)
    return avg[oind]


def compute_neffobs(weight, objid):
    # assumes objid are sorted
    u = util_efs.unique(objid)
    ndet = np.zeros(len(u), dtype='i4')
    ndet[0] = u[0]+1
    ndet[1:] = u[1:]-u[0:-1]
    oind = np.repeat(np.arange(len(u), dtype='i4'), ndet)
    neff = np.zeros(len(u), dtype='f4')
    neff = util_efs.add_arr_at_ind(neff, (weight > 0).astype('i4'), oind)
    return neff[oind]



# @jax.jit
# def model_generic(param, kwargs):
#     objid = kwargs['objid']
#     weight = kwargs['weight']
#     idxlist = kwargs['idxlist']
#     coefflist = kwargs['coefflist']
#     oidx = kwargs['oidx']
#     out = jnp.zeros(len(objid), dtype='f8')
#     calparam = out * 0
#     avgcal = jnp.zeros(len(kwargs['uobjid']), dtype='f8')
#     for idx, coeff in zip(idxlist, coefflist):
#         calparam += param[idx] * coeff
#     avgcal = avgcal.at[oidx].add(weight * calparam)
#     out += -calparam + avgcal[oidx]
#     return out


@numba.jit(nopython=True)
def add_at(arr, idx, val):
    for i in range(len(idx)):
        arr[idx[i]] += val[i]


def compute_zeropoint(param, kwargs):
    """Computes zero point s.t. cal mag = inst mag + Z.

    Roughly a - kx + f.
    """
    calparam = np.zeros(len(kwargs['objid']), dtype='f8')
    for idx, coeff in zip(kwargs['idxlist'], kwargs['coefflist']):
        calparam += param[idx] * coeff
    return calparam


def model_generic_np(param, kwargs):
    objid = kwargs['objid']
    weight = kwargs['weight']
    idxlist = kwargs['idxlist']
    coefflist = kwargs['coefflist']
    oidx = kwargs['oidx']
    out = np.zeros(len(objid), dtype='f8')
    avgcal = np.zeros(len(kwargs['uobjid']), dtype='f8')
    calparam = compute_zeropoint(param, kwargs)
    add_at(avgcal, oidx, weight * calparam)
    out += -calparam + avgcal[oidx]
    return out


def loss_generic_np_global(args):
    param, hp = args
    return loss_generic_np(param, global_saved_data['kwargs'][hp])


def loss_generic_np_global2(param, hp):
    return loss_generic_np(param, global_saved_data['kwargs'][hp])


def grad_loss_generic_np_global2(param, hp):
    return grad_loss_generic(param, global_saved_data['kwargs'][hp])


def grad_loss_generic_np_global3(args):
    param, hp = args
    return grad_loss_generic(param, global_saved_data['kwargs'][hp])


def loss_generic_grouped(param, kwargs):
    fn = loss_generic_np_global
    if kwargs.get('executor', None):
        args = ((param, hp) for hp in kwargs['hplist'])
        terms = kwargs['executor'].map(fn, args)
    else:
        terms = [fn((param, hp)) for hp in kwargs['hplist']]
    return np.sum(terms)


def gradient_loss_generic_grouped_pool(param, kwargs):
    fn = grad_loss_generic_np_global3
    if kwargs.get('executor', None):
        args = ((param, hp) for hp in kwargs['hplist'])
        terms = kwargs['executor'].map(fn, args)
    else:
        terms = [fn((param, hp)) for hp in kwargs['hplist']]
    return np.sum(terms, axis=0)


def loss_generic_grouped2(param, kwargs):
    from functools import partial
    one = partial(loss_generic_one, param=param, kwargs=kwargs)
    kwargs['executor'].map(one, kwargs['hplist'])


def loss_generic_grouped3(param, kwargs):
    if kwargs.get('executor', None):
        import concurrent.futures
        terms = (
            x.result() for x in
            concurrent.futures.as_completed(
            (kwargs['executor'].submit(loss_generic_np_global2, param, hp)
             for hp in kwargs['hplist'])))
    else:
        terms = (loss_generic_np_global2(param, hp) for hp in kwargs['hplist'])
    terms = [x for x in terms]
    return np.sum(terms)


def gradient_loss_generic_grouped(param, kwargs):
    if kwargs.get('executor', None):
        import concurrent.futures
        terms = (
            x.result() for x in
            concurrent.futures.as_completed(
                (kwargs['executor'].submit(kwargs['grad'], param, hp)
                 for hp in kwargs['hplist'])))
    else:
        terms = (kwargs['grad'](param, hp) for hp in kwargs['hplist'])
    terms = [x for x in terms]
    return np.sum(terms, axis=0, dtype='f8')



def damper(chi, damp):
    """Pseudo-Huber loss function."""
    return 2*damp*np.sign(chi)*(np.sqrt(1+np.abs(chi)/damp)-1)


def damper_deriv(chi, damp):
    """First derivative of pseudo-Huber loss function"""
    return (1 + np.abs(chi)/damp)**(-0.5)


# @jax.jit
# def loss(param, kwargs):
#     chi = ((kwargs['instmag'] - kwargs['avginstmag'] -
#             model(param, kwargs)) / kwargs['dmag'])
#     chi = damper(chi, 3)
#     chi2 = jnp.sum(chi**2)
#     # jax.debug.print('{chi2}', chi2=chi2)
#     return chi2


# @jax.jit
# def loss_generic(param, kwargs):
#     chi = ((kwargs['instmag'] - kwargs['avginstmag'] -
#             model_generic(param, kwargs)) / kwargs['dmag'])
#     chi = damper(chi, 5)
#     chi2 = jnp.sum(chi**2)
#     # jax.debug.print('{chi2}', chi2=chi2)
#     return chi2


def loss_generic_np(param, kwargs):
    chi = ((kwargs['instmag'] - kwargs['avginstmag'] -
            model_generic_np(param, kwargs)) / kwargs['dmag'])
    chi *= kwargs['weight'] != 0
    # zero weight observations don't contribute to loss.

    chi = damper(chi, 5)
    chi2 = np.sum(chi**2)
    return chi2


@numba.jit(nopython=True)
def compute_grad(grad, idx, coeff, nobs, prod, weight):
    start = 0
    for nn in nobs:
        end = start + nn
        for i in range(start, end):
            for j in range(start, end):
                grad[idx[j]] += weight[j] * prod[i] * coeff[j]
            grad[idx[i]] -= coeff[i] * prod[i]
        start = start + nn


def grad_loss_generic(param, kwargs):
    model = model_generic_np(param, kwargs)
    chi = (kwargs['instmag'] - kwargs['avginstmag'] - model) / kwargs['dmag']
    chi *= kwargs['weight'] != 0
    prod = -2 * damper(chi, 5) / kwargs['dmag'] * damper_deriv(chi, 5)
    prod *= kwargs['weight'] != 0
    # prod = -2 * chi / kwargs['dmag']
    grad = np.zeros(len(param), dtype='f8')
    idxlist = kwargs['idxlist']
    coefflist = kwargs['coefflist']
    weight = kwargs['weight']
    oidx = kwargs['oidx']
    nobs = np.zeros(len(kwargs['uobjid']), dtype='i4')
    np.add.at(nobs, oidx, 1)
    # should precompute this, but it's conceptually cheaper than the
    # following nested loop.
    for idx, coeff in zip(idxlist, coefflist):
        if np.isscalar(idx):
            idx = np.full(len(prod), idx)
        if np.isscalar(coeff):
            coeff = np.full(len(prod), coeff)
        compute_grad(grad, idx, coeff, nobs, prod, weight)
    return grad


def mjdnum(mjd):
    return np.uint32(mjd + 0.3)


def compute_nidx(mjd, unight):
    nightnum = mjdnum(mjd)
    return np.searchsorted(unight, nightnum)


def compute_unight(obj):
    unights = []
    for hp in obj:
        nightnum = mjdnum(obj[hp]['mjd_obs'])
        unight = np.unique(nightnum)
        unights.append(unight)
    return np.unique(np.concatenate(unights))


# def solve(obj):
#     oidx = np.searchsorted(np.unique(obj['objid']), obj['objid'])
#     avgcal = np.zeros(len(np.unique(obj['objid'])))
#     guess = np.zeros(np.max(obj['idx'])+1, dtype='f4')
#     kwargs = {k: obj[k] for k in ['objid', 'weight', 'idx', 'airmass',
#                                   'instmag', 'dmag', 'avginstmag',
#                                   ]}
#     kwargs['avgcal'] = avgcal
#     kwargs['oidx'] = oidx
#     # from jax.scipy.optimize import minimize
#     # res = minimize(loss, guess, args=(kwargs,), method='BFGS')
#     from scipy.optimize import minimize
#     from jax import grad
#     res = minimize(loss, guess, args=(kwargs,), jac=grad(loss),
#                    method='L-BFGS-B')
#     return res


def solve_generic(obj):
    uobjid = np.unique(obj['objid'])
    oidx = np.searchsorted(uobjid, obj['objid'])
    kwargs = dict()
    kwargs['objid'] = obj['objid']
    kwargs['weight'] = obj['weight']
    kwargs['instmag'] = obj['instmag']
    kwargs['avginstmag'] = obj['avginstmag']
    kwargs['oidx'] = oidx
    kwargs['dmag'] = obj['dmag']
    kwargs['idxlist'] = (0, obj['idx'], obj['fidx'])
    kwargs['coefflist'] = (obj['airmass'], 1, 1)
    kwargs['uobjid'] = uobjid
    maxpar = np.max([np.max(x) for x in kwargs['idxlist']])
    guess = np.zeros(maxpar+1, dtype='f4')
    # from jax.scipy.optimize import minimize
    # res = minimize(loss, guess, args=(kwargs,), method='BFGS')
    from scipy.optimize import minimize
    # from jax import grad
    fun = loss_generic_np
    res = minimize(fun, guess, args=(kwargs,), jac=grad_loss_generic,
                   method='L-BFGS-B')
    return res


def solve_dr11(obj, nproc=1, method='L-BFGS-B',
               options=None, clear_bmask=True,
               guess=None, optiter=30, clipiter=6, badfun=None):
    kwargs = dict()

    kwargs['hplist'] = list(obj.keys())
    unight = compute_unight(obj)
    maxfidx = 0
    if clear_bmask:
        for hp in obj.keys():
            obj[hp]['bmask'] = 0  # clear bad mask
    for hp in obj.keys():
        obj[hp]['bmask'] |= badmask['obs_manual'] * (obj[hp]['ccdnum'] == 31)
        # mask all of S7 / ccd 31.
    compute_weights_and_averages(obj)  # recompute weights
    for hp in obj.keys():
        kwargs[hp] = dict()
        tobj = obj[hp]
        uobjid = np.unique(tobj['objid'])
        oidx = np.searchsorted(uobjid, tobj['objid'])
        kwargs[hp]['objid'] = tobj['objid']
        kwargs[hp]['weight'] = tobj['weight']
        kwargs[hp]['instmag'] = tobj['instmag']
        kwargs[hp]['avginstmag'] = tobj['avginstmag']
        kwargs[hp]['oidx'] = oidx
        kwargs[hp]['dmag'] = tobj['dmag']
        kwargs[hp]['uobjid'] = uobjid
        obj[hp]['idx'] = compute_nidx(tobj['mjd_obs'], unight)
        maxfidx = max(maxfidx, np.max(tobj['fidx'], initial=0))
    maxnidx = len(unight) - 1
    maxpidx = 0
    for hp in obj.keys():
        tobj = obj[hp]
        kwargs[hp]['idxlist'] = (
            0, 1, 2, 3,
            4 + tobj['idx'],
            5 + maxnidx + tobj['idx'],
            6 + 2*maxnidx + tobj['fidx'],
            7 + 2*maxnidx + maxfidx + tobj['fidx'])
        # k term, fwhm, fwhm**2, color-airmass, nightly, flat field,
        # color-flat
        dairmass = tobj['airmass'] - 1.1
        dcolor = np.clip(tobj['bp_rp'] - 0.5, -2, 2)
        dfwhm = np.clip(tobj['fwhm'] - 1.1, -1, 3)
        kwargs[hp]['coefflist'] = (-dairmass,
                                   dfwhm,
                                   dfwhm**2,
                                   dcolor * dairmass,
                                   1,
                                   -dairmass,
                                   1,
                                   dcolor)
        # kwargs[hp]['idxlist'] = (
        #     0, 1 + tobj['idx'], 2 + maxnidx + tobj['fidx'])
        # kwargs[hp]['coefflist'] = (-dairmass, 1, 1)
        maxpidx = max(maxpidx, np.max([
            np.max(x, initial=0) for x in kwargs[hp]['idxlist']]))

    paramdat = np.zeros(maxpidx + 1, dtype=[('descr', 'U80'), ('idx', 'i4')])
    # paramdat[0] = ('k-term', 0)
    # paramdat['descr'][1:maxnidx + 2] = 'nightly a-term'
    # paramdat['idx'][1:maxnidx + 2] = unight
    # paramdat['descr'][2+maxnidx:] = 'f-term'
    # paramdat['idx'][2+maxnidx:] = np.arange(maxfidx + 1)
    paramdat[0] = ('k-term', 0)  # 1
    paramdat[1] = ('fwhm-term-1', 1)  # 2
    paramdat[2] = ('fwhm-term-2', 2)  # 3
    paramdat[3] = ('color-airmass', 3)  # 4
    paramdat['descr'][4:maxnidx + 5] = 'nightly a-term'  # 5
    paramdat['idx'][4:maxnidx + 5] = unight
    paramdat['descr'][5+maxnidx:2*maxnidx + 6] = 'nightly k-term'  # 6
    paramdat['idx'][5+maxnidx:2*maxnidx + 6] = unight
    paramdat['descr'][6+2*maxnidx:7+2*maxnidx+maxfidx] = 'f-term'  # 7
    paramdat['idx'][6+2*maxnidx:7+2*maxnidx+maxfidx] = np.arange(maxfidx + 1)
    paramdat['descr'][7+2*maxnidx+maxfidx:8+2*maxnidx+2*maxfidx] = 'color-flat'  # 8
    paramdat['idx'][7+2*maxnidx+maxfidx:8+2*maxnidx+2*maxfidx] = np.arange(maxfidx + 1)

    print('negated the dairmass term')
    nstar = np.sum([len(kwargs[hp]['uobjid'])
                    for hp in kwargs['hplist']])
    nobs = np.sum([np.sum(kwargs[hp]['weight'] > 0)
                   for hp in kwargs['hplist']])
    print(f'Solving {nobs/10**6:.1f}M observations of {nstar/10**6:.1f}M stars')
    if guess is None:
        guess = np.zeros(maxpidx+1, dtype='f4')
    # from jax.scipy.optimize import minimize
    # res = minimize(loss, guess, args=(kwargs,), method='BFGS')
    from scipy.optimize import minimize
    # from jax import grad
    fun = loss_generic_grouped
    kwargs['grad'] = grad_loss_generic_np_global2
    global_saved_data['kwargs'] = kwargs
    pastinfo = []

    if options is None:
        options = dict(iprint=1)

    for i in range(clipiter):
        if nproc > 1:
            # from concurrent.futures import ProcessPoolExecutor
            # executor = ProcessPoolExecutor(max_workers=nproc)
            executor = multiprocessing.Pool(nproc)
            kwargs['executor'] = executor
        else:
            executor = None

        options['maxiter'] = optiter
        res = minimize(fun, guess, args=(kwargs,),
                       jac=gradient_loss_generic_grouped_pool,
                       method=method,
                       options=options)
        guess = res.x
        imstats = image_stats(obj, res.x, kwargs, nproc=nproc)
        clip_stars(obj, res.x, kwargs, imstats, badfun=badfun,
                   clipscale=2**(clipiter-i-1))
        # need to reach into kwargs again
        compute_weights_and_averages(obj)  # recompute weights
        for hp in kwargs['hplist']:
            kwargs[hp]['weight'] = obj[hp]['weight']
            kwargs[hp]['instmag'] = obj[hp]['instmag']
            kwargs[hp]['avginstmag'] = obj[hp]['avginstmag']
        pastinfo.append((res, imstats))

        if nproc > 1:
            # kwargs['executor'].shutdown()
            kwargs['executor'].terminate()

    return res, imstats, paramdat, pastinfo



def solve_dr11_simple(obj, nproc=1, maxiter=100000, method='L-BFGS-B',
                      options=None):
    kwargs = dict()

    kwargs['hplist'] = list(obj.keys())
    unight = compute_unight(obj)
    maxfidx = 0
    for hp in obj.keys():
        kwargs[hp] = dict()
        tobj = obj[hp]
        uobjid = np.unique(tobj['objid'])
        oidx = np.searchsorted(uobjid, tobj['objid'])
        kwargs[hp]['objid'] = tobj['objid']
        kwargs[hp]['weight'] = tobj['weight']
        kwargs[hp]['instmag'] = tobj['instmag']
        kwargs[hp]['avginstmag'] = tobj['avginstmag']
        kwargs[hp]['oidx'] = oidx
        kwargs[hp]['dmag'] = tobj['dmag']
        kwargs[hp]['uobjid'] = uobjid
        obj[hp]['idx'] = compute_nidx(tobj['mjd_obs'], unight)
        maxfidx = max(maxfidx, np.max(tobj['fidx'], initial=0))
    maxnidx = len(unight) - 1
    maxpidx = 0
    for hp in obj.keys():
        tobj = obj[hp]
        kwargs[hp]['idxlist'] = (
            0, 1, 2, 3 + tobj['idx'])
        # k term, fwhm, fwhm**2, nightly
        dairmass = tobj['airmass'] - 1.1
        dcolor = np.clip(tobj['bp_rp'] - 0.5, -2, 2)
        dfwhm = np.clip(tobj['fwhm'] - 1.1, -1, 3)
        kwargs[hp]['coefflist'] = (dairmass,
                                   dfwhm, dfwhm**2,
                                   1)
        maxpidx = max(maxpidx, np.max([
            np.max(x, initial=0) for x in kwargs[hp]['idxlist']]))
    nstar = np.sum([np.sum(kwargs[hp]['avginstmag'] != 0)
                    for hp in kwargs['hplist']])
    nobs = np.sum([np.sum(kwargs[hp]['weight'] > 0)
                   for hp in kwargs['hplist']])
    print(f'Solving {nobs/10**6:.1f}M observations of {nstar/10**6:.1f}M stars')
    guess = np.zeros(maxpidx+1, dtype='f4')
    # from jax.scipy.optimize import minimize
    # res = minimize(loss, guess, args=(kwargs,), method='BFGS')
    from scipy.optimize import minimize
    # from jax import grad
    fun = loss_generic_grouped
    kwargs['grad'] = grad_loss_generic_np_global2
    global_saved_data['kwargs'] = kwargs

    if nproc > 1:
        # from concurrent.futures import ProcessPoolExecutor
        # executor = ProcessPoolExecutor(max_workers=nproc)
        executor = multiprocessing.Pool(nproc)
        kwargs['executor'] = executor

    if options is None:
        options = dict(iprint=1)
    options['maxiter'] = maxiter
    res = minimize(fun, guess, args=(kwargs,),
                   jac=gradient_loss_generic_grouped_pool,
                   method=method,
                   options=options)

    if nproc > 1:
        # kwargs['executor'].shutdown()
        kwargs['executor'].terminate()

    return res


def solve_dr11_ungrouped(obj, nproc=1):
    kwargs = dict()

    uobjid = np.unique(obj['objid'])
    oidx = np.searchsorted(uobjid, obj['objid'])
    kwargs['objid'] = obj['objid']
    kwargs['weight'] = obj['weight']
    kwargs['instmag'] = obj['instmag']
    kwargs['avginstmag'] = obj['avginstmag']
    kwargs['oidx'] = oidx
    kwargs['dmag'] = obj['dmag']
    kwargs['uobjid'] = uobjid
    unight = np.unique(mjdnum(obj['mjd_obs']))
    obj['idx'] = compute_nidx(obj['mjd_obs'], unight)
    maxnidx = len(unight) - 1
    maxfidx = np.max(obj['fidx'])

    kwargs['idxlist'] = (
        0, 1, 2, 3, 4 + obj['idx'],
        5 + maxnidx + obj['fidx'],
        6 + maxnidx + maxfidx + obj['fidx'])
    # k term, fwhm, fwhm**2, color-airmass, nightly, flat field,
    # color-flat
    dairmass = obj['airmass'] - 1.1
    dcolor = obj['bp_rp'] - 0.5
    dfwhm = obj['fwhm'] - 1.1
    kwargs['coefflist'] = (dairmass,
                           dfwhm, dfwhm**2,
                           dcolor * dairmass,
                           1, 1, dcolor)
    maxpar = np.max([np.max(x) for x in kwargs['idxlist']])
    guess = np.zeros(maxpar+1, dtype='f4')
    # from jax.scipy.optimize import minimize
    # res = minimize(loss, guess, args=(kwargs,), method='BFGS')
    from scipy.optimize import minimize
    # from jax import grad
    fun = loss_generic
    res = minimize(fun, guess, args=(kwargs,),
                   jac=grad_loss_generic,
                   method='L-BFGS-B')
    return res


def calibration_simple(obj, param, nidx=0):
    return (param[1+obj['idx']] + param[1+nidx+obj['fidx']] -
            param[0] * obj['airmass'])


def fake_data(nobj=1000, ndet=2, nnight=3, outlierfrac=0.01, nfidx=8192):
    res = np.zeros(ndet * nobj, dtype=catdtype)
    res['objid'] = (
        np.random.uniform(size=len(res)) * nobj).astype('u8')
    truemag = np.random.uniform(size=nobj) * 10
    res['idx'] = (np.random.uniform(size=len(res)) * nnight).astype('i4')
    res['fidx'] = (np.random.uniform(size=len(res)) * nfidx).astype('i4')
    res['airmass'] = np.random.uniform(1, 2, size=len(res))
    param = np.random.uniform(-1, 1, size=1+nnight+nfidx)
    res['dmag'] = 0.01
    outliers = np.random.uniform(size=len(res)) < outlierfrac
    outliernoise = np.random.randn(len(res)) * 10
    res['instmag'] = (truemag[res['objid']] +
                      -calibration_simple(res, param, nidx=nnight) +
                      np.random.randn(len(res)) * 0.01 +
                      outliers * outliernoise)
    res.sort(order='objid')
    compute_weights_and_averages(res)
    return res, param


def fake_measurements_real_data(obj, scale_errors=1):
    unight = compute_unight(obj)
    nnight = len(unight)
    nfidx = np.max([np.max(obj[i]['fidx']) for i in obj]) + 1
    np.random.seed(1)
    param = np.random.uniform(-1, 1, size=1+nnight+nfidx)
    for i in obj:
        uobjid = np.unique(obj[i]['objid'])
        truemag = np.random.uniform(size=len(uobjid))*10
        oidx = np.searchsorted(uobjid, obj[i]['objid'])
        obj[i]['idx'] = compute_nidx(obj[i]['mjd_obs'], unight)
        instmag = (
            truemag[oidx] - calibration_simple(obj[i], param, nidx=nnight) +
            np.random.randn(len(obj[i])) * obj[i]['dmag'] * scale_errors)
        avginstmag = compute_average(
            instmag, obj[i]['weight'], obj[i]['objid'])
        obj[i]['instmag'] = instmag
        obj[i]['avginstmag'] = avginstmag
    return param


def fill_det(i, param, dtype):
    obj = global_saved_data['obj'][i]
    kwargs = global_saved_data['kwargs'][i]
    n = len(obj['dmag'])
    out = np.zeros(n, dtype=dtype)
    magd = get_mags(obj, param, kwargs)
    out['cal'] = magd['cal']
    out['avg'] = magd['avg']
    out['neff'] = magd['neff']
    out['dmag'] = obj['dmag']
    index = out.dtype.names[4]  # brittle!
    out[index] = obj[index]
    out['bmask'] = obj['bmask']
    out['ccdnum'] = obj['ccdnum']
    return out


def exp_image_stats(shm, ndet, f, l, outdtype, detdtype):
    shm = multiprocessing.shared_memory.SharedMemory(name=shm)
    det = np.ndarray(ndet, dtype=detdtype, buffer=shm.buf)
    ind = det['s'][f:l]
    index = outdtype.names[0]
    nexp = len(np.unique(det[index][ind]))
    if nexp > 1:
        ss = util_efs.subslices(det[index][ind])
        return np.concatenate([
            exp_image_stats(shm.name, ndet, f+f2, f+l2, outdtype,
                            detdtype)
            for (f2, l2) in ss])
    out = np.zeros(1, dtype=outdtype)
    keep = det['neff'][ind] > 0
    # intentionally do not use the bmasks here
    # trust instead median & mad_std to be robust.
    # note that the mean magnitudes _do_ use the bmasks
    # so we shouldn't be including bad data in the means
    # but we may intentionally be doing it in the resids
    # for this exposure.  i.e., if this exposure is bad,
    # we want to and should see it.
    out[index] = det[index][ind[0]]

    resid = det['cal'][ind] - det['avg'][ind]
    if np.sum(keep) != 0:
        out['median'] = np.median(resid[keep])
        out['sd'] = stats.mad_std(resid[keep])
        out['n'] = np.sum(keep)
        out['badfrac'] = (
            np.sum(np.abs(resid[keep]) > 3 * out['sd']) /
            len(keep))
    else:
        out['median'] = -999
        out['sd'] = -999
        out['n'] = 0
        out['badfrac'] = 1.

    ss = util_efs.subslices(det['ccdnum'][ind])
    for f, l in ss:
        m = slice(f, l)
        i = det['ccdnum'][ind[f]]
        keep = det['neff'][ind[m]] > 0
        nn = np.sum(keep)
        out['n_chip'][0, i] = nn
        if nn > 0:
            out['median_chip'][0, i] = np.median(
                resid[m][keep] - out['median'])  # offset relative to chip
            out['sd_chip'][0, i] = stats.mad_std(resid[m][keep])
        else:
            out['median_chip'][0, i] = -999
            out['sd_chip'][0, i] = -999

    if ~np.isfinite(out['median']):
        print('Bad median!')
    return out


def image_stats(obj, param, kwargs, index='expnum', nproc=1):
    print('add something for rejected nights to give them '
          'a default aterm?')
    # start by making a giant flat thing in memory; ugh.
    ndet = np.sum([len(obj[i]['dmag']) for i in obj])
    dtype = [('cal', 'f4'), ('avg', 'f4'), ('neff', 'i4'),
             ('dmag', 'f4'), (index ,'i4'), ('bmask', 'i4'),
             ('ccdnum', 'i4'), ('s', 'i4')]
    size = ndet * np.dtype(dtype).itemsize
    shm = multiprocessing.shared_memory.SharedMemory(
        create=True, size=size)
    det = np.ndarray(ndet, dtype=dtype, buffer=shm.buf)
    # det = np.ndarray(ndet, dtype=dtype)
    starts = np.cumsum([len(obj[i]['dmag']) for i in obj])
    starts = np.concatenate([[0], starts[:-1]])

    global_saved_data['obj'] = obj
    global_saved_data['kwargs'] = kwargs

    if nproc == 1:
        pool = None
    else:
        pool = multiprocessing.Pool(nproc)

    if pool is None:
        for i, start in zip(obj, starts):
            det[start:start+len(obj[i])] = fill_det(
                i, param, det.dtype)
    else:
        args = [(i, param, det.dtype)
                for i in obj]
        for start, det0 in zip(starts, pool.starmap(fill_det, args)):
            det[start:start+len(det0)] = det0

    s = np.lexsort([det['ccdnum'], det[index]])
    det['s'] = s
    ss = util_efs.subslices(det[index][s])
    out = np.zeros(len(ss), dtype=[
        (index, 'i4'), ('mjd_obs', 'f8'), ('median', 'f4'),
        ('sd', 'f4'), ('bmask', 'i4'),
        ('n', 'i4'), ('badfrac', 'f4'),
        ('median_chip', '64f4'), ('sd_chip', '64f4'),
        ('n_chip', '64f4'), ('bmask_chip', '64i4')])
    if pool is None:
        for i, (f, l) in enumerate(ss):
            out[i:i+1] = exp_image_stats(shm.name, ndet, f, l, out.dtype, det.dtype)
    else:
        starts = np.concatenate([[0], ss.uind[:-1] + 1])
        starts = starts[::1000]
        lasts = np.concatenate([starts[1:], [len(det)]])
        args = [(shm.name, ndet, f, l, out.dtype, det.dtype) for f, l in zip(starts, lasts)]
        out = np.concatenate(pool.starmap(exp_image_stats, args))

    for i in obj:
        ind = np.searchsorted(out['expnum'], obj[i]['expnum'])
        uind, loc = np.unique(ind, return_index=True)
        out['mjd_obs'][ind[loc]] = obj[i]['mjd_obs'][loc]

    if pool is not None:
        pool.terminate()

    shm.close()
    shm.unlink()

    return out

    # what do we do when all of the observations on a night
    # or on an image are clipped?
    # -> we still want to get a calibration for the exposure
    # -> any nights with no weight get default parameters??
    # I think these just get zero for all parameters, which is
    # actually a sensible default in all cases right now?


def get_mags(obj, param, kwargs):
    zp = compute_zeropoint(param, kwargs)
    cal = obj['instmag'] + zp
    avg = compute_average(cal, obj['weight'], obj['objid'])
    neff = compute_neffobs(obj['weight'], obj['objid'])
    ntot = compute_neffobs(obj['weight'] * 0 + 1, obj['objid'])
    # lots of duplicated calculations in previous three lines,
    # could be made more efficient
    return dict(cal=cal, avg=avg, neff=neff, ntot=ntot)


def clip_stars(obj, param, kwargs, imstats, index='expnum', badfun=None,
               clipscale=1):
    # image masking
    okstats = imstats['n'] > 0
    medmed = np.median(imstats['median'][okstats])
    medsd = np.median(imstats['sd'][okstats])
    sdmed = stats.mad_std(imstats['median'][okstats])
    sdsd = stats.mad_std(imstats['sd'][okstats])
    imstats['bmask'] = 0
    imstats['bmask'] |= badmask['image_median'] * (
        np.abs(imstats['median'] - medmed) > 5 * sdmed * clipscale)
    imstats['bmask'] |= badmask['image_scatter'] * (
        np.abs(imstats['sd'] - medsd) > 5 * sdsd * clipscale)
    imstats['bmask'] |= badmask['image_noobs'] * (imstats['n'] == 0)
    imstats['bmask'] |= badmask['image_manybad'] * (imstats['badfrac'] > 0.5)
    imstats['bmask_chip'] |= badmask['image_chip'] * (
        (imstats['n_chip'] > 0) &
        (np.abs(imstats['median_chip']) > imstats['sd'][:, None] * 5 * clipscale))
    imclipfrac = np.sum(imstats['bmask'] != 0) / len(imstats)
    print(f'Typical image medians: mu={medmed:.4f}, sd={sdmed:.4f}')
    print(f'Typical image sds: mu={medsd:.4f}, sd={sdsd:.4f}')
    print(f'Clipped {imclipfrac*100:.1f}% of images.')

    if badfun is not None:
        imstats['bmask'] |= badfun(imstats['mjd_obs']) * badmask['image_manual']
        imclipfrac = np.sum(imstats['bmask'] != 0) / len(imstats)
        print(f'Clipped {imclipfrac*100:.1f}% of images after bad night mask.')

    s = np.argsort(imstats['expnum'])
    imstats = imstats[s]

    for i in obj:
        obj[i]['bmask'] &= ~image_masks  # clear image bad mask

    for i in obj:
        ind = np.searchsorted(imstats['expnum'], obj[i]['expnum'])
        obj[i]['bmask'] |= imstats['bmask'][ind]

    # additional chip masking: TODO

    # additional observation-by-observation masking
    # this is a bit janky in that it groups by partition
    # rather than by something physical.
    # It would be better to loop over the exposures,
    # and then iterate over the number of observations each star
    # got on that exposure, but that's not what we do here.
    # nobs = 0
    # nclipobs = 0
    # for i in obj:
    #     magd = get_mags(obj[i], param, kwargs[i])
    #     s = np.argsort(magd['neff'])
    #     for f, l in util_efs.subslices(magd['neff'][s]):
    #         ind = s[f:l]
    #         obj[i]['bmask'][ind] &= ~star_masks  # clear per-star mask
    #         if magd['neff'][ind[0]] <= 1:
    #             # this star has <=1 good observation remaining.
    #             # we can't use it.
    #             obj[i]['bmask'][ind] |= badmask['star_noobs']
    #         else:
    #             eind = np.searchsorted(imstats['expnum'], obj[i]['expnum'][ind])
    #             zpoff = imstats['median'][eind]
    #             imsd = imstats['sd'][eind]
    #             mbadoff = imsd < 0
    #             imsd = np.clip(imsd, 0, np.inf)
    #             zpoff[mbadoff] = 0
    #             resid = magd['cal'][ind] - magd['avg'][ind] - zpoff
    #             med = np.median(resid)
    #             sd = stats.mad_std(resid)
    #             obj[i]['bmask'][ind] |= badmask['star_resid'] * (
    #                 np.abs(resid - med) > 5 * np.hypot(sd, imsd))
    #     nobs += len(obj[i])
    #     nclipobs += np.sum(obj[i]['bmask'] != 0)
    # RELY ON ROBUST CHI2 MINIMIZATION INSTEAD?!
    # the issue here is that one chip can be bad by ~6 mag.
    # we want to detect that case.
    # here all the stars on that chip get removed from all exposures
    # and then that chip ends up looking "okay" but with a weird
    # number of stars.  That's a problem.
    # print(f'Clipped {nclipobs*100/nobs:.1f}% of observations.')

    # # additional observation-by-observation masking
    # s = np.argsort(imstats[index])
    # imstats = imstats[s]
    # magds = []
    # for i in obj:
    #     magds.append(get_mags(obj[i], param, kwargs[i]))
    # calmags = np.concatenate([magd['cal'] for magd in magds])
    # avgmags = np.concatenate([magd['avg'] for magd in magds])
    # neffs = np.concatenate([magd['neff'] for magd in magds])
    # resid = calmags - avgmags
    # dmags = np.concatenate([obj[i]['dmag'] for i in obj])
    # idxs = np.concatenate([obj[i]['expnum'] for i in obj])
    # s = np.lexsort((neffs, idxs))
    # uind = util_efs.unique_multikey((neffs[s], idxs[s]), (0, 1))
    # for f, l in util_efs.subslices(neffs[s], uind=uind):
    #     ind = s[f:l]
    #     med = np.median(resid[ind])
    #     sd = mad_std(resid[ind])
    #     # ugh.  Must put this back in the right obj healpix!  No fun,
    #     # not doing this.
    #







# def compute_stats(obj, param, kwargs):
#     zps = dict()
#     avgs = dict()
#     for i in obj:
#         if len(obj[i]) == 0:
#             zps[i] = []
#             avgs[i] = []
#             continue
#         zps[i] = compute_zeropoint(param, kwargs[i])
#         avgs[i] = compute_average(
#             kwargs[i]['instmag'] + zps[i], kwargs[i]['weight'],
#             kwargs[i]['objid'])
#     calmags = np.concatenate([kwargs[i]['instmag'] + zps[i] for i in obj])
#     avgmags = np.concatenate([avgs[i] for i in obj])
#     resid = calmags - avgmags
#     dmags = np.concatenate([obj[i]['dmag'] for i in obj])
#     stats = dict()
#     stats['median'] = np.median(calmags - avgmags)
#     stats['mad_std'] = stats.mad_std(calmags - avgmags)
#     idxlist = list()
#     for i in obj:
#         for j, idx in enumerate(kwargs[i]['idxlist']):
#             if len(idxlist) <= j:
#                 idxlist.append([idx])
#                 continue
#             if np.ndim(idx) == 0:
#                 continue
#             else:
#                 idxlist[j].append(idx)
#     for i, idx in enumerate(idxlist):
#         if len(idx) == 1:
#             idxlist[i] = idxlist[i][0]
#         else:
#             idxlist[i] = np.concatenate(idxlist[i])
#     # don't really need the coeffs here?
#     dtype = [('median', 'f4'), ('mad_std', 'f4'), ('n', 'i4'), ('idx', 'i4')]
#     stats['by_index'] = list()
#     for idx in idxlist:
#         if np.ndim(idx) == 0:
#             out = np.zeros(1, dtype=dtype)
#             out['median'] = np.median(resid)
#             out['mad_std'] = stats.mad_std(resid)
#             out['n'] = len(resid)
#             stats['by_index'].append(out)
#         else:
#             s = np.argsort(idx)
#             ss = util_efs.subslices(idx[s])
#             resid0 = resid[s]
#             out = np.zeros(len(ss), dtype=dtype)
#             for i, (f, l) in enumerate(ss):
#                 idx0 = idx[s[f]]
#                 out['median'][i] = np.median(resid0[f:l])
#                 out['mad_std'][i] = stats.mad_std(resid0[f:l])
#                 out['n'][i] = l - f
#                 out['idx'][i] = idx0
#             stats['by_index'].append(out)
#     idx = np.concatenate([obj[i]['expnum'] for i in obj])
#     s = np.argsort(idx)
#     ss = util_efs.subslices(idx[s])
#     resid0 = resid[s]
#     out = np.zeros(len(ss), dtype='f4')
#     for i, (f, l) in enumerate(ss):
#         idx0 = idx[s[f]]
#         out['median'][i] = np.median(resid0[f:l])
#         out['mad_std'][i] = stats.mad_std(resid0[f:l])
#         out['n'][i] = l-f
#         out['idx'][i] = idx0
#     return stats

# every hundred iterations we need to go through and remove outlying
# problematic images.


# what approach do we want to use to do the math?
# options:
# - same as we did in ubercal, make some matrices and invert them
#   - somewhat annoying to make this generic if I want to add some
#     new terms.
# - giant sparse matrix
#   - probably most generic and easy to write
#   - will be more memory consuming than necessary
#     but won't be that bad.
#     in principle, can add the resulting A^t C^-1 A's together.
# - non-linear solution
#   - e.g., via least_squares
#   - ideally, would implement the gradients, since most of these are cheap
#   - or via jax; would take me needing to learn whether this kind of
#     operation can be auto-differenced
# ugh.  Let's try the cool jax way.


# Let's start by something like least_squares but evaluate the
# residuals using jax.

# the data is the measured fluxes
# the uncertainties are the uncertainties
# what's the model?  The model is the sum of:
#   - the average flux with some weights
#   - the average of some calibration terms
#   - the calibration terms for this exposure
# it would be good to precompute the relevant weights for each object.


def fpim(fpvec, ccdfac=16*8, nx=256, ny=256, nxv=8, nyv=16):
    imnx = 4094
    imny = 2046
    xpc, ypc = [int(np.ceil(imn / 1. / pn))
                for (imn, pn) in zip([imnx, imny], [nx, ny])]
    nrow = 7
    ncol = 12
    out = np.zeros((nrow*xpc, ncol*ypc), dtype=fpvec.dtype)
    nprow = [3, 4, 5, 6, 6, 7, 7, 6, 6, 5, 4, 3]
    nccd = 62
    row0 = 0
    col0 = 0
    count = 0
    fpvec = fpvec[ccdfac:]  # no CCD 0; first ccdfac are alway all 0.
    if (len(fpvec) % (ccdfac)) != 0:
        newsize = int(np.ceil(len(fpvec)/1./ccdfac)*ccdfac)
        fpvec2 = np.zeros(newsize, dtype=fpvec.dtype)
        fpvec2[:len(fpvec)] = fpvec
        fpvec = fpvec2
    while row0 < len(nprow):
        ccdy = row0*ypc
        startx = int(xpc*(7-nprow[row0])/2.)
        ccdx = startx + col0*xpc
        fpvecout = fpvec[count:count+ccdfac].reshape(nyv, nxv)
        out[ccdx:ccdx+xpc, ccdy:ccdy+ypc] = fpvecout[:xpc, :ypc]
        # reshape(nyv, nxv), and fpvecout.T if x <-> y.
        count += ccdfac
        col0 += 1
        if col0 >= nprow[row0]:
            col0 = 0
            row0 += 1
    return out
