import pickle
import numpy as np
import util_efs


class BadNight:
    def __init__(self, filename, mjd_fun=None):
        self.badlist = pickle.load(open(filename, 'rb'))
        self.filename = filename
        self.mjd_fun = mjd_fun

    def write(self):
        pickle.dump(self.badlist, open(self.filename, 'wb'))

    def isbad(self, night, time=None):
        if night not in self.badlist:
            return False
        for badtime in self.badlist[night]:
            if (time is None or (time > badtime[0] and time < badtime[1]) or
                (time > badtime[0] + 1 and time < badtime[1] + 1) or
                (time > badtime[0] - 1 and time < badtime[1] - 1)):
                return True
        return False

    def add_badnight(self, night, time=None):
        if time is None:
            time = [0.,1.]
        if len(time) != 2:
            raise ValueError('len(time) must equal 2')
        if night not in self.badlist:
            self.badlist[night] = [time]
        else:
            self.badlist[night].append(time)
        self.write()

    def remove_badnight(self, night, time=None):
        if night in self.badlist:
            if time is not None:
                if time in self.badlist[night]:
                    self.badlist[night].remove(time)
                else:
                    print('Cannot find time during night %d' % night)
            else:
                del self.badlist[night]
        self.write()

    def badtimes(self, night):
        return self.badlist.get(night, [ ])

    def checkbad(self, mjd):
        mjd = np.atleast_1d(mjd)
        if len(mjd) <= 0:
            return 0
        if self.mjd_fun is not None:
            nindx = self.mjd_fun(mjd)
        else:
            nindx = mjd.astype('i4')
        bad = np.zeros(len(nindx), dtype='i4')
        nighttime = mjd-np.floor(mjd)
        s = np.argsort(nindx)
        for f,l in util_efs.subslices(nindx[s]):
            ind = s[f:l]
            night = nindx[ind[0]]
            if night not in self.badlist:
                continue
            badtimes = self.badlist[night]
            for start, end in badtimes:
                bad[ind] |= (nighttime[ind] >= start) & (nighttime[ind] <= end)
                bad[ind] |= ((nighttime[ind] >= start+1) &
                             (nighttime[ind] <= end+1))
                bad[ind] |= ((nighttime[ind] >= start-1) &
                             (nighttime[ind] <= end-1))
        return bad

    def writeflat(self, filename='badnight2', clobber=False):
        nrec = np.sum([len(y) for x,y in self.badlist.iteritems()])
        out = np.zeros(nrec, dtype=[('night', 'i4'), ('starttime', 'f4'),
                                       ('endtime', 'f4')])
        i = 0
        for night, badtimes in badlist.iteritems():
            for badtime in badtimes:
                out[i]['night'] = ubercal.mjd_nindx_fun(night)
                out[i]['starttime'] = badtime[0]
                out[i]['endtime'] = badtime[1]
                i += 1
        from astropy.io import fits
        fits.writeto(filename, out, overwrite=True)
