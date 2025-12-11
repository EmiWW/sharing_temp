import numpy as np
import os
import pathlib
from scipy import interpolate
from bisect import bisect_left


# From original JADE code, with 2D interpolator
# E.g., `_rho2d` uses `_rho1d` to linearly interpolate 'rho' (gas density) twice,  first with Pressure and then with Teperature
#
class EOS:
    def __init__(self, path):
        self.read_dat(path)
        self.rho = self._rho2d
        self.adg = self._adg2d

    def read_dat(self, path):
        T = []
        P = []
        rho = []
        adg = []
        for line in open(path, "r"):
            tmp = line.rstrip().split()
            if len(tmp) == 2:
                T.append(float(tmp[0]))
                N = int(tmp[1])
                _P = np.zeros(N)
                _rho = np.zeros(N)
                _adg = np.zeros(N)
                idx = 0
            else:
                _P[idx] = float(tmp[0])
                _rho[idx] = float(tmp[3])
                _adg[idx] = float(tmp[10])
                idx += 1
                if idx == N:
                    P.append(_P)
                    rho.append(_rho)
                    adg.append(_adg)
        self._T = np.asarray(T)
        self._P = np.asarray(P)
        self._rho = np.asarray(rho)
        self._adg = np.asarray(adg)

    def _rho1d(self, idxT, P):
        idxP = bisect_left(self._P[idxT], P)
        if idxP == len(self._P[idxT]):
            rho2 = self._rho[idxT][-1]
            rho1 = self._rho[idxT][-2]
            P2 = self._P[idxT][-1]
            P1 = self._P[idxT][-2]
            return rho2 + (rho2 - rho1) * (P - P2) / (P2 - P1)
        if idxP == 0:
            idxP += 1
        rho2 = self._rho[idxT][idxP]
        rho1 = self._rho[idxT][idxP - 1]
        P2 = self._P[idxT][idxP]
        P1 = self._P[idxT][idxP - 1]
        return rho1 + (rho2 - rho1) * (P - P1) / (P2 - P1)

    def _adg1d(self, idxT, P):
        idxP = bisect_left(self._P[idxT], P)
        if idxP == len(self._P[idxT]):
            adg2 = self._adg[idxT][-1]
            adg1 = self._adg[idxT][-2]
            P2 = self._P[idxT][-1]
            P1 = self._P[idxT][-2]
            return adg2 + (adg2 - adg1) * (P - P2) / (P2 - P1)
        if idxP == 0:
            idxP += 1
        adg2 = self._adg[idxT][idxP]
        adg1 = self._adg[idxT][idxP - 1]
        P2 = self._P[idxT][idxP]
        P1 = self._P[idxT][idxP - 1]
        return adg1 + (adg2 - adg1) * (P - P1) / (P2 - P1)

    def _rho2d(self, T, P):
        idxT = bisect_left(self._T, T)
        if idxT == len(self._T):
            rho2 = self._rho1d(-1, P)
            rho1 = self._rho1d(-2, P)
            T2 = self._T[-1]
            T1 = self._T[-2]
            return rho2 + (rho2 - rho1) * (T - T2) / (T2 - T1)
        if idxT == 0:
            idxT += 1
        rho2 = self._rho1d(idxT, P)
        rho1 = self._rho1d(idxT - 1, P)
        T2 = self._T[idxT]
        T1 = self._T[idxT - 1]
        return rho1 + (rho2 - rho1) * (T - T1) / (T2 - T1)

    def _adg2d(self, T, P):
        idxT = bisect_left(self._T, T)
        if idxT == len(self._T):
            adg2 = self._adg1d(-1, P)
            adg1 = self._adg1d(-2, P)
            T2 = self._T[-1]
            T1 = self._T[-2]
            return adg2 + (adg2 - adg1) * (T - T2) / (T2 - T1)
        if idxT == 0:
            idxT += 1
        adg2 = self._adg1d(idxT, P)
        adg1 = self._adg1d(idxT - 1, P)
        T2 = self._T[idxT]
        T1 = self._T[idxT - 1]
        return adg1 + (adg2 - adg1) * (T - T1) / (T2 - T1)


# Interplate that we created
# This "3D interpolator" makes a list of 2D interpolators (using scipy.interpolate.RectBivariateSpline) for different metallicity tables.
# To estimate opacity for a given temperature, density, and metallicity,
# it selects the 2D interpolator corresponding to the closest metallicity value and performs the interpolation.


class OpacitiesFerguson:
    # Constructor, must give Helium fraction (He_fraction) and global path
    # -------------------------------------------
    def __init__(self, path, He_fraction):
        self.He_fraction = He_fraction
        self.file_lookup_table = []
        self.best_file = None
        # Use the table updated to match Asplund et al. 2021 solar abundance compilcation
        path += "opacities/{}/".format("asplund21")
        self.H_values = []
        self.metal_values = []
        self.temperature = []
        self.all_interpolators = []
        # All opacity (kappa) tables share the same density grid.
        # Set it once for efficiency instead reading from tables repeatedly.
        self.density_grid = np.round(np.linspace(-8.0, 2.0, 51), 1)

        self.read_files_and_create_interpolators(path)

        # Interpolator is now ready to be used with the 'kappa' method:
        # e.g. current_opacity = self.kappa(pressure, gas_density, metallicity)

    def read_files_and_create_interpolators(self, path):
        path = pathlib.Path(path)
        csv_files = list(path.glob("*.csv"))
        # sort tables in ascending order by filename: H fraction & metal fraction
        # because file system read files in arbitary orders, which could change computation
        csv_files.sort()

        # Create a temporary dictionary to map H values to corresponding metal values and
        # interpolators. This keeps the irregular 2D array structure of H and metal values.
        #
        lookup_table = {}
        for file in csv_files:
            temp = os.path.basename(file).split(".")
            H_fraction_str = temp[1]
            metal_fraction = temp[2]
            H_fraction = float("0." + H_fraction_str)
            metal_fraction = float("0." + metal_fraction)

            if H_fraction not in self.H_values:
                self.H_values.append(H_fraction)

            # make interpolator for each file
            # i.e. one interpolator for the (H_fraction, metal_fraction) pair
            interpolator = self.read_csv_file_to_grid(file)
            data = (metal_fraction, interpolator)
            key = int(H_fraction_str)
            if key in lookup_table:
                # key exists → update its list
                lookup_table[key].append(data)
            else:
                # key missing → create a new list
                lookup_table[key] = [data]
        #            break
        # Extract the 2D irregular arrays of metal values and interpolators
        # both are indexed by the corresponding index in self.H_values for the H fraction
        for key, val in lookup_table.items():
            self.metal_values.append([x[0] for x in val])
            self.all_interpolators.append([x[1] for x in val])

    def select_interpolator(self, He_fraction, metal_fraction):
        H_fraction = self.compute_H_fraction(He_fraction, metal_fraction)
        # find the index in H_values closest to the computed H_fraction
        H_index = self.find_index(H_fraction, self.H_values)
        # within the chosen H_index, find the index in metal_values closest to the input metal_fraction
        metal_index = self.find_index(metal_fraction, self.metal_values[H_index])
        # select the interpolar corresponding to the chosen H_index (computed H_fraction) and metal_index (input metal_fraction)
        interpolator = self.all_interpolators[H_index][metal_index]
        # returns index of the best file to use as the interpolator
        return interpolator

    def compute_H_fraction(self, He_fraction, metal_fraction):
        # planetary atmosphere composition: H + He + metals (mostly H2O) = 1
        return round(1.0 - He_fraction - metal_fraction, 5)

    def read_csv_file_to_grid(self, path):
        data = np.genfromtxt(path, delimiter=" ", skip_header=0, comments="#")
        temperature = np.sort(data[:, 0])
        kappa = np.flipud(data[:, 1:])
        return interpolate.RectBivariateSpline(temperature, self.density_grid, kappa).ev

    def find_index(self, target, sorted_list):
        # Compute absolute distance from target to the two closest values
        left_index, right_index = self.find_index_bounds(target, sorted_list)
        # find_index_bounds finds the insertion point between left_index and right_index.
        # hence we need to -1 to get the right index of the closest value
        right_index -= 1
        dist_left = abs(target - sorted_list[left_index])
        dist_right = abs(target - sorted_list[right_index])
        if dist_left < dist_right:
            return left_index
        elif dist_right < dist_left:
            return right_index
        else:
            return right_index

    def find_index_bounds(self, target, sorted_list):
        assert target >= sorted_list[0]
        assert target <= sorted_list[-1]

        left_index = 0
        right_index = len(sorted_list)
        for index, x in enumerate(sorted_list):
            if target >= x:
                left_index = index
            if target <= x:
                right_index = index
                break
        right_index += 1
        return (left_index, right_index)

    def kappa(self, T, _P, rho, metal_fraction):  # P is unused
        logT = np.log10(T)
        logrho = np.log10(rho)
        # R: density / opacity / rosseland paramater (NOT radius)
        # R = rho / (T / 1e6 )^3
        logR = logrho - 3 * (logT - 6.0)
        kappa_func = self.select_interpolator(self.He_fraction, metal_fraction)
        kappa = 10 ** (kappa_func(logT, logR).item())
        return float(kappa)
