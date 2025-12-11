import numpy as np

## Trilinear interplation routine for regular spaced data


def interpol3d_reg(x, y, z, x_arr, y_arr, z_arr, w_table):
    # w_table is a 3D array with shape (len(x_arr), len(y_arr), len(z_arr))
    ix0 = int(np.floor((x - x_arr[0]) / (x_arr[1] - x_arr[0])))
    ix1 = min(ix0 + 1, len(x_arr) - 1)

    iy0 = int(np.floor((y - y_arr[0]) / (y_arr[1] - y_arr[0])))
    iy1 = min(iy0 + 1, len(y_arr) - 1)

    iz0 = int(np.floor((z - z_arr[0]) / (z_arr[1] - z_arr[0])))
    iz1 = min(iz0 + 1, len(z_arr) - 1)

    return interpol3d_common(
        x, ix0, ix1, y, iy0, iy1, z, iz0, iz1, x_arr, y_arr, z_arr, w_table
    )


def interpol3d_ireg(x, y, z, x_arr, y_arr, z_arr, w_table):
    (ix0, ix1) = bitSearch1d_base(0, len(x_arr) - 1, len(x_arr), x, x_arr)
    (iy0, iy1) = bitSearch1d_base(0, len(y_arr) - 1, len(y_arr), y, y_arr)
    (iz0, iz1) = bitSearch1d_base(0, len(z_arr) - 1, len(z_arr), z, z_arr)

    return interpol3d_common(
        x, ix0, ix1, y, iy0, iy1, z, iz0, iz1, x_arr, y_arr, z_arr, w_table
    )


def interpol3d_common(
    x, ix0, ix1, y, iy0, iy1, z, iz0, iz1, x_arr, y_arr, z_arr, w_table
):
    # get indices for interpolation
    # Calculate the lower nd upper indices (ix0,  ix1) for interpolation in x - dimension
    ## Lower index of x,  assuming uniform spacing
    ### 1. x  -  x_arr[0] gives the distance from the first element
    ### 2. x_arr[1]  -  x_arr[0] gives the uniform spacing between consecutive elements
    ### 3. (x  -  x_arr[0])  /  (x_arr[1]  -  x_arr[0]) a fractional value that give relative position of x from the first grid point (x_arr[0]) in term of grid steps
    # INTEGER,  giving the index of the lower grid point
    ### 5. 1  +  np.floor() Adding 1 to adjusts the zero - based index from the Fortran 1 - based index
    ## Upper index of x,  with min() to clamped to array bounds.

    x0 = x_arr[ix0]
    x1 = x_arr[ix1]
    y0 = y_arr[iy0]
    y1 = y_arr[iy1]
    z0 = z_arr[iz0]
    z1 = z_arr[iz1]

    w000 = w_table[ix0, iy0, iz0]
    w001 = w_table[ix0, iy0, iz1]
    w010 = w_table[ix0, iy1, iz0]
    w100 = w_table[ix1, iy0, iz0]
    w011 = w_table[ix0, iy1, iz1]
    w101 = w_table[ix1, iy0, iz1]
    w110 = w_table[ix1, iy1, iz0]
    w111 = w_table[ix1, iy1, iz1]

    w_arr = [w000, w001, w010, w011, w100, w101, w110, w111]

    return interpol_3d_lin_basic(x, y, z, x0, x1, y0, y1, z0, z1, w_arr)


def bitSearch1d_base(imin, imax, n, target, searchlist):
    l = imin
    r = imax

    for i in range(1, n):
        # check if l / r is the target
        if r - l <= 1:
            if searchlist[r] == target:
                index_low = r
                index_high = r
                break
            elif searchlist[l] == target:
                index_low = l
                index_high = l
                break
            else:
                index_low = l
                index_high = l + 1
                break

        # Test if r - l is even or odd
        # Find mid point between l nd r
        if (r + l) % 2 == 0:
            # even
            mid = (r + l) / 2
        else:
            # odd
            mid = (r + l - 1) / 2
        mid = int(mid)
        # check if target is below or above midpoint
        # set new r / l boundary
        if searchlist[mid] < target:
            l = mid
        elif searchlist[mid] > target:
            r = mid
        else:
            index_low = mid
            index_high = mid
            break

    return index_low, index_high


def interpol_1d_lin_basic(x, x0, x1, y0, y1):
    # >  linear interpolation function for scalars
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0)


# *******************************************************************************
def interpol_3d_lin_basic(x, y, z, x0, x1, y0, y1, z0, z1, w_arr):
    # f000, f001, f010, f100, f011, f101, f110, f111)
    # >  trilinear interpolation function for scalars
    # f000 =  f(x0, y0, z0), f001 =  f(x0, y0, z1), f010 =  f(x0, y1, z0), f100 =  f(x1, y0, z0)
    # f011 =  f(x0, y1, z1), f101 =  f(x1, y0, z1), f110 =  f(x1, y1, z0), f111 =  f(x1, y1, z1)

    if x == x0:
        xd = 0.0
    else:
        xd = (x - x0) / (x1 - x0)

    if y == y0:
        yd = 0.0
    else:
        yd = (y - y0) / (y1 - y0)

    if z == z0:
        zd = 0.0
    else:
        zd = (z - z0) / (z1 - z0)

    return interpolate_3d_linear(xd, yd, zd, w_arr)


def interpol_3d_lin_basic2(x, y, z, x0, inv_dx, y0, inv_dy, z0, inv_dz, w_arr):
    # f000 =  f(x0, y0, z0), f001 =  f(x0, y0, z1), f010 =  f(x0, y1, z0), f100 =  f(x1, y0, z0)
    # f011 =  f(x0, y1, z1), f101 =  f(x1, y0, z1), f110 =  f(x1, y1, z0), f111 =  f(x1, y1, z1)

    xd = (x - x0) * inv_dx  # / (x1 - x0)
    yd = (y - y0) * inv_dy  # / (y1 - y0)
    zd = (z - z0) * inv_dz  # / (z1 - z0)

    return interpolate_3d_linear(xd, yd, zd, w_arr)


def interpolate_3d_linear(xd, yd, zd, w_arr):
    f00 = w_arr[0] * (1.0 - xd) + w_arr[4] * xd
    f01 = w_arr[1] * (1.0 - xd) + w_arr[5] * xd
    f10 = w_arr[2] * (1.0 - xd) + w_arr[6] * xd
    f11 = w_arr[3] * (1.0 - xd) + w_arr[7] * xd
    f0 = f00 * (1.0 - yd) + f10 * yd
    f1 = f01 * (1.0 - yd) + f11 * yd

    return f0 * (1.0 - zd) + f1 * zd
