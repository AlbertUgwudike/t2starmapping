from interpol import resize

def polynomial_interpolator(degree):

    def interpolator(B0, n_isochromats):
        r = round(n_isochromats ** (1/3))
        opt = dict(shape=[3*r, 3*r, 3*r], anchor='centers', bound='replicate', interpolation=degree)
        intravoxel = resize(B0, **opt)

        return intravoxel[:, :, r:2*r, r:2*r, r:2*r].reshape(B0.shape[0], n_isochromats).T

    return interpolator