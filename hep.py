#! /usr/bin/env python37

"""Methods for characterizing texture images through
Histogram of Equivalent Patterns (HEP).
"""


import numpy as np

import utils


####################
# HELPER FUNCTIONS #
####################


def square_frame_size(radius):
    """Compute the number of pixels of the external frame of square
    neighbourhoods of the given radii.

    Parameters
    ----------
    radius : list of int
        Radii of the local neighborhoods.

    Returns
    ----------
    points : list of int
        Number of pixels of the local neighborhoods.

    Examples
    --------
    >>> d = RankTransform(radius=[3])
    >>> square_frame_size(d.radius)
    [24]
    >>> square_frame_size([1, 2, 3])
    [8, 16, 24]

    """
    points = [8*r for r in radius]
    return points


def histogram(codes, nbins):
    """Compute the histogram of a map of pattern codes (feature values).

    Parameters
    ----------
    codes : array, dtype=int
        Array of feature values.
        For LCCMSP `codes` is a multidimensional array. It has two layers,
        one for the concave patterns and another for the convex patterns.
    nbins : int
        Number of bins of the computed histogram, i.e. number of
        possible different feature values.

    Returns
    -------
    h_norm : array
        Normalized histogram.

    """
    hist = np.bincount(codes.ravel(), minlength=nbins)
    h_norm = hist/hist.sum()
    return h_norm

#vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv#

#def lexicographic_order(neighbour, central, bandperm=(0, 1, 2), comp=np.less):
#    """
#    Compare a peripheral pixel and the central pixel of the neighbourhood
#    using the lexicographic order.
#
#    Parameters
#    ----------
#    neighbour : array
#        Subimage corresponding to a peripheral pixel of the neighbourhood.
#    central : array
#        Subimage corresponding to the central pixels of the neighbourhood.
#    bandperm : tuple
#        Permutation of the chromatic channels (defined by their indices)
#        which establishes the priority considered in the lexicographic order.
#    comp : Numpy function, optional (default np.less)
#        Comparison function (np.greater, np.less_equal, etc.).
#
#    Returns
#    -------
#    result : boolean array
#        Truth value of comp (neighbour, central) element-wise according to
#        the lexicographic order.
#
#    References
#    ----------
#    .. [1] E. Aptoula and S. Lefêvre
#           A comparative study on multivariate mathematical morphology
#           https://doi.org/10.1016/j.patcog.2007.02.004
#    """
#    weights = np.asarray([256**np.arange(central.shape[2])[::-1]])
#    ord_central = np.sum(central[:, :, bandperm]*weights, axis=-1)
#    ord_neighbour = np.sum(neighbour[:, :, bandperm]*weights, axis=-1)
#    result = comp(ord_neighbour, ord_central)
#    return result
#
#
#def bitmixing_order(neighbour, central, lut, bandperm=(0, 1, 2), comp=np.less):
#    """
#    Compare a peripheral pixel and the central pixel of the neighbourhood
#    using the bit mixing order.
#
#    Parameters
#    ----------
#    neighbour : array
#        Subimage corresponding to a peripheral pixel of the neighbourhood.
#    central : array
#        Subimage corresponding to the central pixels of the neighbourhood.
#    lut : array
#        3D Lookup table
#    bandperm : tuple
#        Permutation of the chromatic channels (defined by their indices)
#        which establishes the priority considered in the bit mixing order.
#    comp : Numpy function, optional (default np.less)
#        Comparison function (np.greater, np.less_equal, etc.).
#
#    Returns
#    -------
#    result : boolean array
#        Truth value of comp (neighbour, central) element-wise according to
#        the bit mixing product order.
#
#    References
#    ----------
#    .. [1] J. Chanussot and P. Lambert
#           Bit mixing paradigm for multivalued morphological filters
#           https://doi.org/10.1049/cp:19971007
#    """
#    ord_central = lut[tuple(central[:, :, bandperm].T)].T
#    ord_neighbour = lut[tuple(neighbour[:, :, bandperm].T)].T
#    result = comp(ord_neighbour, ord_central)
#    return result
#
#
#def refcolor_order(neighbour, central, cref=[0, 0, 0], comp=np.less):
#    """
#    Compare a peripheral pixel and the central pixel of the neighbourhood
#    using the reference color order.
#
#    Parameters
#    ----------
#    neighbour : array
#        Subimage corresponding to a peripheral pixel of the neighbourhood.
#    central : array
#        Subimage corresponding to the central pixels of the neighbourhood.
#    cref : tuple
#        Permutation of the chromatic channels (defined by their indices)
#        which establishes the priority considered in the lexicographic order.
#    comp : Numpy function, optional (default np.less)
#        Comparison function (np.greater, np.less_equal, etc.).
#
#    Returns
#    -------
#    result : boolean array
#        Truth value of comp (neighbour, central) element-wise according to
#        the reference color order.
#
#    .. [1] A. Ledoux, R. Noël, A.-S. Capelle-Laizé and C. Fernandez-Maloigne
#           Toward a complete inclusion of the vector information in
#           morphological computation of texture features for color images
#           https://doi.org/10.1007/978-3-319-07998-1_25
#    """
#    cref = np.asarray(cref).astype(np.int_)
#    dist_central = np.linalg.norm(central - cref)
#    dist_neighbour = np.linalg.norm(neighbour - cref)
#    result =  comp(dist_neighbour, dist_central)
#    return result

#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^#

class HEP(object):
    """Superclass for histogram of equivalent patterns descriptors.

    Notes
    -----
    `order='bitmixing'`
    In this case the triplet of unsigned integers
    [r7r6...r0, g7g6...g0, b7b6...1b0] is converted into an unsigned integer
    r7g7b7r6g6b6...r0g0b0 which is used as the ordering criterion.
    It is important to note that the description above corresponds to the
    default priority (`bands='RGB'`). For other band priority a permutation
    is applied.

    `self.lut` is a look-up table of shape `(256, 256, 256)` which is
    useful to efficiently compute the rank values.


    References
    ----------
    .. [1] A. Fernandez, M. X. Alvarez, and F. Bianconi
           Texture Description through Histograms of Equivalent Patterns
           http://dx.doi.org/10.1007/s10851-012-0349-8

    """
    _canonical_orders = ('linear', 
                         'lexicographic', 
                         'alphamod', 
                         'bitmixing', 
                         'refcolor', 
                         'random')
    _product_orders = ('product',)


    def __init__(self, **kwargs):
        """Initializer of a HEP instance.

        Parameters
        ----------
        kwargs : dictionary
            Keyword arguments listed below:
        order : string
            Order relation used in comparisons.
                'linear':
                    Canonical order for grayscale intensities.
                'product':
                    Product order (is a partial order).
                'lexicographic':
                    Lexicographic order based on priorities between the
                    chromatic components.
                'alphamod':
                    Lexicographic order with the first component
                    divided by alpha.
                'bitmixing':
                    Lexicographic order based on binary representation 
                    of intensities.
                'refcolor':
                    Preorder that relies on the Euclidean distance between
                    a color and a reference color.
                'random':
                    Order based on a random permutation of the
                    lexicographic order.
        radius : list of int, optional (default is [1])
            Radii of the local neighborhoods.
        bands : str, optional (default is 'RGB')
            Color band priorities.
        cref : list of int, optional (default is [0, 0, 0])
            RGB coordinates of the reference color.
        seed : int, optional (default is 0)
            Seed for the pseudo-random number generator.

        """
        self.radius = kwargs.get('radius', [1])
        self.order = kwargs.get('order', 'linear')
        self.points = square_frame_size(self.radius)
        self.dims = self.compute_dims(self.points)
        self.dim = sum(self.dims)

        if self.order in ('lexicographic', 'bitmixing', 'alphamod'):
            self.bands = kwargs.get('bands', 'RGB')
            self.perm = tuple(['RGB'.index(b) for b in self.bands])

        if self.order == 'alphamod':
            self.alpha = kwargs.get('alpha', 2)

        if self.order == 'bitmixing':
            #Generate the lookup table for computing bit mixing order.
            bits = 8*np.dtype(np.uint8).itemsize
            levels = 2**bits
            channels = 3
            indices = np.arange(levels).astype(np.uint8)[:, None]

            exponents = [channels*np.arange(bits)[::-1] + ind
                         for ind in range(channels)[::-1]]

            weights = [2**exp for exp in exponents]

            red = np.sum(np.unpackbits(indices, axis=-1)*weights[0], axis=-1)
            green = np.sum(np.unpackbits(indices, axis=-1)*weights[1], axis=-1)
            blue = np.sum(np.unpackbits(indices, axis=-1)*weights[2], axis=-1)

            self.lut = red[:, None, None] \
                       + green[None, :, None] + blue[None, None, :]

        if self.order == 'random':
            self.seed = kwargs.get('seed', 0)
            np.random.seed(self.seed)
            bits = 8*np.dtype(np.uint8).itemsize
            levels = 2**bits
            channels = 3
            size = tuple(levels for i in range(channels))
            self.lut = np.random.permutation(levels**channels).reshape(size)

        if self.order == 'refcolor':
            self.cref = kwargs.get('cref', [0, 0, 0])


    def compute_dims(self, points):
        """Compute the dimension of the histogram for each neighbourhood."""
        raise NotImplementedError("Subclasses should implement this!")


    def codemap(self, img, radius, points):
        """Return a map of feature values."""
        raise NotImplementedError("Subclasses should implement this!")


    def raise_order_not_supported(self):
        raise ValueError(
                f'{self.order} order not supported for {self.abbrev()}')

    def __call__(self, img):
        """Compute the feature vector of an image through a HEP descriptor.

        Parameters
        ----------
        img : array
            Input image.

        Returns
        -------
        hist : array
            Feature vector (histogram of equivalent patterns).
            
        """
        if len(self.radius) == 1:
            # Single scale
            codes = self.codemap(img, self.radius[0], self.points[0])
            hist = histogram(codes, self.dim)
        else:
            # Multi scale
            lst = []
            for rad in self.radius:
                single_scale_descr = self.get_single_scale(rad)
                lst.append(single_scale_descr(img))
            hist = np.hstack(lst)
        return hist


    def __str__(self):
        """"Return a string representation of the descriptor."""
        name = self.__class__.__name__
        order = self.order
        params = f"order='{order}', radius={self.radius}"
        if order in ('lexicographic', 'bitmixing', 'alphamod'):
            params += f", bands='{self.bands}'"
        if order == 'refcolor':
            params += f", cref={self.cref}"
        elif order == 'random':
            params += f", seed={self.seed}"
        elif order == 'alphamod':
            params += f", alpha='{self.alpha}'"
        return f"{name}({params})"


    def abbrev(self):
        """Return the abbreviated descriptor name.
        
        Parameters
        ----------
        descriptor : HEP.hep
            Instance of a HEP texture descriptor.
            
        Returns
        -------
        out : str
            Abbreviated name of the descriptor which is used to generate 
            the names of the files where the corresponding data (features 
            or/and classification results) are saved.
        
        """
        name = self.__class__.__name__
    
        short_descr = ''.join([letter for letter in name if letter.isupper()])
        order = self.order
        if order == 'linear':
            short_order = ''
        elif order == 'product':
            short_order = 'prod'
        elif order == 'lexicographic':
            short_order = 'lex'
        elif order == 'alphamod':
            short_order = 'alpha'
        elif order == 'bitmixing':
            short_order = 'mix'
        elif order == 'refcolor':
            short_order = 'ref'
        elif order == 'random':
            short_order = 'rand'
        else:
            raise ValueError('invalid order')
    
        if order in ('lexicographic', 'bitmixing'):
            suffix = self.bands
        elif order == 'alphamod':
            suffix = f"{self.alpha}{self.bands}"
        elif order == 'refcolor':
            suffix = ''.join([format(i, '02x') for i in self.cref]).upper()
        elif order == 'random':
            suffix = self.seed
        else:
            suffix = ''
        out = f"{short_descr}{short_order}{suffix}{self.radius}"
        return out.replace(' ', '')


    def get_single_scale(self, radius):
        """Create a single scale instance of the descriptor.

        Parameters
        ----------
        radius : int
            Radius of the neighbourhood.

        Returns
        -------
        descr : `HEP`
            Instance with the same attributes as `self` except `radius`,
            which contains a single scale.
            
        """
        params = {k: v for k, v in self.__dict__.items()}
        params['radius'] = [radius]
        descr = self.__class__(**params)
        return descr


    def compare(self, xarr, yarr, comp=np.less):
        """
        Compare two images according to a given order using the specified
        comparison operator.

        Parameters
        ----------
        xarr, yarr : arrays
            Images to be compared.
        comp : Numpy function, optional (default `np.less`)
            Comparison function (`np.greater`, `np.less_equal`, etc.).

        Returns
        -------
        result : boolean array
            Truth value of `comp(neighbour, central)` element-wise according
            to `order`.

        
        """
        if self.order == 'linear':
            result = comp(xarr, yarr)
        elif self.order == 'product':
            result = np.all(comp(xarr, yarr), axis=-1)
        elif self.order == 'lexicographic':
            weights = np.asarray([256**np.arange(xarr.shape[2])[::-1]])
            lexic_x = np.sum(xarr[:, :, self.perm]*weights, axis=-1)
            lexic_y = np.sum(yarr[:, :, self.perm]*weights, axis=-1)
            result = comp(lexic_x, lexic_y)
        elif self.order == 'alphamod':
            xarr = xarr[:, :, self.perm]
            yarr = yarr[:, :, self.perm]
            xarr[:, :, 0] = xarr[:, :, 0] // self.alpha
            yarr[:, :, 0] = yarr[:, :, 0] // self.alpha
            weights = np.asarray([256**np.arange(xarr.shape[2])[::-1]])
            weights[0] = (256//self.alpha)**(xarr.shape[2] - 1)
            lexic_x = np.sum(xarr*weights, axis=-1)
            lexic_y = np.sum(yarr*weights, axis=-1)
            result = comp(lexic_x, lexic_y)
        elif self.order == 'bitmixing':
            mix_x = self.lut[tuple(xarr[:, :, self.perm].T)].T
            mix_y = self.lut[tuple(yarr[:, :, self.perm].T)].T
            result = comp(mix_x, mix_y)
        elif self.order == 'random':
            rand_x = self.lut[tuple(xarr.T)].T
            rand_y = self.lut[tuple(yarr.T)].T
            result = comp(rand_x, rand_y)
        elif self.order == 'refcolor':
            cref = np.asarray(self.cref).astype(np.int_)
            dist_x = np.linalg.norm(xarr - cref, axis=-1)
            dist_y = np.linalg.norm(yarr - cref, axis=-1)
            result = comp(dist_x, dist_y)
        return result


class Concatenation(object):
    """Class for concatenation of HEP descriptors."""
    def __init__(self, *descriptors):
        self.descriptors = descriptors

    def __call__(self, img):
        return np.concatenate([d(img) for d in self.descriptors])

    def __str__(self):
        return '+'.join([d.__str__() for d in self.descriptors])


class ImprovedCenterSymmetricLocalBinaryPattern(HEP):
    """Return the improved center-symmetric local binary patterns features.


    References
    ----------
    .. [1] Xiaosheng Wu and Junding Sun
           An Effective Texture Spectrum Descriptor
           https://doi.org/10.1109/IAS.2009.126
           
    """
    def compute_dims(self, points):
        """Compute the dimension of the histogram for each neighbourhood."""
        return [2**(p//2) for p in self.points]


    def codemap(self, img, radius, points):
        """Return a map with the pattern codes of an image corresponding
        to the improved center-symmetric local binary pattern coding.

        Parameters
        ----------
        img : array
            Color image.
        radius : int
            Radius of the local neighborhood.
        points : int
            Number of pixels of the local neighborhood.

        Returns
        -------
        codes : array
            Map of feature values.
            
        """
        central = img[radius: -radius, radius: -radius]
        codes = np.zeros(shape=central.shape[:2], dtype=np.int_)

        for exponent, index in enumerate(range(points//2)):

            start = utils.subimage(img, index, radius)
            end = utils.subimage(img, index + points//2, radius)

            ge_1 = self.compare(start, central, comp=np.greater_equal)
            ge_2 = self.compare(central, end, comp=np.greater_equal)
            lt_1 = self.compare(start, central, comp=np.less)
            lt_2 = self.compare(central, end, comp=np.less)

            ge = np.logical_and(ge_1, ge_2)
            lt = np.logical_and(lt_1, lt_2)
            codes += np.logical_or(ge, lt)*2**exponent
        return codes

class RankTransform(HEP):
    """Return the rank transform.

    References
    ----------
    .. [1] R. Zabih and J. Woodfill
           Non-parametric local transforms for computing visual correspondence
           https://doi.org/10.1007/BFb0028345
    .. [2] A. Fernández, D. Lima, F. Bianconi and F. Smeraldi
           Compact color texture descriptor based on rank transform and
           product ordering in the RGB color space
           https://doi.org/10.1109/ICCVW.2017.126
    
    """
    def compute_dims(self, points):
        """Compute the dimension of the histogram for each neighbourhood."""
        if self.order in self._canonical_orders:
            return [p + 1 for p in self.points]
        elif self.order in self._product_orders:
            return [(p + 2)*(p + 1)//2 for p in self.points]
        else:
            self.raise_order_not_supported()


    def codemap(self, img, radius, points):
        """Return a map with the pattern codes of an image corresponding
        to the rank transform.

        Parameters
        ----------
        img : array
            Color image.
        radius : int
            Radius of the local neighborhood.
        points : int
            Number of pixels of the local neighborhood.

        Returns
        -------
        codes : array
            Map of feature values.
            
        """
        central = img[radius: -radius, radius: -radius]
        lt = np.zeros(shape=central.shape[:2], dtype=np.int_)

        if self.order == 'product':
            ge = np.zeros_like(lt)

        for pixel in range(points):
            neighbour = utils.subimage(img, pixel, radius)
            lt += self.compare(neighbour, central, comp=np.less)

            if self.order == 'product':
                ge += self.compare(neighbour, central, comp=np.greater_equal)

        if self.order == 'product':
            dominated = lt
            non_comparable = points - dominated - ge
            codes = non_comparable + dominated*(2*points + 3 - dominated)//2
        else:
            codes = lt

        return codes


if __name__ == '__main__':   
    # Run tests
    import doctest
    doctest.testmod()