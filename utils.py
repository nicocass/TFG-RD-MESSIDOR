#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""utils.py: Utiliy functions for texture classification."""


import os
import pickle
import doctest
import unittest
import numpy as np


def display_message(msg, symbol='#'):
    """Display a message on the screen.

    Parameters
    ----------
    msg : str
        Message to be displayed.
    symbol : str
        Symbol used to print horizontal lines
    """
    length = len(msg)
    hline = symbol*length
    print(hline)
    print(msg.upper())
    print(hline)
    print()


def display_sequence(seq, heading, symbol='-'):
    """Display the items of a sequence.

    Parameters
    ----------
    seq : A sequence such as list or tuple
        Sequence of elements
    heading : str
        Title of the listing.
    symbol : str
        Symbol used to print horizontal lines
    """
    length = len(heading)
    print('{0}\n{1}'.format(heading, symbol*length))
    for item in seq:
        print(item)
    print()


def filepath(folder, *args, ext='pkl'):
    """Returns the full path of the file with the calculated results
    for the given dataset, descriptor, descriptor of the given dataset

    Parameters
    ----------
    folder : string
        Full path of the folder where results are saved.
    args : list or tuple
        Instances of `TextureDataset`, `HEP`, `KNeighborsClassifier`, etc.
    ext : string
        File extension (default pkl).

    Returns
    -------
    fullpath : string
        The complete path of the file where features corresponding to the
        given dataset and descriptor (and estimator) are stored.
    """
    lst = []
    for obj in args:
        if hasattr(obj, 'acronym'):
            item = obj.acronym
        else:
            item = obj.__name__
        lst.append(item)
    lst[-1] = lst[-1] + '.' + ext
    fullpath = os.path.join(folder, '--'.join(lst))
    return fullpath


def load_object(pathname):
    """Load an object from disk using pickle.

    Parameters
    ----------
    pathname : str
        Full path of the file where the object is stored.

    Returns
    -------
    obj : any type
        Object loaded from disk.
    """
    with open(pathname, 'rb') as fid:
        obj = pickle.load(fid)
    return obj


def save_object(obj, pathname):
    """Save an object to disk using pickle.

    Parameters
    ----------
    obj : any type
        Object to be saved.
    pathname : str
        Full path of the file where the object will be stored.
    """
    with open(pathname, 'wb') as fid:
        pickle.dump(obj, fid, 0)


def dos2unix(filename):
    """Replaces the DOS linebreaks (\r\n) by UNIX linebreaks (\n)."""
    with open(filename, 'r') as fid:
        content = fid.read()

    #text.replace(b'\r\n', b'\n')

    with open(filename, 'w', newline='\n') as fid:
        fid.write(content)


def subimage(img, pixel, radius):
    """Return the subimage used to vectorize the comparison between a given
    pixel and the central pixel of the neighbourhood.

    Parameters
    ----------
    img : array
        Input image.
    pixel : int
        Index of the peripheral pixel.
    radius : int
        Radius of the neighbourhood.

    Returns
    -------
    cropped : array
        Image crop corresponding to the given pixel and radius.

    Notes
    -----
    Neighbourhood pixels are numbered as follows:
                                                   R=3
                        R=2
      R=1                              23  22  21  20  19  18  17
                15  14  13  12  11      0   .   .   .   .   .  16
    7  6  5      0   .   .   .  10      1   .   .   .   .   .  15
    0  c  4      1   .   c   .   9      2   .   .   c   .   .  14
    1  2  3      2   .   .   .   8      3   .   .   .   .   .  13
                 3   4   5   6   7      4   .   .   .   .   .  12
                                        5   6   7   8   9  10  11

    Examples
    --------
    >>> x = np.arange(49).reshape(7, 7)
    >>> x
    array([[ 0,  1,  2,  3,  4,  5,  6],
           [ 7,  8,  9, 10, 11, 12, 13],
           [14, 15, 16, 17, 18, 19, 20],
           [21, 22, 23, 24, 25, 26, 27],
           [28, 29, 30, 31, 32, 33, 34],
           [35, 36, 37, 38, 39, 40, 41],
           [42, 43, 44, 45, 46, 47, 48]])
    >>> subimage(x, 2, 1)
    array([[15, 16, 17, 18, 19],
           [22, 23, 24, 25, 26],
           [29, 30, 31, 32, 33],
           [36, 37, 38, 39, 40],
           [43, 44, 45, 46, 47]])
    >>> subimage(x, 8, 2)
    array([[25, 26, 27],
           [32, 33, 34],
           [39, 40, 41]])
    >>> subimage(x, 22, 3)
    array([[1]])
    """
    diam = 2*radius + 1
    total = 4*(diam - 1)
    rows, cols = img.shape[:2]
    if pixel == total - 1:
        limits = 0, -2*radius, 0, -2*radius
    elif pixel in range(total - diam + 1, total - 1):
        limits = 0, -2*radius, total - pixel - 1, -2*radius + total - pixel - 1
    elif pixel == total - diam:
        limits = 0, -2*radius, 2*radius, cols
    elif pixel in range(total - 2*diam + 2, total - diam):
        limits = (total - diam - pixel,
                  total - diam - pixel - 2*radius, 2*radius, cols)
    elif pixel == total - 2*diam + 1:
        limits = 2*radius, rows, 2*radius, cols
    elif pixel in range(diam - 1, total - 2*diam + 1):
        limits = 2*radius, rows, pixel - diam + 2, pixel - diam + 2 - 2*radius
    elif pixel == diam - 2:
        limits = 2*radius, rows, 0, -2*radius
    elif pixel in range(diam - 2):
        limits = pixel + 1, pixel + 1 - 2*radius, 0, -2*radius
    else:
        raise ValueError('Invalid pixel')
    top, down, left, right = limits
    if img.ndim == 2:
        cropped = img[top: down, left: right]
    elif img.ndim == 3:
        cropped = img[top: down, left: right, :]
    else:
        raise ValueError('Invalid image')
    return cropped


def hardcoded_subimage(img, pixel, radius):
    """Utility function for testing `subimage`.

    Parameters
    ----------
    img : array
        Input image (can be single-channnel or multi-channel).
    pixel : int
        Index of the peripheral pixel.
    radius : int
        Radius of the local neighbourhood.

    Returns
    -------
    subimg : array
        Subimage used to vectorize the comparison between a given pixel
        and the central pixel of the neighbourhood.
    """
    hardcoded_limits = {1: {0: (1, -1, 0, -2),
                            1: (2, None, 0, -2),

                            2: (2, None, 1, -1),
                            3: (2, None, 2, None),
                            4: (1, -1, 2, None),
                            5: (0, -2, 2, None),
                            6: (0, -2, 1, -1),
                            7: (0, -2, 0, -2),
                           },
                        2: {0: (1, -3, 0, -4),
                            1: (2, -2, 0, -4),
                            2: (3, -1, 0, -4),
                            3: (4, None, 0, -4),
                            4: (4, None, 1, -3),
                            5: (4, None, 2, -2),
                            6: (4, None, 3, -1),
                            7: (4, None, 4, None),
                            8: (3, -1, 4, None),
                            9: (2, -2, 4, None),
                            10: (1, -3, 4, None),
                            11: (0, -4, 4, None),
                            12: (0, -4, 3, -1),
                            13: (0, -4, 2, -2),
                            14: (0, -4, 1, -3),
                            15: (0, -4, 0, -4),
                           },
                        3: {0: (1, -5, 0, -6),
                            1: (2, -4, 0, -6),
                            2: (3, -3, 0, -6),
                            3: (4, -2, 0, -6),
                            4: (5, -1, 0, -6),
                            5: (6, None, 0, -6),
                            6: (6, None, 1, -5),
                            7: (6, None, 2, -4),
                            8: (6, None, 3, -3),
                            9: (6, None, 4, -2),
                            10: (6, None, 5, -1),
                            11: (6, None, 6, None),
                            12: (5, -1, 6, None),
                            13: (4, -2, 6, None),
                            14: (3, -3, 6, None),
                            15: (2, -4, 6, None),
                            16: (1, -5, 6, None),
                            17: (0, -6, 6, None),
                            18: (0, -6, 5, -1),
                            19: (0, -6, 4, -2),
                            20: (0, -6, 3, -3),
                            21: (0, -6, 2, -4),
                            22: (0, -6, 1, -5),
                            23: (0, -6, 0, -6),
                           },
                       }
    try:
        top, down, left, right = hardcoded_limits[radius][pixel]
        if 1 < img.ndim <= 2:
            subimg = img[top:down, left:right]
        elif img.ndim == 3:
            subimg = img[top:down, left:right, :]
        return subimg
    except (KeyError, NameError):
        print('No unit test available for this radius/pixel pair')
        raise


class TestSubimage(unittest.TestCase):
    """Test class for `subimage`."""

    def test_subimage(self):
        """Tests for `subimage`."""
        self.maxDiff = None
        np.random.seed(0)
        rows, cols = np.random.randint(low=7, high=15, size=2)
        gray = np.random.randint(0, high=255, size=(rows, cols))
        rgb = np.random.randint(0, high=255, size=(rows, cols, 3))
        for img in [gray, rgb]:
            for radius in range(1, 4):
                diam = 2*radius + 1
                total = 4*(diam - 1)
                for pix in range(total):
                    got = subimage(img, pix, radius)
                    expected = hardcoded_subimage(img, pix, radius)
                    self.assertSequenceEqual(got.tolist(), expected.tolist())


if __name__ == "__main__":
    doctest.testmod()
    unittest.main()
