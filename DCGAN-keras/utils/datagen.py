"""Auxiliar methods to deal with loading the dataset."""
import os
import random

import numpy as np

from keras.preprocessing.image import apply_transform, flip_axis
from keras.preprocessing.image import transform_matrix_offset_center
from keras.preprocessing.image import Iterator, load_img, img_to_array

import nibabel as nib
import cv2

# input_size=(260,311,260)

class MyDict(dict):
    """
    Dictionary that allows to access elements with dot notation.

    ex:
        >> d = MyDict({'key': 'val'})
        >> d.key
        'val'
        >> d.key2 = 'val2'
        >> d
        {'key2': 'val2', 'key': 'val'}
    """

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


class TwoImageIterator(Iterator):
    """Class to iterate A and B images at the same time."""

    def __init__(self, directory, target_size, a_dir_name='T1w', b_dir_name='T2w', dim_ordering='th', N=-1,
                 batch_size=32, shuffle=True, seed=None, is_a_grayscale=True, is_b_grayscale=True):
        """
        Iterate through two directories at the same time.

        Files under the directory A and B with the same name will be returned
        at the same time.
        Parameters:
        - directory: base directory of the dataset. Should contain two
        directories with name a_dir_name and b_dir_name;
        - a_dir_name: name of directory under directory that contains the A
        images;
        - b_dir_name: name of directory under directory that contains the B
        images;
        - N: if -1 uses the entire dataset. Otherwise only uses a subset;
        - batch_size: the size of the batches to create;
        - shuffle: if True the order of the images in X will be shuffled;
        - seed: seed for a random number generator.
        """
        self.directory = directory

        self.a_dir = os.path.join(directory, a_dir_name)
        self.b_dir = os.path.join(directory, b_dir_name)

        a_files = set(x for x in os.listdir(self.a_dir))
        b_files = set(x for x in os.listdir(self.b_dir))
        # Files inside a and b should have the same name. Images without a pair are discarded.
        self.filenames = list(a_files.intersection(b_files))

        # Use only a subset of the files. Good to easily overfit the model
        if N > 0:
            random.shuffle(self.filenames)
            self.filenames = self.filenames[:N]
        self.N = len(self.filenames)
        if self.N == 0:
            raise Exception("""Did not find any pair in the dataset. Please check that """
                            """the names and extensions of the pairs are exactly the same. """
                            """Searched inside folders: {0} and {1}""".format(self.a_dir, self.b_dir))

        self.dim_ordering = dim_ordering
        if self.dim_ordering not in ('th', 'default', 'tf'):
            raise Exception('dim_ordering should be one of "th", "tf" or "default". '
                            'Got {0}'.format(self.dim_ordering))

        self.target_size = target_size

        self.is_a_grayscale = is_a_grayscale
        self.is_b_grayscale = is_b_grayscale

        self.image_shape_a = self._get_image_shape(self.is_a_grayscale)
        self.image_shape_b = self._get_image_shape(self.is_b_grayscale)

        if self.dim_ordering in ('th', 'default'):
            self.channel_index = 1
            self.row_index = 2
            self.col_index = 3
            self.depth_index = 4
        if dim_ordering == 'tf':
            self.channel_index = 4
            self.row_index = 1
            self.col_index = 2
            self.depth_index = 3

        if np.isscalar(zoom_range):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2:
            self.zoom_range = [zoom_range[0], zoom_range[1]]

        super(TwoImageIterator, self).__init__(len(self.filenames), batch_size,
                                               shuffle, seed)

    def _get_image_shape(self, is_grayscale):
        """Auxiliar method to get the image shape given the color mode."""
        if self.dim_ordering == 'tf':
            return self.target_size + (3,)
        else:
            return (3,) + self.target_size



    def _load_img_pair(self, idx, load_from_memory):
        """Get a pair of images with index idx."""

        fname = self.filenames[idx]

        a = nib.load(os.path.join(self.a_dir, fname)).get_data()
        b = nib.load(os.path.join(self.b_dir, fname)).get_data()

        a_resize = cv2.resize(a, (256,256,256))
        b_resize = cv2.resize(b, (256,256,256))

        a = img_to_array(a_resize, data_format="channels_first")
        b = img_to_array(b_resize, data_format="channels_first")

        return a, b


    def next(self):
        """Get the next pair of the sequence."""
        # Lock the iterator when the index is changed.
        with self.lock:
            index_array, _, current_batch_size = next(self.index_generator)

        batch_a = np.zeros((current_batch_size,) + self.image_shape_a)
        batch_b = np.zeros((current_batch_size,) + self.image_shape_b)

        for i, j in enumerate(index_array):
            a_img, b_img = self._load_img_pair(j, self.load_to_memory)
            # a = a_img.transpose(1,0,2)
            # b = b_img.transpose(1,0,2)
            # print a_img.shape, a.shape

            batch_a[i] = a_img
            batch_b[i] = b_img

        return [batch_a, batch_b]
