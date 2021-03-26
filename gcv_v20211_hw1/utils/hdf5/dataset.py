import os
from enum import Enum

import h5py
import torch
from torch.utils.data import Dataset


class PreloadTypes(Enum):
    ALWAYS = 'always'
    LAZY = 'lazy'
    NEVER = 'never'


class Hdf5File(Dataset):

    def __init__(self, filename, io, data_label=None, target_label=None, labels=None, preload=PreloadTypes.ALWAYS,
                 transform=None):
        """Represents HDF5 dataset contained in a single HDF5 file.

        :param filename: name of the file
        :param io: HDF5IO object serving as a I/O interface to the HDF5 data files
        :param data_label: string label in HDF5 dataset corresponding to data to train from
        :param target_label: string label in HDF5 dataset corresponding to targets
        :param labels: a list of HDF5 dataset labels to read off the file ('*' for ALL keys)
        :param preload: determines the data loading strategy:
            'always': entire data is read off disk in constructor
            'lazy': entire data is loaded on first access
            'never': entire data never loaded, only the requested data portions are read off disk in getitem
        :param transform: callable implementing data + target transform (e.g., adding noise)
        """
        self.filename = os.path.normpath(os.path.realpath(filename))
        assert not all([value is None for value in [data_label, target_label, labels]]), \
            'Specify value for at least data_label, target_label, or labels'

        self.data_label = data_label
        self.target_label = target_label
        self.transform = transform
        self.items = None  # this is where the data internally is read to
        self.io = io
        assert preload in PreloadTypes, 'unknown preload type: {}'.format(preload)
        self.preload = preload

        with h5py.File(self.filename, 'r') as f:
            self.num_items = self._get_length(f)
            if labels == '*':
                labels = set(f.keys())
            elif None is labels:
                labels = set()
            else:
                labels = set(labels)

        default_labels = set([label for label in [data_label, target_label] if label is not None])
        self.labels = list(default_labels.union(labels))

        if self.preload == PreloadTypes.ALWAYS:
            self.reload()

    def _get_length(self, hdf5_file):
        try:
            num_items = self.io.length(hdf5_file)
        except KeyError:
            import warnings
            warnings.warn('File {} is not compatible with Hdf5File I/O interface {}'.format(
                self.filename, str(self.io.__class__)))
            num_items = 0
        return num_items

    def __len__(self):
        return self.num_items

    def __getitem__(self, index):
        item = self._get_item(index)

        data = None
        if None is not self.data_label:
            data = torch.from_numpy(item[self.data_label])

        target = None
        if None is not self.target_label:
            target = torch.from_numpy(item[self.target_label])

        if self.transform is not None:
            data, target = self.transform(data, target)

        if None is not self.data_label:
            item.update({self.data_label: data})

        if None is not self.target_label:
            item.update({self.target_label: target})

        #print(item)
        return item

    def reload(self):
        with h5py.File(self.filename, 'r') as f:
            self.num_items = self._get_length(f)
            self.items = {label: self.io.read(f, label)
                          for label in self.labels}

    def load_one(self, index):
        with h5py.File(self.filename, 'r') as f:
            self.num_items = self._get_length(f)
            return {label: self.io.read_one(f, label, index)
                    for label in self.labels}

    def is_loaded(self):
        return None is not self.items

    def unload(self):
        self.items = None

    def _get_item(self, index):
        if self.preload in [PreloadTypes.LAZY, PreloadTypes.ALWAYS]:
            if not self.is_loaded():
                self.reload()
            item = {label: self.items[label][index]
                    for label in self.labels}

        else:  # self.preload == PreloadTypes.NEVER
            item = self.load_one(index)

        return item
