#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: rsr2015.py
# Author: Skanda Koppula  <skanda.koppula@gmail.com>
import os
import numpy as np

from tensorpack.dataflow import RNGDataFlow

__all__ = ['Rsr2015', 'RsrMfccFiles']

class RsrMfccFiles(RNGDataFlow):
    """
    Expects data/ folder to be inside base_dir
    and $partition.idx to contain the filenames (without leading path)
    of examples inside the data/ folder
    """
    def __init__(self, partition, base_dir, shuffle=None):
        assert partition in ['train', 'test', 'val']
        assert os.path.isdir(base_dir)
        self.partition = partition
        self.index = os.path.join(dir, partition + '.idx')
        assert os.path.isfile(self.index)

        if shuffle is None:
            shuffle = name == 'train'
        self.shuffle = shuffle

        with open(self.index, 'r') as f:
            lines = f.readlines()

        self.labels = [line.split()[0].strip() for line in lines]
        self.files = [line.split()[1].strip() for line in lines]
        self.mapped_labels = create_label_mapping(labels)

    def create_label_mapping(labels):
        mapping = {label: i for i, label in enumerate(set(labels))}
        mapped_labels = [labels[label] for label in labels]
        return mapped_labels

    def size(self):
        return len(self.files)

    def get_data(self):
        idxs = np.arange(len(self.files))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for i in idxs:
            fname, label = self.files[i], self.labels[i]
            fname = os.path.join(self.base_dir, fname)
            yield [fname, label]


class Rsr2015(RsrMfccFiles):
    """
    Produces MFCC frames of size [context, mfcc_size], and corresponding
    numeric label based on mapping create from RsrMfccFiles. mfcc_size
    is n_mfccs=20 if not including double deltas otherwise n_mfccs*3
    (stacked mfcc, deltas, and double deltas, each of size n_mfccs)
    """
    def __init__(self, base_dir, partition, context, n_mfccs=20, include_dd=False, shuffle=None):
        super(Rsr2015, self).__init__(base_dir, partition, shuffle)
        self.context = context
        self.mfcc_size = n_mfccs*20 if include_dd else n_mfccs
        assert context > 0

    def get_data(self):
        for fname, label in super(Rsr2015, self).get_data():
            utt_data = np.load(fname)
            for i in range(utt_data.shape[0] - self.context)
                yield [utt_data[i:(i+self.context),0:self.mfcc_size].flatten(), label]


if __name__ == '__main__':
    ds = Rsr2015('./fake_data/', 'train', shuffle=False)
    ds.reset_state()

    for k in ds.get_data():
        from IPython import embed
        embed()
        break
