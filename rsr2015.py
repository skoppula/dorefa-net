#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: rsr2015.py
# Author: Skanda Koppula  <skanda.koppula@gmail.com>
import os
import numpy as np
from ast import literal_eval as make_tuple

from tensorpack.dataflow import RNGDataFlow

__all__ = ['Rsr2015', 'RsrMfccFiles']

def create_label_mapping(labels):
    mapping = {label: i for i, label in enumerate(set(labels))}
    mapped_labels = np.array([mapping[label] for label in labels], dtype='int32')
    return mapping, mapped_labels

class RsrMfccFiles(RNGDataFlow):
    """
    Expects a $partition.idx file inside the base_dir fodler
    which contains the path to each example file
    """
    def __init__(self, base_dir, partition, shuffle=None):
        assert partition in ['train', 'test', 'val']
        assert os.path.isdir(base_dir)
        self.base_dir = base_dir
        self.partition = partition
        self.index = os.path.join(base_dir, partition + '.idx')
        assert os.path.isfile(self.index)
        print("Using training example index", self.index)

        if shuffle is None:
            shuffle = name == 'train'
        self.shuffle = shuffle

        with open(self.index, 'r') as f:
            lines = f.readlines()

        self.labels = [line.split()[0].strip() for line in lines]
        self.files = [line.split()[1].strip() for line in lines]
        self.mapping, self.mapped_labels = create_label_mapping(self.labels)

    def size(self):
        return len(self.files)

    def get_data(self):
        idxs = np.arange(len(self.files))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for i in idxs:
            fname, label = self.files[i], self.mapped_labels[i]
            fname = os.path.join(self.base_dir, fname)
            yield [fname, label]

def get_shapes(base_dir, partition):
    shapes_path = os.path.join(base_dir, partition + '.shapes')
    assert os.path.isfile(shapes_path)
    with open(shapes_path, 'r') as f:
        lines = f.readlines()
    return [make_tuple(line.strip().split('.npy ')[-1]) for line in lines]

class Rsr2015(RsrMfccFiles):
    """
    Produces MFCC frames of size [context, mfcc_size], and corresponding
    numeric label based on mapping create from RsrMfccFiles. mfcc_size
    is n_mfccs=20 if not including double deltas otherwise n_mfccs*3
    (stacked mfcc, deltas, and double deltas, each of size n_mfccs)
    """
    def __init__(self, base_dir, partition, context=20, n_mfccs=20, include_dd=False, shuffle=None):
        super(Rsr2015, self).__init__(base_dir, partition, shuffle)
        self.shapes = get_shapes(base_dir, partition)
        self.num_examples_in_epoch = sum([abs(x[0] - context) for x in self.shapes])
        self.context = context
        self.mfcc_size = n_mfccs*20 if include_dd else n_mfccs
        assert context > 0


    def get_data(self):
        for fname, label in super(Rsr2015, self).get_data():
            utt_data = np.load(fname)
            for i in range(utt_data.shape[0] - self.context):
                yield [utt_data[i:(i+self.context),0:self.mfcc_size].flatten(), label]

    def size(self):
        return self.num_examples_in_epoch


if __name__ == '__main__':
    ds = Rsr2015('./fake_data/', 'train', shuffle=False)
    ds.reset_state()

    for k in ds.get_data():
        from IPython import embed
        embed()
        break
