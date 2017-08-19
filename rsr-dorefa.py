#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: rsr-dorefa.py
# Author: Skanda Koppula (skanda.koppula@gmail.com)

import tensorflow as tf
import argparse
import numpy as np
import multiprocessing
import os
import sys

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.tfutils.varreplace import remap_variables
from tensorpack.dataflow import dataset
from tensorpack.utils.gpu import get_nr_gpu
from rsr2015 import Rsr2015

from dorefa import get_dorefa

"""
This implements training quantization algorithm from
DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients
http://arxiv.org/abs/1606.06160

Note that the effective batch size in SyncMultiGPUTrainer is actually
BATCH_SIZE * NUM_GPU. With a different number of GPUs in use, things might
be a bit different, especially for learning rate.

To Train:
    ./rsr-dorefa.py --dorefa 1,2,6 --data PATH --gpu 0,1

To Run Pretrained Model:
    ./alexnet-dorefa.py --load alexnet-126.npy --run a.jpg --dorefa 1,2,6
"""

BITW = 4
BITA = 4
BITG = 32
TOTAL_BATCH_SIZE = 128
BATCH_SIZE = None
NUM_PREFETCH_THREADS=2
shuffle_queue_buffer_size = 100

N_CONTEXT_FRMS = 20
N_MFCCS = 20

class Model(ModelDesc):
    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, N_CONTEXT_FRMS*N_MFCCS], 'input'),
                InputDesc(tf.int32, [None], 'label')]

    def _build_graph(self, inputs):
        image, label = inputs

        fw, fa, fg = get_dorefa(BITW, BITA, BITG)

        old_get_variable = tf.get_variable

        # monkey-patch tf.get_variable to apply fw
        def new_get_variable(v):
            name = v.op.name
            # don't binarize first and last layer
            if not name.endswith('W') or 'fc0' in name or 'fct' in name:
                return v
            else:
                logger.info("Binarizing weight {}".format(v.op.name))
                return fw(v)

        def nonlin(x):
            if BITA == 32:
                return tf.nn.relu(x)    # still use relu for 32bit cases
            return tf.clip_by_value(x, 0.0, 1.0)

        def activate(x):
            return fa(nonlin(x))

        with remap_variables(new_get_variable), \
                argscope(BatchNorm, decay=0.9, epsilon=1e-4), \
                argscope([Conv2D, FullyConnected], use_bias=False, nl=tf.identity):
            logits = (LinearWrap(image)
                      .FullyConnected('fc0', 512)
                      .apply(fg)
                      .BatchNorm('bn_fc0')
                      .apply(activate)

                      .FullyConnected('fc1', 512)
                      .apply(fg)
                      .BatchNorm('bn_fc1')
                      .apply(activate)

                      .FullyConnected('fc2', 512)
                      .apply(fg)
                      .BatchNorm('bnfc2')
                      .apply(nonlin)
                      .FullyConnected('fct', 512, use_bias=True)())

        prob = tf.nn.softmax(logits, name='output')

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')

        wrong = prediction_incorrect(logits, label, 1, name='wrong-top1')
        add_moving_summary(tf.reduce_mean(wrong, name='train-error-top1'))
        wrong = prediction_incorrect(logits, label, 5, name='wrong-top5')
        add_moving_summary(tf.reduce_mean(wrong, name='train-error-top5'))

        # weight decay on all W of fc layers
        wd_cost = regularize_cost('fc.*/W', l2_regularizer(5e-6), name='regularize_cost')

        add_param_summary(('.*/W', ['histogram', 'rms']))
        self.cost = tf.add_n([cost, wd_cost], name='cost')
        add_moving_summary(cost, wd_cost, self.cost)

    def _get_optimizer(self):
        lr = get_scalar_var('learning_rate', 1e-4, summary=True)
        return tf.train.AdamOptimizer(lr, epsilon=1e-5)


def get_data(partition):
    isTrain = partition == 'train'
    rsr_ds = Rsr2015(args.data, partition, shuffle=isTrain)
    ds = LocallyShuffleData(rsr_ds, shuffle_queue_buffer_size, nr_reuse=1)
    print(ds.size())
    ds = BatchData(ds, BATCH_SIZE, remainder=not isTrain)
    ds = PrefetchDataZMQ(ds, min(NUM_PREFETCH_THREADS, multiprocessing.cpu_count()))
    return ds


def get_config():
    if args.output:
        logger.set_logger_dir(args.output, action='d')
    else:
        logger.set_logger_dir('train_logs/', action='d')
    data_train = get_data('train')
    print("Number of examples in epoch:", data_train.size())
    # data_test = get_data('val')

    return TrainConfig(
        dataflow=data_train,
        callbacks=[
            ModelSaver(),
            # HumanHyperParamSetter('learning_rate'),
            ScheduledHyperParamSetter(
                'learning_rate', [(56, 2e-5), (64, 4e-6)])
            # InferenceRunner(data_test,
            #                 [ScalarStats('cost'),
            #                  ClassificationError('wrong-top1', 'val-error-top1'),
            #                  ClassificationError('wrong-top5', 'val-error-top5')])
        ],
        model=Model(),
        steps_per_epoch=data_train.size(),
        max_epoch=100,
    )


def run_example(model, sess_init, inputs):
    pred_config = PredictConfig(
        model=model,
        session_init=sess_init,
        input_names=['input'],
        output_names=['output']
    )
    predictor = OfflinePredictor(pred_config)

    for f in inputs:
        assert os.path.isfile(f)
        utt_data = np.load(f)
        outputs = predictor([utt])[0]
        prob = outputs[0]
        ret = prob.argsort()[-10:][::-1]
        print(f + ":")
        print(list(zip(ret, prob[ret])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='the physical ids of GPUs to use')
    parser.add_argument('--load', help='load a checkpoint, or a npy (given as the pretrained model)')
    parser.add_argument('--data', help='dataset dir', required=True)
    parser.add_argument('--dorefa',
                        help='number of bits for W,A,G, separated by comma', required=True)
    parser.add_argument('--run', help='run on a list of examples with pretrained model', nargs='*')
    parser.add_argument('--output', help='where to output logs')
    parser.add_argument('--num_prefetch_threads', help='where to output logs')
    args = parser.parse_args()

    BITW, BITA, BITG = map(int, args.dorefa.split(','))

    if args.num_prefetch_threads:


    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.run:
        assert args.load.endswith('.npy')
        run_image(Model(), DictRestore(np.load(args.load, encoding='latin1').item()), args.run)
        sys.exit()

    if args.gpu:
        assert args.gpu is not None, "Need to specify a list of gpu for training!"
        nr_tower = max(get_nr_gpu(), 1)
        print(nr_tower)
        BATCH_SIZE = TOTAL_BATCH_SIZE // nr_tower
        logger.info("Using GPU training. Num towers: {} Batch per tower: {}".format(nr_tower, BATCH_SIZE))
    else:
        BATCH_SIZE = TOTAL_BATCH_SIZE
        logger.info("Using CPU. Batch size: {}".format(BATCH_SIZE))

    config = get_config()
    if args.load:
        config.session_init = SaverRestore(args.load)

    if args.gpu:
        config.nr_tower = nr_tower
        SyncMultiGPUTrainer(config).train()
    else:
        QueueInputTrainer(config).train()

