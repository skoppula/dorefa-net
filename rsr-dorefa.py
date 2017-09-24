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
import socket

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.tfutils.varreplace import remap_variables
from tensorpack.dataflow import dataset
from tensorpack.utils.gpu import get_nr_gpu
from rsr2015 import *

from dorefa import get_dorefa

"""
This implements training quantization algorithm from
DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients
http://arxiv.org/abs/1606.06160

Note that the effective batch size in SyncMultiGPUTrainer is actually
BATCH_SIZE * NUM_GPU. With a different number of GPUs in use, things might
be a bit different, especially for learning rate.
"""

flags = tf.app.flags

flags.DEFINE_integer('bit_w', 32, 'Bitwidth of weights')
flags.DEFINE_integer('bit_a', 32, 'Bitwidth of activations')

flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('num_epochs', 100, 'Number of steps to run trainer.')
flags.DEFINE_integer('total_batch_size', 512, 'Total batch size across all GPUs')

flags.DEFINE_integer('n_layers', 5, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('state_size', 512, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('num_prefetch_threads', 4, 'Number of prefetch_threads')

flags.DEFINE_integer('n_context_frms', 50, 'Number of context frames')
flags.DEFINE_integer('n_mfccs', 20, 'Number of MFCC frames')

flags.DEFINE_string('data', '/data/sls/scratch/skoppula/kaldi-rsr/numpy/small_spk_idxs/', 'Directory with train.idx/val.idx/test.idx')
flags.DEFINE_string('trn_cache_dir', '/data/sls/scratch/skoppula/mfcc-nns/rsr-experiments/dorefa/train_cache/rsr_smlspk_512_50_20', 'Directory with train cache')
flags.DEFINE_string('output', './train_logs_rsr_smlspk_dense_l5_ss512_w24_s24/', 'Directory with model output')
flags.DEFINE_string('load', None, 'File with load checkpoint')

flags.DEFINE_string('inference_input', None, 'Files to run inference')

# FOR SOME EXPERIMENTS
flags.DEFINE_boolean('use_clip', True, 'whether to use clip activations')
flags.DEFINE_boolean('force_quantization', True, 'whether to force quantization even with 32-bit')
flags.DEFINE_float('dropout', 1.0, 'whether to use dropout')

flags.DEFINE_string('gpu', None, 'IDs of GPUs, comma seperated: e.g. 2,3')

FLAGS = flags.FLAGS

def print_all_tf_vars():
    logger.info("Model variables:")
    for var in tf.global_variables():
        logger.info("\t{}, {}".format(var.name, var.shape))


class Model(ModelDesc):
    def __init__(self, n_spks):
        super(Model, self).__init__()
        self.n_spks = n_spks

    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, FLAGS.n_context_frms*FLAGS.n_mfccs], 'input'),
                InputDesc(tf.int32, [None], 'label')]

    def _build_graph(self, inputs):
        input, label = inputs

        fw, fa, fg = get_dorefa(FLAGS.bit_w, FLAGS.bit_a, 32)
        logger.info("Using {}-bit activations and {}-bit weights".format(FLAGS.bit_a, FLAGS.bit_w))
        logger.info("Using trn_cache: {}".format(FLAGS.trn_cache_dir))
        logger.info("Using host: {}".format(socket.gethostname()))

        old_get_variable = tf.get_variable

        # monkey-patch tf.get_variable to apply fw
        def new_get_variable(v):
            name = v.op.name
            logger.info("Binarizing weight {}".format(v.op.name))
            return fw(v, FLAGS.force_quantization)

        def nonlin(x):
            if FLAGS.bit_a == 32 and not FLAGS.use_clip:
                return tf.nn.relu(x)    # still use relu for 32bit cases
            return tf.clip_by_value(x, 0.0, 1.0)

        def activate(x):
            return fa(nonlin(x))

        activations = []
        with remap_variables(new_get_variable), \
                argscope([Conv2D, FullyConnected], use_bias=False, nl=tf.identity):
            curr_layer = LinearWrap(input)
            for i in range(FLAGS.n_layers):
                curr_layer = (curr_layer
                            .FullyConnected('fc' + str(i), FLAGS.state_size)
                            .LayerNorm('ln_fc' + str(i))
                            .apply(activate))
                activations.append(curr_layer.tensor())
                curr_layer = (curr_layer
                            .Dropout('dropout', FLAGS.dropout))
            logits = curr_layer.FullyConnected('fct', self.n_spks, use_bias=True)())

        print_all_tf_vars()

        prob = tf.nn.softmax(logits, name='output')

        # used for validation accuracy of utterance
        identity_guesses = flatten(tf.argmax(prob, axis=1))
        uniq_identities, _, count = tf.unique_with_counts(identity_guesses)
        idx_to_identity_with_most_votes = tf.argmax(count)
        chosen_identity = tf.gather(uniq_identities, idx_to_identity_with_most_votes)
        wrong = tf.expand_dims(tf.not_equal(chosen_identity, tf.cast(label[0], tf.int64)), axis=0, name='utt-wrong')

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')

        wrong = prediction_incorrect(logits, label, 1, name='wrong-top1')
        add_moving_summary(tf.reduce_mean(wrong, name='train-error-top1'))

        # weight decay on all W of fc layers
        wd_cost = regularize_cost('fc.*/W', l2_regularizer(5e-6), name='regularize_cost')

        add_param_summary(('.*/W', ['histogram', 'rms']))
        self.cost = tf.add_n([cost, wd_cost], name='cost')
        add_moving_summary(cost, wd_cost, self.cost)

        for activation in activations:
            add_activation_summary(activation)
            tf.summary.histogram(activation.name, activation)

    def _get_optimizer(self):
        lr = get_scalar_var('learning_rate', FLAGS.learning_rate, summary=True)
        return tf.train.AdamOptimizer(lr, epsilon=1e-5)

def get_data(partition, batch_size, context):
    isTrain = partition == 'train'
    if isTrain:
        # rsr_ds = WholeUtteranceAsFrameRsr2015(FLAGS.data, partition, context=context, shuffle=isTrain)
        # ds = LocallyShuffleData(rsr_ds, 5000, shuffle_interval=1000, nr_reuse=1)
        # ds = BatchData(ds, batch_size, remainder=not isTrain)
        rsr_ds = RandomFramesBatchFromCacheRsr2015(FLAGS.trn_cache_dir)
        ds = PrefetchDataZMQ(rsr_ds, min(FLAGS.num_prefetch_threads, multiprocessing.cpu_count()))
        # ds = PrefetchDataZMQ(ds, 1)
        return ds, rsr_ds.size()
    else:
        rsr_ds = WholeUtteranceAsBatchRsr2015(FLAGS.data, partition, context=context, shuffle=isTrain)
        return rsr_ds, rsr_ds.size()


def get_config(batch_size, n_gpus):
    logger.set_logger_dir(FLAGS.output, action='d')
    logger.info("Outputting at: {}".format(FLAGS.output))
    data_train, num_batches_per_trn_epoch = get_data('train', batch_size, FLAGS.n_context_frms)
    logger.info("{} batches per train epoch".format(num_batches_per_trn_epoch))

    data_val, num_batches_per_val_epoch = get_data('val', None, FLAGS.n_context_frms)
    logger.info("{} utterances per val epoch".format(num_batches_per_val_epoch))

    n_spks = get_n_spks()
    logger.info("Using {} speaker".format(n_spks))

    return TrainConfig(
        dataflow=data_train,
        callbacks=[
            ModelSaver(keep_checkpoint_every_n_hours=0.2),
            MinSaver('val-error-top1'),
            # HumanHyperParamSetter('learning_rate'),
            ScheduledHyperParamSetter('learning_rate', [(56, 2e-5), (64, 4e-6)]),
            InferenceRunner(data_val, [ScalarStats('cost'), ClassificationError('wrong-top1', 'val-error-top1'), ClassificationError('utt-wrong', 'val-utt-error')])
        ],
        model=Model(n_spks),
        steps_per_epoch=num_batches_per_trn_epoch/(n_gpus),
        max_epoch=100,
    )


def main(_):

    if FLAGS.inference_input:
        print("Not implemented")
        sys.exit()

    if FLAGS.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
        nr_tower = max(get_nr_gpu(), 1)
        batch_size_per_gpu = FLAGS.total_batch_size // nr_tower
    else:
        nr_tower = 1
        batch_size_per_gpu = FLAGS.total_batch_size

    config = get_config(batch_size_per_gpu, nr_tower)
    if FLAGS.load:
        config.session_init = SaverRestore(FLAGS.load)
    logger.info("Using {} prefetch threads".format(FLAGS.num_prefetch_threads))

    if FLAGS.gpu:
        logger.info("Using GPU training. Num towers: {} Batch per tower: {}".format(nr_tower, batch_size_per_gpu))
        config.nr_tower = nr_tower
        SyncMultiGPUTrainer(config).train()
    else:
        logger.info("Using CPU. Batch size: {}".format(batch_size_per_gpu))
        QueueInputTrainer(config).train()

if __name__ == "__main__":
    tf.app.run()

