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

flags = tf.app.flags

flags.DEFINE_integer('bit_w', 4, 'Bitwidth of weights')
flags.DEFINE_integer('bit_a', 4, 'Bitwidth of gradients')

flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('num_epochs', 100, 'Number of steps to run trainer.')
flags.DEFINE_integer('total_batch_size', 128, 'Total batch size across all GPUs')

flags.DEFINE_integer('n_layers', 4, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('state_size', 256, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('num_prefetch_threads', 2, 'Number of prefetch_threads')

flags.DEFINE_integer('n_context_frms', 20, 'Number of context frames')
flags.DEFINE_integer('n_mfccs', 20, 'Number of MFCC frames')

flags.DEFINE_integer('shuffle_queue_buffer_size', 100, 'Size of shuffle queue')

flags.DEFINE_string('data', './data/', 'Directory with train.idx/val.idx/test.idx')
flags.DEFINE_string('output', './train_logs/', 'Directory with train.idx/val.idx/test.idx')
flags.DEFINE_string('load', None, 'File with load checkpoint')

flags.DEFINE_string('inference_input', None, 'Files to run inference')

flags.DEFINE_string('gpu', None, 'IDs of GPUs, comma seperated: e.g. 2,3')

FLAGS = flags.FLAGS

def print_all_tf_vars():
    logger.info("Model variables:")
    for var in tf.global_variables():
        logger.info("\t{}, {}".format(var.name, var.shape))


class Model(ModelDesc):
    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, FLAGS.n_context_frms*FLAGS.n_mfccs], 'input'),
                InputDesc(tf.int32, [None], 'label')]

    def _build_graph(self, inputs):
        input, label = inputs

        fw, fa, fg = get_dorefa(FLAGS.bit_w, FLAGS.bit_a, 32)

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
            if FLAGS.bit_a == 32:
                return tf.nn.relu(x)    # still use relu for 32bit cases
            return tf.clip_by_value(x, 0.0, 1.0)

        def activate(x):
            return fa(nonlin(x))

        with remap_variables(new_get_variable), \
                argscope(BatchNorm, decay=0.9, epsilon=1e-4), \
                argscope([Conv2D, FullyConnected], use_bias=False, nl=tf.identity):
            curr_layer = LinearWrap(input)
            for i in range(FLAGS.n_layers):
                curr_layer = (curr_layer
                            .FullyConnected('fc' + str(i), FLAGS.state_size)
                            .apply(fg)
                            .BatchNorm('bn_fc' + str(i))
                            .apply(activate))
            logits = (curr_layer.FullyConnected('fc' + str(FLAGS.n_layers), 256)
                      .apply(fg)
                      .BatchNorm('bnfc' + str(FLAGS.n_layers))
                      .apply(nonlin)
                      .FullyConnected('fct', 256, use_bias=True)())

        print_all_tf_vars()

        prob = tf.nn.softmax(logits, name='output')

        identity_guesses = flatten(tf.argmax(prob, axis=1))
        uniq_identities, _, count = tf.unique_with_counts(identity_guesses)
        idx_to_identity_with_most_votes  = tf.argmax(count)
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

    def _get_optimizer(self):
        lr = get_scalar_var('learning_rate', FLAGS.learning_rate, summary=True)
        return tf.train.AdamOptimizer(lr, epsilon=1e-5)


def get_data(partition, batch_size):
    isTrain = partition == 'train'
    rsr_ds = Rsr2015(FLAGS.data, partition, shuffle=isTrain)
    if isTrain:
        ds = LocallyShuffleData(rsr_ds, FLAGS.shuffle_queue_buffer_size, nr_reuse=1)
        ds = BatchData(ds, batch_size, remainder=not isTrain)
        ds = PrefetchDataZMQ(ds, min(FLAGS.num_prefetch_threads, multiprocessing.cpu_count()))
        return ds, rsr_ds.size()
    else:
        return rsr_ds, rsr_ds.size()


def get_config(batch_size, n_gpus):
    logger.set_logger_dir(FLAGS.output, action='d')
    logger.info("Outputting at: {}".format(FLAGS.output))
    data_train, num_egs_per_trn_epoch  = get_data('train', batch_size)
    logger.info("{} examples per train epoch".format(num_egs_per_trn_epoch))

    data_test, num_egs_per_val_epoch = get_data('val', None)
    logger.info("{} examples per val epoch".format(num_egs_per_val_epoch))


    return TrainConfig(
        dataflow=data_train,
        callbacks=[
            ModelSaver(),
            MinSaver('val-error-top1'),
            # HumanHyperParamSetter('learning_rate'),
            ScheduledHyperParamSetter('learning_rate', [(56, 2e-5), (64, 4e-6)]),
            InferenceRunner(data_test, [ScalarStats('cost'), ClassificationError('wrong-top1', 'val-error-top1'), ClassificationError('utt-wrong', 'val-utt-error')])
        ],
        model=Model(),
        steps_per_epoch=num_egs_per_trn_epoch/(batch_size*n_gpus),
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


def main(_):

    if FLAGS.inference_input:
        assert FLAGS.load.endswith('.npy')
        run_example(Model(), DictRestore(np.load(FLAGS.load, encoding='latin1').item()), FLAGS.inference_input)
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

