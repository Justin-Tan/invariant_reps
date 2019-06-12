#!/usr/bin/python3

# Script for adversarial training procedure
# See arXiv 1611.01046

import tensorflow as tf
import numpy as np
import pandas as pd
import time, os, sys
import argparse

# User-defined
from network import Network
from utils import Utils
from data import Data
from model import Model
from config import config_train, directories

tf.logging.set_verbosity(tf.logging.ERROR)

def train(config, args):
    
    config.use_adversary = True
    assert(config.use_adversary), 'use_adversary must be set to True!'
    start_time = time.time()
    pretrain_step, joint_step, v_auc_best, v_cvm = 0, 0, 0., 10.
    ckpt = tf.train.get_checkpoint_state(directories.checkpoints)

    print('Reading data ...')
    if args.input is None:
        input_file = directories.train
        test_file = directories.test
    else:
        input_file = args.input
        test_file = args.test

    features, labels, pivots, pivot_labels = Data.load_data(input_file, adversary=True, parquet=args.parquet)
    print('FSHAPE', features.shape)
    test_features, test_labels, test_pivots, test_pivot_labels = Data.load_data(test_file, adversary=True, parquet=args.parquet)

    # Build graph
    model = Model(config, features=features, labels=labels, args=args)
    saver = tf.train.Saver()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        train_handle = sess.run(model.train_iterator.string_handle())
        test_handle = sess.run(model.test_iterator.string_handle())
        train_feed = {model.training_phase: True, model.handle: train_handle}
        test_feed = {model.training_phase: False, model.handle: test_handle}

        if args.restore_last and ckpt.model_checkpoint_path:
            # Continue training saved model
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('{} restored.'.format(ckpt.model_checkpoint_path))
        else:   
            if args.restore_path:
                new_saver = tf.train.import_meta_graph('{}.meta'.format(args.restore_path))
                new_saver.restore(sess, args.restore_path)
                print('{} restored.'.format(args.restore_path))

        sess.run(model.test_iterator.initializer, feed_dict={
            model.test_features_placeholder:test_features,
            model.test_labels_placeholder:test_labels,
            model.test_pivots_placeholder:test_pivots,
            model.test_pivot_labels_placeholder:test_pivot_labels})

        # Pretrain classifier
        if not args.skip_pretrain:
            print('Pretraining classifer for {} epochs'.format(config.n_epochs_initial))
            for epoch in range(config.n_epochs_initial):
                sess.run(model.train_iterator.initializer, feed_dict={
                    model.features_placeholder:features,
                    model.labels_placeholder:labels,
                    model.pivots_placeholder:pivots,
                    model.pivot_labels_placeholder:pivot_labels})

                # Run utils
                v_auc_best = Utils.run_diagnostics(model, config, directories, sess, saver, train_handle,
                    test_handle, start_time, v_auc_best, epoch, pretrain_step, args.name, v_cvm)

                while True:
                    try:
                        pretrain_step += 1
                        # Update weights
                        sess.run([model.predictor_train_op, model.update_accuracy], 
                            feed_dict=train_feed)

                        if pretrain_step % 1000 == 0:
                            # Periodically show diagnostics
                            v_MI_kraskov, v_pred, v_labels, v_pivots, v_conf = sess.run([model.MI_logits_theta_kraskov,
                                model.pred, model.labels, model.pivots[:,0], model.softmax], feed_dict=test_feed)
                            v_cvm = Utils.cvm_z(v_pivots, v_pred, v_labels, confidence=v_conf, selection_fraction=0.05)
                            v_auc_best = Utils.run_diagnostics(model, config_train, directories, sess, saver, train_handle,
                                test_handle, start_time, v_auc_best, epoch, pretrain_step, args.name, v_cvm)

                    except tf.errors.OutOfRangeError:
                        print('End of epoch!')
                        break

            save_path = saver.save(sess, os.path.join(directories.checkpoints,
                                   'adv_pretrain_{}_end.ckpt'.format(args.name)),
                                   global_step=epoch)

            print("Initial training Complete. Model saved to file: {} Time elapsed: {:.3f} s".format(save_path, time.time()-start_time))
        
        # Begin adversarial training
        print('<<<============================ Pretraining complete. Beginning adversarial training ============================>>>')
        for epoch in range(config.num_epochs):
            sess.run(model.train_iterator.initializer, feed_dict={
                model.features_placeholder:features,
                model.labels_placeholder:labels,
                model.pivots_placeholder:pivots,
                model.pivot_labels_placeholder:pivot_labels})

            if epoch > 0:
                # Run utils
                v_auc_best = Utils.run_adv_diagnostics(model, config, directories, sess, saver, train_handle,
                    test_handle, start_time, v_auc_best, epoch, joint_step, args.name, v_cvm)

            while True:
                try:
                    # Train adversary for adv_iterations relative to predictive model
                    joint_step, *ops = sess.run([model.joint_step, model.joint_train_op, model.update_accuracy], train_feed)

                    for _ in range(config.adv_iterations):
                        sess.run([model.adversary_train_op], test_feed)

                    if joint_step % 1000 == 0:  # Run diagnostics
                        v_MI_kraskov, v_pred, v_labels, v_pivots, v_conf = sess.run([model.MI_logits_theta_kraskov,
                            model.pred, model.labels, model.pivots[:,0], model.softmax], feed_dict=test_feed)
                        v_cvm = Utils.cvm_z(v_pivots, v_pred, v_labels, confidence=v_conf, selection_fraction=0.05)
                        v_auc_best = Utils.run_adv_diagnostics(model, config_train, directories, sess, saver, train_handle,
                            test_handle, start_time, v_auc_best, epoch, joint_step, args.name, v_cvm)

                except tf.errors.OutOfRangeError:
                    print('End of epoch!')
                    break

                except KeyboardInterrupt:
                    save_path = saver.save(sess, os.path.join(directories.checkpoints,
                        'adv_{}_last.ckpt'.format(args.name)), global_step=epoch)
                    print('Interrupted, model saved to: ', save_path)
                    sys.exit()

        save_path = saver.save(sess, os.path.join(directories.checkpoints,
                               'adv_{}_end.ckpt'.format(args.name)),
                               global_step=epoch)

    print("Training Complete. Model saved to file: {} Time elapsed: {:.3f} s".format(save_path, time.time()-start_time))

def main(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("-rl", "--restore_last", help="restore last saved model", action="store_true")
    parser.add_argument("-r", "--restore_path", help="path to model to be restored", type=str)
    parser.add_argument("-opt", "--optimizer", default="adam", help="Selected optimizer", type=str)
    parser.add_argument("-n", "--name", default="adv", help="Checkpoint/Tensorboard label")
    parser.add_argument("-i", "--input", default=None, help="Path to training file", type=str)
    parser.add_argument("-test", "--test", default=None, help="Path to test file", type=str)
    parser.add_argument("-pq", "--parquet", help="Use if dataset in parquet format", action="store_true")
    parser.add_argument("-skip_pt", "--skip_pretrain", help="skip pretraining of classifier", action="store_true")
    parser.add_argument("-lambda", "--adv_lambda", default=0.0, help="Adversary-classification tradeoff parameter",
        type=float)

    args = parser.parse_args()
    config = config_train

    # Launch training
    train(config, args)

if __name__ == '__main__':
    main()
