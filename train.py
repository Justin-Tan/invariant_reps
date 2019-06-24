#!/usr/bin/python3
# python3 train.py -i /home/jtan/gpu/jtan/data/pivot_Mbc_train.h5 -test /home/jtan/gpu/jtan/data/pivot_Mbc_test.h5 -n penalty_alt -lambda 10 -MI -JSD
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

    assert(config.use_adversary is False), 'To use adversarial training, run `adv_train.py`'
    start_time = time.time()
    global_step, n_checkpoints, v_auc_best, v_cvm = 0, 0, 0., 10.
    ckpt = tf.train.get_checkpoint_state(directories.checkpoints)

    print('Reading data ...')
    if args.input is None:
        input_file = directories.train
        test_file = directories.test
    else:
        input_file = args.input
        test_file = args.test

    features, labels, pivots = Data.load_data(input_file, parquet=args.parquet)
    test_features, test_labels, test_pivots = Data.load_data(test_file, parquet=args.parquet)

    # Build graph
    model = Model(config, features=features, labels=labels, args=args)
    saver = tf.train.Saver()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        train_handle = sess.run(model.train_iterator.string_handle())
        test_handle = sess.run(model.test_iterator.string_handle())

        if args.restore_last and ckpt.model_checkpoint_path:
            # Continue training saved model
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('{} restored.'.format(ckpt.model_checkpoint_path))
            start_epoch = args.restart_epoch
        else:   
            if args.restore_path:
                new_saver = tf.train.import_meta_graph('{}.meta'.format(args.restore_path))
                new_saver.restore(sess, args.restore_path)
                print('{} restored.'.format(args.restore_path))
                start_epoch = args.restart_epoch
            else:
                start_epoch = 0
                
        sess.run(model.test_iterator.initializer, feed_dict={
            model.test_features_placeholder:test_features,
            model.test_pivots_placeholder:test_pivots,
            model.pivots_placeholder:test_pivots,
            model.test_labels_placeholder:test_labels})

        for epoch in range(start_epoch, config.num_epochs):
            sess.run(model.train_iterator.initializer, feed_dict={model.features_placeholder:features, 
                model.labels_placeholder:labels, model.pivots_placeholder:pivots})

            v_auc_best = Utils.run_diagnostics(model, config, directories, sess, saver, train_handle,
                test_handle, start_time, v_auc_best, epoch, global_step, args.name, v_cvm)

            if epoch > 0:
                save_path = saver.save(sess, os.path.join(directories.checkpoints, 
                    'MI_reg_{}_epoch{}_step{}.ckpt'.format(args.name, epoch, global_step)), global_step=epoch)
                print('Starting epoch {}, Weights saved to file: {}'.format(epoch, save_path))

            while True:
                try:
                    # Update weights
                    global_step, *ops = sess.run([model.global_step, model.opt_op, model.MINE_labels_train_op, model.auc_op, model.update_accuracy], 
                            feed_dict={model.training_phase: True, model.handle: train_handle})

                    if args.mutual_information_penalty:
                        for _ in range(args.MI_iterations):
                            sess.run(model.MINE_train_op, feed_dict={model.training_phase: True, model.handle: test_handle})

                    if global_step % 1000 == 0:
                        # Periodically show diagnostics
                        v_MI_kraskov, v_MI_MINE, v_pred, v_labels, v_pivots, v_conf = sess.run([model.MI_logits_theta_kraskov, 
                            model.MI_logits_theta, model.pred, model.labels, model.pivots[:,0], model.softmax], 
                                feed_dict={model.training_phase: False, model.handle: test_handle})
                        v_cvm = Utils.cvm_z(v_pivots, v_pred, v_labels, confidence=v_conf, selection_fraction=0.05)
                        v_auc_best = Utils.run_diagnostics(model, config_train, directories, sess, saver, train_handle,
                            test_handle, start_time, v_auc_best, epoch, global_step, args.name, v_cvm)

                    if global_step % 1e5 == 0:
                        save_path = saver.save(sess, os.path.join(directories.checkpoints, 
                            'MI_reg_{}_epoch{}_step{}.ckpt'.format(args.name, epoch, global_step)), global_step=epoch)
                        print('Weights saved to file: {}'.format(save_path))

                except tf.errors.OutOfRangeError:
                    print('End of epoch!')
                    break

                except KeyboardInterrupt:
                    save_path = saver.save(sess, os.path.join(directories.checkpoints,
                        'MI_reg_{}_last.ckpt'.format(args.name)), global_step=epoch)
                    print('Interrupted, model saved to: ', save_path)
                    sys.exit()

        save_path = saver.save(sess, os.path.join(directories.checkpoints,
                               'MI_reg_{}_end.ckpt'.format(args.name)),
                               global_step=epoch)

    print("Training Complete. Model saved to file: {} Time elapsed: {:.3f} s".format(save_path, time.time()-start_time))

def main(**kwargs):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Dataset-related options
    dataset = parser.add_argument_group("Dataset-related options")
    dataset.add_argument("-i", "--input", default=None, help="Path to training file", type=str)
    dataset.add_argument("-test", "--test", default=None, help="Path to test file", type=str)
    dataset.add_argument("-pq", "--parquet", help="Use if dataset is in parquet format", action="store_true")
    dataset.add_argument("-n", "--name", default="MI_reg", help="Checkpoint/Tensorboard label")

    # Optimization-related options
    optim = parser.add_argument_group("Optimization-related options")
    optim.add_argument("-opt", "--optimizer", default="adam", help="Selected optimizer", type=str)
    optim.add_argument("-MI_iters", "--MI_iterations", default=16, help="""Number of gradient steps of the 'discriminator' 
        per encoder gradient step.""", type=int)

    # Penalty-related options
    penalty = parser.add_argument_group("Penalty-related options. Note only one type of penalty should be active.")
    penalty.add_argument("-MI", "--mutual_information_penalty", help="Penalize mutual information between Z and logits", action="store_true")
    penalty.add_argument("-lambda", "--MI_lambda", default=0.0, help="Lagrange multiplier in MI-augmented objective.", type=float)
    penalty.add_argument("-JSD", "--JSD", help="Use Jensen-Shannon approximation of mutual information", action="store_true")
    penalty.add_argument("-heuristic", "--heuristic", help="Use heuristic cost formulation", action="store_true")
    penalty.add_argument("-combined", "--combined", help="Use combined cost formulation", action="store_true")
    penalty.add_argument("-kl", "--kl_update", help="Use approximate D_KL minimization.", action="store_true")

    # Regularization-based options
    reg = parser.add_argument_group("Regularization-related options to stabilize training.")
    reg.add_argument("-jsd_reg", "--jsd_regularizer", help="Toggle gradient-based regularization", action="store_true")
    reg.add_argument("-sn", "--spectral_norm", help="Apply spectral norm to discriminator", action="store_true")

    # Miscellaneous options
    misc = parser.add_argument_group("Miscellaneous options.")
    misc.add_argument("-rl", "--restore_last", help="Restore last saved model", action="store_true")
    misc.add_argument("-r", "--restore_path", help="Path to model to be restored", type=str)
    misc.add_argument("-re", "--restart_epoch", default=0, help="Epoch to restart from", type=int)
    misc.add_argument("-bkg_only", "--bkg_only", help="Apply penalty only to background class.", action="store_true")

    args = parser.parse_args()

    # Launch training
    train(config_train, args)

if __name__ == '__main__':
    main()
