import tensorflow as tf
import numpy as np
import glob, time, os
import functools

from network import Network
from utils import Utils
from config import directories

class Adversary(object):
    def __init__(self, config, classifier_logits, labels, pivots, pivot_labels, 
            training_phase, classifier_opt, evaluate=False):

        # Add ops to the graph for adversarial training
        adversary_losses_dict = {}
        adversary_logits_dict = {}

        for i, pivot in enumerate(config.pivots):
            # Introduce separate adversary for each pivotal variable
            mode = 'background'
            print('Building adversarial network for {} - {} events'.format(pivot, mode))

            with tf.variable_scope('adversary') as scope:
                adversary_logits, adv_hidden = Network.dense_network_ext(
                        x=classifier_logits,
                        config=config,
                        training=training_phase,
                        name='adversary_{}_{}'.format(pivot, mode),
                        actv=getattr(tf.nn, config.adv_activation),
                        n_layers=config.adv_n_layers,
                        n_classes=config.adv_n_classes
                )

            # Mask loss for signal events
            #adversary_loss = tf.reduce_mean(tf.cast((1-labels), 
            #    tf.float32)*tf.nn.sparse_softmax_cross_entropy_with_logits(logits=adversary_logits,
            #        labels=tf.cast(pivot_labels[:,i], tf.int32)))
            adversary_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=adversary_logits,
                    labels=tf.cast(pivot_labels[:,i], tf.int32)))

            adversary_losses_dict[pivot] = adversary_loss
            adversary_logits_dict[pivot] = adversary_logits

            tf.add_to_collection('adversary_losses', adversary_loss)

        self.adversary_combined_loss = tf.add_n(tf.get_collection('adversary_losses'), name='total_adversary_loss')
        self.cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=classifier_logits, labels=(1-tf.one_hot(labels, depth=1)))
        self.predictor_loss = tf.reduce_mean(self.cross_entropy)

        self.total_loss = self.predictor_loss - config.adv_lambda * self.adversary_combined_loss
    
        theta_f = Utils.scope_variables('classifier')
        theta_r = Utils.scope_variables('adversary')

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            predictor_gs = tf.Variable(0, name='predictor_global_step', trainable=False)
            self.joint_step = tf.Variable(0, name='joint_global_step', trainable=False)
            self.predictor_train_op = classifier_opt.minimize(self.predictor_loss, name='classifier_opt', 
                global_step=predictor_gs, var_list=theta_f)
            self.joint_opt_op = classifier_opt.minimize(self.total_loss, name='joint_opt', 
                global_step=self.joint_step, var_list=theta_f)

            adversary_opt = tf.train.AdamOptimizer(config.adv_learning_rate, name='adversary_optimizer')
            adversary_gs = tf.Variable(0, name='adversary_global_step', trainable=False)
            self.adversary_opt_op = adversary_opt.minimize(self.adversary_combined_loss, name='adversary_opt', 
                global_step=adversary_gs, var_list=theta_r)

        self.clf_ema = tf.train.ExponentialMovingAverage(decay=config.ema_decay, num_updates=self.joint_step, name='predictor_ema')
        self.adv_ema = tf.train.ExponentialMovingAverage(decay=config.ema_decay, num_updates=adversary_gs,
                name='adversary_ema')
        maintain_adversary_averages_op = self.adv_ema.apply(theta_r)
        maintain_predictor_averages_op = self.clf_ema.apply(theta_f)

        with tf.control_dependencies([self.adversary_opt_op]):
            self.adversary_train_op = tf.group(maintain_adversary_averages_op)

        with tf.control_dependencies([self.joint_opt_op]):
            self.joint_train_op = tf.group(maintain_predictor_averages_op)

        print('Classifier parameters:', theta_f)
        print('Adversary parameters', theta_r)

        classifier_pred = tf.argmax(classifier_logits, 1)
        true_background = tf.boolean_mask(pivots, tf.cast((1-labels), tf.bool))
        pred_background = tf.boolean_mask(pivots, tf.cast((1-classifier_pred), tf.bool))

        tf.summary.scalar('adversary_loss', self.adversary_combined_loss)
        tf.summary.scalar('total_loss', self.total_loss)

        for i, pivot in enumerate(config.pivots):
            adv_correct_prediction = tf.equal(tf.cast(tf.argmax(adversary_logits_dict[pivot],1), tf.int32), 
                tf.cast(pivot_labels[:,i], tf.int32))
            adv_accuracy = tf.reduce_mean(tf.cast(adv_correct_prediction, tf.float32))
            # tf.summary.scalar('adversary_acc_{}'.format(pivot), adv_accuracy)
            # tf.summary.histogram('true_{}_background_distribution'.format(pivot), true_background[:,i])
            # tf.summary.histogram('pred_{}_background_distribution'.format(pivot), pred_background[:,i])

