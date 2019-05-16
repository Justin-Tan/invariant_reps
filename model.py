#!/usr/bin/python3
import tensorflow as tf
import numpy as np
import glob, time, os, functools

from network import Network
from data import Data
from config import config_test, directories
from adversary import Adversary
from utils import Utils

class Model():
    def __init__(self, config, features, labels, args, evaluate=False):

        # Build the computational graph

        arch = functools.partial(Network.dense_network, actv=tf.nn.elu)
        sequential = False

        self.global_step = tf.Variable(0, trainable=False)
        self.MINE_step = tf.Variable(0, trainable=False)
        self.handle = tf.placeholder(tf.string, shape=[])
        self.training_phase = tf.placeholder(tf.bool)

        self.features_placeholder = tf.placeholder(tf.float32, [features.shape[0], features.shape[1]])
        self.labels_placeholder = tf.placeholder(tf.int32, labels.shape)
        self.test_features_placeholder = tf.placeholder(tf.float32)
        self.test_labels_placeholder = tf.placeholder(tf.int32)
        self.pivots_placeholder = tf.placeholder(tf.float32)
        self.test_pivots_placeholder = tf.placeholder(tf.float32)

        if config.use_adversary and not evaluate:
            self.pivots_placeholder = tf.placeholder(tf.float32)
            self.pivot_labels_placeholder = tf.placeholder(tf.int32)
            self.test_pivots_placeholder = tf.placeholder(tf.float32)
            self.test_pivot_labels_placeholder = tf.placeholder(tf.int32)

            train_dataset = Data.load_dataset(self.features_placeholder, self.labels_placeholder,
                    self.pivots_placeholder, batch_size=config.batch_size, sequential=sequential,
                    adversary=True, pivot_labels_placeholder=self.pivot_labels_placeholder)
            test_dataset = Data.load_dataset(self.test_features_placeholder, self.test_labels_placeholder,
                    self.pivots_placeholder, config_test.batch_size, test=True, sequential=sequential,
                    adversary=True, pivot_labels_placeholder=self.pivot_labels_placeholder)
        else:
            train_dataset = Data.load_dataset(self.features_placeholder, self.labels_placeholder,
                    self.pivots_placeholder, batch_size=config.batch_size, sequential=sequential)
            test_dataset = Data.load_dataset(self.test_features_placeholder, self.test_labels_placeholder,
                    self.pivots_placeholder, config_test.batch_size, test=True, sequential=sequential)


        val_dataset = Data.load_dataset(self.features_placeholder, self.labels_placeholder, self.pivots_placeholder,
            config.batch_size, evaluate=True, sequential=sequential)
        self.iterator = tf.data.Iterator.from_string_handle(self.handle, train_dataset.output_types, train_dataset.output_shapes)

        self.train_iterator = train_dataset.make_initializable_iterator()
        self.test_iterator = test_dataset.make_initializable_iterator()
        self.val_iterator = val_dataset.make_initializable_iterator()

        if config.use_adversary and not evaluate:
            self.example, self.labels, self.pivots, self.pivot_labels = self.iterator.get_next()
            if len(config.pivots) == 1:
                # self.pivots = tf.expand_dims(self.pivots, axis=1)
                self.pivot_labels = tf.expand_dims(self.pivot_labels, axis=1)
        else:
            self.example, self.labels, self.pivots = self.iterator.get_next()

        # if len(config.pivots) == 1:
        #    self.pivots = tf.expand_dims(self.pivots, axis=1)
        self.pivots.set_shape([None, 2*len(config.pivots)])

        if evaluate:
            # embeddings = tf.nn.embedding_lookup(embedding_encoder, ids=self.example)
            with tf.variable_scope('classifier') as scope:
                self.logits, *hreps = arch(self.example, config, self.training_phase)
            self.softmax = tf.nn.sigmoid(self.logits)
            self.pred = tf.cast(tf.greater(self.softmax, 0.5), tf.int32)
            self.ema = tf.train.ExponentialMovingAverage(decay=config.ema_decay, num_updates=self.global_step)
            print('Y shape:', self.labels.shape)
            print('Logits shape:', self.logits.shape)
            self.cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits,
                labels=(1-tf.one_hot(self.labels, depth=1)))

            log_likelihood = -tf.reduce_sum(self.cross_entropy)

        # embeddings = tf.nn.embedding_lookup(embedding_encoder, ids=self.example)

        with tf.variable_scope('classifier') as scope:
            self.logits, self.hrep = arch(self.example, config, self.training_phase)

        self.softmax = tf.nn.sigmoid(self.logits)[:,0]
        self.pred = tf.cast(tf.greater(self.softmax, 0.5), tf.int32)
        self.pred_boolean = tf.cast(tf.greater(self.softmax, 0.5), tf.bool)
        true_background_pivots = tf.boolean_mask(self.pivots, tf.cast((1-self.labels), tf.bool))
        pred_background_pivots = tf.boolean_mask(self.pivots, tf.cast((1-self.pred), tf.bool))

        if config.use_adversary:
            adv = Adversary(config,
                classifier_logits=self.logits,
                labels=self.labels,
                pivots=self.pivots,
                pivot_labels=self.pivot_labels,
                training_phase=self.training_phase,
                predictor_learning_rate=learning_rate,
                args=args)
                
            self.adv_loss = adv.adversary_combined_loss
            self.total_loss = adv.total_loss
            self.predictor_train_op = adv.predictor_train_op
            self.adversary_train_op = adv.adversary_train_op

            self.joint_step, self.ema = adv.joint_step, adv.ema
            self.joint_train_op = adv.joint_train_op

            self.cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits,
                labels=(1-tf.one_hot(self.labels, depth=1)))
            self.cost = tf.reduce_mean(self.cross_entropy)

            if args.fisher_penalty:
                log_likelihood = -tf.reduce_sum(self.cross_entropy)
                dldTheta = tf.gradients(log_likelihood, self.pivots)[0]
                self.observed_fisher_information = tf.reduce_mean(tf.square(tf.squeeze(dldTheta)))
                bkg_dldTheta = tf.boolean_mask(dldTheta, tf.cast((1-self.labels), tf.bool))
                self.observed_bkg_fisher_information = tf.reduce_mean(tf.square(tf.squeeze(bkg_dldTheta)))

            self.MI_logits_theta_kraskov = tf.py_func(Utils.mutual_information_1D_kraskov, inp=[tf.squeeze(self.logits),
                tf.squeeze(self.pivots[:,0])], Tout=tf.float64)
            self.MI_xent_theta_kraskov = tf.py_func(Utils.mutual_information_1D_kraskov, inp=[tf.squeeze(self.cross_entropy),
                tf.squeeze(self.pivots[:,0])], Tout=tf.float64)

            self.MI_logits_labels_kraskov = tf.py_func(Utils.mutual_information_1D_kraskov, inp=[tf.squeeze(self.logits),
                tf.squeeze(self.labels)], Tout=tf.float64)

            X = self.logits
            # Calculate mutual information
            with tf.variable_scope('MINE') as scope:
                # X = tf.stop_gradient(self.logits)
                Z = tf.squeeze(self.pivots[:,0])
                # Z_prime = tf.random_shuffle(Z) # tf.squeeze(self.pivots[:,1])
                Z_prime = tf.random_shuffle(tf.squeeze(self.pivots[:,1]))

                *reg_terms, self.marginal_f, self.MI_logits_theta = Network.MINE(x=X, y=Z, y_prime=Z_prime, batch_size=config.batch_size, dimension=2, training=True, actv=tf.nn.elu)

            with tf.variable_scope('LABEL_MINE') as scope:
                Y = tf.cast(self.labels, tf.float32)
                Y_prime = tf.random_shuffle(Y)

                *reg_terms, self.MI_logits_labels_MINE = Network.MINE(x=X, y=Y, y_prime=Y_prime, batch_size=config.batch_size,
                        dimension=2, training=self.training_phase, actv=tf.nn.elu)


        else:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            # self.cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
            #    labels=self.labels)
            # self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits,
            #    labels=tf.one_hot(self.labels, depth=1))
            self.cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits,
                labels=(1-tf.one_hot(self.labels, depth=1)))
            self.cost = tf.reduce_mean(self.cross_entropy)

            if args.fisher_penalty:
                # Calculate Fisher Information matrix
                bkg_xentropy = tf.boolean_mask(self.cross_entropy, tf.logical_not(tf.cast(self.labels, tf.bool)))
                log_likelihood = -tf.reduce_sum(self.cross_entropy)

                dldTheta = tf.gradients(log_likelihood, self.pivots)[0]
                self.observed_fisher_information = tf.reduce_mean(tf.square(tf.squeeze(dldTheta)))

                bkg_dldTheta = tf.boolean_mask(dldTheta, tf.cast((1-self.labels), tf.bool))
                self.observed_bkg_fisher_information = tf.reduce_mean(tf.square(tf.squeeze(bkg_dldTheta)))

                dfdTheta = tf.gradients(self.logits, self.pivots)[0]
                bkg_dfdTheta = tf.boolean_mask(dldTheta, tf.cast((1-self.labels), tf.bool))

                self.output_gradients = tf.reduce_mean(tf.square(tf.squeeze(dfdTheta)))
                bkg_output_gradients = tf.reduce_mean(tf.square(tf.squeeze(bkg_dfdTheta)))


            X = self.logits
            X_bkg = tf.boolean_mask(X, tf.cast((1-self.labels), tf.bool))
            # Calculate mutual information
            with tf.variable_scope('MINE') as scope:
                # X = tf.stop_gradient(self.logits)
                Z = tf.squeeze(self.pivots[:,0])
                Z_bkg = tf.boolean_mask(Z, tf.cast((1-self.labels), tf.bool))
                # Z_prime = tf.random_shuffle(Z) # tf.squeeze(self.pivots[:,1])
                Z_prime = tf.random_shuffle(tf.squeeze(self.pivots[:,1]))
                Z_prime_bkg = tf.random_shuffle(Z_bkg) 

                #self.MI_logits_theta = Network.MINE(x=X, y=Z, y_prime=Z_prime, batch_size=config.batch_size,
                #        dimension=2, training=True, actv=tf.nn.elu, jensen_shannon=args.JSD)
                 
                (x_joint, x_marginal), (joint_f, marginal_f), self.MI_logits_theta = Network.MINE(x=X_bkg, y=Z_bkg, y_prime=Z_prime_bkg,
                        batch_size=config.batch_size, dimension=2, training=True, actv=tf.nn.elu, labels=self.labels, bkg_only=True, jensen_shannon=args.JSD)

            self.MI_logits_theta_kraskov = tf.py_func(Utils.mutual_information_1D_kraskov, inp=[tf.squeeze(self.logits),
                tf.squeeze(self.pivots[:,0])], Tout=tf.float64)
            self.MI_xent_theta_kraskov = tf.py_func(Utils.mutual_information_1D_kraskov, inp=[tf.squeeze(self.cross_entropy),
                tf.squeeze(self.pivots[:,0])], Tout=tf.float64)
            self.MI_logits_labels_kraskov = tf.py_func(Utils.mutual_information_1D_kraskov, inp=[tf.squeeze(self.logits),
                tf.squeeze(self.labels)], Tout=tf.float64)

            with tf.variable_scope('LABEL_MINE') as scope:
                Y = tf.cast(self.labels, tf.float32)
                Y_prime = tf.random_shuffle(Y)

                *reg_terms, self.MI_logits_labels_MINE = Network.MINE(x=X, y=Y, y_prime=Y_prime, batch_size=config.batch_size,
                        dimension=2, training=self.training_phase, actv=tf.nn.elu)

            # Alternatively, calculate the observed Fisher Information as the negative expected Hessian(ll)
            #hessian_ll = tf.hessians(log_likelihood, self.pivots)
            #FIM = -tf.squeeze(hessian_ll)
            #self.observed_fisher_diagonal_from_hessian = tf.diag_part(FIM)
            #self.observed_bkg_fisher_diagonal_from_hessian = tf.boolean_mask(self.observed_fisher_diagonal_from_hessian, 
            #    tf.cast((1-self.labels), tf.bool))
            #self.observed_fisher_information_from_hessian = tf.trace(FIM)
            #self.observed_bkg_fisher_information_from_hessian = tf.reduce_sum(self.observed_bkg_fisher_diagonal_from_hessian)

            theta_f = Utils.scope_variables('classifier')
            theta_MINE = Utils.scope_variables('MINE')
            theta_MINE_NY = Utils.scope_variables('LABEL_MINE')
            print('Classifier parameters:', theta_f)
            print('mine parameters', theta_MINE)
            print('Label mine parameters', theta_MINE_NY)
            

            with tf.control_dependencies(update_ops):
                # Ensures that we execute the update_ops before performing the train_step
                self.MINE_lower_bound = self.MI_logits_theta
                self.MINE_labels_lower_bound = self.MI_logits_labels_MINE

            if args.fisher_penalty:
                print('Penalizing Fisher Information')
                self.cost += config.fisher_penalty * self.observed_fisher_information

            if args.regularizer and args.JSD:
                print('Using Jensen-Shannon regularizer')
                joint_logit_grads = tf.gradients(self.joint_f, self.x_joint)[0]
                marginal_logit_grads = tf.gradients(self.marginal_f, self.x_marginal)[0]
                gamma_0, alpha_0 = 2.0, 0.1
                gamma = tf.train.exponential_decay(gamma_0, decay_rate=alpha_0, global_step=self.global_step,
                    decay_steps=10**6)
                self.jsd_regularizer = tf.reduce_mean((1.0 - tf.nn.sigmoid(self.joint_f))**2 * tf.square(joint_logit_grads)) + tf.reduce_mean( tf.nn.sigmoid(self.marginal_f)**2 * tf.square(marginal_logit_grads))
                
            if args.mutual_information_penalty:
                print('Penalizing mutual information')
                if args.regularizer and args.JSD: 
                    self.cost += args.MI_lambda * (tf.nn.relu(self.MINE_lower_bound) - config.gamma/2.0 *
                        self.jsd_regularizer)
                else:
                    # 'minmax' loss
                    # heuristic 'non-saturating loss'
                    # heuristic_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=marginal_f, labels=tf.ones_like(marginal_f))) 
                    # heuristic_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=joint_f, labels=tf.zeros_like(joint_f)))
                    # self.cost += args.MI_lambda * heuristic_loss
                    # self.cost += args.MI_lambda * tf.square(self.MINE_lower_bound)

                    # Alternative cost
                    # (Naive) Minimize log(1 - D(E(x),z)) for (x,z) ~ marginals
                    E_update_1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=marginal_f, labels=tf.zeros_like(marginal_f)))

                    # Maximize log(D(E(x),z)) for (x,z) ~ marginals
                    E_update_2 = - tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=marginal_f, labels=tf.ones_like(marginal_f)))
                    self.NS_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=marginal_f, labels=tf.ones_like(marginal_f)))

                    # Minimize log(D(E(x),z)) for (x,z) ~ joint
                    E_update_3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=joint_f, labels=tf.zeros_like(joint_f)))

                    # sum of update 2 and 3
                    alt_update = E_update_2 + E_update_3  # maximize this

                    # self.cost += args.MI_lambda * tf.nn.relu(self.MINE_lower_bound)
                    if args.JSD:
                        # self.cost += args.MI_lambda * tf.nn.relu(self.MINE_lower_bound - E_update_2)
                        if args.new:
                            self.cost += args.MI_lambda * tf.nn.relu(- E_update_2)
                        else:
                            # Combine them?
                            self.cost += args.MI_lambda * tf.nn.relu(self.MINE_lower_bound) # - E_update_2)

                    else:
                        self.cost += args.MI_lambda * tf.nn.relu(self.MINE_lower_bound)


            with tf.control_dependencies(update_ops):

                args.optimizer = args.optimizer.lower()
                if args.optimizer=='adam':
                    self.opt = tf.train.AdamOptimizer(config.learning_rate)
                elif args.optimizer=='momentum':
                    self.opt = tf.train.MomentumOptimizer(config.learning_rate, config.momentum,
                        use_nesterov=True)
                elif args.optimizer == 'rmsprop':
                    self.opt = tf.train.RMSPropOptimizer(config_learning_rate)
                elif args.optimizer == 'sgd':
                    self.opt =  tf.train.GradientDescentOptimizer(config.learning_rate)

                self.grad_loss = tf.get_variable(name='grad_loss', shape=[], trainable=False)
                self.grads = self.opt.compute_gradients(self.MI_logits_theta, grad_loss=self.grad_loss)

                self.opt_op = self.opt.minimize(self.cost, global_step=self.global_step, var_list=theta_f)
            
            MINE_opt = tf.train.AdamOptimizer(config.MINE_learning_rate)

            self.MINE_opt_op = MINE_opt.minimize(-self.MINE_lower_bound, var_list=theta_MINE,
                    global_step=self.MINE_step)

            self.MINE_labels_opt_op = MINE_opt.minimize(-self.MINE_labels_lower_bound, var_list=theta_MINE_NY)
            
            self.ema = tf.train.ExponentialMovingAverage(decay=config.ema_decay, num_updates=self.global_step)
            self.MINE_ema = tf.train.ExponentialMovingAverage(decay=0.95, num_updates=self.MINE_step)
            maintain_averages_clf_op = self.ema.apply(theta_f)
            maintain_averages_MINE_op = self.MINE_ema.apply(theta_MINE)
            maintain_averages_MINE_labels_op = self.ema.apply(theta_MINE_NY)

            with tf.control_dependencies(update_ops+[self.opt_op]):
                self.train_op = tf.group(maintain_averages_clf_op)

            with tf.control_dependencies(update_ops+[self.MINE_opt_op]):
                self.MINE_train_op = tf.group(maintain_averages_MINE_op)

            with tf.control_dependencies(update_ops+[self.MINE_labels_opt_op]):
                self.MINE_labels_train_op = tf.group(maintain_averages_MINE_labels_op)

        self.str_accuracy, self.update_accuracy = tf.metrics.accuracy(self.labels, self.pred)
        correct_prediction = tf.equal(self.labels, tf.cast(self.pred, tf.int32))
        _, self.auc_op = tf.metrics.auc(predictions=self.pred, labels=self.labels, num_thresholds=2048)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar('accuracy', self.accuracy)
        tf.summary.scalar('learning_rate', learning_rate)
        tf.summary.scalar('cost', self.cost)
        tf.summary.scalar('auc', self.auc_op)
        tf.summary.scalar('logits_theta_MI', self.MI_logits_theta_kraskov)
        tf.summary.scalar('xent_theta_MI', self.MI_xent_theta_kraskov)    
        tf.summary.scalar('logits_labels_MI', self.MI_logits_labels_kraskov)

        self.merge_op = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(
            os.path.join(directories.tensorboard, '{}_train_{}'.format(args.name, time.strftime('%d-%m_%I:%M'))), graph=tf.get_default_graph())
        self.test_writer = tf.summary.FileWriter(
            os.path.join(directories.tensorboard, '{}_test_{}'.format(args.name, time.strftime('%d-%m_%I:%M'))))
