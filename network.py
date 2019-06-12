""" Network wiring """

import tensorflow as tf
import numpy as np
import glob, time, os
import functools

from utils import Utils

class Network(object):

    @staticmethod
    def dense_network(x, config, training, name='fully_connected', actv=tf.nn.relu, **kwargs):
        # Toy dense network for binary classification
        
        init = tf.contrib.layers.xavier_initializer()
        shape = [512,512,512,512,512]
        kwargs = {'center': True, 'scale': True, 'training': training, 'fused': True, 'renorm': True}
        # x = tf.reshape(x, [-1, num_features])
        # x = x[:,:-1]
        print('Input X shape', x.get_shape())

        with tf.variable_scope(name, initializer=init, reuse=tf.AUTO_REUSE) as scope:
            h0 = tf.layers.dense(x, units=shape[0], activation=actv)
            h0 = tf.layers.batch_normalization(h0, **kwargs)

            h1 = tf.layers.dense(h0, units=shape[1], activation=actv)
            h1 = tf.layers.batch_normalization(h1, **kwargs)

            h2 = tf.layers.dense(h1, units=shape[2], activation=actv)
            h2 = tf.layers.batch_normalization(h2, **kwargs)

            h3 = tf.layers.dense(h2, units=shape[3], activation=actv)
            h3 = tf.layers.batch_normalization(h3, **kwargs)

            h4 = tf.layers.dense(h3, units=shape[3], activation=actv)
            h4 = tf.layers.batch_normalization(h4, **kwargs)

        out = tf.layers.dense(h4, units=1, kernel_initializer=init)
        
        return out, h4

    @staticmethod
    def dense_network_ext(x, config, training, n_layers, n_classes, name='fcn', actv=tf.nn.relu, **kwargs):
        # Toy dense network for binary classification
        
        init = tf.contrib.layers.xavier_initializer()
        shape = [128 for _ in range(int(n_layers))]
        assert n_layers <= len(shape), 'Number of requested layers too high.'
        kwargs = {'center': True, 'scale': True, 'training': training, 'fused': True, 'renorm': True}
        print('Input X shape', x.get_shape())

        with tf.variable_scope(name, initializer=init, reuse=tf.AUTO_REUSE) as scope:
            h0 = tf.layers.dense(x, units=shape[0], activation=actv)
            h0 = tf.layers.batch_normalization(h0, **kwargs)
            h = h0
            current_layer = 1

            while current_layer < n_layers:
                h = tf.layers.dense(h, units=shape[current_layer], activation=actv)
                h = tf.layers.batch_normalization(h, **kwargs)
                current_layer += 1

            out = tf.layers.dense(h, units=n_classes, kernel_initializer=init)

        return out, h

    @staticmethod
    def MINE(x, y, y_prime, training, batch_size, name='MINE', actv=tf.nn.elu, 
            n_layers=2, dimension=2, labels=None, jensen_shannon=True, 
            standardize=False, **kwargs):
        """
        Mutual Information Neural Estimator
        (x,y):      Drawn from joint p(x,y)
        y_prime:    Drawn from marginal p(y)

        returns
        MI:         Lower bound on mutual information between x,y
        """

        init = tf.contrib.layers.xavier_initializer()
        drop_rate = 0.0
        shape = [64 for _ in range(int(n_layers))]
        assert n_layers <= len(shape), 'Number of requested layers too high.'
        kwargs = {'center': True, 'scale': True, 'training': training, 'fused': True, 'renorm': False}
        # y_prime = tf.random_shuffle(y)

        # Standardize inputs
        x_mu, x_sigma = tf.nn.moments(x, axes=0)
        y_mu, y_sigma = tf.nn.moments(y, axes=0)
        y_prime_mu, y_prime_sigma = tf.nn.moments(y_prime, axes=0)

        if standardize:
            x = (x - x_mu) / x_sigma
            y = (y - y_mu) / y_sigma
            y_prime = (y_prime - y_prime_mu) / y_prime_sigma

        if dimension == 2:
            y, y_prime = tf.expand_dims(y, axis=1), tf.expand_dims(y_prime, axis=1)
        if len(x.get_shape().as_list()) < 2:
            x = tf.expand_dims(x, axis=1)

        z = tf.concat([x,y], axis=1)
        z_prime = tf.concat([x,y_prime], axis=1)
        z.set_shape([None, dimension])
        z_prime.set_shape([None, dimension])
        print('X SHAPE:', x.get_shape().as_list())
        print('Z SHAPE:', z.get_shape().as_list())
        print('Z PRIME SHAPE:', z_prime.get_shape().as_list())

        def statistic_network(t, name='MINE', reuse=False):
            with tf.variable_scope(name, initializer=init, reuse=reuse) as scope:

                h0 = tf.layers.dense(t, units=shape[0], activation=None)
                # h0 = tf.layers.batch_normalization(h0, **kwargs)
                h0 = tf.contrib.layers.layer_norm(h0, center=True, scale=True, activation_fn=actv)

                h = h0
                current_layer = 1

                while current_layer < n_layers:
                    h = tf.layers.dense(h, units=shape[current_layer], activation=None)
                    h = tf.contrib.layers.layer_norm(h, center=True, scale=True, activation_fn=actv)
                    current_layer += 1

                out = tf.layers.dense(h, units=1, kernel_initializer=init)

            return out

        def log_sum_exp_trick(x, batch_size, axis=1):
            # Compute along batch dimension
            x = tf.squeeze(x)
            x_max = tf.reduce_max(x)
            # lse = x_max + tf.log(tf.reduce_mean(tf.exp(x-x_max)))
            lse = x_max + tf.log(tf.reduce_sum(tf.exp(x-x_max))) - tf.log(batch_size)
            return lse

        joint_f = statistic_network(z)
        marginal_f = statistic_network(z_prime, reuse=True)
        print('Joint shape', joint_f.shape)
        print('marginal shape', marginal_f.shape)

        # MI_lower_bound = tf.reduce_mean(joint_f) - tf.log(tf.reduce_mean(tf.exp(marginal_f)) + 1e-5)
        MI_lower_bound = tf.squeeze(tf.reduce_mean(joint_f)) - tf.squeeze(log_sum_exp_trick(marginal_f,
            tf.cast(batch_size, tf.float32)))

        # H(p,q) = - E_p[log q]
        joint_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=joint_f,
            labels=tf.ones_like(joint_f)))
        marginal_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=marginal_f,
            labels=tf.zeros_like(marginal_f)))

        JSD_lower_bound = -(marginal_loss + joint_loss) + tf.log(4.0)
        # JSD_lower_bound = tf.squeeze(tf.reduce_mean(-tf.nn.softplus(-tf.squeeze(joint_f)))) - tf.squeeze(tf.reduce_mean(tf.nn.softplus(tf.squeeze(marginal_f))))
        # GAN_lower_bound = tf.reduce_mean(tf.log(tf.nn.sigmoid(joint_f))) + tf.reduce_mean(tf.log(1.0-tf.nn.sigmoid(marginal_f)))

        if jensen_shannon:
            lower_bound = JSD_lower_bound
        else:
            lower_bound = MI_lower_bound

        return (z, z_prime), (joint_f, marginal_f), lower_bound


    @staticmethod
    def kernel_MMD(x, y, y_prime, batch_size, name='kernel_MMD', actv=tf.nn.elu, dimension=2, labels=None, bkg_only=True):
        """
        Kernel MMD 
        (x,y):      Drawn from joint
        y_prime:    Drawn from marginal

        returns
        mmd2:       MMD distance between two distributions
        """

        def gaussian_kernel_mmd2(X, Y, gamma):
            """
            Parameters
            ____
            X: Matrix, shape: (n_samples, features)
            Y: Matrix, shape: (m_samples, features)

            Returns
            ____
            mmd: MMD under Gaussian kernel
            """

            XX = tf.matmul(X, X, transpose_b=True)
            XY = tf.matmul(X, Y, transpose_b=True)
            YY = tf.matmul(Y, Y, transpose_b=True)

            M, N = tf.cast(XX.get_shape()[0], tf.float32), tf.cast(YY.get_shape()[0], tf.float32)

            X_sqnorm = tf.reduce_sum(X**2, axis=-1)
            Y_sqnorm = tf.reduce_sum(Y**2, axis=-1)

            row_bc = lambda x: tf.expand_dims(x,0)
            col_bc = lambda x: tf.expand_dims(x,1)

            K_XX = tf.exp( -gamma * (col_bc(X_sqnorm) - 2 * XX + row_bc(X_sqnorm)))
            K_XY = tf.exp( -gamma * (col_bc(X_sqnorm) - 2 * XY + row_bc(Y_sqnorm)))
            K_YY = tf.exp( -gamma * (col_bc(Y_sqnorm) - 2 * YY + row_bc(Y_sqnorm)))

            mmd2 = tf.reduce_sum(K_XX) / M**2 - 2 * tf.reduce_sum(K_XY) / (M*N) + tf.reduce_sum(K_YY) / N**2

            return mmd2

        def rbf_mixed_mmd2(X, Y, M, N, sigmas=[1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0]):
            """
            Parameters
            ____
            X:      Matrix, shape: (n_samples, features)
            Y:      Matrix, shape: (m_samples, features)
            sigmas: RBF parameter

            Returns
            ____
            mmd2:   MMD under Gaussian mixed kernel
            """

            XX = tf.matmul(X, X, transpose_b=True)
            XY = tf.matmul(X, Y, transpose_b=True)
            YY = tf.matmul(Y, Y, transpose_b=True)

            X_sqnorm = tf.reduce_sum(X**2, axis=-1)
            Y_sqnorm = tf.reduce_sum(Y**2, axis=-1)

            row_bc = lambda x: tf.expand_dims(x,0)
            col_bc = lambda x: tf.expand_dims(x,1)

            K_XX, K_XY, K_YY = 0,0,0

            for sigma in sigmas:
                gamma = 1 / (2 * sigma**2)
                K_XX += tf.exp( -gamma * (col_bc(X_sqnorm) - 2 * XX + row_bc(X_sqnorm)))
                K_XY += tf.exp( -gamma * (col_bc(X_sqnorm) - 2 * XY + row_bc(Y_sqnorm)))
                K_YY += tf.exp( -gamma * (col_bc(Y_sqnorm) - 2 * YY + row_bc(Y_sqnorm)))

            mmd2 = tf.reduce_sum(K_XX) / M**2 - 2 * tf.reduce_sum(K_XY) / (M*N) + tf.reduce_sum(K_YY) / N**2

            return mmd2

        init = tf.contrib.layers.xavier_initializer()

        if bkg_only:
            batch_size_bkg_only = tf.cast(batch_size - tf.reduce_sum(labels), tf.float32)

        if dimension == 2:
            y, y_prime = tf.expand_dims(y, axis=1), tf.expand_dims(y_prime, axis=1)
        if len(x.get_shape().as_list()) < 2:
            x = tf.expand_dims(x, axis=1)

        z = tf.concat([x,y], axis=1)
        z_prime = tf.concat([x,y_prime], axis=1)
        z.set_shape([None, dimension])
        z_prime.set_shape([None, dimension])
        print('X SHAPE:', x.get_shape().as_list())
        print('Z SHAPE:', z.get_shape().as_list())
        print('Z PRIME SHAPE:', z_prime.get_shape().as_list())

        mmd2 = tf.nn.relu(rbf_mixed_mmd2(z, z_prime, M=batch_size_bkg_only, N=batch_size))

        return z, z_prime, tf.sqrt(mmd2)
