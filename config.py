#!/usr/bin/env python3

class config_train(object):
    mode = 'beta'
    n_layers = 5
    num_epochs = 4096
    batch_size = 1024
    ema_decay = 0.999
    learning_rate = 1e-4
    momentum = 0.9
    n_classes = 2
    MI_penalty = 8.0
    MI_iterations = 8 
    MINE_learning_rate = 8e-4
    D_learning_rate = 1e-5
    MI_label_lagrange_multiplier = 5.0
    gamma = 0.1
    lr_schedule = 'constant'
    n_features = 105
    kl_update = True

    # Adversary
    use_adversary = False
    pivots = ['B_Mbc']
    adv_n_layers = 2
    adv_activation = 'elu'
    adv_keep_prob = 1.0
    adv_hidden_nodes = [128,128,128]
    adv_learning_rate = 1e-4
    adv_lambda = 4.0
    adv_iterations = 16
    adv_n_classes = 8  # number of bins for discretized predictions
    n_epochs_initial = 10


class config_test(object):
    mode = 'beta_test'
    n_layers = 5
    num_epochs = 4096
    batch_size = 1024
    ema_decay = 0.999
    learning_rate = 1e-4
    momentum = 0.9
    n_classes = 2
    n_features = 105

    # Adversary
    use_adversary = False
    pivots = ['B_Mbc'] #, 'B_deltaE']
    adv_n_layers = 3
    adv_keep_prob = 1.0
    adv_hidden_nodes = [128, 128, 128]
    adv_learning_rate = 4e-3
    adv_lambda = 4
    adv_iterations = 24
    adv_n_classes = 8  # number of bins for discretized predictions
    n_epochs_initial = 4

class directories(object):
    train = '/data/train.h5'
    test = '/data/test.h5'
    val = '/data/val.h5'
    tensorboard = 'tensorboard'
    checkpoints = 'checkpoints'
    checkpoints_best = 'checkpoints/best'
    results = 'results'
