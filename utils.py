# -*- coding: utf-8 -*-
# Diagnostic helper functions for Tensorflow session
import tensorflow as tf
from tensorflow.python.client import device_lib
import numpy as np
import pandas as pd
import scipy
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve, auc
import os, time, datetime, math
from lnc import MI

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import rc
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-darkgrid')
plt.style.use('seaborn-talk')

class Utils(object):
    
    @staticmethod
    def get_available_gpus():
        local_device_protos = device_lib.list_local_devices()
        #return local_device_protos
        print('Available GPUs:')
        print([x.name for x in local_device_protos if x.device_type == 'GPU'])

    @staticmethod
    def scope_variables(name):
        with tf.variable_scope(name):
            return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name)

    @staticmethod
    def run_diagnostics(model, config, directories, sess, saver, train_handle,
            test_handle, start_time, v_auc_best, epoch, step, name, v_cvm):
        t0 = time.time()
        improved = ''
        sess.run(tf.local_variables_initializer())
        feed_dict_train = {model.training_phase: False, model.handle: train_handle}
        feed_dict_test = {model.training_phase: False, model.handle: test_handle}

        try:
            t_auc, t_acc, t_loss, t_summary = sess.run([model.auc_op, model.accuracy, model.cost, model.merge_op], 
                feed_dict=feed_dict_train)
            model.train_writer.add_summary(t_summary)
        except tf.errors.OutOfRangeError:
            t_auc, t_loss, t_acc = float('nan'), float('nan'), float('nan')

        v_MI_kraskov, v_MI_MINE, v_MI_labels_kraskov, v_MI_labels_MINE, v_auc, v_acc, v_loss, v_summary, y_true, y_pred = sess.run([model.MI_logits_theta_kraskov, model.MI_logits_theta, model.MI_logits_labels_kraskov, model.MI_logits_labels_MINE, model.auc_op, model.accuracy, model.cost, model.merge_op, model.labels, model.pred], feed_dict=feed_dict_test) # TEST
        model.test_writer.add_summary(v_summary)

        if v_auc > v_auc_best:
            v_auc_best = v_auc
            improved = '[*]'
            if epoch>5:
                save_path = saver.save(sess,
                            os.path.join(directories.checkpoints_best, 'MI_reg_{}_epoch{}.ckpt'.format(name, epoch)),
                            global_step=epoch)
                print('Weights saved to file: {}'.format(save_path))


        print('Epoch {} | Training Acc: {:.3f} | Test Acc: {:.3f} | Test auc: {:.3f} | MI_kraskov: {:.3f} | MI_MINE: {:.3f} | MI_labels_kraskov: {:.3f} | MI_labels_MINE: {:.3f} | Train Loss: {:.3f} | Test Loss: {:.3f} | Test cvm: {:.3f} | Rate: {} examples/s ({:.2f} s) {}'.format(epoch, t_acc, v_acc, v_auc, v_MI_kraskov, v_MI_MINE, v_MI_labels_kraskov, v_MI_labels_MINE, t_loss, v_loss, v_cvm, int(config.batch_size * 1000 /(time.time()-t0)), time.time() - start_time, improved))

        return v_auc_best

    @staticmethod
    def run_adv_diagnostics(model, config, directories, sess, saver, train_handle,
            test_handle, start_time, v_auc_best, epoch, step, name, v_cvm):
        t0 = time.time()
        improved = ''
        sess.run(tf.local_variables_initializer())
        feed_dict_train = {model.training_phase: False, model.handle: train_handle}
        feed_dict_test = {model.training_phase: False, model.handle: test_handle}

        t_acc, t_loss, t_auc, t_summary = sess.run([model.accuracy, model.cost, model.auc_op, model.merge_op], feed_dict = feed_dict_train)
        v_ops = [model.accuracy, model.cost, model.MI_logits_theta_kraskov, model.adv_loss, model.auc_op, model.total_loss, model.merge_op]
        v_acc, v_loss, v_MI, v_adv_loss, v_auc, v_total, v_summary = sess.run(v_ops, feed_dict=feed_dict_test)
        model.train_writer.add_summary(t_summary)
        model.test_writer.add_summary(v_summary)

        if v_auc > v_auc_best:
            v_auc_best = v_auc
            improved = '[*]'
            if epoch>0:
                save_path = saver.save(sess,
                            os.path.join(directories.checkpoints_best, 'adv_{}_epoch{}.ckpt'.format(name, epoch)),
                            global_step=epoch)
                print('Weights saved to file: {}'.format(save_path))

        print('Epoch {} | Training Acc: {:.3f} | Test Acc: {:.3f} | Test Loss: {:.3f} | Test AUC: {:.3f} | Mutual Info: {:.3f} | Test cvm: {:.3f} | Adv. loss: {:.3f} | Total loss: {:.3f} | Rate: {} examples/s ({:.2f} s) {}'.format(epoch, t_acc, v_acc, v_loss, v_auc, v_MI, v_cvm, v_adv_loss, v_total, int(config.batch_size * 1000 /(time.time()-t0)), time.time() - start_time, improved))


        return v_auc_best

    @staticmethod
    def jsd_metric(df, block, name, selection_fraction=0.01):
        """
        Attempt to quantify sculpting.
        Evaluates mass decorrelation on some blackbox learner by evaluating a discrete
        approximation of the Jensen-Shannon divergence between the distributions of interest
        (here a mass-related quantity) passing and failing some learner threshold. If the 
        learned representation used for classification is noninformative of the variable of
        interest this should be low.
        """
        mbc_cutoff = 5.2425
        mbc_upper = 5.29
        df = df[df.B_Mbc > mbc_cutoff]
        df = df[df.B_Mbc < mbc_upper]

        v_auc = roc_auc_score(df.label.values, df.y_prob.values)
        df_sig, df_bkg = df[df.label>0.5], df[df.label<0.5]
        select_bkg = df_bkg.nlargest(int(df_bkg.shape[0]*selection_fraction), columns='y_prob')

        min_threshold = select_bkg.y_prob.min()
        df_tight = df[df.y_prob > min_threshold].query('B_deltaE < 0.1')
        df_tight = df_tight[df_tight.B_deltaE > -0.25]
        
        return jsd_discrete

    @staticmethod
    def online_fit(df, block, name, plot_components=True):

        import iminuit
        import probfit

        rc('text', usetex=False)
        sel_frac = 0.005
        v_auc = roc_auc_score(df.label.values, df.y_prob.values)
        df_sig, df_bkg = df[df.label>0.5], df[df.label<0.5]
        select_bkg = df_bkg.nlargest(int(df_bkg.shape[0]*sel_frac), columns='y_prob')

        min_threshold = select_bkg.y_prob.min()
        df_tight = df[df.y_prob > min_threshold].query('B_deltaE < 0.1')
        df_tight = df_tight[df_tight.B_deltaE > -0.25]

        q = df_tight[df_tight.B_mctype < 4]
        s = df_tight[df_tight.B_mctype > 6]

        mbc_cutoff = 5.2425
        fit_range = (mbc_cutoff, 5.29)
        s, q = s[s.B_Mbc > mbc_cutoff], q[q.B_Mbc > mbc_cutoff]
        N_sig_true = int(s._weight_.sum())
        N_bkg_true = int(q._weight_.sum())
        data = pd.concat([q,s])

        extended_crystalball = probfit.Extended(probfit.Normalized(probfit.crystalball, fit_range), extname='N_sig')
        cb_pars = dict(alpha=1.277, n=11.66, mean=5.27934, sigma=0.003135, N_sig=s.shape[0] * s._weight_.mean(), 
                       error_N_sig=10, fix_alpha=True, fix_n=True, fix_mean=True, fix_sigma=True)

        extended_argus = probfit.Extended(probfit.Normalized(probfit.argus, fit_range), extname='N_bkg')
        argus_pars = dict(chi=8.0, c=5.29, p=0.5, N_bkg=q.shape[0] * q._weight_.mean(), fix_c=True,
                          error_chi=0.1, error_p=0.1, error_N_bkg=10, limit_chi=(0.,10.)) # , limit_p=(0.2,2.))
        pdf = probfit.AddPdf(extended_crystalball, extended_argus)

        unbinned_likelihood = probfit.UnbinnedLH(pdf, data.B_Mbc.values, extended=True, extended_bound=fit_range, weights=data._weight_.values)
        start_pars = {**cb_pars, **argus_pars}
        minuit = iminuit.Minuit(unbinned_likelihood, pedantic=False, print_level=0, **start_pars)

        # MLE
        try:
            start_time = time.time()
            print('Starting fit | # fit points', df_tight.shape[0])
            minuit.migrad()
            print('Fit complete ({:.3f} s)'.format(time.time()-start_time))
            print('ML Parameters OK?', minuit.migrad_ok())
            print('Cov matrix OK?', minuit.matrix_accurate())
        except RuntimeError:
            print('MLE fit failure.')
            return -1.0, -999., v_auc, N_sig_true, 0, 10**4

        minuit_converge = minuit.migrad_ok()
        minuit_pos_def = minuit.matrix_accurate()

        if not (minuit_converge and minuit_pos_def):
            print('MLE fit failure.')
            return -1.0, -999., v_auc, N_sig_true, 0, 10**4
        
        # minuit.minos(var='N_sig')
        h = minuit.hesse()
        N_sig = [d for d in h if d['name']=='N_sig'][0]

        print('N_sig: {} ({})| Error_sig: {}'.format(N_sig['value'], N_sig_true, N_sig['error']))
        sig_value_error_ratio = N_sig['value'] / N_sig['error']
        weighted_mse = ((N_sig['value'] - N_sig_true) * N_sig['error']/N_sig['value'])**2
        weighted_mse = (N_sig['value'] / N_sig['error'])**2 / abs(N_sig['value'] - N_sig_true)
        N_bkg = [d for d in h if d['name']=='N_bkg'][0]

        ((data_edges, datay), (errorp, errorm), (total_pdf_x, total_pdf_y), parts) = unbinned_likelihood.draw(minuit, parts=True)
        plt.clf()

        m = probfit.mid(data_edges)
        rc('text', usetex=True)
        plt.errorbar(m, datay, errorp, fmt='.', capsize=1, color='Gray', label='Data', alpha=0.8)
        plt.plot(total_pdf_x, total_pdf_y, lw=4, label='Total Model')
        labels = [r'Signal PDF: {}$\pm${} ({})'.format(int(N_sig['value']), int(N_sig['error']), N_sig_true),
                  r'Background PDF: {}$\pm${} ({})'.format(int(N_bkg['value']), int(N_bkg['error']), N_bkg_true)]

        sea_green = '#54ff9f'
        crimson_tide = '#e50005'
        steel_blue = '#4e6bbd'
        colors = [sea_green, crimson_tide]

        for label, part, c in zip(labels, parts, colors):
            x, y = part
            plt.plot(x, y, ls='--', lw=2, label=label, color=c)

        if plot_components:
            nbins=100
            sns.distplot(q.B_Mbc, color=crimson_tide, hist=True, kde=False, norm_hist=False, label = r'$e^+e^-\rightarrow q\bar{q}$', 
                    bins=nbins, hist_kws=dict(linewidth=1.5, alpha=0.5, weights=q._weight_, histtype='step'))
            sns.distplot(s.B_Mbc, color=sea_green, hist=True, kde=False, norm_hist=False, label = r'$b \rightarrow s \gamma$',
                    bins=nbins, hist_kws=dict(linewidth=1.5, alpha=0.5, weights=s._weight_, histtype='step'))

        fs_title = 18
        plt.grid(True)
        plt.legend(loc='upper left')
        plt.ylim((0,700))
        # plt.legend(loc='best')
        plt.xlabel(r'$M_{bc}$ (GeV)', fontsize=fs_title)
        plt.ylabel('Events', fontsize=fs_title)
        plt.title(r'AUC: {:.3f} | $N_S / \delta N_S$: {:.3f}'.format(v_auc, sig_value_error_ratio), fontsize=fs_title)
        plt.savefig(os.path.join('/data/cephfs/punim0011/jtan/ray_results',
            'graphs/block_{}-{}-{}.pdf'.format(block, datetime.datetime.now().isoformat(), name)), 
            bbox_inches='tight',format='pdf', dpi=128)
        rc('text', usetex=False)

        return sig_value_error_ratio, weighted_mse, v_auc, N_sig_true, N_sig['value'], N_sig['error']

    @staticmethod
    def run_tune_diagnostics(model, config, directories, sess, saver, train_handle, test_handle, 
            start_time, v_auc_best, block, step, name, v_auc, v_reward=None, adversary=False):
        t0 = time.time()
        improved = ''
        sess.run(tf.local_variables_initializer())
        feed_dict_train = {model.training_phase: False, model.handle: train_handle}
        feed_dict_test = {model.training_phase: False, model.handle: test_handle}

        try:
            t_auc, t_acc, t_loss, t_summary, t_true, t_prob = sess.run([model.auc_op, model.accuracy, model.cost, model.merge_op,
                model.labels, model.softmax], feed_dict=feed_dict_train)
            model.train_writer.add_summary(t_summary)
            t_auc = roc_auc_score(y_true=t_true, y_score=t_prob)
        except tf.errors.OutOfRangeError:
            t_auc, t_loss, t_acc = float('nan'), float('nan'), float('nan')

        v_MI_kraskov, v_MI_MINE, v_MI_labels_kraskov, v_adv_loss, v_acc, v_loss, v_summary, y_true, y_pred, v_pivots, y_prob = sess.run([model.MI_logits_theta_kraskov, 
            model.MI_logits_theta, model.MI_logits_labels_kraskov, model.adv_loss, model.accuracy, model.cost, 
            model.merge_op, model.labels, model.pred, model.pivots[:,0], model.softmax], feed_dict=feed_dict_test) # TEST

        model.test_writer.add_summary(v_summary)

        # Calculate MMD between Z spectrum pre/post selection
        # v_mmd = np.sqrt(Utils.mmd2_z(v_pivots, y_pred, y_true, y_prob, selection_fraction=0.1))
        # v_auc = roc_auc_score(y_true=y_true, y_score=y_prob)

        if v_auc > v_auc_best:
            v_auc_best = v_auc
            improved = '[*]'

        if adversary:
            print("Block {} | Test Acc: {:.3f} | Train auc: {:.3f} | Test auc: {:.3f} | MI_kraskov: {:.3f} | Adv_loss: {:.3f} | " 
                  "MI_labels_kraskov: {:.3f} | Train Loss: {:.3f} | Test Loss: {:.3f} | Reward: {:.3f} | "
                  "Rate: {} examples/s ({:.2f} s) {}".format(block, v_acc, t_auc, v_auc, v_MI_kraskov, v_adv_loss, v_MI_labels_kraskov, 
                      t_loss, v_loss, v_reward, int(config.batch_size/(time.time()-t0)), time.time() - start_time, improved))
        else:
            print("Block {} | Test Acc: {:.3f} | Train auc: {:.3f} | Test auc: {:.3f} | MI_kraskov: {:.3f} | MI_MINE: {:.3f} | " 
                  "MI_labels_kraskov: {:.3f} | Train Loss: {:.3f} | Test Loss: {:.3f} | Reward: {:.3f} | "
                  "Rate: {} examples/s ({:.2f} s) {}".format(block, v_acc, t_auc, v_auc, v_MI_kraskov, v_MI_MINE, v_MI_labels_kraskov, 
                      t_loss, v_loss, v_reward, int(config.batch_size/(time.time()-t0)), time.time() - start_time, improved))

        return v_auc_best, v_MI_kraskov, v_acc, v_loss

    @staticmethod
    def plot_ROC_curve(y_true, y_pred, out, meta = ''):


        plt.style.use('seaborn-darkgrid')
        plt.style.use('seaborn-talk')
        plt.style.use('seaborn-pastel')

        # Compute ROC curve, integrate
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        print('Val AUC:', roc_auc)

        plt.figure()
        plt.axes([.1,.1,.8,.7])
        plt.figtext(.5,.9, r'$\mathrm{Receiver \;Operating \;Characteristic}$', fontsize=15, ha='center')
        plt.figtext(.5,.85, meta, fontsize=10,ha='center')
        plt.plot(fpr, tpr, # color='darkorange',
                         lw=2, label='ROC (area = %0.4f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=1.0, linestyle='--')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel(r'$\mathrm{False \;Positive \;Rate}$')
        plt.ylabel(r'$\mathrm{True \;Positive \;Rate}$')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join('results', '{}_ROC.pdf'.format(out)), format='pdf', dpi=1000)
        plt.gcf().clear()


    @staticmethod
    def mutual_information_1D_kraskov(x, y):
        # k-NN based estimate of mutual information

        mi = MI.mi_LNC([x,y],k=5,base=np.exp(1),alpha=0.2)
        return mi


    @staticmethod
    def rbf_mixed_mmd2(X, Y, sigmas=[1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0]):
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

        XX = X @ X.T
        XY = X @ Y.T
        YY = Y @ Y.T

        M = np.shape(X)[0]
        N = np.shape(Y)[0]
        assert ((M > 10) and (N > 10)), 'Insufficient samples for mmd estimation.'

        X_sqnorm = np.sum(X**2, axis=-1)
        Y_sqnorm = np.sum(Y**2, axis=-1)

        row_bc = lambda x: np.expand_dims(x,0)
        col_bc = lambda x: np.expand_dims(x,1)

        K_XX, K_XY, K_YY = 0,0,0

        for sigma in sigmas:
            gamma = 1 / (2 * sigma**2)
            K_XX += np.exp( -gamma * (col_bc(X_sqnorm) - 2 * XX + row_bc(X_sqnorm)))
            K_XY += np.exp( -gamma * (col_bc(X_sqnorm) - 2 * XY + row_bc(Y_sqnorm)))
            K_YY += np.exp( -gamma * (col_bc(Y_sqnorm) - 2 * YY + row_bc(Y_sqnorm)))

        mmd2 = np.sum(K_XX) / M**2 - 2 * np.sum(K_XY) / (M*N) + np.sum(K_YY) / N**2

        return mmd2

    @staticmethod
    def cvm_z(z, prediction, labels, confidence, selection_fraction):

        z = np.squeeze(z)
        df = pd.DataFrame([z, prediction, labels, confidence]).T
        df.columns = ['z', 'pred', 'labels', 'confidence']
        df_bkg = df[df['labels']<0.5]
        select_bkg = df_bkg.nlargest(int(df_bkg.shape[0]*selection_fraction), columns='confidence')

        z_bkg = np.squeeze(df_bkg.z.values)
        z_bkg_postsel = np.squeeze(select_bkg.z.values)

        cvm = scipy.stats.energy_distance(z_bkg, z_bkg_postsel)

        return cvm

    @staticmethod
    def mmd2_z(z, prediction, labels, confidence, selection_fraction):

        z = np.squeeze(z)
        df = pd.DataFrame([z, prediction, labels, confidence]).T
        df.columns = ['z', 'pred', 'labels', 'confidence']
        df_bkg = df[df['labels']<0.5]
        select_bkg = df_bkg.nlargest(int(df_bkg.shape[0]*selection_fraction), columns='confidence')

        z_bkg = np.expand_dims(df_bkg.z.values, axis=1)
        mmd2 = Utils.rbf_mixed_mmd2(z_bkg, z_bkg_postsel)

        return mmd2

    @staticmethod
    def reweight(event_counts, integrated_lumi=10**9):
        """
        Inputs: Integrated luminosity in nb^{-1}
        1 ab^{-1} = 10**9 1 nb^{-1}
        Outputs: Dict containing event normalization to lumi
        Default luminosity is 1 ab^{-1}
        """
        weights = {}
        xsections = {'uu': 1.61, 'dd': 0.4, 'cc': 1.30, 'ss': 0.38, 'charged': 0.514*1.05, 'mixed': 0.486*1.05, 'signal':
                3.43*10**(-4)*1.05, 'Xs': 2.575*10**(-4)*1.05, 'KStarplus': 4.31*10**(-5)*1.05*0.514, 'KStar0': 4.24*10**(-5)*0.486*1.05}
        xsections['Bu'] = xsections['Xs'] * 0.514
        xsections['Bd'] = xsections['Xs'] * 0.486
        for k in event_counts.keys():
            weights[k] = integrated_lumi * xsections[k] / event_counts[k]

        return weights

    @staticmethod
    def jsd_metric(df, selection_fraction=0.005, nbins=32, mbc_min=5.2425, mbc_max=5.29):
        """
        Attempt to quantify sculpting.
        Evaluates mass decorrelation on some blackbox learner by evaluating a discrete
        approximation of the Jensen-Shannon divergence between the distributions of interest
        (here a mass-related quantity) passing and failing some learner threshold. If the 
        learned representation used for classification is noninformative of the variable of
        interest this should be low.
        """

        def _one_hot_encoding(x, nbins):
            x_one_hot = np.zeros((x.shape[0], nbins))
            x_one_hot[np.arange(x.shape[0]), x] = 1
            x_one_hot_sum = np.sum(x_one_hot, axis=0)/x_one_hot.shape[0]

            return x_one_hot_sum

        df_bkg = df[df.label<0.5]
        df_bkg = df_bkg[df_bkg.B_deltaE > -0.25].query('B_deltaE < 0.1')
        select_bkg = df_bkg.nlargest(int(df_bkg.shape[0]*selection_fraction), columns='y_prob')
        min_threshold = select_bkg.y_prob.min()

        df_pass = df_bkg[df_bkg.y_prob > min_threshold]
        df_bkg_pass = df_pass[df_pass.label < 0.5]

        df_fail = df_bkg[df_bkg.y_prob < min_threshold]
        df_bkg_fail = df_fail[df_fail.label < 0.5]

        N_bkg_pass = int(df_bkg_pass._weight_.sum())
        N_bkg_fail = int(df_bkg_fail._weight_.sum())
        print('N_bkg_pass / N_bkg_fail: {}'.format(N_bkg_pass/N_bkg_fail))

        # Discretization
        mbc_bkg_pass_discrete = np.digitize(df_bkg_pass.B_Mbc, bins=np.linspace(mbc_min,mbc_max,nbins+1), right=False)-1
        mbc_bkg_fail_discrete = np.digitize(df_bkg_fail.B_Mbc, bins=np.linspace(mbc_min,mbc_max,nbins+1), right=False)-1

        mbc_bkg_pass_sum = _one_hot_encoding(mbc_bkg_pass_discrete, nbins)
        mbc_bkg_fail_sum = _one_hot_encoding(mbc_bkg_fail_discrete, nbins)

        M = 0.5*mbc_bkg_pass_sum + 0.5*mbc_bkg_fail_sum

        kld_pass = scipy.stats.entropy(mbc_bkg_pass_sum, M)
        kld_fail = scipy.stats.entropy(mbc_bkg_fail_sum, M)

        jsd_discrete = 0.5*kld_pass + 0.5*kld_fail

        return jsd_discrete


    @staticmethod
    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    @staticmethod
    def get_parameter_overview(variables, title, limit=40):
        """Returns a string with variables names, their shapes, count, and types.
        To get all trainable parameters pass in `tf.trainable_variables()`.
        Args:
            variables: List of `tf.Variable`(s).
            limit: If not `None`, the maximum number of variables to include.
        Returns:
            A string with a table like in the example.
        +----------------+---------------+------------+---------+
        | Name           | Shape         | Size       | Type    |
        +----------------+---------------+------------+---------+
        | FC_1/weights:0 | (63612, 1024) | 65,138,688 | float32 |
        | FC_1/biases:0  |       (1024,) |      1,024 | float32 |
        | FC_2/weights:0 |    (1024, 32) |     32,768 | float32 |
        | FC_2/biases:0  |         (32,) |         32 | float32 |
        +----------------+---------------+------------+---------+
        Total: 65,172,512
        """
        print(title)
        max_name_len = max([len(v.name) for v in variables] + [len("Name")])
        max_shape_len = max([len(str(v.get_shape())) for v in variables] + [len(
                "Shape")])
        max_size_len = max([len("{:,}".format(v.get_shape().num_elements()))
                                                for v in variables] + [len("Size")])
        max_type_len = max([len(v.dtype.base_dtype.name) for v in variables] + [len(
                "Type")])

        var_line_format = "| {: <{}s} | {: >{}s} | {: >{}s} | {: <{}s} |"
        sep_line_format = var_line_format.replace(" ", "-").replace("|", "+")

        header = var_line_format.replace(">", "<").format("Name", max_name_len,
                                                          "Shape", max_shape_len,
                                                          "Size", max_size_len,
                                                          "Type", max_type_len)
        separator = sep_line_format.format("", max_name_len, "", max_shape_len, "",
                                           max_size_len, "", max_type_len)

        lines = [separator, header, separator]

        total_weights = sum(v.get_shape().num_elements() for v in variables)

        # Create lines for up to 80 variables.
        for v in variables:
            if limit is not None and len(lines) >= limit:
                lines.append("[...]")
                break
            lines.append(var_line_format.format(
                    v.name, max_name_len,
                    str(v.get_shape()), max_shape_len,
                    "{:,}".format(v.get_shape().num_elements()), max_size_len,
                    v.dtype.base_dtype.name, max_type_len))

        lines.append(separator)
        lines.append("Total: {:,}".format(total_weights))

        print("\n".join(lines))


