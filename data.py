#!/usr/bin/python3
import tensorflow as tf
import numpy as np
import pandas as pd

class Data(object):

    @staticmethod
    def load_data(filename, evaluate=False, adversary=False, parquet=False, tune=False,
                  pivots=['_pivot'], auxillary=None, adv_n_classes=10):

        if parquet:
            import pyarrow.parquet as pq
            dataset = pq.ParquetDataset(filename)
            df = dataset.read(nthreads=4).to_pandas()
        else:
            df = pd.read_hdf(filename, key='df')

        if not evaluate:
            df = df.sample(frac=1).reset_index(drop=True)

        if auxillary is None:
            # Cleanup + omit variables prefixed with an underscore from training
            auxillary = [col for col in df.columns if col.startswith('_')]

        df_features = df.drop(auxillary, axis=1)
        print('Data shape:', df_features.shape)
        print('Features', df_features.columns.tolist())

        pivot_df = df[pivots]
        pivot_features = pivot_df[pivots]

        if adversary:
            # Bin variable -> discrete classification problem
            # Each protected variable must be binned separately
            if len(pivots) == 1:
                pivot = pivots[0]
            pivot_df = pivot_df.assign(pivot_labels=pd.qcut(df[pivot], q=adv_n_classes, labels=False))
            pivot_labels = pivot_df.pivot_labels

        marginal_df = pivot_features[pivots]
        marginal_df = marginal_df.sample(frac=1)
        marginal_df = marginal_df.reset_index(drop=True)
        pivot_features[[pivot + '_marginal' for pivot in pivots]] = marginal_df

        if evaluate:
            if tune:
                from ray.tune.util import pin_in_object_store as pin
                return pin(np.nan_to_num(df_features.values)), pin(df._label.values.astype(np.int32)), \
                    pin(pivot_features.values.astype(np.float32)), pin(df[auxillary])
            else:
                return df, np.nan_to_num(df_features.values), df._label.values.astype(np.int32), \
                    pivot_features.values.astype(np.float32)
        else:
            if adversary:
                return np.nan_to_num(df_features.values), df._label.values.astype(np.int32), \
                    pivot_features.values.astype(np.float32), pivot_labels.values.astype(np.int32)
            else:
                if tune:
                    from ray.tune.util import pin_in_object_store as pin
                    return pin(np.nan_to_num(df_features.values)), pin(df._label.values.astype(np.int32)), \
                        pin(pivot_features.values.astype(np.float32))
                else:
                    return np.nan_to_num(df_features.values), df._label.values.astype(np.int32), \
                        pivot_features.values.astype(np.float32)


    @staticmethod
    def load_dataset(features_placeholder, labels_placeholder, pivots_placeholder, batch_size, test=False,
            evaluate=False, sequential=False, prefetch_size=2, adversary=False, pivot_labels_placeholder=None):
    
        if adversary:
            dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder, 
                pivots_placeholder, pivot_labels_placeholder))
            padded_shapes = (tf.TensorShape([None]), tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([]))
            padding_values = (0.,0,0.,0)
        else:
            dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder,
                pivots_placeholder))
            padded_shapes = (tf.TensorShape([None]), tf.TensorShape([]), tf.TensorShape([]))
            padding_values = (0.,0,0.)
        
        # Retain order if evaluate=True
        if evaluate is False:
            dataset = dataset.shuffle(buffer_size=10**5)

        if sequential:
            dataset = dataset.padded_batch(
                batch_size,
                padded_shapes=padded_shapes, 
                padding_values=padding_values,
                drop_remainder=True)
        else:
            # Don't bottleneck GPU with data loading
            dataset = dataset.batch(batch_size, drop_remainder=True)
            dataset = dataset.prefetch(prefetch_size)

        if test is True:
            dataset = dataset.repeat()

        return dataset


    @staticmethod
    def load_tfr_dataset(filenames_placeholder, n_features, batch_size, test=False, evaluate=False):

        # TODO: Parallelize tfrecord decoding, fuse map/batch
        num_parallel_calls = 4
        prefetch_size = 4

        def _parse_function(example_proto):
            features_desc = {"data": tf.FixedLenFeature([n_features], tf.float32),
                             "labels": tf.FixedLenFeature((), tf.float32),
                             "meta": tf.FixedLenFeature((), tf.float32)}
            parsed_features = tf.parse_single_example(example_proto, features_desc)

            return parsed_features["data"], parsed_features["labels"], parsed_features["meta"]

        dataset = tf.data.TFRecordDataset(filenames_placeholder)
         # Parse the record into tensors.
        dataset = dataset.map(_parse_function, num_parallel_calls=num_parallel_calls)

        if test:  # Repeat input indefinitely
            dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=int(1e6), count=None))
        else:
            if evaluate is False:  # Retain order for evaluation
                dataset = dataset.shuffle(buffer_size=int(1e6))

        dataset = dataset.batch(batch_size, drop_remainder=True)

        # Enqueue batches on CPU 
        dataset = dataset.prefetch(prefetch_size)

        return dataset
