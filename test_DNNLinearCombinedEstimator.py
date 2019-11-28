__NUM_EPOCHS__=5
__EVAL_STEPS__=1

import numpy as np
import pandas as pd
import tensorflow as tf
# from tensorflow import contrib
tf.enable_eager_execution()

training_df: pd.DataFrame = pd.DataFrame(
    data={
        'feature1': np.random.rand(10),
        'feature2': np.random.rand(10),
        'feature3': np.random.rand(10),
        'feature4': np.random.randint(0, 3, 10),
        'feature5': np.random.randint(0, 3, 10),
        'feature6': np.random.randint(0, 3, 10),
        'target1': np.random.rand(10),
        'target2': np.random.randint(2,size=10)
    }
)
features = ['feature1', 'feature2', 'feature3','feature4', 'feature5', 'feature6']
targets = ['target1', 'target2']
Categorical_Cols = ['feature4', 'feature5', 'feature6']
Numerical_Cols = ['feature1', 'feature2', 'feature3']


wide_columns = [tf.feature_column.categorical_column_with_vocabulary_list(key=x, vocabulary_list=[0, 1, -1])
                                    for x in Categorical_Cols]


deep_columns = [tf.feature_column.numeric_column(x) for x in Numerical_Cols]


def input_fn(df):
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
  continuous_cols = {k: tf.constant(df[k].values)
                     for k in Numerical_Cols}
  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
  categorical_cols = {k: tf.SparseTensor(
      indices=[[i, 0] for i in range(df[k].size)],
      values=df[k].values,
      dense_shape=[df[k].size, 1])
                      for k in Categorical_Cols}
  # Merges the two dictionaries into one.
  feature_cols = continuous_cols.copy()
  feature_cols.update(categorical_cols)

  labels =tf.convert_to_tensor(training_df.as_matrix(training_df[targets].columns.tolist()), dtype=tf.int32)

  return feature_cols, labels



def train_input_fn():
  return input_fn(training_df)

def eval_input_fn():
  return input_fn(training_df)


def loss_fn(labels, logits, features):
    target2 = tf.reshape(labels[:, 1], [-1, 1])
    dist = tf.distributions.Normal(loc=0.0, scale=1.0)

    target1 = labels[:, 0]
    p_target1 = tf.exp(logits[:, 0])

    e = tf.reshape((p_target1 - target1), [-1,1])

    # prevent log(0)
    e = e + 1e-8

    error1 = -dist.log_prob(-e)
    error2 = -dist.log_cdf(e)

    error3 = error2 * (1. - target2)
    error4 = error1 * target2

    error5 = error3 + error4
    error5 = error5 * [1, 0]

    with tf.control_dependencies([tf.compat.v1.assert_non_negative(error5)]):
        error = error5

    return error


def get_head():
    head = tf.contrib.estimator.regression_head(
        label_dimension=2,
        loss_fn=loss_fn,
        inverse_link_fn=tf.exp,
        name='target',
        loss_reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE
    )

    return head

def get_model():
    return tf.estimator.DNNLinearCombinedEstimator(
              head=get_head(),
              # wide settings
              linear_feature_columns=wide_columns,
              #     linear_optimizer=tf.train.FtrlOptimizer(...),
              # deep settings
              dnn_feature_columns=deep_columns,
              #     dnn_optimizer=tf.train.ProximalAdagradOptimizer(...),
              dnn_hidden_units=[10, 10])


def train_and_evaluate():
    """ Train the model """
    estimator = get_model()
    serving_feature_spec = tf.feature_column.make_parse_example_spec(
        wide_columns + deep_columns)
    serving_input_receiver_fn = (
        tf.estimator.export.build_parsing_serving_input_receiver_fn(serving_feature_spec))

    exporter = tf.estimator.BestExporter(
        name="predict",
        #       event_file_pattern='*.tfevents.*',
        serving_input_receiver_fn=serving_input_receiver_fn,
        exports_to_keep=2)

    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: train_input_fn(),
        max_steps=__NUM_EPOCHS__)
    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: eval_input_fn(),
        steps=__EVAL_STEPS__,
        exporters=exporter,
        start_delay_secs=1,  # start evaluating after N seconds
        throttle_secs=10)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    print("Finished training")
    estimator.export_saved_model(
        './',
        serving_input_receiver_fn,
        assets_extra=None,
        as_text=False,
        checkpoint_path=None
    )
    
train_and_evaluate()
