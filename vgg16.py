import numpy as np
import tensorflow as tf

def vgg16_model_fn(features, labels, mode, params):
    n_classes = params['n_classes']

    input_layer = tf.reshape(features["image_data"], [-1, 224, 224, 3])
    conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=64,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)
    
    conv2 = tf.layers.conv2d(
      inputs=conv1,
      filters=64,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    
    conv3 = tf.layers.conv2d(
      inputs=pool1,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)
    
    conv4 = tf.layers.conv2d(
      inputs=conv3,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)
    
    pool2 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

    conv5 = tf.layers.conv2d(
      inputs=pool2,
      filters=256,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)
    
    conv6 = tf.layers.conv2d(
      inputs=conv5,
      filters=256,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

    conv7 = tf.layers.conv2d(
      inputs=conv6,
      filters=256,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

    pool3 = tf.layers.max_pooling2d(inputs=conv7, pool_size=[2, 2], strides=2)

    conv8 = tf.layers.conv2d(
      inputs=pool3,
      filters=512,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

    conv9 = tf.layers.conv2d(
      inputs=conv8,
      filters=512,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

    conv10 = tf.layers.conv2d(
      inputs=conv9,
      filters=512,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

    pool4 = tf.layers.max_pooling2d(inputs=conv10, pool_size=[2, 2], strides=2)

    conv11 = tf.layers.conv2d(
      inputs=pool4,
      filters=512,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

    conv12 = tf.layers.conv2d(
      inputs=conv11,
      filters=512,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

    conv13 = tf.layers.conv2d(
      inputs=conv12,
      filters=512,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

    pool5 = tf.layers.max_pooling2d(inputs=conv13, pool_size=[2, 2], strides=2)
    pool5_flat = tf.reshape(pool5, [-1, 7 * 7 * 512])
    
    fc1 = tf.layers.dense(inputs=pool5_flat, units=4096, activation=tf.nn.relu)
    fc1_dropout = tf.layers.dropout(
        inputs=fc1, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN
    )

    fc2 = tf.layers.dense(inputs=fc1_dropout, units=4096, activation=tf.nn.relu)
    fc2_dropout = tf.layers.dropout(
        inputs=fc2, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN
    )

    logits = tf.layers.dense(inputs=fc2_dropout, units=n_classes)

    predictions = {
      "classes": tf.argmax(input=logits, axis=1),
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
      "logits": logits
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.MomentumOptimizer(momentum=0.9, learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
          labels=labels,
          predictions=predictions["classes"],
          name="accuracy_eval"
        ),
        "mean_per_class_accuracy": tf.metrics.mean_per_class_accuracy(
          labels=labels,
          predictions=predictions["classes"],
          num_classes=n_classes,
          name="mean_per_class_accuracy_eval"
        )
    }

    accuracy_tensor = tf.identity(eval_metric_ops["accuracy"][0], "accuracy_tensor")
    mean_per_class_accuracy_tensor = tf.identity(
        eval_metric_ops["mean_per_class_accuracy"][0],
        "mean_per_class_accuracy_tensor"
    )

    tf.summary.scalar("accuracy", eval_metric_ops["accuracy"][1])
    tf.summary.scalar("mean_per_class_accuracy", eval_metric_ops["mean_per_class_accuracy"][1])

    # the only case left : evaluation
    return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
    
