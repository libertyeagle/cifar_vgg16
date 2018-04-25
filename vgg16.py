import numpy as np
import tensorflow as tf

weight_decay = 0.0005

def _activation_summary(x):
    tf.summary.histogram(x.op.name + '/activations', x)
    tf.summary.scalar(x.op.name + '/sparsity', tf.nn.zero_fraction(x))
    
def vgg16_model_fn(features, labels, mode, params):
    n_classes = params['n_classes']
    learning_rate = params['learning_rate']

    input_layer = tf.reshape(features["image_data"], [-1, 32, 32, 3])
    conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=64,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
      name="conv1",
    )
    _activation_summary(conv1)

    conv2 = tf.layers.conv2d(
      inputs=conv1,
      filters=64,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
      name="conv2"
    )
    _activation_summary(conv2)

    pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, name="pool1")
    
    conv3 = tf.layers.conv2d(
      inputs=pool1,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
      name="conv3"
    )
    _activation_summary(conv3)
    
    conv4 = tf.layers.conv2d(
      inputs=conv3,
      filters=128,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
      name="conv4"
    )
    _activation_summary(conv4)

    pool2 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2, name="pool2")

    conv5 = tf.layers.conv2d(
      inputs=pool2,
      filters=256,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      name="conv5"
    )
    _activation_summary(conv5)
    
    conv6 = tf.layers.conv2d(
      inputs=conv5,
      filters=256,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
      name="conv6"
    )
    _activation_summary(conv6)
    
    conv7 = tf.layers.conv2d(
      inputs=conv6,
      filters=256,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
      name="conv7"
    )
    _activation_summary(conv7)

    pool3 = tf.layers.max_pooling2d(inputs=conv7, pool_size=[2, 2], strides=2, name="pool3")

    conv8 = tf.layers.conv2d(
      inputs=pool3,
      filters=512,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
      name="conv8"
    )
    _activation_summary(conv8)

    conv9 = tf.layers.conv2d(
      inputs=conv8,
      filters=512,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
      name="conv9"
    )
    _activation_summary(conv9)

    conv10 = tf.layers.conv2d(
      inputs=conv9,
      filters=512,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      name="conv10"
    )
    _activation_summary(conv10)

    pool4 = tf.layers.max_pooling2d(inputs=conv10, pool_size=[2, 2], strides=2, name="pool4")

    conv11 = tf.layers.conv2d(
      inputs=pool4,
      filters=512,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
      name="conv11"
    )
    _activation_summary(conv11)

    conv12 = tf.layers.conv2d(
      inputs=conv11,
      filters=512,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
      name="conv12"
    )
    _activation_summary(conv12)

    conv13 = tf.layers.conv2d(
      inputs=conv12,
      filters=512,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu,
      kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
      name="conv13"
    )
    _activation_summary(conv13)

    pool5 = tf.layers.max_pooling2d(inputs=conv13, pool_size=[2, 2], strides=2, name="pool5")
    pool5_flat = tf.reshape(pool5, [-1, 1 * 1 * 512])
    
    fc1 = tf.layers.dense(
      inputs=pool5_flat,
      units=1024,
      activation=tf.nn.relu,
      kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
      name="fc1"
    )

    fc1_dropout = tf.layers.dropout(
        inputs=fc1, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN, name="fc1_dropout"
    )
    _activation_summary(fc1_dropout)

    logits = tf.layers.dense(inputs=fc1_dropout, units=n_classes, name="logits")

    predictions = {
      "classes": tf.argmax(input=logits, axis=1),
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor"),
      "logits": logits
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits) + tf.losses.get_regularization_loss()

    if mode == tf.estimator.ModeKeys.TRAIN:
        for var in tf.trainable_variables():
          tf.summary.histogram(var.op.name, var)

        optimizer = tf.train.MomentumOptimizer(momentum=0.9, learning_rate=learning_rate)

        grads = optimizer.compute_gradients(loss)

        for grad, var in grads:
          if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

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
    
