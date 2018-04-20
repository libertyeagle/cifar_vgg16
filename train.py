from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from vgg16 import vgg16_model_fn

import numpy as np
import tensorflow as tf
import csv
import os

# control your usage of GPUs
# lock on the GPU No.0 & No.1 
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

batch_size = 32
num_epoches = 74

def _parse_function(filename, label):
  image_string = tf.read_file(filename)
  image_decoded = tf.image.decode_png(image_string)
  image_resized = tf.image.resize_images(image_decoded, [224, 224])
  return {"image_data" : image_resized}, label

def train_input_fn():
    description_file = open("dataset_original/train/data_description.csv", 'r')
    csv_reader = csv.reader(description_file)
    dataset_filenames = []
    dataset_labels = []
    
    for item in csv_reader:
        dataset_filenames.append("dataset_original/train/dataset/{:s}".format(item[0]))
        dataset_labels.append(int(item[1]))
    
    filenames = tf.constant(dataset_filenames)
    labels = tf.constant(dataset_labels)
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(_parse_function)
    dataset = dataset.repeat().batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    ret_features, ret_labels = iterator.get_next()
    
    return ret_features, ret_labels

def eval_input_fn():
    description_file = open("dataset_original/test/data_description.csv", 'r')
    csv_reader = csv.reader(description_file)
    dataset_filenames = []
    dataset_labels = []
    
    for item in csv_reader:
        dataset_filenames.append("dataset_original/test/dataset/{:s}".format(item[0]))
        dataset_labels.append(int(item[1]))
        
    filenames = tf.constant(dataset_filenames)
    labels = tf.constant(dataset_labels)
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(_parse_function)
    dataset = dataset.batch(batch_size)
    
    iterator = dataset.make_one_shot_iterator()
    ret_features, ret_labels = iterator.get_next()
    
    return ret_features, ret_labels

def main(unused_argv):
    tensors_to_log_train = {
        "probabilities": "softmax_tensor",
    }

    logging_hook_train = tf.train.LoggingTensorHook(
        tensors=tensors_to_log_train,
        every_n_secs=60
    )

    tensors_to_log_eval = {
        "accuracy_eval": "accuracy_tensor",
        "mean_per_class_accuracy_eval": "mean_per_class_accuracy_tensor"
    }

    logging_hook_eval = tf.train.LoggingTensorHook(
        tensors=tensors_to_log_eval,
        every_n_iter=1
    )
    
    print("initializing estimator...")

    cifar10_classifier = tf.estimator.Estimator(
        model_fn=vgg16_model_fn, model_dir="cifar_vgg16_model",
        params={"n_classes" : 10}
    )
    
    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn,
        max_steps=370000,
        hooks=[logging_hook_train]
    )


    eval_spec = tf.estimator.EvalSpec(
        input_fn=eval_input_fn,
        start_delay_secs=120,
        throttle_secs=120,
        hooks=[logging_hook_eval]
    )

    print("start training...")

    tf.estimator.train_and_evaluate(cifar10_classifier, train_spec, eval_spec)

    print("training finished.")

    eval_results = cifar10_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()