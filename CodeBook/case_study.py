import os
import tensorflow as tf
import numpy as np
import sys
import argparse
import pickle
from tensorflow.keras.models import load_model
from functools import reduce
from operator import mul


# def get_num_params():
#     num_params = 0
#     for variable in tf.trainable_variables():
#         shape = variable.get_shape()
#         num_params += reduce(mul, [dim.value for dim in shape], 1)
#     return num_params


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute coverage.')
    parser.add_argument('--base', '-bs', default='./Evaluation', help='Base directory.')
    parser.add_argument('--dataset', '-ds', default='MNIST', help="Dataset.",
                        # choices=["MNIST", "MNIST2", "CIFAR-10", "Blob", "Circle", "Reuters", "IMDB"]
                        )
    parser.add_argument('--iterate', '-iter', type=int, default=1)
    args = parser.parse_args()
    parent_dir = args.base
    dataset_name = args.dataset
    dir = (parent_dir + '/' + dataset_name + '/normal')
    i = args.iterate

    model_num = 0
    para_num = 0
    for model in os.listdir(dir):
        if os.path.isdir(os.path.join(dir, model)):
            print("\nModel : {}".format(model))
            model_path = os.path.join(dir, model)
            # pkl_dir = os.path.join(model_path, "training_config.pkl")
            # with open(pkl_dir, "rb") as f:
            #     config = pickle.load(f)
            #     print(config)
            model = load_model(model_path + "/model.h5")
            # model.summary()
            # print(model.trainable_variables)
            model_num += 1
            cur_params = np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables])
            print(cur_params)
            para_num += cur_params
            # for variable in model.trainable_variables:
            #     shape = variable.get_shape()
            #     print(shape)
            #     num_params += reduce(mul, [dim.value for dim in shape], 1)
            # print(num_params)
        else:
            continue
    print("Average param:{}".format(para_num/model_num))


