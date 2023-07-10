import pickle
import os

dataset='MNIST'
dir=('Evaluation/'+dataset+'/normal')
import tensorflow as tf
a = tf.test.is_built_with_cuda()  # 判断CUDA是否可以用
b = tf.test.is_gpu_available(
    cuda_only=False,
    min_cuda_compute_capability=None
)  # 判断GPU是否可以用

print(a) # 显示True表示CUDA可用
print(b) # 显示True表示GPU可用

# 查看驱动名称
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")


'''for model in os.listdir(dir):
    if os.path.isdir(os.path.join(dir, model)):
        print("\nModel : {}".format(model))
        model_path = os.path.join(dir, model)
    else:
        continue
    for fault in os.listdir(model_path):
        print('\nfault : {}'.format(fault))
        fault_path = os.path.join(model_path,fault)
        if os.path.isdir(fault_path):
            config_dir = [os.path.join(fault_path, f) for f in os.listdir(fault_path) if f.endswith("pkl")][0]
            file = open(config_dir, "rb")
            data = pickle.load(file)
            print(data)
            file.close()
    else:
        continue
    config_dir = [os.path.join(model_path, f) for f in os.listdir(model_path) if f.endswith("pkl")][0]
    file=open(config_dir,"rb")
    data=pickle.load(file)
    print(data)
    file.close()
'''