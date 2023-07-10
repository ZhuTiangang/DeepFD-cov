import pickle
import os
import keras.optimizers as O

dataset='MNIST'
dir=('../Evaluation/'+dataset+'/normal')
def validate_opt_kwargs(training_config):
    opt_cls = getattr(O, training_config["optimizer"])
    optimizer = opt_cls()
    kwargs = optimizer.get_config()
    print(kwargs)
    training_config["opt_kwargs"] = {k: v for k, v in training_config["opt_kwargs"].items() if (k in kwargs or k=='lr')}
    print(training_config["opt_kwargs"])
    return training_config

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
            with open(config_dir, 'rb') as f:
                training_config = pickle.load(f)

            print(training_config)
            training_config = validate_opt_kwargs(training_config)

            with open(os.path.join(fault_path, 'config.pkl'), 'wb') as f:
                pickle.dump(training_config, f)
    else:
        continue'''
dir = ('../Evaluation/' + dataset + '/normal')
iter = [1]
for model in os.listdir(dir):
    if os.path.isdir(os.path.join(dir, model)):
        print("\nModel : {}".format(model))
        model_path = os.path.join(dir, model)
    else:
        continue
    for fault in os.listdir(model_path):
        if fault == 'origin':
            continue
        else:
            print('\nfault : {}'.format(fault))
            fault_path = os.path.join(model_path, fault)
            if os.path.isdir(fault_path):
                for i in iter:
                    result_dir = (fault_path + '/iter_{}'.format(i) + '/result_dir')
                    # print(result_dir)
                    if os.path.isdir(result_dir):
                        f=[os.path.join(result_dir, f) for f in os.listdir(result_dir) if f.endswith("xlsx")]
                        for each in f:
                            print(each)
                            os.remove(each)


