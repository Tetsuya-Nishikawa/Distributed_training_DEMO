from Model import Model
import numpy as np
import tensorflow as tf
import io_data
import MyLibrary 

gpus = tf.config.experimental.list_physical_devices('GPU')
#GPUのメモリを制限する。
if gpus:
  try:
    for gpu in gpus:
        print(gpu)
        tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    print(e)
print("GPUのメモリを制限する")
def tensor_cast(inputs, labels):
    inputs = tf.cast(inputs, tf.float32)/255.0
    labels = labels 
    #inputs = tf.reshape(inputs, [1, inputs.shape[0], inputs.shape[1], inputs.shape[2]])
    return inputs, tf.cast(labels, tf.int64)


if __name__ == '__main__':
    alpha = [10**(-2), 10**(-3), 10**(-4)]
    lambd = [0, 1, 10, 100]
    hparam_list = MyLibrary.make_hparam_list(alpha, lambd)
    batch_size = 10
    epochs = 5
    
    #グリッドリサーチ
    print(hparam_list)
    for hparm_key in hparam_list:

        tf.random.set_seed(seed=1234)
        a = hparam_list[hparm_key]["alpha"]
        l = hparam_list[hparm_key]["lambd"]
        model = Model("Adam", a, l, batch_size, epochs)
        train_dataset, test_dataset = io_data.read_dataset(batch_size)
        train_ds =   model.mirrored_strategy.experimental_distribute_dataset(train_dataset)
        test_ds  =   model.mirrored_strategy.experimental_distribute_dataset(test_dataset)
        
        model.train(train_ds, test_ds)
