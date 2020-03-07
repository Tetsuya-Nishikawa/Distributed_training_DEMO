from Model import Model
import numpy as np
import tensorflow as tf
import in_data
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

if __name__ == '__main__':
    alpha = [1**(-2), 1**(-3), 1**(-4)]
    lambd = [0, 1, 10, 100]
    hparam_list = MyLibrary.make_hparam_list(alpha, lambd)
    batch_size = 8
    epochs = 5
    
    #グリッドリサーチ
    for hparm in hparam_list:
        tf.random.set_seed(seed=1234)
        alpha = hparm["alpha"]
        lambd = hparm["lambd"]
        model = Model("Adam", alpha, lambd, batch_size, epochs)

        train_dataset, test_dataset = in_data.read_dataset(batch_size)
        train_ds =   model.mirrored_strategy.experimental_distribute_dataset(train_dataset)
        test_ds  =   model.mirrored_strategy.experimental_distribute_dataset(test_dataset)

        model.train(train_ds)