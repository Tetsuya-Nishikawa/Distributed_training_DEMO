from Model import Model
import numpy as np
import tensorflow as tf
import io_data
import MyLibrary 
def tensor_cast(inputs, labels):
    inputs = tf.cast(inputs, tf.float32)
    labels = labels - 1
    inputs = tf.reshape(inputs, [1, inputs.numpy().shape[0], inputs.numpy().shape[1], inputs.numpy().shape[2]])
    return inputs, tf.cast(labels, tf.int64)

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

        #train_dataset, test_dataset = in_data.read_dataset(batch_size)
        train_images, train_labels, test_images, test_labels = MyLibrary.ReadMnistDataset()
        #padded_shape = (tf.constant(-1.0, dtype=tf.float32), tf.constant(0, dtype=tf.int64), tf.constant(False, dtype=tf.bool))
        train_dataset = tf.data.Dataset.zip((train_images, train_labels)).map(tensor_cast).shuffle(buffer_size=10, seed=100).prefetch(tf.data.experimental.AUTOTUNE)
        test_dataset  = tf.data.Dataset.zip((test_images, test_labels)).map(tensor_cast).shuffle(buffer_size=10, seed=100).prefetch(tf.data.experimental.AUTOTUNE)

        train_ds =   model.mirrored_strategy.experimental_distribute_dataset(train_dataset)
        test_ds  =   model.mirrored_strategy.experimental_distribute_dataset(test_dataset)

        model.train(train_ds, test_dataset)