from Model import Model
import numpy as np
import tensorflow as tf
import io_data
import MyLibrary 
def tensor_cast(inputs, labels):
    inputs = tf.cast(inputs, tf.float32)
    labels = labels - 1
    inputs = tf.reshape(inputs, [1, inputs.shape[0], inputs.shape[1], inputs.shape[2]])
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
    alpha = [10**(-2), 10**(-3), 10**(-4)]
    lambd = [0, 1, 10, 100]
    hparam_list = MyLibrary.make_hparam_list(alpha, lambd)
    batch_size = 8
    epochs = 5
    
    #グリッドリサーチ
    print(hparam_list)
    for hparm_key in hparam_list:

        tf.random.set_seed(seed=1234)
        a = hparam_list[hparm_key]["alpha"]
        l = hparam_list[hparm_key]["lambd"]
        model = Model("Adam", a, l, batch_size, epochs)

        #train_dataset, test_dataset = in_data.read_dataset(batch_size)
        train_images, train_labels, test_images, test_labels = MyLibrary.ReadMnistDataset()
        #padded_shape = (tf.constant(-1.0, dtype=tf.float32), tf.constant(0, dtype=tf.int64), tf.constant(False, dtype=tf.bool))
        train_images = tf.data.Dataset.from_tensors(train_images)
        train_labels = tf.data.Dataset.from_tensors(train_labels)
        test_images = tf.data.Dataset.from_tensors(test_images)
        test_labels = tf.data.Dataset.from_tensors(test_labels)

        train_dataset = tf.data.Dataset.zip((train_images, train_labels)).map(tensor_cast).shuffle(buffer_size=10, seed=100).prefetch(tf.data.experimental.AUTOTUNE)
        test_dataset  = tf.data.Dataset.zip((test_images, test_labels)).map(tensor_cast).shuffle(buffer_size=10, seed=100).prefetch(tf.data.experimental.AUTOTUNE)

        train_ds =   model.mirrored_strategy.experimental_distribute_dataset(train_dataset)
        test_ds  =   model.mirrored_strategy.experimental_distribute_dataset(test_dataset)

        model.train(train_ds, test_ds)