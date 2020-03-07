import tensorflow as tf
import numpy as np
import os
import sys
import MyLayer

class Model(tf.keras.Model):
    def __init__(self, opt_name, alpha, lambd, batch_size, epochs):
        self.lambd = lambd
        self.alpha = alpha
        self.batch_size = batch_size
        self.epochs = epochs
        self.mirrored_strategy = tf.distribute.MirroredStrategy()
        
        with self.mirrored_strategy.scope():
            self.train_acc = tf.keras.losses.SparseCategoricalCrossentropy()
            self.test_acc  = tf.keras.losses.SparseCategoricalCrossentropy()

        self.bn1         = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization(), input_shape=(112, 200, 3))
        self.conv_1   = tf.keras.layers.TimeDistributed(MyLayer.ConvLayer([3, 3], 3, 16, [2, 2], "VALID"))
        self.pool1    = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D())
        self.flatten1 =  tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())

        self.bn2         = tf.keras.layers.BatchNormalization()
        self.dense1  = MyLayer.DenseLayer(2048, False)
        
        self.bn3         = tf.keras.layers.BatchNormalization()
        self.dense2  = MyLayer.DenseLayer(1024, False)
        
        self.bn4         = tf.keras.layers.BatchNormalization()
        self.dense3  = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(64, activation='none'))

        def call(self, inputs, training=None):
            outputs = inputs


            return outputs

        def train_step(self, images, labels):
            with tf.GradientTape() as tape:
                pred = self(images, True)
                loss = tf.compat.v1.losses.softmax_cross_entropy(onehot_labels=labels, logits=pred, reduction=tf.compat.v1.losses.Reduction.NONE)
            grads = tape.gradient(loss, self.trainable_variables)
            self.opt.apply_gradients(zip(grads, self.trainable_variables))
            loss = tf.nn.compute_average_loss(loss, global_batch_size=self.batch_size)
            return loss

        def test_step(self, images, labels, mask):
            pred = self(images, mask, False)
            loss  = tf.compat.v1.losses.softmax_cross_entropy(onehot_labels=labels, logits=pred, reduction=tf.compat.v1.losses.Reduction.NONE)
            loss =  tf.nn.compute_average_loss(loss, global_batch_size=self.batch_size)

            return loss

        @tf.function
        def distributed_train_step(self, images, labels, mask):
                from main import mirrored_strategy#この汚い部分は、後で修正
                return  self.mirrored_strategy.experimental_run_v2(self.train_step, args=(images, labels, mask))

        @tf.function
        def distributed_test_step(self, images, labels, mask):
                from main import mirrored_strategy
                return  self.mirrored_strategy.experimental_run_v2(self.test_step,  args=(images, labels, mask))

        def train(self, train_ds, test_ds):
            with self.mirrored_strategy.scope():
                for epoch in range(self.epochs):

                    train_mean_loss = 0.0
                    test_mean_loss = 0.0
                    num_batches = 0.0
                    
                    if (epoch+1)%10==0 :
                        self.opt.learning_rate = self.opt.learning_rate*0.1

                    for (batch, (train_images, train_labels, masks)) in enumerate(train_ds): 
                        print("hparam : ", params[param], "epoch : ", epoch+1, "batch : ",batch+1)
                        losses            = self.distributed_train_step(train_images, train_labels, masks)
                        train_mean_loss       = train_mean_loss +mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM,losses,axis=None)
                        num_batches += 1.0 
                
                    train_mean_loss = train_mean_loss / num_batches
                    print("the number of train batch : ", batch+1, "train_mean_loss : ", train_mean_loss, "train acc : ", train_acc.numpy()*100)
                    num_batches = 0.0
                    for (batch, (test_images, test_labels, masks)) in enumerate(test_ds):
                        losses           = self.distributed_test_step(test_images, test_labels, masks)
                        test_mean_loss  =  test_mean_loss  + mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, losses,axis=None)
                        num_batches += 1.0
        
                    #dataplotmanager.y_list.append(train_accuracy.result().numpy()*100)
                    #dataplotmanager.t_list.append(test_accuracy.result().numpy()*100)
                    test_mean_loss = test_mean_loss / num_batches
                    print("the number of test batch : ", batch+1)
                    print("epoch : ", epoch+1, " | train loss value : ", train_mean_loss.numpy(), ", test loss value : ", test_mean_loss.numpy())
                    print("train acc : ", train_acc.numpy()*100, "test acc : ", test_acc.numpy()*100)
                    self.accuracy_reset()
                    #weights_filename = "Model" + str(a) + str(l) + ".bin"
                    #self.save_weights(weights_filename)#定期的に保存

        def accuracy_reset(self):
            self.train_acc.reset_states()
            self.test_acc.reset_states()
    
