import tensorflow as tf
import numpy as np
import os
import sys
import MyLayer

class Model(tf.keras.Model):
    def __init__(self, opt_name, alpha, lambd, batch_size, epochs):
        super(Model, self).__init__()
        self.lambd = lambd
        self.alpha = alpha
        self.batch_size = batch_size
        self.epochs = epochs
        self.mirrored_strategy = tf.distribute.MirroredStrategy()
        #self.loss_object  = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

        with self.mirrored_strategy.scope():
            self.train_acc =  tf.keras.metrics.SparseCategoricalAccuracy()
            self.test_acc  =  tf.keras.metrics.SparseCategoricalAccuracy()

        self.bn1         = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization(), input_shape=(112, 200, 3))
        self.conv1   = tf.keras.layers.TimeDistributed(MyLayer.ConvLayer([5, 5], 3, 32, [1, 1], "SAME"))
        self.pool1    = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D())
        
        self.bn2         = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())
        self.conv2 = tf.keras.layers.TimeDistributed(MyLayer.ConvLayer([3, 3], 32, 64, [1, 1], "SAME"))
        self.pool2    = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D())        

        self.bn3         = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())
        self.conv3 = tf.keras.layers.TimeDistributed(MyLayer.ConvLayer([3, 3],64, 128, [1, 1], "SAME"))
        self.pool3    = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D())        

        self.bn4         = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())
        self.conv4 = tf.keras.layers.TimeDistributed(MyLayer.ConvLayer([3, 3], 128, 128, [1, 1], "SAME"))
        self.pool4    = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D())
        
        self.flatten1 =  tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())
        self.bn5        = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())
        self.dense1  = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1024, activation=None))
        self.lstm      = tf.keras.layers.LSTM(1000, time_major=False, activation='tanh', return_sequences=True)
        self.bn6        = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())
        self.dense2  = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(64, activation=None))

        if opt_name=="Adam":
            self.opt             = tf.keras.optimizers.Adam(alpha)
        if opt_name=="Sgd":
            self.opt             = tf.keras.optimizers.SGD(alpha)

    def call(self, inputs, training=None):
        outputs = inputs
        outputs = self.bn1(outputs, mask=mask, training=training)
        outputs = self.conv1(outputs,mask=mask)  
        outputs = self.pool1(outputs,mask=mask)
        
        outputs = self.bn2(outputs, mask=mask, training=training)        
        outputs = self.conv2(outputs, mask=mask)    
        outputs = self.pool2(outputs,mask=mask)

        outputs = self.bn3(outputs, mask=mask, training=training)
        outputs = self.conv3(outputs, mask=mask)
        outputs = self.pool3(outputs, mask=mask)

        outputs = self.bn4(outputs, mask=mask, training=training)
        outputs = self.conv4(outputs, mask=mask)
        outputs = self.pool4(outputs, mask=mask)

        outputs = self.flatten1(outputs,mask=mask)
        outputs = self.bn5(outputs, training=training,mask=mask)
        outputs = self.dense1(outputs, mask=mask) 

        outputs = self.lstm(outputs, mask=mask)   

        outputs = self.bn6(outputs, training=training,mask=mask)
        outputs = self.dense2(outputs, mask=mask) 

        return outputs

    def train_step(self, videos, labels):
        #lossの計算方法は、https://danijar.com/variable-sequence-lengths-in-tensorflow/の"Masking the Cost Function"を参考にしました。
        with tf.GradientTape() as tape:
            pred = self(images, mask, True)
        #,weights=tf.compat.v1.to_float(mask),reduction=Reduction.None
            loss  = tf.compat.v1.losses.softmax_cross_entropy(labels,pred, weights=tf.compat.v1.to_float(mask),reduction=tf.compat.v1.losses.Reduction.NONE)
   
            loss  = tf.reduce_sum(loss, axis=-1)
            #Batchの要素それぞれの有効なSequenceの長さ分だけ割る。
            loss  /= tf.reduce_sum(tf.compat.v1.to_float(mask), axis=1)
            loss_l2 = 0.0
            for v in self.trainable_variables:
                 loss_l2 = loss_l2 + self.lambd*tf.reduce_sum(v**2)/2
            loss = loss + loss_l2
        grads   = tape.gradient(loss, self.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.trainable_variables))
        #input->pred : (BatchSize, sequences, Labels)
        #output->pred : ((BatchSize, sequences)
        a = tf.argmax(pred, axis=2)
        #print("after_argmax")
        #print("aの形状は、", a.numpy().shape)
        #input->pred : (BatchSize, sequences)
        #output->pred : ((BatchSize, sequences)
        a = MyLibrary.bincount(a, 64, mask, axis=-1)
        #input->pred : (BatchSize, sequences)
        #output->pred : ((BatchSize, labels)
        #print("after_bincount")
        #print("aの形状は、", a.numpy().shape)
        #print("bの形状は、", labels.numpy().shape)
        a = tf.argmax(a, axis=1)
        b = tf.reduce_max(tf.argmax(labels, axis=-1), axis=-1)
        #print("after_argmax")
        #print("aの形状は、", a.numpy().shape)
        #print("bの形状は、", b.numpy().shape)
        self.train_acc.update_state(b, a)    
        #self.train_acc.update_state(labels, pred)
        loss = tf.nn.compute_average_loss(loss, global_batch_size=self.BATCH_SIZE)
        return loss

    def test_step(self, videos, labels):
        pred = self(images, mask, False)
        loss  = tf.compat.v1.losses.softmax_cross_entropy(labels,pred, weights=tf.compat.v1.to_float(mask),reduction=tf.compat.v1.losses.Reduction.NONE)

        #時間軸に沿って、合計する。
        #入力の形状(BatchSize, Sequence)
        loss  = tf.reduce_sum(loss, axis=-1)
        #Batchの要素それぞれの有効なSequenceの長さ分だけ割る。
        loss  /= tf.reduce_sum(tf.compat.v1.to_float(mask), axis=1)
        loss =  tf.nn.compute_average_loss(loss, global_batch_size=self.BATCH_SIZE)
        #input->pred : (BatchSize, sequences, Labels)
        #output->pred : ((BatchSize, sequences)
        a = tf.argmax(pred, axis=2)
        #input->pred : (BatchSize, sequences)
        #output->pred : ((BatchSize, sequences)
        a = MyLibrary.bincount(a, 64, mask, axis=-1)
        #input->pred : (BatchSize, sequences)
        #output->pred : ((BatchSize, labels)
        a = tf.argmax(a, axis=1)
        b = tf.reduce_max(tf.argmax(labels, axis=-1), axis=-1)
        self.test_acc.update_state(b, a)        
        #self.test_acc.update_state(labels, pred)

        return loss

    @tf.function
    def distributed_train_step(self, videos, labels):
            return  self.mirrored_strategy.experimental_run_v2(self.train_step, args=(videos, labels))

    @tf.function
    def distributed_test_step(self, videos, labels):
            return  self.mirrored_strategy.experimental_run_v2(self.test_step,  args=(videos, labels))

    def train(self, train_ds, test_ds):
        with self.mirrored_strategy.scope():
            for epoch in range(self.epochs):

                train_mean_loss = 0.0
                test_mean_loss = 0.0
                num_batches = 0.0
                
                if (epoch+1)%10==0 :
                    self.opt.learning_rate = self.opt.learning_rate*0.1

                for (batch, (train_videos, train_labels)) in enumerate(train_ds): 

                    print("hparam : ", "epoch : ", epoch+1, "batch : ",batch+1)
                    losses            = self.distributed_train_step(train_videos, train_labels)
                    train_mean_loss       = train_mean_loss +self.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM,losses,axis=None)
                    num_batches += 1.0 
            
                train_mean_loss = train_mean_loss / num_batches
                print("the number of train batch : ", batch+1, "train_mean_loss : ", train_mean_loss, "train acc : ", self.train_acc.result())
                num_batches = 0.0
                for (batch, (test_videos, test_labels)) in enumerate(test_ds):
                    losses           = self.distributed_test_step(test_videos, test_labels)
                    test_mean_loss  =  test_mean_loss  + self.mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, losses,axis=None)
                    num_batches += 1.0
    
                #dataplotmanager.y_list.append(train_accuracy.result().result())
                #dataplotmanager.t_list.append(test_accuracy.result().result())
                test_mean_loss = test_mean_loss / num_batches
                print("the number of test batch : ", batch+1)
                print("epoch : ", epoch+1, " | train loss value : ", train_mean_loss.numpy(), ", test loss value : ", test_mean_loss.numpy())
                print("train acc : ", self.train_acc.result(), "test acc : ", self.test_acc.result())
                self.accuracy_reset()
                #weights_filename = "Model" + str(a) + str(l) + ".bin"
                #self.save_weights(weights_filename)#定期的に保存

    def accuracy_reset(self):
        self.train_acc.reset_states()
        self.test_acc.reset_states()

