import tensorflow as tf
import numpy as np
import MyLibrary

class DenseLayer(tf.keras.layers.Layer):
    def __init__(self, output_dimention, isoutputlayer):
        super(DenseLayer, self).__init__()
        self.output_dimention = output_dimention 
        self.isoutputlayer = isoutputlayer
    def build(self, input_shape):

        self.w = self.add_weight(shape=[input_shape[1], self.output_dimention], initializer = tf.keras.initializers.he_normal(), trainable=True)
        self.b = self.add_weight(shape=[1,], initializer = tf.keras.initializers.he_normal(), trainable=True)
    def compute_mask(self, inputs, mask=None):
        # Just pass the received mask from previous layer, to the next layer or 
        # manipulate it if this layer changes the shape of the input
        return mask
    def call(self, inputs, mask=None):
        if self.isoutputlayer==True:
            return tf.nn.softmax(tf.matmul(inputs, self.w) + self.b)
        else:
            return tf.nn.relu(tf.matmul(inputs, self.w) + self.b)

class ConvLayer(tf.keras.layers.Layer):
    def __init__(self, filter_shape, input_channels, filters, strides_shape, padding):
        super(ConvLayer, self).__init__()
        self.conv_filt_shape = [filter_shape[0], filter_shape[1],  input_channels, filters]
        self.filters = filters
        self.strides             = strides_shape
        self.padding = padding
    def build(self, input_shape):
        self.w = self.add_weight(shape=self.conv_filt_shape, initializer = tf.keras.initializers.he_normal(), trainable=True)
        self.b = self.add_weight(shape=[self.filters,], initializer = tf.keras.initializers.he_normal(), trainable=True)
    
    def compute_mask(self, inputs, mask=None):
        return mask
 
    def call(self, inputs, mask=None):
        filter_shape = [self.conv_filt_shape[0], self.conv_filt_shape[1]]
        #ストライドが2の時は、元の画像サイズに対しての半分に調整
        if self.strides[0] == 2:
            pad = -(-(-2+filter_shape[0])//2)
            inputs = MyLibrary.zero_padding(inputs, pad)
        #inputs   = batch_layer(inputs, 3)
        
        outputs = tf.nn.conv2d(inputs, self.w, strides=self.strides, padding=self.padding) + self.b
        return tf.nn.relu(outputs)

class ResNetLayer(tf.keras.layers.Layer):
    def __init__(self, filters, strides):
        super(ResNetLayer, self).__init__()
        self.filters = filters
        self.strides = strides
        
    def build(self, input_shape):
        input_dimention = input_shape[3]
        if self.strides[0]==2:
            self.conv1                = ConvLayer([1, 1], input_dimention, self.filters//4, self.strides, "VALID")
        else:
            self.conv1                = ConvLayer([1, 1], input_dimention, self.filters//4, self.strides, "SAME")          
        self.conv2                = ConvLayer([3, 3], self.filters//4,                 self.filters//4, [1, 1], "SAME")
        self.conv3                = ConvLayer([1, 1], self.filters//4,                 self.filters     , [1, 1], "SAME")
        if self.strides[0]==2:
            self.shortcut = ConvLayer([3, 3], input_dimention, self.filters, [2, 2], "VALID")
        else:
            self.shortcut = ConvLayer([3, 3], input_dimention, self.filters, [1, 1], "SAME")


    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, inputs, mask=None):

        block1 = self.conv1(inputs)
        block2 = self.conv2(block1)
        outputs = self.conv3(block2)
      
        outputs = outputs + self.shortcut(inputs)

        return tf.nn.relu(outputs)

#ConvLSTM(https://arxiv.org/pdf/1506.04214v1.pdf)
class ConvLSTM2D(tf.compat.v1.nn.rnn_cell.RNNCell):

    def __init__(self, units, filters, strides,**kwargs):
        super(ConvLSTM2D, self).__init__(**kwargs)
        self.filters = filters
        self.strides = strides
        #units->(height, width, channels)
        #strides=[1, 1]、paddingなしの場合のhiddenの形状計算
        units[0] = (units[0] - 3)//self.strides[0] + 1
        units[1] = (units[1] - 3)//self.strides[1] + 1
        units[2] = self.filters
        state_size  = tf.TensorShape(units)
        self._state_size = tf.compat.v1.nn.rnn_cell.LSTMStateTuple(state_size, state_size)
        self._output_size = tf.TensorShape(state_size)
    
    def build(self, input_shape):
        
        #tf.keras.layers.Conv2Dに対して、activationを指定しない場合、活性化関数を使用しないことになる。
        self.conv_xi = tf.keras.layers.Conv2D(self.filters, 3, strides=self.strides, kernel_initializer='glorot_uniform')
        self.conv_hi = tf.keras.layers.Conv2D(self.filters, 3, strides=[1,1], kernel_initializer='glorot_uniform', padding="same")
        self.conv_xf = tf.keras.layers.Conv2D(self.filters, 3, strides=self.strides, kernel_initializer='glorot_uniform')
        self.conv_hf = tf.keras.layers.Conv2D(self.filters, 3, strides=[1,1], kernel_initializer='glorot_uniform', padding="same")
        self.conv_xo = tf.keras.layers.Conv2D(self.filters, 3, strides=self.strides, kernel_initializer='glorot_uniform')
        self.conv_ho = tf.keras.layers.Conv2D(self.filters, 3, strides=[1,1], kernel_initializer='glorot_uniform', padding="same")
        self.conv_xg = tf.keras.layers.Conv2D(self.filters, 3, strides=self.strides, kernel_initializer='glorot_uniform')
        self.conv_hg = tf.keras.layers.Conv2D(self.filters, 3, strides=[1,1], kernel_initializer='glorot_uniform', padding="same")
        #RECURRENT BATCH NORMALIZATION(https://arxiv.org/pdf/1603.09025.pdf)
        self.batch_xi = tf.keras.layers.BatchNormalization()
        self.batch_hi = tf.keras.layers.BatchNormalization()
        self.batch_xf = tf.keras.layers.BatchNormalization()
        self.batch_hf = tf.keras.layers.BatchNormalization()
        self.batch_xo = tf.keras.layers.BatchNormalization()
        self.batch_ho = tf.keras.layers.BatchNormalization()
        self.batch_xg = tf.keras.layers.BatchNormalization()
        self.batch_hg = tf.keras.layers.BatchNormalization()
        self.batch_cell = tf.keras.layers.BatchNormalization()
        self.build = True
    
    @property
    def output_size(self):
        return self._output_size
    @property
    def state_size(self):
        return self._state_size   

    def compute_mask(self, inputs, mask=None):
        # Just pass the received mask from previous layer, to the next layer or 
        # manipulate it if this layer changes the shape of the input
        return mask
    
    def call(self, inputs, states, mask=None, training=True):
        cell, hidden = states
        f = tf.nn.sigmoid(self.batch_xf(self.conv_xf(inputs)) + self.batch_hf(self.conv_hf(hidden)))
        i = tf.nn.sigmoid(self.batch_xi(self.conv_xi(inputs)) + self.batch_hi(self.conv_hi(hidden)))
        o = tf.nn.sigmoid(self.batch_xo(self.conv_xo(inputs)) + self.batch_hi(self.conv_ho(hidden)))
        g = tf.nn.tanh(self.batch_xg(self.conv_xg(inputs)) + self.batch_hg(self.conv_hg(hidden)))
        #statesの更新!!       
        new_cell      = f * cell + (i * g)
        new_cell      = self.batch_cell(new_cell, training=training)
        
        new_hidden = o * tf.nn.tanh(new_cell)
        new_state   = tf.compat.v1.nn.rnn_cell.LSTMStateTuple(new_cell, new_hidden)
        
        return new_hidden, new_state

#https://www.tensorflow.org/api_docs/python/tf/compat/v1/nn/rnn_cell/LSTMCell
#ConvLSTM(https://arxiv.org/pdf/1506.04214v1.pdf)
#論文の(3)式を実装

#https://github.com/tensorflow/tensorflow/blob/v2.1.0/tensorflow/python/ops/rnn_cell_impl.py#L804-L1081のLSTMCellを参考にしました。
#上のgithubに記載されているLSTMCellを大雑把に眺めたところ、
#1.hiddenとcellの形状をtf.compat.v1.nn.rnn_cell.LSTMStateTuple()で指定する。
#2.call関数の引数statesに、hiddenとcellが格納されるらしい。
#3.build関数には、self.build=Trueを必ず書く。
#4.新しいhiddenとcellは、tf.compat.v1.nn.rnn_cell.LSTMStateTuple(new_cell, new_hidden)で登録する。
#上の4つを守れば、LSTMCellは自由にカスタマイズしても良さそうだと思った。
class ConvLSTM2D(tf.compat.v1.nn.rnn_cell.RNNCell):

    #__init__関数に渡されるunitsには、inputの形状を入力する。
    def __init__(self, units, filters, strides,**kwargs):
        super(ConvLSTM2D, self).__init__(**kwargs)
        self.filters = filters
        self.strides = strides
        #units->(height, width, channels)
        #strides=[1, 1]、paddingなしの場合のhiddenの形状計算
        #inputsに対して畳み込みを適用した後の次元とhiddenの次元を合わせるために調整
        units[0] = (units[0] - 3)//self.strides[0] + 1
        units[1] = (units[1] - 3)//self.strides[1] + 1
        units[2] = self.filters
        state_size  = tf.TensorShape(units)
        self._state_size = tf.compat.v1.nn.rnn_cell.LSTMStateTuple(state_size, state_size)
        self._output_size = tf.TensorShape(state_size)
    
    def build(self, input_shape):
        
        #tf.keras.layers.Conv2Dに対して、activationを指定しない場合、活性化関数を使用しないことになる。
        #とりあえす、フィルターの形状を[3,3]で固定
        self.conv_xi = tf.keras.layers.Conv2D(self.filters, 3, strides=self.strides, kernel_initializer='glorot_uniform')
        self.conv_hi = tf.keras.layers.Conv2D(self.filters, 3, strides=[1,1], kernel_initializer='glorot_uniform', padding="same")
        self.conv_ci = tf.keras.layers.Conv2D(self.filters, 3, strides=[1,1], kernel_initializer='glorot_uniform', padding="same")

        self.conv_xf = tf.keras.layers.Conv2D(self.filters, 3, strides=self.strides, kernel_initializer='glorot_uniform')
        self.conv_hf = tf.keras.layers.Conv2D(self.filters, 3, strides=[1,1], kernel_initializer='glorot_uniform', padding="same")
        self.conv_cf = tf.keras.layers.Conv2D(self.filters, 3, strides=[1,1], kernel_initializer='glorot_uniform', padding="same")

        self.conv_xo = tf.keras.layers.Conv2D(self.filters, 3, strides=self.strides, kernel_initializer='glorot_uniform')
        self.conv_ho = tf.keras.layers.Conv2D(self.filters, 3, strides=[1,1], kernel_initializer='glorot_uniform', padding="same")
        self.conv_co = tf.keras.layers.Conv2D(self.filters, 3, strides=[1,1], kernel_initializer='glorot_uniform', padding="same")

        self.conv_xg = tf.keras.layers.Conv2D(self.filters, 3, strides=self.strides, kernel_initializer='glorot_uniform')
        self.conv_hg = tf.keras.layers.Conv2D(self.filters, 3, strides=[1,1], kernel_initializer='glorot_uniform', padding="same")
        self.conv_cg = tf.keras.layers.Conv2D(self.filters, 3, strides=[1,1], kernel_initializer='glorot_uniform', padding="same")

        #RECURRENT BATCH NORMALIZATION(https://arxiv.org/pdf/1603.09025.pdf)
        self.batch_xi = tf.keras.layers.BatchNormalization()
        self.batch_hi = tf.keras.layers.BatchNormalization()
        self.batch_ci = tf.keras.layers.BatchNormalization()

        self.batch_xf = tf.keras.layers.BatchNormalization()
        self.batch_hf = tf.keras.layers.BatchNormalization()
        self.batch_cf = tf.keras.layers.BatchNormalization()

        self.batch_xo = tf.keras.layers.BatchNormalization()
        self.batch_ho = tf.keras.layers.BatchNormalization()
        self.batch_co = tf.keras.layers.BatchNormalization()

        self.batch_xg = tf.keras.layers.BatchNormalization()
        self.batch_hg = tf.keras.layers.BatchNormalization()
        self.batch_cg = tf.keras.layers.BatchNormalization()

        self.batch_cell = tf.keras.layers.BatchNormalization()
        self.build = True
        self.b_i = self.add_weight(shape=[self.filters,], initializer = tf.keras.initializers.he_normal(), trainable=True, name="bf")
        self.b_f = self.add_weight(shape=[self.filters,], initializer = tf.keras.initializers.he_normal(), trainable=True, name="bi")
        self.b_o = self.add_weight(shape=[self.filters,], initializer = tf.keras.initializers.he_normal(), trainable=True, name="bo")
        self.b_g = self.add_weight(shape=[self.filters,], initializer = tf.keras.initializers.he_normal(), trainable=True, name="bg")
  
    @property
    def output_size(self):
        return self._output_size
    @property
    def state_size(self):
        return self._state_size   

    #maskを次の層に伝えるために、この関数は必要らしい。(この辺りのバックエンドの動きをよく分かっていません。)
    def compute_mask(self, inputs, mask=None):
        return mask
    
    def call(self, inputs, states, mask=None, training=True):
        cell, hidden = states
        f = tf.nn.sigmoid(self.batch_xf(self.conv_xf(inputs), training=training) + self.batch_hf(self.conv_hf(hidden), training=training) + self.batch_cf(self.conv_hf(hidden), training=training) + self.b_f)
        i = tf.nn.sigmoid(self.batch_xi(self.conv_xi(inputs), training=training) + self.batch_hi(self.conv_hi(hidden), training=training) + self.batch_ci(self.conv_hf(hidden), training=training) + self.b_i)
        o = tf.nn.sigmoid(self.batch_xo(self.conv_xo(inputs), training=training) + self.batch_ho(self.conv_ho(hidden), training=training) + self.batch_co(self.conv_hf(hidden), training=training) + self.b_o)
        g = tf.nn.tanh(self.batch_xg(self.conv_xg(inputs), training=training) + self.batch_hg(self.conv_hg(hidden), training=training) + self.batch_cg(self.conv_hf(hidden), training=training) + self.b_g)
        #statesの更新   
        new_cell      = f * cell + (i * g)
        new_cell      = self.batch_cell(new_cell, training=training)
        
        new_hidden = o * tf.nn.tanh(new_cell)
        new_state   = tf.compat.v1.nn.rnn_cell.LSTMStateTuple(new_cell, new_hidden)
        
        return new_hidden, new_state
