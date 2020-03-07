import tensorflow as tf
import scipy.stats as stats

def zero_padding(inputs, pad):
    padding = tf.constant([[0, 0], [pad, pad], [pad, pad], [0, 0]])
    padded_inputs = tf.pad(inputs, padding)
    return padded_inputs

#onehot表現に変換する。
def make_onehot_labels(labels, depth):
    one_hot_labels = tf.one_hot(labels, depth)
    return one_hot_labels

#最頻値を求める
def compute_mode(data):
    arg_max = tf.argmax(data, axis=-1)
    return np.array(stats.mode(arg_max, axis=-1)[0])

def ReadMnistDataset():
    """
    mnist datasetを読み込む
    コードが正しく動作しているかをチェックするためのデータセット
    ラベル数10(0~9)
    """
    mnist_dataset = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = mnist_dataset.load_data()
    return train_images, train_labels, test_images, test_labels

#https://stackoverflow.com/questions/50882282/tensorflow-bincount-with-axis-option
def bincount(arr, length, axis=-1):
    """Count the number of ocurrences of each value along an axis."""
    n = tf.equal(tf.cast(arr[..., tf.newaxis], tf.int32), tf.cast(tf.range(length), tf.int32))
    n = tf.cast(n, tf.int32)
    return tf.math.count_nonzero(n, axis=axis - 1 if axis < 0 else axis)

#ハイパーパラメータのリストを作成する。
def make_hparam_list(alpha, lambd):
    index = 0
    params ={}
    for a in alpha:
        for l in lambd:
            params[index] = {}
            params[index]['alpha'] = a
            params[index]['lambd'] = l
            index = index + 1
    return params
class DataPlotManager(object):
    def __init__(self, epochs, hparam):
        self.epochs = epochs
        self.y_list    = [] 
        self.x         = [epoch+1 for epoch in range(self.epochs)]
        self.t_list = []
        self.hparam = []
    
    def CollectResultData(train_result, test_result):
        """
        結果を収集する
            引数：
                reulst->実数(認識精度)
        """
        self.y_list.append(train_result)
        self.t_list.append(test_result)

    def CollectHparam(hp):
        """
        ハイパーパラメータを収集する
            引数：
                hp->１組のハイパーパラメータ
        """
        self.hparam.append(hp)

    def graph(self):
        self.num_hparam = len(self.hparam_list)
        self.y_list    = np.array(self.y_list).reshape(self.num_hparam, epochs) 
        self.x          = [epoch+1 for epoch in range(self.epochs)]
        self.hparam    = self.hparam_list
        self.t_list = np.array(self.t_list).reshape(self.num_hparam, epochs) 

        fig = plt.figure(figsize=(10, 10))
        plt.subplots_adjust(wspace=2, hspace=2)
        x_ax = self.num_opts
        for i in range(self.opt_times):

            ax = plt.subplot(x_ax, 1, i+1)
            ax.set_title(self.hparam[i])
            ax.set_ylim(0.0, 1.0)
            ax.plot(self.x, self.y_list[i], self.t_list[i])
        fig.suptitle('red:val_data blue:train_data', fontsize=20)
        plt.tight_layout()
        try:
            plt.show()
        except UnicodeDecodeError:
            plt.show()