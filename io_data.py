import tensorflow as tf
import skvideo.io
import imutils

#input_path = "/home/ubuntu/nfs/cybozu/Temporal_Transformation/SampledVideo"
#/home/ubuntu/nfs/cybozu/video_dataset/train_dataset
#/home/ubuntu/nfs/cybozu/video_dataset/test_dataset
tfrecord_train_dir = "/home/ubuntu/nfs/cybozu/video_dataset/tfrecords/train.tfrecords"
tfrecord_test_dir  = "/home/ubuntu/nfs/cybozu/video_dataset/tfrecords/test.tfrecords"

def tensor_cast(inputs, labels, mask):
    inputs = tf.cast(inputs, tf.float32)
    labels = labels - 1
    labels = tf.one_hot(labels, 64)
    labels = tf.reshape(labels, [-1,])
    labels = tf.stack([labels]*201)
    return inputs, tf.cast(labels, tf.int64), mask

def GenerateMaskTensor(video):
    timesteps = 201
    l = [False]*timesteps
    frames = video.shape[0]
    for index in range(frames):
        l[index] = True

    return l

#tfrecordの処理の参考URL
#https://www.tensorflow.org/tutorials/load_data/tfrecord
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    # If the value is an eager tensor BytesList won't unpack a string from an EagerTensor.
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() 
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(video, label, video_shape, mask):
    feature = {
        'video': _bytes_feature(video),
        'label': _int64_feature(label),
        'len_seq':_int64_feature(video_shape[0]),
        'height': _int64_feature(video_shape[1]),
        'width': _int64_feature(video_shape[2]),
        'depth': _int64_feature(video_shape[3]),
        'mask' : _bytes_feature(mask),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def parse_tfrecord(serialized_example):
    feature_description = {
        'video' : tf.io.FixedLenFeature((), tf.string),
        'label': tf.io.FixedLenFeature((), tf.int64),
        'len_seq':tf.io.FixedLenFeature((), tf.int64),
        'height': tf.io.FixedLenFeature((), tf.int64),
        'width': tf.io.FixedLenFeature((), tf.int64),
        'depth': tf.io.FixedLenFeature((), tf.int64),
        'mask' : tf.io.FixedLenFeature((), tf.string),
    }
    example = tf.io.parse_single_example(serialized_example, feature_description)
    
    video = tf.io.parse_tensor(example['video'], out_type = tf.float64)
    video_shape = [example['len_seq'], example['height'], example['width'], example['depth']]
    video = tf.reshape(video, video_shape)

    mask = tf.io.parse_tensor(example['mask'], out_type = tf.bool)

    return video, example['label'], mask

def write_dataset():
    train_input_path = "/home/ubuntu/nfs/cybozu/video_dataset/train_dataset/*.mp4"
    test_input_path = "/home/ubuntu/nfs/cybozu/video_dataset/test_dataset/*.mp4"

    with tf.io.TFRecordWriter(tfrecord_train_dir) as writer:

        video_list = glob.glob(train_input_path)

        for video_path in video_list:
            video = skvideo.io.vread(video_path)
            mask = GenerateMaskTensor(video) 

            mask = tf.io.serialize_tensor(mask)
            video_bytes = tf.io.serialize_tensor(video)

            label = video_path[0]
            label = int(label)
            print("現在処理しているビデオのインデックスは、", video_list.index(video_path))
            example = serialize_example(video_bytes, label, mask, len(video))
            writer.write(example)
    with tf.io.TFRecordWriter(tfrecord_test_dir) as writer:

        video_list = glob.glob(test_input_path)

        for video_path in video_list:
            video = skvideo.io.vread(video_path)
            mask = GenerateMaskTensor(video) 

            mask = tf.io.serialize_tensor(mask)
            video_bytes = tf.io.serialize_tensor(video)

            label = video_path[0]
            label = int(label)
            print("現在処理しているビデオのインデックスは、", video_list.index(video_path))
            example = serialize_example(video_bytes, label, mask, len(video))
            writer.write(example)


def read_dataset(BATCH_SIZE):
    tfrecord_train_dataset = tf.data.TFRecordDataset(train_path)
    parsed_train_dataset = tfrecord_train_dataset.map(parse_tfrecord)
    tfrecord_test_dataset = tf.data.TFRecordDataset(test_path)
    parsed_test_dataset = tfrecord_test_dataset.map(parse_tfrecord)

    padded_shape = (tf.constant(-1.0, dtype=tf.float32), tf.constant(0, dtype=tf.int64), tf.constant(False, dtype=tf.bool))
    train_dataset = parsed_train_dataset.map(tensor_cast).shuffle(buffer_size=10, seed=100).padded_batch(BATCH_SIZE,padded_shapes=([201, 112, 200, 3], [201, 64], [201]), padding_values=padded_shape).prefetch(tf.data.experimental.AUTOTUNE)
    test_dataset =  parsed_test_dataset.map(tensor_cast).shuffle(buffer_size=10,  seed=100).padded_batch(BATCH_SIZE,  padded_shapes=([201, 112, 200, 3], [201, 64], [201]), padding_values=padded_shape).prefetch(tf.data.experimental.AUTOTUNE)
    return train_dataset, test_dataset

if __name__ == '__main__':
    write_dataset()