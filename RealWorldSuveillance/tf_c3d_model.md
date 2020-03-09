```python
import cv2
import numpy as np
import tensorflow as tf

NUM_CLASSES = 101
WIDTHD = 171
HEIGHTS = 128
CROP_SIZE = 112
CHANNELS = 3
NUM_FRAMES_PER_CLIP = 16

def conv3d(name, l_input, w, b):
    return tf.nn.bias_add(
        tf.nn.conv3d(l_input, w, strides=[1]*5, padding='SAME'), b)

def max_pool(name, l_input, k):
    return tf.nn.max_pool3d(l_input, ksize=[1, k, 2, 2, 1],
                            strides=[1, k, 2, 2, 1], padding='SAME', name=name)

def _variable_with_weight_decay(name, shape, wd):
    var = _variable_on_cpu(name, shape, tf.contrib.layers.xavier_initializer())
    if wd is not None:
        weight_decay = tf.nn.l2_loss(var) * wd
        tf.add_to_collection('weightdecay_losses', weight_decay)
    return var
    
def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var
    
weights = {
      'wc1': _variable_with_weight_decay('wc1', [3, 3, 3, 3, 64], 0.0005),
      'wc2': _variable_with_weight_decay('wc2', [3, 3, 3, 64, 128], 0.0005),
      'wc3': _variable_with_weight_decay('wc3a', [3, 3, 3, 128, 256], 0.0005),
      'wc4': _variable_with_weight_decay('wc4a', [3, 3, 3, 256, 256], 0.0005),
      'wc5': _variable_with_weight_decay('wc5a', [3, 3, 3, 256, 256], 0.0005),
      'wd1': _variable_with_weight_decay('wd1', [4096, 2048], 0.0005),
      'wd2': _variable_with_weight_decay('wd2', [2048, 2048], 0.0005),
      'out': _variable_with_weight_decay('wout', [2048, 101], 0.0005)
      }
biases = {
      'bc1': _variable_with_weight_decay('bc1', [64], 0.000),
      'bc2': _variable_with_weight_decay('bc2', [128], 0.000),
      'bc3': _variable_with_weight_decay('bc3a', [256], 0.000),
      'bc4': _variable_with_weight_decay('bc4a', [256], 0.000),
      'bc5': _variable_with_weight_decay('bc5a', [256], 0.000),
      'bd1': _variable_with_weight_decay('bd1', [2048], 0.000),
      'bd2': _variable_with_weight_decay('bd2', [2048], 0.000),
      'out': _variable_with_weight_decay('bout', 101, 0.000),
      }
      
_weights = weights
_biases = biases

def placeholder_inputs(batch_size):
    images_placeholder = tf.placeholder(tf.float32, 
                                        shape=(batch_size,
                                               NUM_FRAMES_PER_CLIP,
                                               CROP_SIZE,
                                               CROP_SIZE,
                                               CHANNELS))
    labels_placeholder = tf.placeholder(tf.int64, shape=(batch_size))
    return images_placeholder, labels_placeholder
    
images_placeholder, labels_placeholder = placeholder_inputs(10)

_X = images_placeholder

batch_size = 10
_dropout = 0.5

conv1 = conv3d('conv1', _X, _weights['wc1'], _biases['bc1'])
pool1 = max_pool('pool1', conv1, k=1)

conv2 = conv3d('conv2', pool1, _weights['wc2'], _biases['bc2'])
pool2 = max_pool('pool2', conv2, k=2)

conv3 = conv3d('conv3a', pool2, _weights['wc3'], _biases['bc3'])
pool3 = max_pool('pool3', conv3, k=2)

conv4 = conv3d('conv4a', pool3, _weights['wc4'], _biases['bc4'])
pool4 = max_pool('pool4', conv4, k=2)

conv5 = conv3d('conv5a', pool4, _weights['wc5'], _biases['bc5'])
pool5 = max_pool('pool5', conv5, k=2)

pool5 = tf.transpose(pool5, perm=[0, 1, 4, 2, 3])
dense1 = tf.reshape(pool5, [batch_size, _weights['wd1'].get_shape().as_list()[0]])
dense1 = tf.matmul(dense1, _weights['wd1']) + _biases['bd1']

dense1 = tf.nn.relu(dense1, name='fc1') # Relu activation
dense1 = tf.nn.dropout(dense1, _dropout)

dense2 = tf.nn.relu(tf.matmul(dense1, _weights['wd2']) + _biases['bd2'], name='fc2') # Relu activation
dense2 = tf.nn.dropout(dense2, _dropout)

out = tf.matmul(dense2, _weights['out']) + _biases['out']

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
init = tf.global_variables_initializer()
sess.run(init)

video_path = 'E:/UCF101/UCF-101/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi'
vidcap = cv2.VideoCapture(video_path)

res = []
while True:
    ret, frame = vidcap.read()
    if ret:
        frame = cv2.resize(frame, (171, 128), 
                   interpolation = cv2.INTER_AREA).astype(np.float32)
        scale = float(CROP_SIZE) / float(HEIGHTS)
        frame = cv2.resize(frame, (int(WIDTHS*scale+1), CROP_SIZE))
        crop_x = int((frame.shape[0] - CROP_SIZE) / 2)
        crop_y = int((frame.shape[1] - CROP_SIZE) / 2)
        frame = frame[crop_x:crop_x+CROP_SIZE, crop_y:crop_y+CROP_SIZE, :]
        noise = np.random.randint(0, 50, (CROP_SIZE, CROP_SIZE, 3))
        frame -= noise
        res.append(frame)
    else:
        break
        
video_clips = np.stack(
    [np.stack(res[i*16:(i+1)*16], axis=3).reshape(
        CHANNELS, -1, CROP_SIZE, CROP_SIZE) for i in range(164 // 16)],
    axis=0
)

video_clips = video_clips.reshape(10, 16, 112, 112, 3)

print(sess.run(conv1, feed_dict={_X:video_clips}).shape)
print(sess.run(pool1, feed_dict={_X:video_clips}).shape)
print(sess.run(conv2, feed_dict={_X:video_clips}).shape)
print(sess.run(pool2, feed_dict={_X:video_clips}).shape)
print(sess.run(conv3, feed_dict={_X:video_clips}).shape)
print(sess.run(pool3, feed_dict={_X:video_clips}).shape)
print(sess.run(conv4, feed_dict={_X:video_clips}).shape)
print(sess.run(pool4, feed_dict={_X:video_clips}).shape)
print(sess.run(conv5, feed_dict={_X:video_clips}).shape)
print(sess.run(pool5, feed_dict={_X:video_clips}).shape)
print(sess.run(dense1, feed_dict={_X:video_clips}).shape)
print(sess.run(dense2, feed_dict={_X:video_clips}).shape)
print(sess.run(out, feed_dict={_X:video_clips}).shape)
```
```
(10, 16, 112, 112, 64)
(10, 16, 56, 56, 64)
(10, 16, 56, 56, 128)
(10, 8, 28, 28, 128)
(10, 8, 28, 28, 256)
(10, 4, 14, 14, 256)
(10, 4, 14, 14, 256)
(10, 2, 7, 7, 256)
(10, 2, 7, 7, 256)
(10, 1, 256, 4, 4)
(10, 2048)
(10, 2048)
(10, 101)
```
```python
weights = {
      'wc1': _variable_with_weight_decay('wc1', [7, 7, 7, 3, 64], 0.0005),
      'wc2': _variable_with_weight_decay('wc2', [5, 5, 5, 64, 128], 0.0005),
      'wc3': _variable_with_weight_decay('wc3a', [5, 5, 5, 128, 256], 0.0005),
      'wc4': _variable_with_weight_decay('wc4a', [3, 3, 3, 256, 256], 0.0005),
      'wc5': _variable_with_weight_decay('wc5a', [3, 3, 3, 256, 256], 0.0005),
      'wd1': _variable_with_weight_decay('wd1', [4096, 2048], 0.0005),
      'wd2': _variable_with_weight_decay('wd2', [2048, 2048], 0.0005),
      'out': _variable_with_weight_decay('wout', [2048, 101], 0.0005)
      }
biases = {
      'bc1': _variable_with_weight_decay('bc1', [64], 0.000),
      'bc2': _variable_with_weight_decay('bc2', [128], 0.000),
      'bc3': _variable_with_weight_decay('bc3a', [256], 0.000),
      'bc4': _variable_with_weight_decay('bc4a', [256], 0.000),
      'bc5': _variable_with_weight_decay('bc5a', [256], 0.000),
      'bd1': _variable_with_weight_decay('bd1', [2048], 0.000),
      'bd2': _variable_with_weight_decay('bd2', [2048], 0.000),
      'out': _variable_with_weight_decay('bout', 101, 0.000),
      }

print(sess.run(conv1, feed_dict={_X:video_clips}).shape)
print(sess.run(pool1, feed_dict={_X:video_clips}).shape)
print(sess.run(conv2, feed_dict={_X:video_clips}).shape)
print(sess.run(pool2, feed_dict={_X:video_clips}).shape)
print(sess.run(conv3, feed_dict={_X:video_clips}).shape)
print(sess.run(pool3, feed_dict={_X:video_clips}).shape)
print(sess.run(conv4, feed_dict={_X:video_clips}).shape)
print(sess.run(pool4, feed_dict={_X:video_clips}).shape)
print(sess.run(conv5, feed_dict={_X:video_clips}).shape)
print(sess.run(pool5, feed_dict={_X:video_clips}).shape)
print(sess.run(dense1, feed_dict={_X:video_clips}).shape)
print(sess.run(dense2, feed_dict={_X:video_clips}).shape)
print(sess.run(out, feed_dict={_X:video_clips}).shape)
```
```
(10, 16, 112, 112, 64)
(10, 16, 56, 56, 64)
(10, 16, 56, 56, 128)
(10, 8, 28, 28, 128)
(10, 8, 28, 28, 256)
(10, 4, 14, 14, 256)
(10, 4, 14, 14, 256)
(10, 2, 7, 7, 256)
(10, 2, 7, 7, 256)
(10, 1, 256, 4, 4)
(10, 2048)
(10, 2048)
(10, 101)
```
