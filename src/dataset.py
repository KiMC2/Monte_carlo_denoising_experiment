import numpy as np
import tensorflow as tf

def decode_example(seralized_example, shape):
  features = tf.parse_single_example(seralized_example,
    features={
      'image/noisy_img'     : tf.FixedLenFeature([], 
                                dtype=tf.string, default_value=''),
      'image/reference'     : tf.FixedLenFeature([], 
                                dtype=tf.string, default_value=''),
    }
  )

  h, w, in_c, out_c = shape

  noisy_img      = tf.decode_raw(features['image/noisy_img'], tf.float32)
  reference      = tf.decode_raw(features['image/reference'], tf.float32)

  noisy_img      = tf.reshape(noisy_img, [h, w, in_c])
  reference      = tf.reshape(reference, [h, w, out_c])

  return noisy_img, reference

def next_batch_tensor(tfrecord_path, shape, batch_size=1,
                      shuffle_buffer=0, prefetch_size=1, repeat=0):
  ''' 다음 데이터를 출력하기 위한 텐서를 출력한다. 
  Args:
    tfrecord_path  : 읽을 tfrecord 경로(---/---/파일이름.tfrecord)
    shape          : 높이, 폭, 입력 채널, 출력 채널의 시퀀스
                     ex) [65, 65, 66, 3] <-- h, w, in_c, out_c
    batch_size     : 미니배치 크기
    shuffle_buffer : 데이터 섞기 위한 버퍼 크기
    prefetch_size  : 모름. 그냥 1 씀
    repeat         : 데이터가 다 읽힌 경우 Exception이 발생한다.
                     이를 없애기 위해서는 몇 번 더 반복할지 정해줘야 한다.
  Returns:
    noisy_img      : noise가 있는 이미지 tensor
    reference      : noise가 없는 이미지 tensor(조금은 있겠지만..)
  '''

  dataset = tf.data.TFRecordDataset(tfrecord_path)
  dataset = dataset.map(lambda x: decode_example(x, shape))
  dataset = dataset.batch(batch_size)

  if shuffle_buffer != 0:
    dataset = dataset.shuffle(buffer_size=shuffle_buffer)
  if prefetch_size != 0:
    dataset = dataset.prefetch(buffer_size=prefetch_size)
  if repeat != 0:
    dataset = dataset.repeat(repeat)
  
  iterator = dataset.make_one_shot_iterator()

  next_noise_image, next_reference = iterator.get_next()
  
  noisy_img = next_noise_image 
  reference = next_reference 
  
  return noisy_img, reference

























  # noisy_img      = preprocessing(next_noise_image)
  # noisy_img_grad = calc_grad(next_noise_image)
  # noisy_img      = tf.concat((noisy_img, noisy_img_grad), axis=-1)


# def preprocessing(data):
#   return data
#   log_data = tf.log(data[:, :, :, :12] + 1)
#   cliped_depth = tf.clip_by_value(data[:, :, :, 20:], 0, np.max(data[:, :, :, 20:]))

#   return tf.concat((log_data, data[:, :, :, 12:20], cliped_depth), axis=3)

# def calc_grad(data):
#   _, h, w, _ = data.shape
  
#   dx_zero = tf.zeros_like(data[:,:,0:1,:])
#   dy_zero = tf.zeros_like(data[:,0:1,:,:])

#   dx = data[:, :, 1:, :] - data[:, :, :w - 1, :]
#   dy = data[:, 1:, :, :] - data[:, :h - 1, :, :]

#   dx = tf.concat((dx_zero, dx), axis=2)
#   dy = tf.concat((dy_zero, dy), axis=1)

#   gradient = tf.concat((dx, dy), axis=3)

#   return gradient
  
# def postprocessing(data):
#   return data
#   exp_data = tf.exp(data[:, :, :, :12]) - 1

#   return tf.concat((exp_data, data[:, :, :, 12:]), axis=3)
  