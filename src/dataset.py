import tensorflow as tf

def decode_example(seralized_example, img_shape):
  features = tf.parse_single_example(seralized_example,
    features={
      'image/noisy_img'     : tf.FixedLenFeature([], 
                                dtype=tf.string, default_value=''),
      'image/reference'     : tf.FixedLenFeature([], 
                                dtype=tf.string, default_value=''),
    }
  )

  h, w, c = img_shape

  noisy_img      = tf.decode_raw(features['image/noisy_img'], tf.float32)
  reference      = tf.decode_raw(features['image/reference'], tf.float32)

  noisy_img      = tf.reshape(noisy_img, [h, w, c])
  reference      = tf.reshape(reference, [h, w, c])

  return noisy_img, reference

def next_batch_tensor(tfrecord_path, img_shape, batch_size=1,
               shuffle_buffer=0, prefetch_size=0, repeat=0):
      
  dataset = tf.data.TFRecordDataset(tfrecord_path)
  dataset = dataset.map(lambda x: decode_example(x, img_shape))
  dataset = dataset.batch(batch_size)

  if shuffle_buffer != 0:
    dataset = dataset.shuffle(buffer_size=shuffle_buffer)
  if prefetch_size != 0:
    dataset = dataset.prefetch(buffer_size=prefetch_size)
  if repeat != 0:
    dataset = dataset.repeat(repeat)
  
  iterator = dataset.make_one_shot_iterator()

  next_noise_image, next_reference = iterator.get_next()
  
  return next_noise_image, next_reference