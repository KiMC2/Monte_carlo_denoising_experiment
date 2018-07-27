import tensorflow as tf

def batch_norm(inputs, data_format="channels_last"):
  return tf.layers.batch_normalization(inputs, \
    axis = -1 if data_format=='channels_last' else 1)

def conv2d(inputs, filters, kernel_size=3, strides=1, 
           activation=None, padding='same', tf_pad='CONSTANT', 
           data_format='channels_last'):
  
  layer = inputs
  if tf_pad != 'CONSTANT':
    paddings = tf.constant([[0,0], [1, 1], [1, 1], [0,0]])
    layer = tf.pad(layer, paddings, "SYMMETRIC")

  return tf.layers.conv2d(
    inputs             = layer,
    filters            = filters,
    kernel_size        = kernel_size,
    strides            = strides,
    data_format        = data_format,
    activation         = activation,
    padding            = padding,
    kernel_initializer = tf.contrib.layers.xavier_initializer()
    #tf.truncated_normal_initializer(stddev=0.01)
  )

def deconv2d(inputs, filters, kernel_size=3, strides=1, 
           activation=None, padding='same', data_format='channels_last'):
  return tf.layers.conv2d_transpose(
    inputs             = inputs,
    filters            = filters,
    kernel_size        = kernel_size,
    strides            = strides,
    data_format        = data_format,
    activation         = activation,
    padding            = padding,
    kernel_initializer = tf.contrib.layers.xavier_initializer()
    #tf.truncated_normal_initializer(stddev=0.01)
  )

def max_pool(inputs, pool_size=2, strides=2, 
             padding='valid', data_format='channels_last', name=None):
  return tf.layers.max_pooling2d(
    inputs = inputs,
    pool_size = pool_size,
    strides = strides,
    padding = padding,
    data_format = data_format,
    name=name
  )

def SSIM(a, b):

  # Generate filter kernel
  _, _, _, d = a.get_shape().as_list()
  window = generate_weight(5, 1.5)
  window = window / np.sum(window)
  window = window.astype(np.float32)
  window = window[:,:,np.newaxis,np.newaxis]
  window = tf.constant(window)
  window = tf.tile(window,[1, 1, d, 1])

  # Find means
  mA = tf.nn.depthwise_conv2d(a, window, strides=[1, 1, 1, 1], padding='VALID')
  mB = tf.nn.depthwise_conv2d(b, window, strides=[1, 1, 1, 1], padding='VALID')

  # Find standard deviations
  sA = tf.nn.depthwise_conv2d(a*a, window, strides=[1, 1, 1, 1], padding='VALID') - mA**2
  sB = tf.nn.depthwise_conv2d(b*b, window, strides=[1, 1, 1, 1], padding='VALID') - mB**2
  sAB = tf.nn.depthwise_conv2d(a*b, window, strides=[1, 1, 1, 1], padding='VALID') - mA*mB

  # Calc SSIM constants 
  L = 1.0
  k1 = 0.01
  k2 = 0.03
  c1 = (k1 * L)**2
  c2 = (k2 * L)**2

  # Plug into SSIM equation
  assert(c1 > 0 and c2 > 0)
  p1 = (2.0*mA*mB + c1)/(mA*mA + mB*mB + c1)
  p2 = (2.0*sAB + c2)/(sA + sB + c2)

  # We want to maximize SSIM or minimize (1-SSIM)
  return 1-tf.reduce_mean(p1*p2)