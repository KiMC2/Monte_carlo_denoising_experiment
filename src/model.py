import tensorflow as tf

from src.tf_ops import conv2d, deconv2d, batch_norm, SSIM

class DenoisingModel():
  def __init__(self, n_channel,
                     selected_loss,
                     start_lr,
                     lr_decay_step,
                     lr_decay_rate,
                     global_step):

    self.noisy_img      = tf.placeholder(tf.float32, [None, None, None, n_channel], name='noisy_img')
    self.reference      = tf.placeholder(tf.float32, [None, None, None, n_channel], name='reference')
    
    self.n_channel           = n_channel

    self.denoised_img   = self.build_model(self.noisy_img, n_channel)
    self.loss           = self.build_loss(selected_loss,
                                          self.denoised_img, self.reference)

    self.global_step    = global_step
    self.lr             = tf.train.exponential_decay( start_lr, 
                                                      global_step,
                                                      lr_decay_step, 
                                                      lr_decay_rate)
    self.optim          = self.build_optim()

  def build_model(self, inputs, n_channel):
    ''' 입력용 tensor inputs과 출력 크기 n_channel을 꼭 사용해야함 '''
    raise Exception('Model should be implemented')

  def build_loss(self, selected_loss, denoised_img, reference):

    if selected_loss == "L2":
      diff_square = tf.square(tf.subtract(denoised_img, reference))
      loss = tf.reduce_mean(diff_square)
    
    if selected_loss == 'HUBER':
      huber_loss = tf.losses.huber_loss(reference, denoised_img)
      loss = tf.reduce_mean(huber_loss)

    elif selected_loss == "LMLS":
      diff = tf.subtract(denoised_img, reference)
      diff_square_ch_mean = tf.reduce_mean(tf.square(diff), axis=-1)
      loss = tf.reduce_mean(tf.log(1 + (0.5 * diff_square_ch_mean)))
    
    elif selected_loss == "RELMSE":
      L2 = tf.square(tf.subtract(denoised_img, reference))
      denom = tf.square(reference) + 1.0e-2
      loss = tf.reduce_mean(L2 / denom)
      
    elif selected_loss == "L1":
      diff = tf.abs(tf.subtract(denoised_img, reference))
      loss = tf.reduce_mean(diff)

    elif selected_loss == 'MAPE':
      diff = tf.abs(tf.subtract(denoised_img, reference))
      diff = tf.div(diff, reference + 1.0+1e-2)
      loss = tf.reduce_mean(diff)

    elif selected_loss == 'SSIM':
      loss = SSIM(denoised_img, reference)

    return loss

  def build_optim(self):
    optim = tf.train.AdamOptimizer(self.lr)
    grads = optim.compute_gradients(self.loss)

    # Clip gradients to avoid exploding weights
    grads = [(None, var) if grad is None else (tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in grads]

    # Apply gradients
    apply_gradient_op = optim.apply_gradients(grads, global_step=self.global_step)
    with tf.control_dependencies([apply_gradient_op]):
        train_op = tf.no_op(name='Train')

    return train_op

  def get_pred(self, sess, noisy_img, reference):
    return sess.run(self.denoised_img, feed_dict={self.noisy_img:noisy_img,
                                                  self.reference:reference})
  def get_loss(self, sess, noisy_img, reference):
    return sess.run(self.loss, feed_dict={self.noisy_img:noisy_img,
                                          self.reference:reference})

  def optimize(self, sess, noisy_img, reference):
    sess.run(self.optim, feed_dict={self.noisy_img: noisy_img,
                                    self.reference: reference})


class VGGNet(DenoisingModel):
  def build_model(self, inputs, n_channel):
    layer = conv2d(inputs, 64)

    for _ in range(18):
      layer = conv2d(layer, 64)
    
    layer = conv2d(layer, n_channel)

    return layer

