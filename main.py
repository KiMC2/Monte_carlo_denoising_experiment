import os
import time
import cv2
import numpy as numpy
import tensorflow as tf

import src.exr as exr
import src.utils as utils
from src.dataset import next_batch_tensor
from src.saver import Saver
from src.model import VGGNet
from src.initialize import build_option

def main(argc):
  
  option = build_option()

  global_step = tf.train.get_or_create_global_step()

  model = VGGNet(n_channel=option.n_channel,
                 selected_loss=option.loss,
                 start_lr=option.start_lr,
                 lr_decay_step=option.lr_decay_step,
                 lr_decay_rate=option.lr_decay_rate,
                 global_step=global_step)

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:

    # Saver
    # ==========================================================================
    
    saver = Saver(os.path.join(option.checkpoint_dir, option.experiment_name))
    tf.global_variables_initializer().run()
    saver.restore(sess)
    
    # ==========================================================================

    # Saver
    # ==========================================================================

    train_dir = os.path.join(option.logs_dir, option.experiment_name, 'train')
    valid_dir = os.path.join(option.logs_dir, option.experiment_name, 'valid')
    utils.make_dir(train_dir)
    utils.make_dir(valid_dir)
    train_writer = tf.summary.FileWriter(train_dir, sess.graph)
    valid_writer = tf.summary.FileWriter(valid_dir, sess.graph)

    tf.summary.scalar('loss', model.loss)
    tf.summary.scalar('learning_rate', model.lr)
    merged_summary = tf.summary.merge_all()
  
    # ==========================================================================

    # valid data
    # ==========================================================================
    valid_noisy_img, valid_reference = \
      next_batch_tensor(
        tfrecord_path = option.tfrecord_valid_path, 
        img_shape     = [option.img_height, option.img_width, option.n_channel],
        repeat        = 10000
      )
    # ==========================================================================

    for epoch in range(option.epoch_num):
      print(" Epoch :", epoch)
      print(" ================================================================")

      train_noisy_img, train_reference = \
        next_batch_tensor(
          tfrecord_path = option.tfrecord_train_path, 
          img_shape     = [option.patch_size, option.patch_size, option.n_channel],
          batch_size    = option.batch_size,
          shuffle_buffer= option.shuffle_buffer, 
          prefetch_size = option.prefetch_size, 
          repeat        = 0
        )

      while True:
        # 더이상 읽을 것이 없으면 빠져 나오고 다음 epoch으로
        try:
          noisy_img, reference = sess.run([train_noisy_img, train_reference])
        except tf.errors.OutOfRangeError as e:
          break
        
        step = sess.run(global_step)

        model.optimize(sess, noisy_img, reference)

        # 일정 주기마다 logging
        if step % option.log_period == 0:
          loss_val = model.get_loss(sess, noisy_img, reference)

          print(f'epoch {epoch}, step {step}] loss : {loss_val}')

        summary = sess.run(merged_summary, feed_dict={model.noisy_img:noisy_img,
                                                      model.reference:reference})
        train_writer.add_summary(summary, step)

        # 일정주기마다 모델 저장
        if step % option.save_period == 0:
          saver.save(sess, step)

        if step % option.valid_period == 0:
          noisy_img, reference = sess.run([valid_noisy_img, valid_reference])
          loss_val, denoised_img, summary = \
                sess.run([model.loss, model.denoised_img, merged_summary],
                          feed_dict={model.noisy_img:noisy_img,
                                     model.reference:reference})
          valid_writer.add_summary(summary, step)

          debug_dir = os.path.join(option.debug_image_dir, 
                                   option.experiment_name, 'valid')
          exr.save_debug_img(debug_dir + f'step', denoised_img[0, :, :, :])

  tf.reset_default_graph()   

  
if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)

  tf.app.run()
