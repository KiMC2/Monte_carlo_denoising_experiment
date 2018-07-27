import os
import time
import cv2
import numpy as np
import tensorflow as tf

import src.exr as exr
import src.utils as utils
import src.model as model
from src.dataset import next_batch_tensor
from src.saver import Saver
from src.initialize import build_option

def main(argc):
  
  option = build_option()

  global_step = tf.train.get_or_create_global_step()

  denoising_model = eval("model." + option.model_name)(
                        n_input_channel  = option.n_input_channel,
                        n_output_channel = option.n_output_channel,
                        selected_loss    = option.loss,
                        start_lr         = option.start_lr,
                        lr_decay_step    = option.lr_decay_step,
                        lr_decay_rate    = option.lr_decay_rate,
                        global_step      = global_step
                      )

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  with tf.Session(config=config) as sess:

    # Model write
    model.write(sess, option.text_dir + '/model.txt')

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

    tf.summary.scalar('loss', denoising_model.loss)
    tf.summary.scalar('learning_rate', denoising_model.lr)
    merged_summary = tf.summary.merge_all()
  
    # ==========================================================================

    # valid data
    # ==========================================================================
    valid_noisy_img, valid_reference = \
      next_batch_tensor(
        tfrecord_path = option.tfrecord_valid_path, 
        shape         = [option.img_height, option.img_width, 
                         option.n_input_channel, option.n_output_channel],
        repeat        = 10000
      )
    # ==========================================================================

    for epoch in range(option.n_epoch):
      train_noisy_img, train_reference = \
        next_batch_tensor(
          tfrecord_path = option.tfrecord_train_path, 
          shape         = [option.patch_size, option.patch_size, 
                           option.n_input_channel, option.n_output_channel],
          batch_size    = option.batch_size,
          shuffle_buffer= option.shuffle_buffer, 
          prefetch_size = option.prefetch_size, 
        )

      while True:
        # 더이상 읽을 것이 없으면 빠져 나오고 다음 epoch으로
        try:
          noisy_img, reference = sess.run([train_noisy_img, train_reference])
        except tf.errors.OutOfRangeError as e:
          print("Done")
          break

        step = sess.run(global_step)

        denoising_model.optimize(sess, noisy_img, reference)

        # 일정 주기마다 logging
        if step % option.log_period == 0:
          loss_val = denoising_model.get_loss(sess, noisy_img, reference)

          print(f'epoch {epoch}, step {step}] loss : {loss_val}')

          summary = sess.run(merged_summary, feed_dict={denoising_model.noisy_img:noisy_img,
                                                        denoising_model.reference:reference})
          train_writer.add_summary(summary, step)

          denoised_img = denoising_model.get_pred(sess, noisy_img, reference)
          

          for b in range(option.n_save_patch_img):
            exr.save_debug_img(option.debug_image_dir + f'/{option.experiment_name}' + f'/train/denoised/{step}_{b}',
                               denoised_img[b, :, :, :])
            exr.save_debug_img(option.debug_image_dir + f'/{option.experiment_name}' + f'/train/noisy/{step}_{b}',
                               noisy_img[b, :, :, :])
            exr.save_debug_img(option.debug_image_dir + f'/{option.experiment_name}' + f'/train/refer/{step}_{b}',
                               reference[b, :, :, :])

        # 일정주기마다 모델 저장
        if step % option.save_period == 0:
          saver.save(sess, step)

        if step % option.valid_period == 0:
          print("=====================================================================")
          
          noisy_img, reference = sess.run([valid_noisy_img, valid_reference])
          loss_val, denoised_img, summary = \
                sess.run([denoising_model.loss, denoising_model.denoised_img, merged_summary],
                          feed_dict={denoising_model.noisy_img:noisy_img,
                                     denoising_model.reference:reference})

          print(" Test ] loss ", loss_val)

          valid_writer.add_summary(summary, step)

          debug_dir = os.path.join(option.debug_image_dir, 
                                   option.experiment_name, 'valid')
          exr.save_debug_img(debug_dir + f'/valid/{step}', denoised_img[0, :, :, :])
          print("=====================================================================")

  tf.reset_default_graph()   

  
if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)

  tf.app.run()
