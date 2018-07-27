import os
import unittest
import numpy as np
import tensorflow as tf

from glob import glob


import src.exr as exr
import src.model as model
from src.tf_ops import conv2d
from src.dataset import next_batch_tensor


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

class TestNet(model.DenoisingModel):

  def build_model(self, inputs, output_ch):
    
    # First Layer
    layer = conv2d(inputs, 64)
    for _ in range(20):
      layer = conv2d(layer, 64)

    layer = conv2d(layer, output_ch)

    return layer


class TestDataset(unittest.TestCase):
  def setUp(self):
    self.sess            = tf.Session()
    self.test_dir        = './debug/unittest/'
    self.data_dir        = os.path.join(self.test_dir , 'data')
    self.noisy_dir       = os.path.join(self.data_dir , 'noisy_img')
    self.refer_dir       = os.path.join(self.data_dir , 'reference')
    self.test_exr_path   = os.path.join(self.noisy_dir, '10064185-00128spp.exr')
    self.tfrecord_path   = os.path.join(self.data_dir , 'test.tfrecord')
    self.temp_result_dir = os.path.join(self.data_dir , 'temp_result')
    
  def tearDown(self):
    self.sess.close()
  
  def test_exr_to_record_and_get_next_tensor_and_tset_model_training(self):    
    print("\n\nTesting...[test_exr_to_record_and_get_next_tensor_and_tset_model_training]")
    n_input_channel  = 66
    n_output_channel = 3
    grad_add         = True
    
    exr.to_tfrecord(self.noisy_dir, self.refer_dir, self.tfrecord_path,
                    n_file=3, patch_size=0, n_patch=0, 
                    is_training=False, grad_add=grad_add, 
                    save_original_img=False, save_patch_img=False)

    

    noisy_img, reference = next_batch_tensor(
                  tfrecord_path=self.tfrecord_path,
                  shape=[720, 1280, n_input_channel, n_output_channel])

    global_step = tf.train.get_or_create_global_step()
    denoising_model = TestNet(
                        n_input_channel  = n_input_channel,
                        n_output_channel = n_output_channel,
                        selected_loss    = "L1",
                        start_lr         = 0.001,
                        lr_decay_step    = 2000,
                        lr_decay_rate    = 0.9,
                        global_step      = global_step
                      )
    self.sess.run(tf.global_variables_initializer())

    ni, re = self.sess.run([noisy_img, reference])

    exr.save_debug_img('./tests/data/temp_result/test_exr_to_record/noisy_img', ni[0])
    exr.save_debug_img('./tests/data/temp_result/test_exr_to_record/reference', re[0])

    loss = denoising_model.get_loss(self.sess, ni, re)

    pred = denoising_model.get_pred(self.sess, ni, re)

    denoising_model.optimize(self.sess, ni, re)

    print('loss : ', loss, 'pred.shape : ', pred.shape)