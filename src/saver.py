import os
import tensorflow as tf

import src.utils as utils


class Saver():
  
  def __init__(self, model_dir, max_to_keep=5):
    """ Save parameters of the model
    Args:
      dir: 모델을 저장할 폴더
      max_to_keep: 모델의 저장할 갯수
                  (그 이상이면 가장 이전 것을 자동으로 지운다.)  
    """

    self.path  = os.path.join(model_dir, 'model.ckpt')
    self.saver = tf.train.Saver(
                    max_to_keep=max_to_keep,
                    var_list= tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
                  )
    
    # Folder 생성
    utils.make_dir(model_dir)

  def restore(self, sess):
    recent_ckpt_job_path = tf.train.latest_checkpoint(os.path.dirname(self.path))
        
    if recent_ckpt_job_path is not None:
      self.saver.restore(sess, recent_ckpt_job_path)

  def save(self, sess, global_step):
    self.saver.save(sess, self.path, global_step)

    

