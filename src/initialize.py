from argparse import ArgumentParser

def _bool(s):
  # 일정 단어로 True False를 표현
  # parser는 bool을 str로 받기 때문에 무조건 True가 나온다.
  # 따라서 bool 대신 임의의 함수로 type을 대신한다.(parser.register)
  return s.lower() in ("yes", "true", "t", "1")

def build_option():
  parser = ArgumentParser()
  parser.register('type', 'bool', _bool)

  parser.add_argument('-ex', '--experiment_name', type=str,
                      default='test_1',
                      help='실험 이름. 이것에 따라 log와 checkpoint 이름 결정')

  # Model
  # ============================================================================

  parser.add_argument('-ls', '--loss', type=str,
                      default="L1",
                      help='model 학습할 때 쓰일 loss')

  parser.add_argument('-sl', '--start_lr', type=float,
                      default=0.001,
                      help='model 학습할 때 처음 learning rate 값')

  parser.add_argument('-lds', '--lr_decay_step', type=int,
                      default=10000,
                      help='learning rate decay할 주기(step)')

  parser.add_argument('-ldr', '--lr_decay_rate', type=float,
                      default=0.9,
                      help='learning rate decay 비율')

  # ============================================================================

  # Path
  # ============================================================================
  parser.add_argument('-l', '--logs_dir', type=str,
                      default='./logs',
                      help='tensorflow model log를 저장할 폴더')

  parser.add_argument('-c', '--checkpoint_dir', type=str,
                      default='./checkpoint',
                      help='model의 paramter를 저장할 폴더')

  parser.add_argument('-t', '--tfrecord_train_path', type=str,
                      default='./data/tfrecord/train.tfrecord',
                      help="train set을 읽을 tfrecord 파일 이름과 경로")

  parser.add_argument('-e', '--tfrecord_test_path', type=str,
                      default='./data/tfrecord/test.tfrecord',
                      help="test set을 읽을 tfrecord 파일 이름과 경로")

  parser.add_argument('-v', '--tfrecord_valid_path', type=str,
                      default='./data/tfrecord/valid.tfrecord',
                      help="valid set을 읽을 tfrecord 파일 이름과 경로")

  parser.add_argument('-d', '--debug_image_dir', type=str,
                      default='./debug/image',
                      help="학습하면서 나오는 이미지들을 저장하는 폴더")

  # ============================================================================

  # Dataset
  # ============================================================================
  parser.add_argument('-nc', '--n_channel', type=int,
                      default=22,
                      help='이미지가 가지고 있는 채널 수')

  parser.add_argument('-wi', '--img_width', type=int,
                      default=1280,
                      help='이미지 가로 크기')

  parser.add_argument('-he', '--img_height', type=int,
                      default=720,
                      help='이미지 세로 크기')

  parser.add_argument('-p', '--patch_size', type=int,
                      default=65,
                      help="학습용 패치 크기")

  parser.add_argument('-b', '--batch_size', type=int,
                      default=32,
                      help="minibatch 크기")

  parser.add_argument('-sf', '--shuffle_buffer', type=int,
                      default=100,
                      help="tfrecord 파일을 읽을 때 섞기 위한 shuffle buffer 크기")

  parser.add_argument('-ps', '--prefetch_size', type=int,
                      default=1,
                      help="미리 읽는 거라는데 잘 모르겠음")

  # ============================================================================

  # Training
  # ============================================================================
  parser.add_argument('-ep', '--epoch_num', type=int,
                      default=1000000000,
                      help='epoch 수. 보통은 그냥 크게!')
  
  parser.add_argument('-lp', '--log_period', type=int,
                      default=20,
                      help='일정 주기마다 logging(print, log)')

  parser.add_argument('-vp', '--valid_period', type=int,
                      default=1000,
                      help='일정 주기마다 모델 parameter 저장')

  parser.add_argument('-sp', '--save_period', type=int,
                      default=5000,
                      help='일정 주기마다 모델 parameter 저장')
  
  # ============================================================================

  option = parser.parse_args()

  return option