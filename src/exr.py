import os
import sys
import time
import Imath
import OpenEXR
import numpy as np
import tensorflow as tf
from datetime import datetime
from PIL import Image
from glob import glob

channel_list = [  
                'R',          'G',          'B',    'colorVariance.Z', 
        'specular.R', 'specular.G', 'specular.B', 'specularVariance.Z',
        'diffuse.R',  'diffuse.G',  'diffuse.B',  'diffuseVariance.Z', 
          'normal.R',   'normal.G',   'normal.B',   'normalVariance.Z', 
          'albedo.R',   'albedo.G',   'albedo.B',   'albedoVariance.Z', 
          'depth.Z',                                'depthVariance.Z',
    ]

# 실행을 root(projectfoler)에서 해야 src.utils를 불러올 수 있다.
sys.path.append('.')

import src.utils as utils

# Convert bytes(image) to features (for tfrecord)
def _bytes_to_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _ch_np_precision(channel):
  if channel in ['depth.Z', 'depthVariance.Z']:
    return np.float32
  else:
    return np.float16

def _ch_precision(channel):
  if channel in ['depth.Z', 'depthVariance.Z']:
    return Imath.PixelType(Imath.PixelType.FLOAT)
  else:
    return Imath.PixelType(Imath.PixelType.HALF)

def _find_gt_file(exr_file_name, gt_dir):
  ''' exr_file_name과 동일한 이름의 reference를 찾아서 파일 경로로 반환한다. '''
  return glob(os.path.join(gt_dir, f"{exr_file_name}-*.exr"))[0]

def _write_tfrecord(writer, noisy_img, reference):

  example = tf.train.Example(
    features=tf.train.Features(feature={
      'image/noisy_img'     : _bytes_to_feature(noisy_img.tostring()),
      'image/reference'     : _bytes_to_feature(reference.tostring()),
      }
    )
  )
  writer.write(example.SerializeToString())

def write(path, data):
  assert data.ndim == 3, "In write(), data ndim must be 3"

  utils.make_dir(os.path.dirname(path))

  h, w, _ = data.shape
  header = OpenEXR.Header(w, h)
  header['compression'] = Imath.Compression(Imath.Compression.PIZ_COMPRESSION)
  header['channels'] = {c: Imath.Channel(_ch_precision(c)) for c in channel_list}
  
  out = OpenEXR.OutputFile(path, header)
  ch_data = {ch: data[:, :, index].\
                 astype(_ch_np_precision(channel_list[index])).tostring()
                 for index, ch in enumerate(channel_list)}

  out.writePixels(ch_data)

def to_jpg(exrfile, jpgfile):
  ''' exrfile을 jpgfile로 변환합니다. '''
  utils.make_dir(os.path.dirname(jpgfile))
  
  color_ch = channel_list[:3]
  
  File = OpenEXR.InputFile(exrfile)
  PixType = Imath.PixelType(Imath.PixelType.FLOAT)
  DW = File.header()['dataWindow']
  Size = (DW.max.x - DW.min.x + 1, DW.max.y - DW.min.y + 1)

  rgb = [np.fromstring(File.channel(c, PixType), dtype=np.float32) \
                  for c in color_ch]
  
  for i in range(len(color_ch)):
      rgb[i] = np.where(rgb[i]<=0.0031308,
              (rgb[i]*12.92)*255.0,
              (1.055*(rgb[i]**(1.0/2.4))-0.055) * 255.0)
  
  rgb8 = [Image.frombytes("F", Size, c.tostring()).convert("L") for c in rgb]
  Image.merge("RGB", rgb8).save(jpgfile, "JPEG", quality=95)

def save_debug_img(file_name_with_path, image):
  write(file_name_with_path + '.exr', image)
  to_jpg(file_name_with_path + '.exr', file_name_with_path + '.jpg')
  
def _calc_grad(data):
  ''' 3차원 데이터 각 채널의 gradient를 계산한다. '''
  
  if data.ndim != 3:
    raise Exception('in _calc_grad(data), '
                    'data n_dims must be 3, but', data.ndim)
  
  h, w, c = data.shape
  dx = data[:, 1:, :] - data[:, :w - 1, :]
  dy = data[1:, :, :] - data[:h - 1, :, :]
  dx = np.concatenate((np.zeros([h,1,c]),dx), axis=1)
  dy = np.concatenate((np.zeros([1,w,c]),dy), axis=0)

  gradient = np.concatenate((dx, dy), axis=2)
  
  return gradient.astype(np.float32)  # float32로 통일 시켜줘야한다.

def _crop(image, x, y, patch_size):
  ''' x, y 기준으로 patch_size크기만큼을 잘라낸다.'''
  return image[y:y+patch_size, x:x+patch_size, :]

def _make_patches(noisy_img, reference, patch_size, n_sample):
  ''' patch를 출력해주는 generator
      random한 수가 들어가기 때문에 image 하나만 넣는게 아니라 
      noisy_img, reference 다 들어가야한다.
  Args:
    noisy_img       : 3차원 노이즈 이미지
    reference       : 3차원 ground truth 이미지
    patch_size      : patch 크기
    n_sample        : patch 출력할 개수(샘플할 개수)
  '''

  assert noisy_img.ndim == 3, \
        'noisy_img ndim must be 3 but {}'.format(noisy_img.ndim)
  assert reference.ndim == 3, \
        'reference ndim must be 3 but {}'.format(reference.ndim)
    
  # sampling 하기 위한 최대 x, y위치 구함
  # x + patch_size 했을 때 넘지 않기 위함
  h, w, _ = np.shape(noisy_img)
  max_x = w - (patch_size - 1)
  max_y = h - (patch_size - 1)

  for _ in range(n_sample):
    x, y = np.random.randint(max_x), np.random.randint(max_y)

    # 임의의 x, y좌표를 가지고 crop
    patched_noisy_img      = _crop(noisy_img, x, y, patch_size)
    patched_reference      = _crop(reference, x, y, patch_size)

    yield patched_noisy_img, patched_reference

def read(exr_file):
  ''' 데이터 읽어서 반환
  Args:
    exr_file: exr 파일 경로
    channels: 읽을 채널들
  Returns:
    exr_data: exr 채널들의 값
  '''
  
  input_file = OpenEXR.InputFile(exr_file)

  header    = input_file.header()
  dw        = header['dataWindow']

  width     = dw.max.x - dw.min.x + 1
  height    = dw.max.y - dw.min.y + 1
  n_channel = len(channel_list)

  # channels 값 읽기
  strings = input_file.channels(channel_list)

  # 값을 저장할 곳
  exr_data = np.zeros([height, width, n_channel], dtype=np.float32)

  # 각 채널의 string을 처리
  for index, string in enumerate(strings):
    # 읽은 string 값을 각 채널의 data type으로 변환(FLOAT, HALF)
    data = np.fromstring(string, dtype=_ch_np_precision(channel_list[index]))

    # height, width로 변환
    data = data.reshape(height, width)

    # nan-> 0, inf -> big number
    data = np.nan_to_num(data)

    # data 저장
    exr_data[:, :, index] = data

  return exr_data.astype(np.float32)

def to_tfrecord(noisy_dir, refer_dir, tfrecord_path, 
                n_file=-1, patch_size=0, n_sample=0, is_training=False,
                save_original_img=False, save_patch_img=False):
  ''' exr 데이터를 불러와서 tfrecord 파일 하나에 모두 넣는다.
      train일 경우는 patch로 crop하고 넣고 아닌 경우(valid, test)는 이미지
      그대로 집어 넣는다.
  Args:
    noisy_dir         : noisy image의 폴더 경로
    refer_dir         : reference의 폴더 경로
    tfrecord_path     : 저장할 tfrecord 파일 경로    
    patch_size        : 잘라낼 patch 크기 (train만 적용)
    n_sample          : 잘라낼 patch 개수 (train만 적용)
    n_file            : 폴더 안에서 읽고싶은 파일 개수(-1이면 마지막거 뺴고 다)
    is_training       : train(true) / valid, test(false)
    save_original_img : 읽은 파일들을 이미지로 저장하는 지 여부
    save_patch_img    : 패치들을 저장하는지 여부
  '''
    
  print("=====================================================================")
  print('noisy image in {} will be read...'.format(noisy_dir))
  print('reference in {} will be read.....'.format(refer_dir))
  print('Make patch : ', is_training)
  print("=====================================================================")
  
  # 나중에 출력할 총 시간과 패치 개수
  total_time = 0
  total_patch = 0

  with tf.python_io.TFRecordWriter(tfrecord_path) as writer:
    # exr input, gt 파일 이름들 모두 모으기
    exr_files = glob(os.path.join(noisy_dir, "*.exr"))[:n_file]

    assert len(exr_files) > 0, "There is no files in {}".format(noisy_dir)
    
    f_index = 0
    for exr_file in exr_files:
      f_index += 1
      start_time = time.time()

      # exr_file = path/(scene_name)-(n_spp)spp.exr
      exr_file_name = os.path.basename(exr_file).split('-')[0];
      exr_gt_file = _find_gt_file(exr_file_name, refer_dir)

      # exr 파일의 데이터들 읽기
      noisy_img      = read(exr_file)
      reference      = read(exr_gt_file)

      # 읽은 파일들을 저장
      if save_original_img:
        dir_name = './data/dataset/'
        save_debug_img(dir_name + 'noisy_img/' + exr_file_name, noisy_img)
        save_debug_img(dir_name + 'reference/' + exr_file_name, reference)
        
      # Train
      # ========================================================================
      if is_training:
        patch_num = 0
        for p_ni, p_re in _make_patches(noisy_img, reference, patch_size, n_sample):
          patch_num += 1
          # if is_appropriate(pr):
          _write_tfrecord(writer, p_ni, p_re)

          # 패치들을 저장
          if save_patch_img:
            dir_name = './data/dataset/patch'
            save_debug_img(dir_name + '/noisy_img/' + exr_file_name + '_' + str(patch_num), p_ni)
            save_debug_img(dir_name + '/reference/' + exr_file_name + '_' + str(patch_num), p_re)

        total_patch += patch_num
        

      # ========================================================================
      
      # Valid, Test
      # ========================================================================
      else:
        _write_tfrecord(writer, noisy_img, reference)
      # ========================================================================

      # print Result
      end_time = time.time() - start_time
      total_time += end_time
      h = int(total_time // 3600)
      m = int(total_time % 3600 // 60)
      s = int(total_time % 3600 % 60)
      now = datetime.now()

      print(f'{now.hour:02}:{now.minute:02}:{now.second:02} | ', end="")
      print(f'({exr_file_name}) file read, {f_index} of {len(exr_files)}, ', end="")
      print(f'elapsed time {end_time:.3f}s({h:02}h {m:02}m {s:02}s)')

def make_dataset(mode, tf_record_name,  
                n_file=-1, n_sample=0, patch_size=0,
                save_img=False, save_patch_img=False):
  print('=====================================================================')
  print('Making {} dataset ({}.tfrecord)'.format(mode, tf_record_name))
  print("The number of file to read : ", n_file, "(-1 is all)")
  print("The number of sample : ", n_sample)
  print("The size of patch : ", patch_size)
  print('=====================================================================')
  
  if mode == 'train':
    dataset_dir = 'data/train'
    is_training=True
  elif mode == 'valid':
    dataset_dir = 'data/valid'
    is_training=False
  elif mode == 'test':
    dataset_dir = 'data/test'
    is_training=False
  else:
    ValueError("mode is one of ['train', 'valid', 'test']")

  noisy_dir = os.path.join(dataset_dir, 'noisy_img')
  refer_dir = os.path.join(dataset_dir, 'reference')
  tfrecord  = 'data/tfrecord/{}.tfrecord'.format(tf_record_name)

  to_tfrecord(noisy_dir         = noisy_dir, 
              refer_dir         = refer_dir,
              tfrecord_path     = tfrecord, 
              n_file            = n_file,
              patch_size        = patch_size,
              n_sample          = n_sample,
              is_training       = is_training,
              save_original_img = save_img,
              save_patch_img    = save_patch_img)

if __name__ == '__main__':
  print('''
  =====================================================
  make_dataset('train', 'train', 
                n_file=-1, n_sample=400, patch_size=65, 
                save_img=False, save_patch_img=False)
  make_dataset('valid', 'valid', n_file=-1)
  make_dataset('test', 'test', n_file=-1)
  =====================================================
  ''')  

  input('지금 이렇게 데이터 셋 만들려고 하는데 확실해요?')
  make_dataset('train', 'train', 
                n_file=-1, n_sample=400, patch_size=65, 
                save_img=False, save_patch_img=False)
  make_dataset('valid', 'valid', n_file=-1)
  make_dataset('test', 'test', n_file=-1)