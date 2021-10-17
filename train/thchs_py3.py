# For python 3
import os, codecs, sys
import numpy as np
import pdb
from scipy.fftpack import fft
import scipy.io.wavfile as wav
from random import shuffle
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D
from tensorflow.keras.layers import Reshape, Dense, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model

import matplotlib.pyplot as plt

SRC_FILE = "./data/data_thchs30/"

g_plotted = False

def compute_fbank_data(wavsignal):
  x=np.linspace(0, 400 - 1, 400, dtype = np.int64)
  w = 0.54 - 0.46 * np.cos(2 * np.pi * (x) / (400 - 1) ) # 汉明窗
  fs = 16000
  # wav波形 加时间窗以及时移10ms
  time_window = 25 # 单位ms
  window_length = fs / 1000 * time_window # 计算窗长度的公式，目前全部为400固定值
  wav_arr = np.array(wavsignal)
  wav_length = len(wavsignal)
  range0_end = int(len(wavsignal)/fs*1000 - time_window) // 10 # 计算循环终止的位置，也就是最终生成的窗数
  data_input = np.zeros((range0_end, 200), dtype = np.float) # 用于存放最终的频率特征数据
  data_line = np.zeros((1, 400), dtype = np.float)
  for i in range(0, range0_end):
    p_start = i * 160
    p_end = p_start + 400
    data_line = wav_arr[p_start:p_end]    
    data_line = data_line * w # 加窗
    data_line = np.abs(fft(data_line))
    data_input[i]=data_line[0:200] # 设置为400除以2的值（即200）是取一半数据，因为是对称的
  data_input = np.log(data_input + 1)
  #data_input = data_input[::]
  return data_input

# 获取信号的时频图
def compute_fbank(file):
  fs, wavsignal = wav.read(file)
  if fs != 16000:
    assert False
  return compute_fbank_data(wavsignal)

def source_get(path):
  train_path = path + "/data"
  label_list = []
  wav_list = []
  for root, dirs, files in os.walk(train_path):
    for fn in files:
      if fn.endswith(".wav") or fn.endswith(".WAV"):
        wav_fn = os.sep.join([root, fn])
        label_fn = wav_fn + ".trn"
        wav_list.append(wav_fn)
        label_list.append(label_fn)
  return label_list, wav_list

def read_label(label_fn):
  with codecs.open(label_fn, "r", encoding="utf-8") as f:
    data = f.readlines()
    return data[1]

def gen_label_data(label_list):
  label_data = []
  for fn in label_list:
    x = read_label(fn)
    label_data.append(x.strip("\n"))
  return label_data

def make_vocab(label_data):
  vocab = []
  for line in label_data:
    line = line.split(" ")
    for x in line:
      if x not in vocab:
        vocab.append(x)
  vocab.append("_")
  return vocab

def word2id(line, vocab):
  return [vocab.index(x) for x in line.split(" ")]

# data is spectrogram
def wav_pad_one(data):
  wav_len = len(data)
  wav_len = ((wav_len-1) // 8 + 1) * 8
  ret = np.zeros((1, wav_len, 200, 1))
  ret[0, :len(data), :, 0] = data
  return ret

def wav_padding(wav_data_list):
  wav_lens = [len(data) for data in wav_data_list]
  wav_max_len = max(wav_lens)
  wav_lens = np.array([leng//8 for leng in wav_lens])
  new_wav_data_lst = np.zeros((len(wav_data_list), wav_max_len, 200, 1))
  for i in range(len(wav_data_list)):
    new_wav_data_lst[i, :wav_data_list[i].shape[0], :, 0] = wav_data_list[i]
  return new_wav_data_lst, wav_lens

def label_padding(label_data_list):
  label_lens = np.array([len(label) for label in label_data_list])
  max_label_len = max(label_lens)
  new_label_data_lst = np.zeros((len(label_data_list), max_label_len))
  for i in range(len(label_data_list)):
    new_label_data_lst[i][:len(label_data_list[i])] = label_data_list[i]
  return new_label_data_lst, label_lens

def data_generator(batch_size, shuffle_list, wav_list, label_data, vocab):
  for i in range(len(shuffle_list)//batch_size):
    wav_data_lst = []
    label_data_lst = []
    begin = i * batch_size
    end = begin + batch_size
    sub_list = shuffle_list[begin:end]
    for index in sub_list:
      #print("computing fbank for %s" % wav_list[index])
      fbank = compute_fbank(wav_list[index])
      pad_fbank = np.zeros((fbank.shape[0]//8*8+8, fbank.shape[1]))
      pad_fbank[:fbank.shape[0], :] = fbank
      label = word2id(label_data[index], vocab)
      wav_data_lst.append(pad_fbank)
      label_data_lst.append(label)
    pad_wav_data, input_length = wav_padding(wav_data_lst)
    pad_label_data, label_length = label_padding(label_data_lst)
    inputs = {'the_inputs': pad_wav_data,
              'the_labels': pad_label_data,
              'input_length': input_length,
              'label_length': label_length,
              }
    outputs = {'ctc': np.zeros(pad_wav_data.shape[0],)} 
    yield inputs, outputs

# ========================================

def conv2d(size):
  return Conv2D(size, (3,3), use_bias=True, activation='relu',
  padding='same', kernel_initializer='he_normal')

def norm(x):
  return BatchNormalization(axis=-1)(x)

def maxpool(x):
  return MaxPooling2D(pool_size=(2, 2), strides=None, padding="valid")(x)

def dense(units, activation="relu"):
  return Dense(units, activation=activation, use_bias=True,
  kernel_initializer="he_normal")

def cnn_cell(size, x, pool=True):
  x = norm(conv2d(size)(x))
  x = norm(conv2d(size)(x))
  if pool:
    x = maxpool(x)
  return x

def ctc_lambda(args):
  labels, y_pred, input_length, label_length = args
  y_pred = y_pred[:, :, :]
  return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

class AModel():
  def __init__(self, vocab_size):
    self.vocab_size = vocab_size
    self._model_init()
    self._ctc_init()
    self.opt_init()

  def _model_init(self):
    self.inputs = Input(name="the_inputs", shape=(None, 200, 1))
    self.h1 = cnn_cell(32, self.inputs)
    self.h2 = cnn_cell(64, self.h1)
    self.h3 = cnn_cell(128, self.h2)
    self.h4 = cnn_cell(128, self.h3, pool=False)
    self.h6 = Reshape((-1, 3200))(self.h4)
    self.h7 = dense(256)(self.h6)
    self.outputs = dense(self.vocab_size, activation="softmax")(self.h7)
    self.model = Model(inputs=self.inputs, outputs=self.outputs)

  def _ctc_init(self):
    self.labels = Input(name="the_labels", shape=[None], dtype="float32")
    self.input_length = Input(name="input_length", shape=[1], dtype="int64")
    self.label_length = Input(name="label_length", shape=[1], dtype="int64")
    self.loss_out = Lambda(ctc_lambda, output_shape=(1,), name="ctc")\
      ([self.labels, self.outputs, self.input_length, self.label_length])
    self.ctc_model = Model(inputs=[self.labels, self.inputs,
      self.input_length, self.label_length], outputs=self.loss_out)

  def opt_init(self):
    opt = Adam(lr = 0.0008, beta_1 = 0.9, beta_2 = 0.999, decay = 0.01, epsilon = 10e-8)
    self.ctc_model.compile(loss = {"ctc": lambda y_true, output: output}, optimizer=opt)

label_list, wav_list = source_get(SRC_FILE)
label_data = gen_label_data(label_list)
vocab = make_vocab(label_data)
vocab_size = len(vocab)
assert(len(label_data) == len(label_list))

total_nums = len(label_data)
batch_size = 5
batch_num = total_nums // batch_size
epochs = 1

shuffle_list = [i for i in range(total_nums)]
shuffle(shuffle_list)

print("%d labels and wav files." % len(label_list))
print("%d label data entries." % len(label_data))
print("%d entries in vocabulary." % len(vocab))


def decode_ctc(num_result, num2word):
  result = num_result[:, :, :]
  in_len = np.zeros((1), dtype = np.int32)
  in_len[0] = result.shape[1];
  r = K.ctc_decode(result, in_len, greedy = True, beam_width=10, top_paths=1)
  r1 = K.get_value(r[0][0])
  r1 = r1[0]
  text = []
  for i in r1:
    text.append(num2word[i])
  return r1, text

am = AModel(vocab_size)
am.model.summary()
am.ctc_model.summary()

WEIGHT_PATH = "weights_my_py3/weights.h5"
MODEL_PATH = "weights_my_py3/model.h5"
is_train = False
is_test  = False

def RenderInput(x):
  blah = np.rollaxis(x, 3, 0)[0][0]
  fig, (ax0, ax1) = plt.subplots(ncols=2)
  im = ax0.pcolormesh(np.arange(-0.5,blah.shape[1],1),
                      np.arange(-0.5,blah.shape[0],1),
                      blah)
  fig.colorbar(im, ax1)
  fig.tight_layout()
  fig.savefig("/tmp/x.png")

for arg in sys.argv:
  if arg == "load":
    print("[py3] 将从 %s 中读入权重并测试" % WEIGHT_PATH)
    if os.path.exists(WEIGHT_PATH):
      am.model.load_weights(WEIGHT_PATH, by_name=True)
      is_test = True
    else:
      print(Exception("%s does not exist" % WEIGHT_PATH))
      exit(0)
  elif arg == "train":
    print("[py3] 将继续训练")
    if os.path.exists(WEIGHT_PATH):
      am.model.load_weights(WEIGHT_PATH, by_name=True)
    is_train = True
    is_test = True
  elif arg == "testonefile":
    if os.path.exists(WEIGHT_PATH):
      am.model.load_weights(WEIGHT_PATH, by_name=True)
    else:
      print("Weights %s does not exist." % WEIGHT_PATH)
      exit(0)
    print("Will test a single file")
    fn = sys.argv[2]
    if not os.path.exists(fn):
      print("%s does not exist." % fn)
      exit(0)
    sr, data = wav.read(fn)
    if sr != 16000:
      print("Sample rate is not 16KHz")
      exit(0)
    if len(data.shape) > 1:
      print("Not single channel")
      exit(0)
    ffts = compute_fbank_data(data)
    print(ffts)
    fbanks = wav_pad_one(ffts)
    print(fbanks.shape)
    RenderInput(fbanks)
    result = am.model.predict(fbanks, steps=1)
    result, text = decode_ctc(result, vocab)
    print("Result: %s" % result)
    print("Text  : %s" % text)
  elif arg == "testfbankinput" and len(sys.argv) > 2:
    if os.path.exists(WEIGHT_PATH):
      am.model.load_weights(WEIGHT_PATH, by_name=True)
    else:
      print("Weights %s does not exist." % WEIGHT_PATH)
      exit(0)
    print("Will use feature bank input.")
    with open(sys.argv[2]) as f:
      ffts = []
      for line in f:
        line = line.strip()
        if len(line) > 0:
          fftline = [float(x) for x in line.split(",")]
          fftline = fftline[0:200]
          ffts.append(fftline)
      print("len(ffts)=%d" % len(ffts))
      theinput = np.zeros((len(ffts), 200))
      for i in range(len(ffts)):
        for j in range(0, 200):
          theinput[i][j] = ffts[i][j]
      print(theinput)
      theinput = wav_pad_one(theinput)
      RenderInput(theinput)
      result = am.model.predict(theinput, steps=1)
      result, text = decode_ctc(result, vocab)
      print("Result: %s" % result)
      print("Text  : %s" % text)
    exit(0)

if is_train:
  for k in range(epochs):
    print("Epoch #%d/%d" % (k+1, epochs))
    batch = data_generator(batch_size, shuffle_list, wav_list, label_data, vocab)
    am.ctc_model.fit_generator(batch, steps_per_epoch=batch_num, epochs=1)
  print("将把权重存入 %s 中" % WEIGHT_PATH)
  am.model.save_weights(WEIGHT_PATH)
  print("Saving model to %s" % MODEL_PATH);
  am.model.save(MODEL_PATH);
if is_test:
  batch = data_generator(1, shuffle_list, wav_list, label_data, vocab)
  for i in range(10):
    # 载入训练好的模型，并进行识别
    inputs, outputs = next(batch)
    x = inputs['the_inputs']
    y = inputs['the_labels'][0]
    if i == 1:
      print(x)
      RenderInput(x)
    result = am.model.predict(x, steps=1)
    # 将数字结果转化为文本结果
    result, text = decode_ctc(result, vocab)
    print("Data ID=%s" % wav_list[shuffle_list[i]])
    print("数字结果： %s" % str(result))
    print("文本结果： %s" % str(text))
    print("原文结果： %s" % str([vocab[int(i)] for i in y]))
