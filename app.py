from flask import Flask, render_template,request,url_for
from flask_bootstrap import Bootstrap 
from tensorflow import keras
import pandas as pd
import tensorflow as tf
from underthesea import word_tokenize
import os
import numpy as np
from keras.preprocessing.sequence import pad_sequences

ROOT_DIR = os.path.abspath(os.curdir)

print(ROOT_DIR)

# Doc du lieu
data2 = pd.read_excel(ROOT_DIR+"/dataset_28600.xlsx", header=None)

data_train = []
j=0
for i in data2[0]:
  if len(i.split())>110 or len(i.split())<1:
    continue
  else :
    data_train.append(i)
data_train = np.array(data_train)

# tạo từ điển
x = []
word2id = {}
id2word = {}
vocab_size = 0
max_length = 0
for i in data_train:
  # kiem tra max length
  x.append(i)
  if max_length<len(i.split()):
    max_length = len(i.split())
  for j in i.split():
    if j not in word2id.keys():
      vocab_size+=1
      word2id[j] = vocab_size
      id2word[vocab_size] = j

# thêm ký tự pad 
word2id['[PAD]'] = 0
id2word[0]='[PAD]'
vocab_size+=1

# padding (k cần ghi do t làm bằng thư viện)
def padding(s, max_length=40):
  s+=' [PAD] '*(max_length-len(s.split()))
  return s.strip()

# chuyển từ thành số
def encode(s):
  s = word_tokenize(s)
  text = []
  for i in s:
    if (word2id.get(i) == None)  :
      text.append(word2id['[PAD]'])
    else :
      text.append(word2id[i])
  return text


def predict(x):
  switcher={
        1:'phục vụ tệ',
        2:'món ăn tệ',
        3:'không hợp vệ sinh',
        4:'hợp vệ sinh',
        5:'phục vụ tốt',
        6:'món ăn ngon',
        7:'khác'
      }
  return switcher.get(x, "Lỗi")

app = Flask(__name__,static_url_path = "/static", static_folder = "static")
Bootstrap(app)
model = keras.models.load_model(ROOT_DIR+'/Model2.h5')

@app.route('/',methods = ["GET","POST"])
def index():
  if request.method =="GET":
	  return render_template('TrangChủ.html')
  else :
    text = request.form.get("text")
    text = word_tokenize(text, format='text')
    text = [encode(text)]
    text = np.array(text)
    text = pad_sequences(text, maxlen=max_length, padding='post')
    
    pred = model.predict_classes(text)
    return render_template("pred.html",text =predict(pred[0]))

@app.route('/<string:text>',methods = ["GET","POST"])
def pred(text):
	return text


if __name__ == '__main__':
	app.debug=True
	app.run()