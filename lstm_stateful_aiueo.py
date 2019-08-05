from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM
import csv

import cv2
import pyaudio
from scipy import signal
from scipy.fftpack import fft, ifft
import wave
import struct


fr=64*64*4 #44100
p=pyaudio.PyAudio()
sec=0.25
CHUNK=int(fr*sec)  #1024*N
p=pyaudio.PyAudio()
fn=fr*sec
#t=np.linspace(0,sec,fn)

stream=p.open(	format = pyaudio.paInt16,
		channels = 1,
		rate = fr,
		frames_per_buffer = CHUNK,
		input = True,
		output = True) # inputとoutputを同時にTrueにする

# length of input
input_len = 1000

# e.g. if tsteps=2 and input=[1, 2, 3, 4, 5],
#      then output=[1.5, 2.5, 3.5, 4.5]
tsteps = 2

# The input sequence length that the LSTM is trained on for each output point
lahead = 1

# training parameters passed to "model.fit(...)"
batch_size = 1
epochs = 10

# ------------
# MAIN PROGRAM
# ------------

print("*" * 33)
if lahead >= tsteps:
    print("STATELESS LSTM WILL ALSO CONVERGE")
else:
    print("STATELESS LSTM WILL NOT CONVERGE")
print("*" * 33)

np.random.seed(1986)

print('Generating Data...')

"""
def gen_uniform_amp(amp=1, xn=10000):
    data_input = np.random.uniform(-1 * amp, +1 * amp, xn)
    data_input = pd.DataFrame(data_input)
    return data_input

def gen_uniform_amp(amp=1, xn=10000):
    x0=0
    step=1
    period=200
    k=0.0001
    cos = np.zeros(((xn - x0) * step))
    for i in range(len(cos)):
        idx = x0 + i * step
        cos[i] = amp * (np.cos(20 * np.pi * idx / period))+amp * (np.cos(40 * np.pi * idx / period))
        cos[i] = cos[i]* np.exp(-k * idx)
    data_input = pd.DataFrame(cos)    
    return data_input


def sin_wav(A,f0,fs,t):
    point = np.arange(0,fs*t)
    sin_wav =A* np.sin(2*np.pi*f0*point/fs)
    return sin_wav

def create_sin_wav(A,f0,fs,t):
    sin_wave=0
    print(A[0])
    #int_f0=int(f0[0])
    for i in range(0,len(A),1):
        f1=f0[i]
        sw=sin_wav(A[i],f1,fs,t)
        sin_wave += sw
    sin_wave = [x * 1.0 for x in sin_wave]  #32767.0
    return sin_wave

def gen_uniform_amp(amp=1, xn=10000):
    x0=0
    step=1
    period=200
    k=0.0001
    #cos = np.zeros(((xn - x0) * step))
    A=(0.07,0.09,0.08,0.19,0.08,0.07) #a
    f0=261
    f=(f0,2*f0,3*f0,4*f0,5*f0,6*f0) #a
    fs=44100
    t=0.25 #sec
    sin_wav=create_sin_wav(A,f,fs,t)
    return sin_wav

# in order to maintain generated data length = input_len after pre-processing,
# add a few points to account for the values that will be lost.
to_drop = max(tsteps - 1, lahead - 1)
sin_wave = gen_uniform_amp(amp=0.1, xn=input_len + to_drop)
print(sin_wave)
data_input = pd.DataFrame(sin_wave)
sin_wave = [int(float(x)* 32767.0 ) for x in sin_wave] 


binwave = struct.pack("h" * len(sin_wave), *sin_wave)
output =stream.write(binwave)

"""
def readcsv_(path,sk):
    with open(path+str(sk)+'.txt', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',') #, quotechar=',')
        sin_wav=[]
        
        for row in spamreader:
            #print("row",row)
            sin_wav = (float(x) for x in row) #" ".join

    #print(type(sin_wav),sin_wav,"=sin_wav")
    sin_wave = [int(float(x)* 32767)  for x in sin_wav] #32767.0   
    binwave = struct.pack("h" * len(sin_wave), *sin_wave)        
    #print(binwave)
    return binwave,sin_wave

def readcsv(path,sk):
    #print("Input data path:  " + path)
    with open(path+str(sk)+'.txt', 'r', newline='\n', encoding="utf-8") as f:
        reader = csv.reader(f, lineterminator='\n') # 改行コード（\n）を指定しておく
        Row=[]
        for row in reader:
            Row.append(row)
    return Row

to_drop = max(tsteps - 1, lahead - 1)
sin_wave=[]
list_=('a','e','i','o','u')
for sky in range(0,5,1):
    sky=list_[sky]
    for sk in range(0,10,1):
        b,x_ = readcsv_("./aiueo/sig_0730/"+sky+"_64x64/boin_fig",sk)
        sin_wave += x_
for i in range(len(sin_wave)):
    sin_wave[i]=sin_wave[i]/32762.
sky='_all'
print(sin_wave)        
#sin_wave = [int(float(x)* 32767.0) for x in sin_wave]        
data_input = pd.DataFrame(sin_wave)
#data_input = np.array(sin_wave)

# set the target to be a N-point average of the input
expected_output = data_input  #.rolling(window=tsteps, center=False).mean()

# when lahead > 1, need to convert the input to "rolling window view"
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.repeat.html
if lahead > 1:
    data_input = np.repeat(data_input.values, repeats=lahead, axis=1)
    data_input = pd.DataFrame(data_input)
    for i, c in enumerate(data_input.columns):
        data_input[c] = data_input[c].shift(i)

# drop the nan
expected_output = expected_output[to_drop:]
data_input = data_input[to_drop:]

print('Input shape:', data_input.shape)
print('Output shape:', expected_output.shape)
print('Input head: ')
print(data_input.head())
print('Output head: ')
print(expected_output.head())
print('Input tail: ')
print(data_input.tail())
print('Output tail: ')
print(expected_output.tail())

print('Plotting input and expected output')
plt.plot(data_input[0][:100], '.')  #10
plt.plot(expected_output[0][:100], '-')  #10
plt.legend(['Input', 'Expected output'])
plt.title('Input')
#plt.show()
plt.savefig('./aiueo/sig_0730/inputExpected'+str(sky)+'.jpg')
plt.show()

def create_model(stateful):
    model = Sequential()
    model.add(LSTM(20,
              input_shape=(lahead, 1),
              batch_size=batch_size,
              stateful=stateful))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

print('Creating Stateful Model...')
model_stateful = create_model(stateful=True)


# split train/test data
def split_data(x, y, ratio=0.8):
    to_train = int(input_len * ratio)
    # tweak to match with batch_size
    to_train -= to_train % batch_size

    x_train = x[:to_train]
    y_train = y[:to_train]
    x_test = x[to_train:]
    y_test = y[to_train:]

    # tweak to match with batch_size
    to_drop = x.shape[0] % batch_size
    if to_drop > 0:
        x_test = x_test[:-1 * to_drop]
        y_test = y_test[:-1 * to_drop]

    # some reshaping
    reshape_3 = lambda x: x.values.reshape((x.shape[0], x.shape[1], 1))
    x_train = reshape_3(x_train)
    x_test = reshape_3(x_test)

    reshape_2 = lambda x: x.values.reshape((x.shape[0], 1))
    y_train = reshape_2(y_train)
    y_test = reshape_2(y_test)

    return (x_train, y_train), (x_test, y_test)


(x_train, y_train), (x_test, y_test) = split_data(data_input, expected_output)
print('x_train.shape: ', x_train.shape)
print('y_train.shape: ', y_train.shape)
print('x_test.shape: ', x_test.shape)
print('y_test.shape: ', y_test.shape)

sin_wave=y_test
sin_wave = [np.clip(int(float(x)* 32767.0 ),-32768,32767) for x in sin_wave] 
binwave = struct.pack("h" * len(sin_wave), *sin_wave)
for i in range(10):
    output =stream.write(binwave)

#model_stateful.load_weights('./aiueo/sig_0730/lstm_stateful'+str(sky)+'_epoch_009.hdf5')
    
print('Training')
for i in range(epochs):
    print('Epoch', i + 1, '/', epochs)

    model_stateful.fit(x_train,
                       y_train,
                       batch_size=batch_size,
                       epochs=1,
                       verbose=1,
                       validation_data=(x_test, y_test),
                       shuffle=False)
    model_stateful.save_weights('./aiueo/sig_0730/lstm_stateful'+str(sky)+'_epoch_{0:03d}.hdf5'.format(i), True)
    model_stateful.reset_states()

print('Predicting')
predicted_stateful = model_stateful.predict(x_test, batch_size=batch_size)

print('Creating Stateless Model...')
model_stateless = create_model(stateful=False)
#model_stateless.load_weights('./aiueo/sig_0730/lstm_stateless'+str(sky)+'_epoch_.hdf5')

print('Training')
model_stateless.fit(x_train,
                    y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test),
                    shuffle=False)
model_stateless.save_weights('./aiueo/sig_0730/lstm_stateless'+str(sky)+'_epoch_.hdf5', True)

print('Predicting')
predicted_stateless = model_stateless.predict(x_test, batch_size=batch_size)

# ----------------------------
"""
sin_wave=y_test
sin_wave = [int(float(x)* 32767.0 ) for x in sin_wave] 
binwave = struct.pack("h" * len(sin_wave), *sin_wave)
output =stream.write(binwave)
"""
sin_wave=(predicted_stateful).flatten()[tsteps - 1:]
sin_wave = [np.clip(int(float(x)* 32767.0),-32767,32767) for x in sin_wave]  #32767.0
binwave = struct.pack("h" * len(sin_wave), *sin_wave)
for i in range(10):
    output =stream.write(binwave)

sin_wave=(predicted_stateless).flatten()
sin_wave = [np.clip(int(float(x)* 32767.0),-32767,32767) for x in sin_wave] 
binwave = struct.pack("h" * len(sin_wave), *sin_wave)
for i in range(10):
    output =stream.write(binwave)

print('Plotting Results')
plt.subplot(3, 1, 1)
plt.plot(y_test[:1000])
plt.title('Expected')
y_min=min(y_test)
y_max=max(y_test)
plt.ylim(y_min-0.01,y_max+0.01)
plt.subplot(3, 1, 2)
# drop the first "tsteps-1" because it is not possible to predict them
# since the "previous" timesteps to use do not exist
#plt.plot((y_test - predicted_stateful).flatten()[tsteps - 1:1000])
plt.plot((predicted_stateful).flatten()[tsteps - 1:1000])
plt.title('Stateful: Expected - Predicted')
#plt.ylim(y_min-0.01,y_max+0.01)
plt.subplot(3, 1, 3)
#plt.plot((y_test - predicted_stateless).flatten()[0:1000])
plt.plot((predicted_stateless).flatten()[0:1000])
plt.title('Stateless: Expected - Predicted')
#plt.ylim(y_min-0.01,y_max+0.01)
#plt.show()
plt.savefig('./aiueo/sig_0730/Stateful'+str(sky)+'.jpg')
plt.show()