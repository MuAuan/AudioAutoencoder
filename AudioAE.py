# -*- coding:utf-8 -*-

import pyaudio
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import wave
import librosa
import wave
import struct


from keras.layers import Input, Dense
from keras.models import Model


# this is the size of our encoded representations
encoding_dim = 320  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(10240,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(10240, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.summary()

RATE=44100
N=10
CHUNK=1024*N
p=pyaudio.PyAudio()
fr = RATE
fn=51200*N/50
fs=fn/fr
FORMAT = pyaudio.paInt16
CHANNELS=1
fs = 44100#サンプリング周波数
f0 = 440#基本周波数(今回はラ)
sec = 10 #秒

stream=p.open(format = pyaudio.paInt16,
        channels = 1,
        rate = RATE,
        frames_per_buffer = CHUNK,
        input = True,
        output = True) # inputとoutputを同時にTrueにする
# Figureの初期化
fig = plt.figure(figsize=(16, 8)) #...1
# Figure内にAxesを追加()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
frames=[]
s=0
while stream.is_active():
    input = stream.read(CHUNK)
    #print("input",input)
    #output = stream.write(input)
    sig =[]
    sig = np.frombuffer(input, dtype="int16") /32768.0
    encoded_imgs = encoder.predict(sig.reshape(1,10240,))
    sig = decoder.predict(encoded_imgs)
    sig=sig.reshape(10240,)
    sig_x=sig
    sin_wave = [int(x * 32767.0) for x in sig_x]#16bit符号付き整数に変換
    #バイナリ化
    binwave = struct.pack("h" * len(sin_wave), *sin_wave)
    output = stream.write(binwave)
    #サイン波をwavファイルとして書き出し
    w = wave.Wave_write("input_AE_"+str(s)+".wav")
    p = (1, 2, fs, len(binwave), 'NONE', 'not compressed')
    #(チャンネル数(1:モノラル,2:ステレオ)、サンプルサイズ(バイト)、サンプリング周波数、フレーム数、圧縮形式(今のところNONEのみ)、圧縮形式を人に判読可能な形にしたもの？通常、 'NONE' に対して 'not compressed' が返されます。)
    w.setparams(p)
    w.writeframes(binwave)
    w.close()
    if s>10:
        break
    s+=1
    
    """
    nperseg = 1024*N
    
    f, t, Zxx = signal.stft(sig, fs=fn, nperseg=nperseg)
    ax2.pcolormesh(fs*t, f/fs, np.abs(Zxx), cmap='hsv')
    ax2.set_xlim(0,fs)
    ax2.set_ylim(2,20000)
    ax2.set_yscale('log')
    ax2.set_axis_off()
    x = np.linspace(0, 100, nperseg)
    ax1.plot(x,sig)
    ax1.set_ylim(-0.1,0.6)
    plt.pause(0.01)
    plt.clf()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    """
    
stream.stop_stream()
stream.close()

print( "Stop Streaming")