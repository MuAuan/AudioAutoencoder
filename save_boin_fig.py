# -*- coding:utf-8 -*-

import pyaudio
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.fftpack import fft, ifft
import cv2
import csv
import wave
import struct

fr=10000 #44100
p=pyaudio.PyAudio()
sec=0.25
CHUNK=int(fr*sec)  #1024*N
p=pyaudio.PyAudio()
fn=fr*sec
t=np.linspace(0,sec,fn)

stream=p.open(	format = pyaudio.paInt16,
		channels = 1,
		rate = fr,
		frames_per_buffer = CHUNK,
		input = True,
		output = True) # inputとoutputを同時にTrueにする
# Figureの初期化
fig = plt.figure(figsize=(6, 6)) #...1
# Figure内にAxesを追加()
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)
sk=0

path = './aiueo/sig_0730/boin_fig'

# listをCSVファイルで出力
def writecsv(path,sk,output_data):
    #print("Output data path:  " + path)
    #print(output_data)
    with open(path+str(sk)+'.txt', 'w', newline='\n', encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator='\n') # 改行コード（\n）を指定しておく
        writer.writerow(output_data)

def readcsv(path,sk):
    with open(path+str(sk)+'.txt', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',') #, quotechar=',')
        sin_wav=[]
        
        for row in spamreader:
            #print("row",row)
            sin_wav = (float(x) for x in row) #" ".join

    #print(type(sin_wav),sin_wav,"=sin_wav")
    sin_wave = [int(float(x)* 32767.0 ) for x in sin_wav]    
    binwave = struct.pack("h" * len(sin_wave), *sin_wave)        
    #print(binwave)
    return binwave,sin_wave        
        
#with open('./aiueo/sig/boin_fig.txt', 'w', newline='\n', encoding="utf-8") as f:
while stream.is_active():
    input = stream.read(CHUNK)
    #output = stream.write(input)
    sig =[]
    sig = np.frombuffer(input, dtype="int16")  /32768.0
    #print("sig",sig)
    
    writecsv(path,sk,sig)
    bin_wave,sin_wave = readcsv(path,sk)
    output =stream.write(bin_wave)

    sig1=sig.reshape(50,50) #50000*0.05=2500
    #print(sig1.shape)
    ax1.plot(t,sig)
    ax2.plot(t,sin_wave)
    ax3.imshow(np.clip(sig1,0,1))
    
    plt.pause(0.01)
    plt.savefig('./aiueo/sig_0730/IntensityvsFreq_'+str(sk)+'.jpg')
    plt.clf()
        
    ax3 = fig.add_subplot(111)
    img=cv2.imread('./aiueo/sig_0730/IntensityvsFreq_'+str(sk)+'.jpg')
    sk+=1
    cv2.imshow('image',img)
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.imwrite('./aiueo/sig_0730/IntensityvsFreq_real_'+str(sk)+'.jpg',img)
        cv2.destroyAllWindows()
        break    
	
stream.stop_stream()
stream.close()
p.terminate()

print( "Stop Streaming")