# -*- coding:utf-8 -*-

import pyaudio
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.fftpack import fft, ifft
import cv2

CHUNK=1024
RATE=44100
p=pyaudio.PyAudio()
N=10
CHUNK=1024*N
p=pyaudio.PyAudio()
fr = RATE
fn=51200*N/50
fs=fn/fr

stream=p.open(	format = pyaudio.paInt16,
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
while stream.is_active():
    input = stream.read(CHUNK)
    output = stream.write(input)
    sig =[]
    sig = np.frombuffer(input, dtype="int16")  /32768.0

    nperseg = 1024*N
    freq =fft(sig,int(fn))
    Pyy = np.sqrt(freq*freq.conj())/fn
    f = np.arange(20,20000,(20000-20)/int(fn)) #RATE11025,22050;N50,100
    ax2.set_ylim(0,0.05)
    ax2.set_xlim(20,20000)
    ax2.set_xlabel('Freq[Hz]')
    ax2.set_ylabel('Power')
    ax2.set_xscale('log')
    ax2.plot(2*f*RATE/44100,Pyy)
    
    #ax2.set_axis_off()
    x = np.linspace(0, 100, nperseg)
    ax1.plot(x,sig)
    ax1.set_ylim(-0.5,0.5)
    plt.pause(0.01)
    plt.savefig('IntensityvsFreq.jpg')
    plt.clf()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    img=cv2.imread('IntensityvsFreq.jpg')
    cv2.imshow('image',img)
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.imwrite('IntensityvsFreq_real.jpg',img) #z0,z1
        cv2.destroyAllWindows()
        break    
	
stream.stop_stream()
stream.close()
p.terminate()

print( "Stop Streaming")