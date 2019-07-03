import numpy as np
from matplotlib import pyplot as plt
import wave
import struct
import pyaudio
from scipy.fftpack import fft, ifft
import cv2

#パラメータ
RATE=44100
N=1
CHUNK=1024*N
p=pyaudio.PyAudio()
fn=RATE
nperseg=fn*N

stream=p.open(format = pyaudio.paInt16,
        channels = 1,
        rate = RATE,
        frames_per_buffer = CHUNK,
        input = True,
        output = True) # inputとoutputを同時にTrueにする

A=0.6#振幅
B=1
C=0.3
D=0.1 
fs = 44100#サンプリング周波数
f0 = 130 #440#基本周波数(今回はラ)
f1 = 290
f2 = 420
f3 = 530
sec = 0.1 #秒

def create_wave(A,f0,fs,t):#A:振幅,f0:基本周波数,fs:サンプリング周波数,再生時間[s]
    point = np.arange(0,fs*t)
    sin_wave =(A* np.sin(2*np.pi*f0*point/fs)+B* np.sin(2*np.pi*f1*point/fs)+C* np.sin(2*np.pi*f2*point/fs)+D* np.sin(2*np.pi*f3*point/fs))/4

    sin_wave = [int(x * 32767.0) for x in sin_wave]
    binwave = struct.pack("h" * len(sin_wave), *sin_wave)
    w = wave.Wave_write("130+290+420+530Hz.wav")
    p = (1, 2, fs, len(binwave), 'NONE', 'not compressed')
    w.setparams(p)
    w.writeframes(binwave)
    w.close()

def sound_wave():
    wavfile = "130+290+420+530Hz.wav"
    wr = wave.open(wavfile, "rb")
    input = wr.readframes(wr.getnframes())
    output = stream.write(input)
    sig =[]
    sig = np.frombuffer(input, dtype="int16")  /32768.0
    return sig

fig = plt.figure(figsize=(12, 8)) #...1
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
while True:
    create_wave(A, f0, fs, sec)
    sig = sound_wave()
    #cv2.imshow('image',img)
    freq =fft(sig,int(fn))
    Pyy = np.sqrt(freq*freq.conj())/fn
    f = np.arange(20,20000,(20000-20)/int(fn))
    ax2.set_ylim(0,0.05)
    ax2.set_xlim(20,20000)
    ax2.set_xlabel('Freq[Hz]')
    ax2.set_ylabel('Power')
    ax2.set_xscale('log')
    ax2.plot(2*f*RATE/44100,Pyy)
    
    #ax2.set_axis_off()
    x = np.linspace(0, 1, sec*nperseg)
    ax1.plot(x,sig)
    ax1.set_ylim(-1,1)
    plt.savefig('IntensityvsFreq.jpg')
    plt.clf()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    img=cv2.imread('IntensityvsFreq.jpg')
    cv2.imshow('image',img)
    print("f0_{},f1_{},f2_{},f3_{}".format(f0,f1,f2,f3))
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()
        break
    elif k == ord('j'): # 
        A=A+0.1
    elif k == ord('h'): # 
        A=A-0.1
    elif k == ord('u'): # 
        C=C+0.1
    elif k == ord('n'): # 
        C=C-0.1
    elif k == ord('i'): # 
        D=D+0.1
    elif k == ord('m'): # 
        D=D-0.1        
    elif k == ord('f'): # 
        cv2.imwrite('./data/f_{}_{}_{}_{}.jpg'.format(f0,f1,f2,f3),img)
        f1=f1+1
    elif k == ord('r'): # 
        cv2.imwrite('./data/f_{}_{}_{}_{}.jpg'.format(f0,f1,f2,f3),img)
        f1=f1-1           