import wave
import numpy as np
from matplotlib import pylab as plt
import struct
import pyaudio
import matplotlib

a = 1     #振幅
fs = 44100 #サンプリング周波数
f0 = 400  #周波数
f1 = f0-398
sec = 5   #秒
CHUNK=1024
p=pyaudio.PyAudio()

stream=p.open(	format = pyaudio.paInt16,
        channels = 1,
        rate = fs,
        frames_per_buffer = CHUNK,
        input = True,
        output = True) # inputとoutputを同時にTrueにする

swav=[]
for n in np.arange(fs * sec):
    #サイン波を生成
    s = (a * np.sin(2.0 * np.pi * f0 * n / fs)+ a * np.sin(2.0 * np.pi * f1 * n / fs))/2
    swav.append(s)
#サイン波を表示
matplotlib.rcParams.update({'font.size': 18, 'font.family': 'sans', 'text.usetex': False}) 
fig = plt.figure(figsize=(8,6))  #(width,height)
x_offset=np.round(0.05*4,decimals=2)
y_offset=np.round(0.05*10,decimals=2)
width=0.3
height=0.3
axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # main axes
axes2 = fig.add_axes([x_offset, y_offset, width, height]) # insert axes

x = np.linspace(0, sec, int(fs*sec)) #sec;サンプリング時間、fs*sec;サンプリング数
axes1.set_ylim(-1.2,1.2)
axes1.set_xlim(0,0.1)
axes1.plot(x,swav)

axes2.set_ylim(-1.2,1.2)
axes2.set_xlim(0,5)
axes2.plot(x,swav)
plt.pause(1)
plt.savefig('./fft/sound_'+str(f0)+'_'+str(f1)+'.jpg')

#サイン波を-32768から32767の整数値に変換(signed 16bit pcmへ)
swav = [int(x * 32767.0) for x in swav]
#バイナリ化
binwave = struct.pack("h" * len(swav), *swav)
#サイン波をwavファイルとして書き出し
w = wave.Wave_write("./fft/output"+str(f0)+"_"+str(f1)+".wav")
params = (1, 2, fs, len(binwave), 'NONE', 'not compressed')
w.setparams(params)
w.writeframes(binwave)
w.close()

input = binwave
output = stream.write(input)