import numpy as np
from matplotlib import pyplot as plt
import wave
import struct
import pyaudio
from scipy.fftpack import fft, ifft
import cv2
from scipy import signal
import matplotlib

#パラメータ
RATE=44100
CHUNK=1024
p=pyaudio.PyAudio()
sa= 'jyoyu' #'u' #'o' #'i' #'e' #'a'
stream=p.open(format = pyaudio.paInt16,
        channels = 1,
        rate = RATE,
        frames_per_buffer = CHUNK,
        input = True,
        output = True) # inputとoutputを同時にTrueにする

fr = RATE #サンプリング周波数
sec =1 #秒

def sin_wav(A,f0,fr,t):
    point = np.arange(0,fr*t)
    sin_wav =A* np.sin(2*np.pi*f0*point/fr)
    return sin_wav

def create_wave(A,f0,fr,t):#A:振幅,f0:基本周波数,fr:サンプリング周波数,再生時間[s]
    sin_wave=0
    #print(A[0])
    int_f0=int(f0[0])
    for i in range(0,len(A),1):
        f1=f0[i]
        sw=sin_wav(A[i],f1,fr,t)
        sin_wave += sw
    sin_wave = [int(x * 32767.0) for x in sin_wave]    
    binwave = struct.pack("h" * len(sin_wave), *sin_wave)
    w = wave.Wave_write('./fft_sound/'+sa+'_'+str(sec)+'Hz.wav')
    p = (1, 2, fr, len(binwave), 'NONE', 'not compressed')
    w.setparams(p)
    w.writeframes(binwave)
    w.close()

def sound_wave(fu):
    int_f0=int(fu)
    wavfile = './fft_sound/'+sa+'_'+str(sec)+'Hz.wav'
    wr = wave.open(wavfile, "rb")
    input = wr.readframes(wr.getnframes())
    output = stream.write(input)
    sig =[]
    sig = np.frombuffer(input, dtype="int16")  /32768.0
    return sig

a = 1     #振幅
fr = 44100 #サンプリング周波数
f0 = 400  #周波数
f1 = f0-50
CHUNK=int(sec*fr) #1024
fn=fr * sec

t = np.linspace(0,sec, fn)
input = stream.read(CHUNK)

sig =[]
sig = np.frombuffer(input, dtype="int16")  /32768.0

#サイン波を-32768から32767の整数値に変換(signed 16bit pcmへ)
swav = [int(x * 32767.0) for x in sig]
#バイナリ化
binwave = struct.pack("h" * len(swav), *swav)
w = wave.Wave_write("./fft_sound/output_"+str(sec)+"_"+sa+".wav")
params = (1, 2, fr, len(binwave), 'NONE', 'not compressed')
w.setparams(params)
w.writeframes(binwave)
w.close()

def FFT(sig,fn,fr):
    freq =fft(sig,fn)
    Pyy = np.sqrt(freq*freq.conj())/fn
    f = np.arange(0,fr,fr/fn)
    ld = signal.argrelmax(Pyy, order=10) #相対強度の最大な番号をorder=10で求める
    ssk=0
    fsk=[]
    Psk=[]
    maxPyy=max(np.abs(Pyy))
    for i in range(len(ld[0])):  #ピークの中で以下の条件に合うピークの周波数fと強度Pyyを求める
        if np.abs(Pyy[ld[0][i]])>0.1*maxPyy and f[ld[0][i]]<20000 and f[ld[0][i]]>20:
            fssk=f[ld[0][i]]
            Pssk=np.abs(Pyy[ld[0][i]])
            fsk.append(fssk)
            Psk.append(Pssk)
            ssk += 1
    #print('{}'.format(np.round(fsk[:len(fsk)],decimals=3))) #標準出力にピーク周波数fskを小数点以下二桁まで出力する
    #print('{}'.format(np.round(Psk[:len(fsk)],decimals=4))) #標準出力にピーク強度Pskを小数点以下6桁まで出力する
    return  freq,Pyy,fsk,Psk,f

def draw_pic(freq,Pyy,fsk,Psk,f,sk,sig):
    matplotlib.rcParams.update({'font.size': 18, 'font.family': 'sans', 'text.usetex': False}) 
    fig = plt.figure(figsize=(12,12))  #(width,height)

    axes1 = fig.add_axes([0.1, 0.55, 0.8, 0.4]) # main axes
    axes2 = fig.add_axes([0.15, 0.7, 0.2, 0.2]) # insert axes
    axes3 = fig.add_axes([0.1, 0.1, 0.8, 0.35]) # main axes

    axes1.plot(t, sig)
    axes1.grid(True)
    axes1.set_xlim([0, 0.1])

    axes2.set_ylim(-1.2,1.2)
    axes2.set_xlim(0,sec)
    axes2.plot(t,sig)

    Pyy_abs=np.abs(Pyy)
    axes3.plot(f,Pyy_abs)
    axes3.axis([min(fsk)*0.9, max(fsk)*1.1, 0,max(Pyy_abs)*1.5])  #0.5, 2
    axes3.grid(True)
    axes3.set_xscale('log')
    axes3.set_ylim(0,max(Pyy_abs)*1.5)

    axes3.set_title('{}'.format(np.round(fsk[:len(fsk)],decimals=1))+'\n'+'{}'.format(np.round(Psk[:len(fsk)],decimals=4)),size=10)  #グラフのタイトルにピーク周波数とピーク強度を出力する
    axes3.plot(fsk[:len(fsk)],Psk[:len(fsk)],'ro')  #ピーク周波数、ピーク強度の位置に〇をつける
    # グラフにピークの周波数をテキストで表示
    for i in range(len(fsk)):
        axes3.annotate('{0:.1f}'.format(fsk[i]),  #np.round(fsk[i],decimals=2) でも可 '{0:.0f}(Hz)'.format(fsk[i])
                     xy=(fsk[i], Psk[i]),
                     xytext=(10, 20),
                     textcoords='offset points',
                     arrowprops=dict(arrowstyle="->",connectionstyle="arc3,rad=.2")
                    )

    plt.pause(1)
    wavfile=str(sec)+'_'+sa+str(sk)
    plt.savefig('./fft_sound/figure_'+wavfile+'.jpg')
    plt.close()
    #return Psk, fsk

#入力音声をFFTして描画する    
freq,Pyy,fsk,Psk,f=FFT(sig,fn,fr)
draw_pic(freq,Pyy,fsk,Psk,f,1,sig)
#マイク入力を出力
input = binwave
output = stream.write(input)

#FFTで得られた周波数Pskと振幅fsk
A=Psk/max(Psk)/len(fsk)
f=fsk
print(A)

#上記のAとｆを使ってサイン波でフォルマント合成
sigs =[]
create_wave(A, f, fr, sec)
sigs = sound_wave(f[0])

#フォルマント合成音声をFFTして描画する
freqs,Pyys,fsks,Psks,fss=FFT(sigs,fn,fr)
draw_pic(freqs,Pyys,fsks,Psks,fss,2,sigs)    

while True:
    #永続的にフォルマント合成音を発生する
    sig =[]
    create_wave(A, f, fr, sec)
    sig = sound_wave(f[0])
    
    
    