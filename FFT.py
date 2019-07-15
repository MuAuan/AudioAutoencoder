from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import wave
from scipy.fftpack import fft, ifft
import matplotlib

#wavfile = 'hirakegoma' #.wav'
#wavfile = 'akasatana'
#wavfile = 'ohayo.wav'
#wavfile = 'output400_450'
#wavfile = 'output400_405'
#wavfile = 'output400_410'
#wavfile = 'output400_420'
#wavfile = 'output400_402'
#wavfile = 'output400_401'
#wavfile = 'output400_400.5'
#wavfile = 'output400_400.2'
#wavfile = 'output400_400.1'
#wavfile = 'output400_500'
#wavfile = 'output400_1'
#wavfile = 'output400_10'
#wavfile = 'output400_5'
#wavfile = 'output400_1600'
#wavfile = 'output400_0.5'
#wavfile = 'output400_2'
#wavfile = 'output400_600'
#wavfile = 'output400_800'
#wavfile = 'output400_1000'
#wavfile = 'output400_1200'
#wavfile = 'output400_1400'
#wavfile = 'output400_50'
wavfile = 'output400_350'
#wavfile = 'output400_300'
#wavfile = 'output400_200'
#wavfile = 'output400_100'
#wavfile = 'output400'
#wavfile = 'output400_0.1'
#wavfile = 'output400_0.01'

def wave_input(wavfile):
    wr = wave.open(wavfile+'.wav', "rb")
    ch = wr.getnchannels()
    width = wr.getsampwidth()
    fr = wr.getframerate() #sampling freq ; RATE
    fn = wr.getnframes()  #sampling No. of frames; CHUNK
    fs = fn / fr  #sampling time
    origin = wr.readframes(fn)
    data = origin  #[:fn]
    print("fs",fs)
    print("fr",fr)
    # ステレオ前提 > monoral
    sig = np.frombuffer(data, dtype="int16")  /32768.0
    t1 = np.linspace(0,fs, fn) #, endpoint=False)
    return t1,sig,fn,fr 

t,sig,fn,fr=wave_input(wavfile)
"""
amp=np.max(sig)
carrier = amp * (np.sin(2*np.pi*(3.333e2)*t)+np.sin(2*np.pi*(8.80e2)*t))/2
sig= sig  + carrier
"""
#サイン波を表示
matplotlib.rcParams.update({'font.size': 18, 'font.family': 'sans', 'text.usetex': False}) 
fig = plt.figure(figsize=(12,12))  #(width,height)

axes1 = fig.add_axes([0.1, 0.55, 0.8, 0.4]) # main axes
axes2 = fig.add_axes([0.15, 0.7, 0.2, 0.2]) # insert axes
axes3 = fig.add_axes([0.1, 0.1, 0.8, 0.35]) # main axes

axes1.plot(t, sig)
axes1.grid(True)
axes1.set_xlim([0, 0.1])
#plt.savefig("./fft/sig_fig_"+wavfile+".jpg", dpi=200)
#plt.pause(1)
#plt.close()

axes2.set_ylim(-1.2,1.2)
axes2.set_xlim(0,5)
axes2.plot(t,sig)
#plt.pause(1)
#plt.savefig("./fft/sig_fig_"+wavfile+".jpg", dpi=200)

def FFT(sig,fn,fr):
    freq =fft(sig,fn)
    Pyy = np.sqrt(freq*freq.conj())/fn
    f = np.arange(0,fr,fr/fn)
    ld = signal.argrelmax(Pyy, order=1) #相対強度の最大な番号をorder=10で求める
    ssk=0
    fsk=[]
    Psk=[]
    maxPyy=max(np.abs(Pyy))
    for i in range(len(ld[0])):  #ピークの中で以下の条件に合うピークの周波数fと強度Pyyを求める
        if np.abs(Pyy[ld[0][i]])>0.25*maxPyy and f[ld[0][i]]<20000: # and f[ld[0][i]]>20:
            fssk=f[ld[0][i]]
            Pssk=np.abs(Pyy[ld[0][i]])
            fsk.append(fssk)
            Psk.append(Pssk)
            ssk += 1
    print('{}'.format(np.round(fsk[:len(fsk)],decimals=3))) #標準出力にピーク周波数fskを小数点以下二桁まで出力する
    print('{}'.format(np.round(Psk[:len(fsk)],decimals=4))) #標準出力にピーク強度Pskを小数点以下6桁まで出力する
    return  freq,Pyy,fsk,Psk,f

freq,Pyy,fsk,Psk,f=FFT(sig,fn,fr)

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
plt.savefig('./fft/figure_'+wavfile+'.jpg')

