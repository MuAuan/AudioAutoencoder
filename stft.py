from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import wave
import matplotlib
from scipy.fftpack import fft, ifft

wavfile = 'hirakegoma'
#wavfile = 'ohayo'
#wavfile = 'output400_450'
#wavfile='akasatana'
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
#wavfile = 'output400_350'
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

t1,sig,fn,fr=wave_input(wavfile)

"""
carrier = amp * (np.sin(2*np.pi*(3.333e2)*t)+np.sin(2*np.pi*(8.80e2)*t))/2
sig= sig  + carrier
"""
sig=1*sig
amp=np.max(sig)
#サイン波を表示
matplotlib.rcParams.update({'font.size': 18, 'font.family': 'sans', 'text.usetex': False}) 
fig = plt.figure(figsize=(12,12))  #(width,height)

axes1 = fig.add_axes([0.1, 0.6, 0.7, 0.4]) # main axes
axes2 = fig.add_axes([0.15, 0.75, 0.2, 0.2]) # insert axes
axes3 = fig.add_axes([0.1, 0.1, 0.7, 0.4]) # main axes
axes4 = fig.add_axes([0.81, 0.1, 0.03, 0.4])
axes5 = fig.add_axes([0.15, 0.25, 0.2, 0.2]) # insert axes
axes6 = fig.add_axes([0.91, 0.1, 0.03, 0.4])
axes7 = fig.add_axes([0.55, 0.25, 0.2, 0.2]) # insert axes

axes1.plot(t1, sig)
axes1.grid(True)
axes2.set_xlim([1, 1.1])
axes2.set_ylim(-amp*1.1,amp*1.1)
axes1.set_xlim(0,5)
axes2.plot(t1,sig)

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
sig1=sig[int(2.5*fr):int(2.8*fr)]
fn1=int(fn*0.3/5)
freq,Pyy,fsk,Psk,f=FFT(sig1,fn1,fr)

Pyy_abs=np.abs(Pyy)
axes7.plot(f,Pyy_abs)
axes7.axis([min(fsk)*0.9, max(fsk)*1.1, 0,max(Pyy_abs)*1.5])  #0.5, 2
axes7.grid(True)
axes7.set_xscale('log')
axes7.set_ylim(0,max(Pyy_abs)*1.5)

axes7.set_title('{}'.format(np.round(fsk[:len(fsk)],decimals=1))+'\n'+'{}'.format(np.round(Psk[:len(fsk)],decimals=4)),size=10)  #グラフのタイトルにピーク周波数とピーク強度を出力する
axes7.plot(fsk[:len(fsk)],Psk[:len(fsk)],'ro')  #ピーク周波数、ピーク強度の位置に〇をつける
# グラフにピークの周波数をテキストで表示
for i in range(len(fsk)):
    axes7.annotate('{0:.1f}'.format(fsk[i]),  #np.round(fsk[i],decimals=2) でも可 '{0:.0f}(Hz)'.format(fsk[i])
                 xy=(fsk[i], Psk[i]),
                 xytext=(10, 20),
                 textcoords='offset points',
                 arrowprops=dict(arrowstyle="->",connectionstyle="arc3,rad=.2")
                )

nperseg=200
f, t, Zxx = signal.stft(sig, fr, nperseg=nperseg)
rr=np.abs(Zxx)
max_rr=np.max(rr)
img=axes3.pcolormesh(t, f, rr, vmin=0, vmax=max_rr,cmap="hsv")
axes3.set_yscale('log')
axes3.set_ylim([20, 20000])
axes3.set_xlim([0, 5])
axes3.set_title('STFT Magnitude')
axes3.set_ylabel('Frequency [Hz]')
axes3.set_xlabel('Time [sec]')

img1 = axes5.pcolormesh(t, f, rr, vmin=0, vmax=max_rr,cmap="jet")
axes5.set_yscale('log')
axes5.set_ylim([20, 20000])
axes5.set_xlim([1, 1.1])

fig.colorbar(img, cax=axes4)
fig.colorbar(img1, cax=axes6)

plt.pause(1)
plt.savefig("./stft/stft_fig_"+wavfile+"_"+str(nperseg)+".jpg", dpi=200)
plt.close()
"""
matplotlib.rcParams.update({'font.size': 18, 'font.family': 'sans', 'text.usetex': False}) 
plt.rcParams['figure.figsize'] = (12, 12)
fig = plt.figure()
ax1 = fig.add_axes([0.1, 0.6, 0.7, 0.4])
ax2 = fig.add_axes([0.1, 0.1, 0.7, 0.4], sharex=ax1)
ax3 = fig.add_axes([0.81, 0.1, 0.03, 0.4])
ax4 = fig.add_axes([0.15, 0.75, 0.2, 0.2]) # insert axes
ax5 = fig.add_axes([0.15, 0.25, 0.2, 0.2]) # insert axes
ax6 = fig.add_axes([0.91, 0.1, 0.03, 0.4])

ax1.plot(t1, sig, 'k')

img = ax2.imshow(np.flipud(rr),extent=[min(t1),max(t1), 20, 20000],  aspect='auto', cmap='hsv') #extent=[0, 5,20, 20000],
#img=ax2.pcolormesh(t, f, rr, vmin=0, vmax=max_rr,cmap="hsv")
twin_ax = ax2
twin_ax.set_yscale('log')
#twin_ax.set_xlim(0, 5) #5 0.1
twin_ax.set_ylim(20, 20000)
ax2.set_title('STFT Magnitude')
ax2.set_ylabel('Frequency [Hz]')
ax2.set_xlabel('Time [sec]')
ax2.tick_params(which='both', labelleft=False, left=False)
twin_ax.tick_params(which='both', labelleft=True, left=True, labelright=False)
fig.colorbar(img, cax=ax3)

ax4.set_xlim([1, 1.1])
ax4.set_ylim(-amp*1.1,amp*1.1)
ax4.plot(t1,sig)

img1 = ax5.pcolormesh(t, f, rr, vmin=0, vmax=max_rr,cmap="jet")
ax5.set_yscale('log')
ax5.set_ylim([20, 20000])
ax5.set_xlim([1, 1.1])
fig.colorbar(img1, cax=ax6)

fig.savefig("./stft/stft_fig2_"+wavfile+"_"+str(nperseg)+".jpg", dpi=200)
plt.pause(1)
"""