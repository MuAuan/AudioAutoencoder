# -*- coding:utf-8 -*-

import pyaudio
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import wave
import librosa
import wave
import struct

RATE=44100
CHUNK = 22050
p=pyaudio.PyAudio()
f0=261.626 #ド	C4	
f1=293.665 #レ	D4	
f2=329.628 #ミ	E4	
f3=349.228 #ファ	F	
f4=391.995 #ソ	G4	
f5=440.000 #ラ	A4	
f6=493.883 #シ	B4	
f7=523.251 #ド	C5
f_list=(f0,f1,f2,f3,f4,f5,f6,f7,f6,f5,f4,f3,f2,f1,f0,f0)

stream=p.open(format = pyaudio.paInt16,
        channels = 1,
        rate = RATE,
        frames_per_buffer = CHUNK,
        input = True,
        output = True) # inputとoutputを同時にTrueにする

w = wave.Wave_write("./doremi/merody.wav")
p = (1, 2, RATE, CHUNK, 'NONE', 'not compressed')
w.setparams(p)
    
for i in f_list:
    wavfile = './doremi/Doremi{}.wav'.format(i)
    print(wavfile)
    wr = wave.open(wavfile, "rb")
    input = wr.readframes(wr.getnframes())
    output = stream.write(input)
    w.writeframes(input)
