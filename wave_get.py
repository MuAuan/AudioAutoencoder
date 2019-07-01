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
N=10
CHUNK=1024*N
p=pyaudio.PyAudio()

stream=p.open(format = pyaudio.paInt16,
        channels = 1,
        rate = RATE,
        frames_per_buffer = CHUNK,
        input = True,
        output = True) # inputとoutputを同時にTrueにする

s=0
while stream.is_active():
    wavfile = 'input'+str(s)+'.wav'
    wr = wave.open(wavfile, "rb")
    input = wr.readframes(wr.getnframes())
    output = stream.write(input)
    if s>10:
        break
    s+=1