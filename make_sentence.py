# -*- coding:utf-8 -*-

import pyaudio
import numpy as np
import wave
import struct

RATE=44100
CHUNK = 22050
p=pyaudio.PyAudio()

#f_list=['a','i','si','te','ru','u','wa','n','sa','n']
#sentence='あ い し て る う わ ん さ ん ま る'
file1 = open("sentence2.txt","r",encoding="utf-8") 
sentence=file1.read()
print(sentence)
f_list=sentence.split(" ")
print(f_list)
#f_list=['あ','い','し','て','る','う','わ','ん','さ','ん','ま','る']
stream=p.open(format = pyaudio.paInt16,
        channels = 1,
        rate = int(RATE*1.15),
        frames_per_buffer = CHUNK,
        input = True,
        output = True) # inputとoutputを同時にTrueにする

w = wave.Wave_write("./fft_sound/Success_a-n_05/a-n-iine/aiueo/ohayo005_sin.wav")
p = (1, 2, RATE, CHUNK, 'NONE', 'not compressed')
w.setparams(p)

def kana2a(i):
    j='n'
    if i=='あ':
        j='a'
    elif i=='い':
        j='i'
    elif i=='し':
        j='si'
    elif i=='て':
        j='te'
    elif i=='る':
        j='ru'
    elif i=='う':
        j='u'
    elif i=='わ':
        j='wa'        
    elif i=='ん':
        j='n'        
    elif i=='さ':
        j='sa'  
    elif i=='ま':
        j='ma'          
    return j

#C:\Users\user\Onsei_AE\fft_sound\Success_a-n_05\a-n-iine\aiueo
for i in f_list:
    i=kana2a(i)
    wavfile = './fft_sound/Success_a-n_05/a-n-iine/aiueo/'+i+'.wav'
    print(wavfile)
    wr = wave.open(wavfile, "rb")
    input = wr.readframes(wr.getnframes())
    output = stream.write(input)
    w.writeframes(input)
