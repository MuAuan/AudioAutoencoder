import cv2
import csv
import pyaudio
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.fftpack import fft, ifft
import wave
import struct

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D,Conv2DTranspose
from keras.models import Model
from keras import backend as K
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Reshape, Embedding,InputLayer


fr=16384 #44100
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

sk=0

path = './aiueo/sig_0730/boin_fig'

def writecsv(path,sk,output_data):
    #print("Output data path:  " + path)
    #print(output_data)
    with open(path+str(sk)+'.txt', 'w', newline='\n', encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator='\n') # 改行コード（\n）を指定しておく
        writer.writerow(output_data)

def readcsv_(path,sk):
    with open(path+str(sk)+'.txt', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',') #, quotechar=',')
        sin_wav=[]
        
        for row in spamreader:
            #print("row",row)
            sin_wav = (float(x) for x in row) #" ".join

    sin_wave = [int(float(x)* 32767.0 ) for x in sin_wav]    
    binwave = struct.pack("h" * len(sin_wave), *sin_wave)        
    #print(binwave)
    return binwave,sin_wave

# this is the size of our encoded representations
encoding_dim =1024 #256 #64  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(64*64,))

# "encoded" is the encoded representation of the input
def encoder(input_img):
    encoded = Dense(encoding_dim, activation='relu')(input_img)
    return encoded

def decoder(encoded):
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(64*64, activation='sigmoid')(encoded)
    return decoded

encoded=encoder(input_img)
decoded=decoder(encoded)
# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1] #-1
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

encoder.summary()
autoencoder.summary()

# listをCSVファイルから入力
def readcsv(path,sk):
    #print("Input data path:  " + path)
    with open(path+str(sk)+'.txt', 'r', newline='\n', encoding="utf-8") as f:
        reader = csv.reader(f, lineterminator='\n') # 改行コード（\n）を指定しておく
        Row=[]
        for row in reader:
            Row.append(row)
    return Row     

from keras.datasets import mnist
import numpy as np
x_train=[]
y_train=[]
list_=('a','e','i','o','u')
for sky in range(5):
    sky=list_[sky]
    for sk in range(100):
        x_ = readcsv("./aiueo/sig_0730/"+sky+"_64x64/boin_fig",sk)
        x_train.append(x_)
x_train=np.array(x_train)    
x_train = x_train.astype('float32')
print(x_train[0])
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_train0=x_train
#x_train = np.reshape(x_train, (len(x_train), 64, 64, 1))

#x_train = x_train.reshape(50,50,)
print(x_train.shape)
#autoencoder.load_weights('./aiueo/sig_0730/boin_AE'+'o'+'.hdf5')
#autoencoder.load_weights('./aiueo/sig_0730/boin_AE64_'+str(encoding_dim)+'.hdf5')

autoencoder.fit(x_train, x_train,
                epochs=5000,
                batch_size=16,
                shuffle=True,
                validation_data=(x_train, x_train))

autoencoder.save_weights('./aiueo/sig_0730/boin_AE64_'+str(encoding_dim)+'.hdf5', True)
#autoencoder.save_weights('./aiueo/sig_0730/boin_AE_dense.hdf5', True)

# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = encoder.predict(x_train)
#decoded_imgs = decoder.predict(encoded_imgs)
decoded_imgs = autoencoder.predict(x_train)
decoded_imgs = decoded_imgs.reshape((len(decoded_imgs), np.prod(decoded_imgs.shape[1:])))

for j in range(0,500,25):
    sk=j
    writecsv(path,sk,x_train0[j])
    bin_wave,sin_wave = readcsv_(path,sk)
    output =stream.write(bin_wave)    

for j in range(0,500,25):
    sk=j
    writecsv(path,sk,decoded_imgs[j])
    bin_wave,sin_wave = readcsv_(path,sk)
    output =stream.write(bin_wave)

# use Matplotlib (don't ask)
import matplotlib.pyplot as plt

n = 10  # how many digits we will display

ax_list=('ax1','ax2','ax3','ax4','ax5','ax6','ax7','ax8','ax9','ax10')
plt.figure(figsize=(20, 20))
for i in range(n):
    for j in range(5):
        ax=ax_list[2*j]
        # display original
        ax = plt.subplot(10, n, i + 1+ 2*j*n)
        plt.imshow(x_train[i+j*n].reshape(64, 64))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax=ax_list[2*j+1]
        # display reconstruction
        ax = plt.subplot(10, n, i + 1 + (2*j+1)*n)
        plt.imshow(decoded_imgs[i+j*n].reshape(64, 64))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)    
plt.pause(1)
plt.savefig('./aiueo/sig_0730/training_64_dense'+str(encoding_dim)+'.jpg')
plt.close()
