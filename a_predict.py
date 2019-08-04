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
# Figureの初期化
#fig = plt.figure(figsize=(6, 6)) #...1
# Figure内にAxesを追加()
#ax1 = fig.add_subplot(311)
#ax2 = fig.add_subplot(312)
#ax3 = fig.add_subplot(313)
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

    #print(type(sin_wav),sin_wav,"=sin_wav")
    sin_wave = [int(float(x)* 32767.0 ) for x in sin_wav]    
    binwave = struct.pack("h" * len(sin_wave), *sin_wave)        
    #print(binwave)
    return binwave,sin_wave

# this is the size of our encoded representations
encoding_dim = 64  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
#input_img = Input(shape=(50*50,))
input_img = Input(shape=(64, 64, 1)) 


# "encoded" is the encoded representation of the input
def encoder(input_img):
    #encoded = Dense(encoding_dim, activation='relu')(input_img)
    """
    x = Dense(256, activation='relu')(input_img)
    x = Dense(128, activation='relu')(x)
    encoded = Dense(encoding_dim, activation='relu')(x)
    """
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    return encoded

def decoder(encoded):
    # "decoded" is the lossy reconstruction of the input
    #decoded = Dense(50*50, activation='sigmoid')(encoded)
    """
    x = Dense(encoding_dim, activation='relu')(encoded)
    x = Dense(128, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    decoded = Dense(50*50, activation='sigmoid')(x)
    """
    #x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    x = Conv2DTranspose(16, (3, 3), activation='relu', strides=2, padding='same')(encoded)
    x = Conv2DTranspose(32, (3, 3), activation='relu', strides=2, padding='same')(x)
    x = Conv2DTranspose(64, (3, 3), activation='relu', strides=2, padding='same')(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
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
decoder_layer = autoencoder.layers[0] #-1
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

decoder.summary()
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
        #y_ = sky
        #print(x_)
        x_train.append(x_)
x_train=np.array(x_train)    
x_train = x_train.astype('float32') # / 255.
print(x_train[0])
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_train0=x_train
x_train = np.reshape(x_train, (len(x_train), 64, 64, 1))

autoencoder.load_weights('./aiueo/sig_0730/boin_AE64_'+str(encoding_dim)+'.hdf5')
"""
autoencoder.fit(x_train, x_train,
                epochs=5000,
                batch_size=16,
                shuffle=True,
                validation_data=(x_train, x_train))

#autoencoder.save_weights('./aiueo/sig_0730/boin_AE'+sky+'.hdf5', True)
autoencoder.save_weights('./aiueo/sig_0730/boin_AE.hdf5', True)
"""
# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = encoder.predict(x_train)
#decoded_imgs = decoder.predict(encoded_imgs)
decoded_imgs = autoencoder.predict(x_train)
decoded_imgs = decoded_imgs.reshape((len(decoded_imgs), np.prod(decoded_imgs.shape[1:])))

for j in range(0,len(x_train),5):
    sk=j
    writecsv(path,sk,x_train0[j])
    bin_wave,sin_wave = readcsv_(path,sk)
    output =stream.write(bin_wave)    

for j in range(0,len(x_train),5):
    sk=j
    writecsv(path+'pr_',sk,decoded_imgs[j])
    bin_wave,sin_wave = readcsv_(path+'pr_',sk)
    output =stream.write(bin_wave)

# use Matplotlib (don't ask)
import matplotlib.pyplot as plt

n = 10  # how many digits we will display
plt.figure(figsize=(20, 20))
for i in range(n):
    # display original
    ax1 = plt.subplot(10, n, i + 1)
    plt.imshow(x_train[i].reshape(64, 64))
    plt.gray()
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)

    # display reconstruction
    ax2 = plt.subplot(10, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(64, 64))
    plt.gray()
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    
    ax3 = plt.subplot(10, n, i + 1 +2*n)
    plt.imshow(x_train[i+n].reshape(64, 64))
    plt.gray()
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)

    # display reconstruction
    ax4 = plt.subplot(10, n, i + 1 + 3*n)
    plt.imshow(decoded_imgs[i+n].reshape(64, 64))
    plt.gray()
    ax4.get_xaxis().set_visible(False)
    ax4.get_yaxis().set_visible(False)

    ax5 = plt.subplot(10, n, i + 1 + 4*n)
    plt.imshow(x_train[i+2*n].reshape(64, 64))
    plt.gray()
    ax5.get_xaxis().set_visible(False)
    ax5.get_yaxis().set_visible(False)

    # display reconstruction
    ax6 = plt.subplot(10, n, i + 1 + 5*n)
    plt.imshow(decoded_imgs[i+2*n].reshape(64, 64))
    plt.gray()
    ax6.get_xaxis().set_visible(False)
    ax6.get_yaxis().set_visible(False)

    ax7 = plt.subplot(10, n, i + 1 + 6*n)
    plt.imshow(x_train[i+3*n].reshape(64, 64))
    plt.gray()
    ax7.get_xaxis().set_visible(False)
    ax7.get_yaxis().set_visible(False)

    # display reconstruction
    ax8 = plt.subplot(10, n, i + 1 + 7*n)
    plt.imshow(decoded_imgs[i+3*n].reshape(64, 64))
    plt.gray()
    ax8.get_xaxis().set_visible(False)
    ax8.get_yaxis().set_visible(False)

    ax9 = plt.subplot(10, n, i + 1 +8*n)
    plt.imshow(x_train[i+4*n].reshape(64, 64))
    plt.gray()
    ax9.get_xaxis().set_visible(False)
    ax9.get_yaxis().set_visible(False)

    # display reconstruction
    ax10 = plt.subplot(10, n, i + 1 + 9*n)
    plt.imshow(decoded_imgs[i+4*n].reshape(64, 64))
    plt.gray()
    ax10.get_xaxis().set_visible(False)
    ax10.get_yaxis().set_visible(False)    
plt.pause(1)
plt.savefig('./aiueo/sig_0730/training_64_'+str(encoding_dim)+'.jpg')
plt.close()
