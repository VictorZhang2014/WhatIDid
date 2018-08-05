# https://github.com/Zackory/Keras-MNIST-GAN/blob/master/mnist_dcgan.py
    
import os 
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
import matplotlib.pyplot as plt
import math

from keras.models import Sequential, Model
from keras.layers import Input
from keras.layers.core import Reshape, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.optimizers import Adam
from keras.datasets import mnist
from keras import initializers
from keras import backend as K

K.set_image_dim_ordering('th')

# Tired of seeing the same results every time? Remove the line below.
np.random.seed(1000)

random_dim = 100

(x_train, _), (_, _) = mnist.load_data()
x_train = (x_train.astype(np.float32) - 127.5) / 127.5
x_train = x_train[:,np.newaxis,:,:]

# Optimizer
adam = Adam(lr=0.0002, beta_1=0.5)

# Generator
generator = Sequential()
generator.add(Dense(128*7*7, input_dim=random_dim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
generator.add(LeakyReLU(0.2))
generator.add(Reshape((128, 7, 7)))
generator.add(UpSampling2D(size=(2, 2)))
generator.add(Conv2D(64, kernel_size=(5, 5), padding='same'))
generator.add(LeakyReLU(0.2))
generator.add(UpSampling2D(size=(2, 2)))
generator.add(Conv2D(1, kernel_size=(5, 5), padding='same', activation='tanh'))
generator.compile(loss='binary_crossentropy', optimizer=adam)
print("------------------------------------- Generator Model Architecture ----------------------------------------")
generator.summary()

# Discriminator
discriminator = Sequential()
discriminator.add(Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same', input_shape=(1, 28, 28), kernel_initializer=initializers.RandomNormal(stddev=0.02)))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))
discriminator.add(Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding='same'))
discriminator.add(LeakyReLU(0.2))
discriminator.add(Dropout(0.3))
discriminator.add(Flatten())
discriminator.add(Dense(1, activation='sigmoid'))
discriminator.compile(loss='binary_crossentropy', optimizer=adam)
print("------------------------------------- Discriminator Model Architecture ----------------------------------------")
discriminator.summary()

# Combined network
discriminator.trainable = False

# The generator takes noise as input and generates imgs
gan_input = Input(shape=(random_dim,))
x = generator(gan_input)

# The discriminator take generated images as input and determines validity
gan_output = discriminator(x)

gan = Model(inputs=gan_input, outputs=gan_output)
gan.compile(loss='binary_crossentropy', optimizer=adam)
print("------------------------------------- GAN Model Architecture ----------------------------------------")
gan.summary()


d_losses = []
g_losses = []

def train(epochs=1, batch_size=128):
    batches = int(math.ceil(x_train.shape[0] / float(batch_size)))
    
    for e in range(epochs):
        for batch_i in range(batches):
            
            # get a random set of input real images
            real_image = x_train[np.random.randint(0, x_train.shape[0], batch_size)]
            
            # get a random set of input noise and generate fake MNIST images
            noise = np.random.normal(0, 1, (batch_size, random_dim))
            fake_image = generator.predict(noise)
            
            # combine two image matrixes
            X = np.concatenate([real_image, fake_image])
            
            # Labels for generated and real data
            y_dis = np.zeros(2 * batch_size)
            # One-sided label smoothing
            y_dis[:batch_size] = 0.9
            
            # train discriminator
            discriminator.trainable = True
            d_loss = discriminator.train_on_batch(X, y_dis)

            # Train generator
            noise = np.random.normal(0, 1, size=[batch_size, random_dim])
            y_gen = np.ones(batch_size)
            discriminator.trainable = False
            g_loss = gan.train_on_batch(noise, y_gen)
                    
            if batch_i % 10 == 0:
                cur_batch_progress = float(batch_i)/float(batches) * 100
                print("Epoch: {}/{}, current batch: {:.2f}%, d_loss: {:.5f}, g_loss: {:.5f}"
                      .format(e,epochs,cur_batch_progress,d_loss,g_loss))
                
        d_losses.append(d_loss)
        g_losses.append(g_loss)
            
        if e % 5 == 0:
            plot_generated_image(e)
            save_model(e)
    
    plot_losses(e)
            
          
def plot_generated_image(epoch, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, random_dim])
    generated_img = generator.predict(noise)
    generated_img = generated_img.reshape(examples, 28, 28)
    
    plt.figure(figsize=figsize)
    for i in range(generated_img.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_img[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('images/generated_img_epoch_{}.png'.format(epoch))
    
    
def save_model(epoch):
    generator.save('models/gan_generator_epoch_{}.h5'.format(epoch))
    discriminator.save('models/gan_discriminator_epoch_{}.h5'.format(epoch))        
    

def plot_losses(epoch):
    plt.figure(figsize=(10, 8))
    plt.plot(d_losses, label='Discriminative Loss')
    plt.plot(g_losses, label='Generative Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('images/gan_loss_epoch_{}.png'.format(epoch))

    
train(50, 128)
