import numpy as np
import os

from keras.optimizers import Adam, SGD
from networks.generator import UNETGenerator
from networks.discriminator import PatchGanDiscriminator
from networks.DCGAN import DCGAN
from utils import patch_utils
from utils import logger
import time
from utils.datagen import TwoImageIterator, MyDict

from keras.utils import generic_utils as keras_generic_utils

# WORKING_DIR = os.path.dirname(os.path.realpath(__file__))
WORKING_DIR = '/host/hamlet/local_raid/data/'
DATASET = 'brainhack'

import keras.backend as K
K.set_image_data_format('channels_first')

# ---------------------------------------------
# HYPER PARAMS
# ---------------------------------------------
# width, height of images to work with. Assumes images are square
im_width = im_height = im_depth = 256

# inpu/output channels in image
input_channels = 1
output_channels = 1

# image dims
input_img_dim = (input_channels, im_width, im_height, im_depth)
output_img_dim = (output_channels, im_width, im_height, im_depth)

# We're using PatchGAN setup, so we need the num of non-overlaping patches
# this is how big we'll make the patches for the discriminator
# for example. We can break up a 256x256 image in 16 patches of 64x64 each
sub_patch_dim = (256,256,256)
nb_patch_patches, patch_gan_dim = patch_utils.num_patches(output_img_dim=output_img_dim, sub_patch_dim=sub_patch_dim)


# ---------------------------------------------
# TRAINING ROUTINE
# ---------------------------------------------

# ----------------------
# GENERATOR
# Our generator is an AutoEncoder with U-NET skip connections
# ----------------------
generator_nn = UNETGenerator(input_img_dim=input_img_dim, num_output_channels=output_channels)
generator_nn.summary()

# ----------------------
# PATCH GAN DISCRIMINATOR
# the patch gan averages loss across sub patches of the image
# it's fancier than the standard gan but produces sharper results
# ----------------------
discriminator_nn = PatchGanDiscriminator(output_img_dim=output_img_dim,
        patch_dim=patch_gan_dim, nb_patches=nb_patch_patches)
discriminator_nn.summary()

# disable training while we put it through the GAN
discriminator_nn.trainable = False

# ------------------------
# Define Optimizers
# opt_discriminator = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
opt_discriminator = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
opt_dcgan = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

# -------------------------
# compile generator
generator_nn.compile(loss='mae', optimizer=opt_discriminator)

# ----------------------
# MAKE FULL DCGAN
# ----------------------
dc_gan_nn = DCGAN(generator_model=generator_nn,
                  discriminator_model=discriminator_nn,
                  input_img_dim=input_img_dim,
                  patch_dim=sub_patch_dim)

dc_gan_nn.summary()

# ---------------------
# Compile DCGAN
# we use a combination of mae and bin_crossentropy
loss = ['mae', 'binary_crossentropy']
loss_weights = [1E2, 1]
dc_gan_nn.compile(loss=loss, loss_weights=loss_weights, optimizer=opt_dcgan)

# ---------------------
# ENABLE DISCRIMINATOR AND COMPILE
discriminator_nn.trainable = True
discriminator_nn.compile(loss='binary_crossentropy', optimizer=opt_discriminator)


params = MyDict({
    # Model
    'nfd': 16,  # Number of filters of the first layer of the discriminator
    'nfatob': 32,  # Number of filters of the first layer of the AtoB model
    'alpha': 100,  # The weight of the reconstruction loss of the atob model
    # Train
    'epochs': 100,  # Number of epochs to train the model
    'batch_size': 1,  # The batch size
    'samples_per_batch': 20,  # The number of samples to train each model on each iteration
    'save_every': 10,  # Save results every 'save_every' epochs on the log folder
    'lr': 2e-4,  # The learning rate to train the models
    'beta_1': 0.5,  # The beta_1 value of the Adam optimizer
    'continue_train': False,  # If it should continue the training from the last checkpoint
    # File system
    'log_dir': 'log',  # Directory to log
    'expt_name': 'test1',  # The name of the experiment. Saves the logs into a folder with this name
    'train_dir': 'training',  # Directory inside base_dir that contains training data
    'val_dir': 'validation',  # Directory inside base_dir that contains validation data
    'train_samples': -1,  # The number of training samples. Set -1 to be the same as training examples
    'val_samples': -1,  # The number of validation samples. Set -1 to be the same as validation examples
    # Image
    'target_size': 256,  # The size of the images loaded by the iterator. DOES NOT CHANGE THE MODELS

})

# ------------------------
# RUN ACTUAL TRAINING
batch_size = params.batch_size
# data_path = WORKING_DIR + '/data/' + DATASET
data_path = WORKING_DIR + DATASET + '/HCPdataset/'
nb_epoch = 50
n_images_per_epoch = 100


print('Training starting...')
for epoch in range(0, nb_epoch):

    print('Epoch {}'.format(epoch))
    batch_counter = 1
    start = time.time()
    progbar = keras_generic_utils.Progbar(n_images_per_epoch)

    # init the datasources again for each epoch
    # tng_gen = facades_generator(data_dir_name=data_path, data_type='training', im_width=im_width, batch_size=batch_size)
    # val_gen = facades_generator(data_dir_name=data_path, data_type='validation', im_width=im_width, batch_size=batch_size)
    ts = params.target_size
    train_dir = os.path.join(data_path, 'training')
    tng_gen = TwoImageIterator(train_dir, batch_size=params.batch_size, target_size=(ts,ts,ts))
    # val_dir = os.path.join(data_path, 'validation')
    # val_gen = TwoImageIterator(val_dir, batch_size=params.batch_size, target_size=(ts,ts,ts))

    # go through 1... n_images_per_epoch (which will go through all buckets as well
    for mini_batch_i in range(0, n_images_per_epoch, params.batch_size):

        # load a batch of decoded and original images
        # both for training and validation
        X_train_decoded_imgs, X_train_original_imgs = next(tng_gen)
        # X_val_decoded_imgs, X_val_original_imgs = next(val_gen)

        # generate a batch of data and feed to the discriminator
        # some images that come out of here are real and some are fake
        # X is image patches for each image in the batch
        # Y is a 1x2 vector for each image. (means fake or not)
        X_discriminator, y_discriminator = patch_utils.get_disc_batch(X_train_original_imgs,
                                                          X_train_decoded_imgs,
                                                          generator_nn,
                                                          batch_counter,
                                                          patch_dim=sub_patch_dim)

        # Update the discriminator
        # print('calculating discriminator loss')
        disc_loss = discriminator_nn.train_on_batch(X_discriminator, y_discriminator)

        # create a batch to feed the generator
        X_gen_target, X_gen = next(patch_utils.gen_batch(X_train_original_imgs, X_train_decoded_imgs, batch_size))
        print('X_gen')
        print X_gen.shape
        print X_gen_target.shape
        y_gen = np.zeros((X_gen.shape[0], 2), dtype=np.uint8)
        y_gen[:, 1] = 1
        print y_gen.shape

        # Freeze the discriminator
        discriminator_nn.trainable = False

        # training GAN
        # print('calculating GAN loss...')
        gen_loss = dc_gan_nn.train_on_batch(X_gen[:3], [X_gen_target[:3], y_gen])

        # Unfreeze the discriminator
        discriminator_nn.trainable = True

        # counts batches we've ran through for generating fake vs real images
        batch_counter += 1

        # print losses
        D_log_loss = disc_loss
        gen_total_loss = gen_loss[0].tolist()
        gen_total_loss = min(gen_total_loss, 1000000)
        gen_mae = gen_loss[1].tolist()
        gen_mae = min(gen_mae, 1000000)
        gen_log_loss = gen_loss[2].tolist()
        gen_log_loss = min(gen_log_loss, 1000000)

        progbar.add(batch_size, values=[("Dis logloss", D_log_loss),
                                        ("Gen total", gen_total_loss),
                                        ("Gen L1 (mae)", gen_mae),
                                        ("Gen logloss", gen_log_loss)])

        # ---------------------------
        # Save images for visualization every 2nd batch
        if batch_counter % 2 == 0:

            # print images for training data progress
            logger.plot_generated_batch(X_train_original_imgs, X_train_decoded_imgs, generator_nn, epoch, 'tng', mini_batch_i)

            # print images for validation data
            # X_full_val_batch, X_sketch_val_batch = next(patch_utils.gen_batch(X_val_original_imgs, X_val_decoded_imgs, batch_size))
            # logger.plot_generated_batch(X_full_val_batch, X_sketch_val_batch, generator_nn, epoch, 'val', mini_batch_i)

    # -----------------------
    # log epoch
    print("")
    print('Epoch %s/%s, Time: %s' % (epoch + 1, nb_epoch, time.time() - start))

    # ------------------------------
    # save weights on every 2nd epoch
    if epoch % 2 == 0:
        gen_weights_path = os.path.join('./pix2pix_out/weights/gen_weights_epoch_%s.h5' % (epoch))
        generator_nn.save_weights(gen_weights_path, overwrite=True)

        disc_weights_path = os.path.join('./pix2pix_out/weights/disc_weights_epoch_%s.h5' % (epoch))
        discriminator_nn.save_weights(disc_weights_path, overwrite=True)

        DCGAN_weights_path = os.path.join('./pix2pix_out/weights/DCGAN_weights_epoch_%s.h5' % (epoch))
        dc_gan_nn.save_weights(DCGAN_weights_path, overwrite=True)
