import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as nd
import skimage.transform as skt

def inverse_normalization(X):
    return X * 255.0

def plot_generated_batch(X_full, X_sketch, generator_model, epoch_num, dataset_name, batch_num):

    # Generate images
    X_gen = generator_model.predict(X_sketch)

    # X_sketch = inverse_normalization(X_sketch)
    # X_full = inverse_normalization(X_full)
    # X_gen = inverse_normalization(X_gen)
    target_size = (256,256)
    # limit to 8 images as output
    Xs = X_sketch[:8]
    Xg = X_gen[:8]
    Xr = X_full[:8]

    Xsr = skt.resize(Xs[0,:,:,64], target_size, mode='constant')
    Xgr = skt.resize(Xg[0,:,:,64], target_size, mode='constant')
    Xrr = skt.resize(Xg[0,:,:,64], target_size, mode='constant')

    # put |decoded, generated, original| images next to each other
    # X = np.concatenate((Xs[:,:,:,64*2], Xg[:,:,:,64*2], Xr[:,:,:,64*2]), axis=3)
    X = np.concatenate((Xsr, Xgr, Xrr), axis=2)

    # make one giant block of images
    X = np.concatenate(np.expand_dims(X, axis=0), axis=1)

    # save the giant n x 3 images
    plt.imsave('./pix2pix_out/progress_imgs/{}_epoch_{}_batch_{}.png'.format(dataset_name, epoch_num, batch_num), X[0], cmap='Greys_r')
