# this code can turn horse to zebra at around 20000 iteration
# when batch_size = 5 save_iter = 200 niter = 100000 lmbd = 10 pic_dir = args.pic_dir idloss = 0.0 lr = 0.00005
# can turn horse to zebra around 14000 iteration
# when batch_size = 1 save_iter = 200 niter = 100000 lmbd = 5 pic_dir = args.pic_dir idloss = 0.0 lr = 0.0002

experiment = 'brain2d_test2'

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--cuda", type=str, help="cuda", default='0')
parser.add_argument("--pic_dir", type=str, help="picture dir", default='./quickshots/'+experiment)
args = parser.parse_args()
print args

import os
os.environ['THEANO_FLAGS']=os.environ.get('THEANO_FLAGS','')+',gpuarray.preallocate=0.50,device=cuda{}'.format(args.cuda)
os.environ['CUDA_VISIBLE_DEVICES']='{}'.format(args.cuda)

if not os.path.exists(args.pic_dir):
    os.mkdir(args.pic_dir)

from CycleGAN.utils.data_utils import ImageGenerator
from CycleGAN.models import CycleGAN
from CycleGAN.utils import Option

import keras.backend as K
K.set_image_data_format('channels_first')
print K.image_data_format()

data_dir    = '/host/silius/local_raid/ravnoor/07_lab/brainhack/CycleGAN-keras/datasets/'
convert     = 'brain2D'
# A = T1, B= T2

if __name__ == '__main__':
    opt = Option()
    opt.batch_size = 5
    opt.save_iter = 250
    opt.niter = 100000
    opt.lmbd = 10
    opt.pic_dir = args.pic_dir
    opt.idloss = 0.0
    opt.lr = 00005
    opt.d_iter = 1

    opt.shapeA = (3,240,240)
    opt.shapeB = (3,240,240)
    # opt.shapeA = (4,233,197)
    # opt.shapeB = (4,233,197)
    opt.resize = (240,240)
    opt.crop = None
    # opt.crop = (128,128)

    opt.__dict__.update(args.__dict__)
    opt.summary()


    cycleGAN = CycleGAN(opt)

    IG_A = ImageGenerator(root=data_dir+convert+'/trainA',
                resize=opt.resize, crop=opt.crop)
    IG_B = ImageGenerator(root=data_dir+convert+'/trainB',
                resize=opt.resize, crop=opt.crop)

    cycleGAN.fit(IG_A, IG_B)
