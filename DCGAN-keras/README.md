# pix2pix-keras
Pix2pix implementation in keras.    

Original paper: [Image-to-Image Translation with Conditional Adversarial Networks (pix2pix)](https://arxiv.org/pdf/1611.07004.pdf)    
Paper Authors and Researchers: Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros    

### To run    
```bash

# setup conda
# install requirements
conda env create -f environment.yml

# start training
KERAS_BACKEND=theano THEANO_FLAGS='floatX=float32,device=cuda0' python main.py # for nvidia GPU
KERAS_BACKEND=theano THEANO_FLAGS='floatX=float32,device=cpu' python main.py # for CPU

```    
