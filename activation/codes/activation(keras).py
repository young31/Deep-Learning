import keras.backend as K
import numpy as np

def swish(x):
    return x*K.sigmoid(x)


def gelu(x):
    return 0.5*x*(1+K.tanh(np.sqrt(2/np.pi)*(x+0.044715*K.pow(x, 3))))


def mish(x):
    return x*K.tanh(K.softplus(x))