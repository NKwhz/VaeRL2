# -*- coding: utf-8 -*-
from keras.layers import Dense, multiply, add


def get_pho_rep(pho_0, pho_1, pho_dim):
    a_0 = Dense(units=pho_dim)(pho_0)
    a_1 = Dense(units=pho_dim)(pho_1)
    _x_0 = multiply([pho_0, a_0])
    _x_1 = multiply([pho_1, a_1])
    x = add([_x_0, _x_1])
    return x