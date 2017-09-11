# -*- coding: utf-8 -*-
import GetResult as GR
import numpy as np
from keras.layers import Input, Dense, Lambda, multiply, Layer, add, concatenate
from keras.models import Model
from keras import backend as K
from keras import objectives
from pho_generate import get_pho_rep

# 小数据
train_doc = 'x_train/doc_20000.npy'
train_pho0 = 'x_train/pho_0_20000.npy'
train_pho1 = 'x_train/pho_1_20000.npy'
train_y = 'x_train/y_20000.npy'

test_num = 1000
test_doc = 'x_test/doc_1000.npy'
test_pho0 = 'x_test/pho_0_1000.npy'
test_pho1 = 'x_test/pho_1_1000.npy'

# 大数据
# train_doc = 'x_train/doc.npy'
# train_pho0 = 'x_train/pho_0.npy'
# train_pho1 = 'x_train/pho_1.npy'
# train_y = 'x_train/y.npy'
#
# test_num = 10000
# test_doc = 'x_test/doc.npy'
# test_pho0 = 'x_test/pho_0.npy'
# test_pho1 = 'x_test/pho_1.npy'

pho_dim = 125
doc_dim = 125
con_dim = 250

activ = 'tanh'
optim = 'adagrad'


def slice(x, start, end):
    return x[:, start:end]


class CustomVariationalLayer(Layer):
    def __init__(self, alpha, beta, **kwargs):
        self.is_placeholder = True
        self.alpha = alpha
        self.beta = beta
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def loss(self, x, y):
        return K.mean(self.alpha * x + self.beta * y)

    def call(self, inputs):
        x = inputs[0]
        y = inputs[1]
        loss = self.loss(x, y)
        self.add_loss(loss, inputs=inputs)
        # We won't actually use the output.
        return x


def rhyme2vec(alpha=0, beta=0, use_bias=False, latent_dim=0, epochs=5, batch_size=711240):
    # recall@k
    rec_k_list = [1, 5, 30, 150]
    # the input dimension
    input_shape = pho_dim

    x_train0 = np.load(train_pho0)
    x_train1 = np.load(train_pho1)
    y_train = np.load(train_y)  # label file

    # load testing dataset
    x_test0 = np.load(test_pho0)  # feature file
    x_test1 = np.load(test_pho1)
    print('====load dataset done====' + '\n')

    x_0 = Input(shape=(input_shape,))
    x_1 = Input(shape=(input_shape,))
    x = get_pho_rep(x_0, x_1, pho_dim)

    sig = Dense(1, activation='sigmoid', use_bias=use_bias)(x)

    rhyme = Model(outputs=sig, inputs=[x_0, x_1])
    print('=======Model Information=======' + '\n')
    rhyme.summary()
    rhyme.compile(optimizer=optim, loss='binary_crossentropy')
    rhyme.fit([x_train0, x_train1], y_train,
              shuffle=False,
              epochs=epochs,
              batch_size=batch_size
              )

    rank = rhyme.predict([x_test0, x_test1])

    rank = rank.reshape(test_num, 300)
    rank_list = GR.get_rank_matrix(rank)
    result = GR.get_result_by_ranks(rank_list, rec_k_list)

    print('=======Rhyme2vec Result=======' + '\n')
    return result, rhyme


def doc2vec(alpha=0, beta=0, use_bias=False, latent_dim=0, epochs=5, batch_size=711240,
            test_doc_fname=test_doc):
    # recall@k
    rec_k_list = [1, 5, 30, 150]
    # the input dimension
    input_shape = doc_dim

    x_train = np.load(train_doc)  # feature file
    y_train = np.load(train_y)  # label file

    x_test = np.load(test_doc_fname)  # feature file

    # Model
    x = Input(shape=(input_shape,))
    sig = Dense(1, activation='sigmoid', use_bias=use_bias)(x)

    doc = Model(outputs=sig, inputs=x)
    print('=======Model Information=======' + '\n')
    doc.summary()
    doc.compile(optimizer=optim, loss='binary_crossentropy')

    doc.fit(x_train, y_train,
            shuffle=False,
            epochs=epochs,
            batch_size=batch_size
            )

    rank = doc.predict(x_test)

    rank = rank.reshape(test_num, 300)
    rank_list = GR.get_rank_matrix(rank)
    result = GR.get_result_by_ranks(rank_list, rec_k_list)
    print('=======Doc2vec Result=======' + '\n')
    return result, doc


def con(alpha=0, beta=0, use_bias=True, latent_dim=0, epochs=5, batch_size=711240):
    # recall@k
    rec_k_list = [1, 5, 30, 150]

    # load training dataset
    x_train0 = np.load(train_pho0)
    x_train1 = np.load(train_pho1)
    x_train_doc = np.load(train_doc)
    y_train = np.load(train_y)  # label file

    # load testing dataset
    x_test_doc = np.load(test_doc)
    x_test0 = np.load(test_pho0)  # feature file
    x_test1 = np.load(test_pho1)

    # Model
    x_doc = Input(shape=(doc_dim,))
    x_0 = Input(shape=(pho_dim,))
    x_1 = Input(shape=(pho_dim,))
    x_pho = get_pho_rep(x_0, x_1, pho_dim)

    encoder = concatenate([x_doc, x_pho])
    sig = Dense(1, activation='sigmoid', use_bias=use_bias)(encoder)

    con = Model(outputs=sig, inputs=[x_doc, x_0, x_1])
    print('=======Model Information=======' + '\n')
    con.summary()
    con.compile(optimizer=optim, loss='binary_crossentropy')

    con.fit([x_train_doc, x_train0, x_train1], y_train,
            shuffle=False,
            epochs=epochs,
            batch_size=batch_size
            )

    rank = con.predict([x_test_doc, x_test0, x_test1])
    rank = rank.reshape(test_num, 300)
    rank_list = GR.get_rank_matrix(rank)
    result = GR.get_result_by_ranks(rank_list, rec_k_list)

    print('=======Concatenate Result=======' + '\n')
    return result, con


def conAE(alpha, beta, activation=activ, use_bias=True, latent_dim=100, epochs=5, batch_size=711240):
    # recall@k
    rec_k_list = [1, 5, 30, 150]

    # load training dataset
    x_train0 = np.load(train_pho0)
    x_train1 = np.load(train_pho1)
    x_train_doc = np.load(train_doc)
    y_train = np.load(train_y)  # label file

    # load testing dataset
    x_test0 = np.load(test_pho0)  # feature file
    x_test1 = np.load(test_pho1)
    x_test_doc = np.load(test_doc)

    # AE Model
    x_doc = Input(shape=(doc_dim,))
    x_0 = Input(shape=(pho_dim,))
    x_1 = Input(shape=(pho_dim,))
    x_pho = get_pho_rep(x_0, x_1, pho_dim)
    encoder = concatenate([x_doc, x_pho])

    answer = Dense(units=latent_dim, activation=activation, use_bias=use_bias)(encoder)

    decoder = Dense(units=doc_dim + pho_dim, activation=activation, use_bias=use_bias)(answer)

    _x_doc = Lambda(slice, arguments={'start': 0, 'end': 125})(decoder)
    _x_pho = Lambda(slice, arguments={'start': 125, 'end': 250})(decoder)

    y = Input(shape=(1,))
    sig = Dense(1, activation='sigmoid', use_bias=use_bias)(answer)

    def bi_loss(args):
        x, y = args
        loss = objectives.binary_crossentropy(x, y)
        return loss

    def ae_loss(args):
        xd, _xd, xp, _xp = args
        return objectives.binary_crossentropy(xd, _xd) \
               + objectives.binary_crossentropy(xp, _xp)

    # Label loss
    label_loss = Lambda(bi_loss)([y, sig])

    # AE loss
    sae_loss = Lambda(ae_loss)([x_doc, _x_doc, x_pho, _x_pho])

    # Custom loss layer
    L = CustomVariationalLayer(alpha=alpha, beta=beta)([label_loss, sae_loss])

    AE = Model(outputs=L, inputs=[x_doc, x_0, x_1, y])
    print('=======Model Information=======' + '\n')
    AE.summary()
    AE.compile(optimizer=optim, loss=None)

    AE.fit([x_train_doc, x_train0, x_train1, y_train],
           shuffle=False,
           epochs=epochs,
           batch_size=batch_size
           )

    AE_sig = Model(inputs=[x_doc, x_0, x_1], outputs=sig)
    rank = AE_sig.predict([x_test_doc, x_test0, x_test1])

    rank = rank.reshape(test_num, 300)
    rank_list = GR.get_rank_matrix(rank)
    result = GR.get_result_by_ranks(rank_list, rec_k_list)

    print('=======conAE Result=======' + '\n')
    return result, AE


def conVAE(alpha, beta, activation=activ, use_bias=True, epochs=5, batch_size=560000, units=200, latent_dim=100,
           epsilon_std=1.0):
    # recall@k
    rec_k_list = [1, 5, 30, 150]

    # load training dataset
    x_train0 = np.load(train_pho0)
    x_train1 = np.load(train_pho1)
    x_train_doc = np.load(train_doc)
    y_train = np.load(train_y)  # label file

    # load testing dataset
    x_test0 = np.load(test_pho0)  # feature file
    x_test1 = np.load(test_pho1)
    x_test_doc = np.load(test_doc)

    # VAE Model
    x_doc = Input(shape=(doc_dim,))
    x_0 = Input(shape=(pho_dim,))
    x_1 = Input(shape=(pho_dim,))
    x_pho = get_pho_rep(x_0, x_1, pho_dim)
    concat = concatenate([x_doc, x_pho])

    z_mean = Dense(units=latent_dim, activation=activation, use_bias=use_bias)(concat)
    z_log_var = Dense(units=latent_dim, activation=activation, use_bias=use_bias)(concat)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                                  stddev=epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    z = Lambda(sampling)([z_mean, z_log_var])

    decoder = Dense(units=doc_dim + pho_dim, activation=activation, use_bias=use_bias)(z)

    _x_doc = Lambda(slice, arguments={'start': 0, 'end': 125})(decoder)
    _x_pho = Lambda(slice, arguments={'start': 125, 'end': 250})(decoder)

    y = Input(shape=(1,))
    sig = Dense(1, activation='sigmoid', use_bias=use_bias)(z_mean)

    # Label loss
    def bi_loss(args):
        x, y = args
        loss = objectives.binary_crossentropy(x, y)
        return loss

    label_loss = Lambda(bi_loss)([y, sig])

    # VAE loss
    def vae_loss(args):
        xd, _xd, xp, _xp = args
        xent_loss = objectives.binary_crossentropy(xd, _xd) \
                    + objectives.binary_crossentropy(xp, _xp)
        kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return xent_loss + kl_loss

    vae_loss = Lambda(vae_loss)([x_doc, _x_doc, x_pho, _x_pho])

    # Custom loss layer
    L = CustomVariationalLayer(alpha=alpha, beta=beta)([label_loss, vae_loss])

    con_vae = Model(outputs=L, inputs=[x_doc, x_0, x_1, y])
    print('=======Model Information=======' + '\n')
    con_vae.summary()

    con_vae.compile(optimizer=optim, loss=None)
    con_vae.fit([x_train_doc, x_train0, x_train1, y_train],
                shuffle=False,
                epochs=epochs,
                batch_size=batch_size
                )

    con_vae_sig = Model(inputs=[x_doc, x_0, x_1], outputs=sig)
    rank = con_vae_sig.predict([x_test_doc, x_test0, x_test1])

    rank = rank.reshape(test_num, 300)
    rank_list = GR.get_rank_matrix(rank)
    result = GR.get_result_by_ranks(rank_list, rec_k_list)

    print('=======conVAE Result=======' + '\n')
    return result, conVAE


def VaeRL2(alpha=1.0, beta=1.0, activation=activ, use_bias=True, epochs=5, batch_size=1000,
           units=200, latent_dim=100, epsilon_std=1.0):
    # recall@k
    rec_k_list = [1, 5, 30, 150]

    # load training dataset
    x_train_doc = np.load(train_doc)  # doc2vec feature file
    x_train0 = np.load(train_pho0)
    x_train1 = np.load(train_pho1)
    y_train = np.load(train_y)

    # load testing dataset
    x_test_doc = np.load(test_doc)  # doc2vec feature file
    x_test0 = np.load(test_pho0)  # feature file
    x_test1 = np.load(test_pho1)

    # Input
    x_doc = Input(shape=(doc_dim,))
    x_0 = Input(shape=(pho_dim,))
    x_1 = Input(shape=(pho_dim,))
    x_pho = get_pho_rep(x_0, x_1, pho_dim)

    # Attention Model
    x_a = concatenate([x_doc, x_pho])
    attention = Dense(units=doc_dim + pho_dim)(x_a)
    x_r = multiply([x_a, attention])

    # VAE model
    z_mean = Dense(units=latent_dim, activation=activation, use_bias=use_bias, name='output')(x_r)
    z_log_var = Dense(units=latent_dim, activation=activation, use_bias=use_bias)(x_r)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                                  stddev=epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    z = Lambda(sampling)([z_mean, z_log_var])

    decoder = Dense(units=doc_dim + pho_dim, activation=activation, use_bias=use_bias)(z)
    _attention = Dense(units=doc_dim + pho_dim)(decoder)
    _x_a = multiply([decoder, _attention])

    # Output
    _x_doc = Lambda(slice, arguments={'start': 0, 'end': 125})(_x_a)
    _x_pho = Lambda(slice, arguments={'start': 125, 'end': 250})(_x_a)

    y = Input(shape=(1,), name='y_in')
    sig = Dense(1, activation='sigmoid', use_bias=use_bias)(z_mean)

    # Label loss
    def loss(args):
        x, y = args
        loss = objectives.binary_crossentropy(x, y)
        return loss

    label_loss = Lambda(loss)([y, sig])

    # Vae loss
    x_doc_loss = Lambda(loss)([x_doc, _x_doc])
    x_pho_loss = Lambda(loss)([x_pho, _x_pho])

    def vae_loss(args):
        x, y = args
        kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        xent_loss = x + y
        return xent_loss + kl_loss

    vae_loss = Lambda(vae_loss)([x_doc_loss, x_pho_loss])

    # Custom loss layer

    L = CustomVariationalLayer(alpha=alpha, beta=beta)([label_loss, vae_loss])

    vaerl2 = Model(outputs=L, inputs=[x_doc, x_0, x_1, y])
    print('=======Model Information=======' + '\n')
    vaerl2.summary()

    vaerl2.compile(optimizer='adadelta', loss=None)
    vaerl2.fit([x_train_doc, x_train0, x_train1, y_train],
               shuffle=False,
               epochs=epochs,
               batch_size=batch_size
               )

    vaerl2_sig = Model(inputs=[x_doc, x_0, x_1], outputs=sig)
    rank = vaerl2_sig.predict([x_test_doc, x_test0, x_test1])

    rank = rank.reshape(test_num, 300)
    rank_list = GR.get_rank_matrix(rank)
    result = GR.get_result_by_ranks(rank_list, rec_k_list)

    print('=======ASVAE Result=======' + '\n')
    return result, vaerl2


def func(model, name, epoch, log_f, turns, dim):
    r = []
    log_f.write('\n======{}======\n'.format(name))
    for j in range(turns):
        result, m_f = model(alpha=1, beta=1, epochs=epoch, latent_dim=dim)
        line = '{}:> {}\n'.format(j, ' '.join([str(f1) for f1 in result]))
        log_f.write(line)
        r.append(result)
        log_f.flush()
        model_filename = 'model/{}_{}.h5'.format(name, j)
        m_f.save_weights(model_filename)
    t = [0] * len(r[0])
    for l in r:
        for j in range(6):
            t[j] = t[j] + l[j]
    for j in range(6):
        t[j] /= turns
    log_f.write('Average: {}\n'.format(' '.join([str(fl) for fl in t])))
    log_f.write('======{} end.======\n'.format(name))
    log_f.flush()
    return t


if __name__ == '__main__':
    import time

    log = open('log', 'a')
    log.write("\n{}\n"
              "test size:{}\n".format(time.asctime(time.localtime(time.time())), test_num))

    # dims
    models = [doc2vec, rhyme2vec, con, conAE, conVAE, VaeRL2]
    names = ['doc2vec', 'rhyme2vec', 'con', 'conAE', 'conVAE', 'VaeRL2']
    turn = [5] * 6
    res = []
    # dims = list(range(50, 251, 50))
    dims = [50]
    times = []
    iset = [3]
    '''
    0: doc2vec
    1: rhyme2vec
    2: con
    3: conAE
    4: conVAE
    5: VaeRL2
    '''
    print('================Dimension Discussion=================')
    for i in iset:
        for d in dims:
            time_start = time.time()
            res_t = func(models[i], names[i], 20, log, turn[i], d)
            time_used = time.time() - time_start
            log.write('{} dims take {} seconds in average.\n'.format(d, time_used))
            log.write('result: {}\n'.format(res_t))
            log.flush()
    log.close()
