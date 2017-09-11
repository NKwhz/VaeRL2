import numpy as np
import GetResult as GR
from keras.layers import Dense, Input, multiply, add
from keras.models import Model

x_train = np.load(open('x_train/pho_sum_20000.npy', 'rb'))
x_train_0 = np.load(open('x_train/pho_0_20000.npy', 'rb'))
x_train_1 = np.load(open('x_train/pho_1_20000.npy', 'rb'))
y_train = np.load(open('x_train/y_20000.npy', 'rb'))
x_test = np.load(open('x_test/pho_sum_1000.npy', 'rb'))
x_test_0 = np.load(open('x_test/pho_0_1000.npy', 'rb'))
x_test_1 = np.load(open('x_test/pho_1_1000.npy', 'rb'))

# recall@k
rec_k_list = [1, 5, 30, 150]
# number of test samples
test_num = 1000
# the input dimension
input_shape = 125
# parameters of neural network
activation = 'tanh'  # relu, softmax, sigmoid, linear, elu
use_bias = True  # False
units = 100  # units of the single dense layer
epochs = 20
batch_size1 = 100
batch_size2 = 100 * 126  # according to how many parameters
optimizer = 'adagrad'

x = Input(shape=(input_shape,))
sig = Dense(1, activation='sigmoid', use_bias=use_bias)(x)

pho_0 = Model(outputs=sig, inputs=[x])
print('=======Model Information=======' + '\n')
pho_0.summary()
pho_0.compile(optimizer=optimizer, loss='binary_crossentropy')
pho_0.fit([x_train_0], y_train,
          shuffle=False,
          epochs=epochs,
          batch_size=batch_size2
          )
rank = pho_0.predict([x_test_0])

rank = rank.reshape(test_num, 300)
rank_list = GR.get_rank_matrix(rank)
result_0 = GR.get_result_by_ranks(rank_list, rec_k_list)

pho_1 = Model(outputs=sig, inputs=[x])
print('=======Model Information=======' + '\n')
pho_1.summary()
pho_1.compile(optimizer=optimizer, loss='binary_crossentropy')
pho_1.fit([x_train_1], y_train,
          shuffle=False,
          epochs=epochs,
          batch_size=batch_size2
          )

rank = pho_1.predict([x_test_1])

rank = rank.reshape(test_num, 300)
rank_list = GR.get_rank_matrix(rank)
result_1 = GR.get_result_by_ranks(rank_list, rec_k_list)

DA = Model(outputs=sig, inputs=[x])
print('=======Model Information=======' + '\n')
DA.summary()
DA.compile(optimizer=optimizer, loss='binary_crossentropy')
DA.fit([x_train], y_train,
       shuffle=False,
       epochs=epochs,
       batch_size=batch_size2
       )

rank = DA.predict([x_test])

rank = rank.reshape(test_num, 300)
rank_list = GR.get_rank_matrix(rank)
result_DA = GR.get_result_by_ranks(rank_list, rec_k_list)

x_0 = Input(shape=(input_shape,))
a_0 = Dense(input_shape)(x_0)
x_1 = Input(shape=(input_shape,))
a_1 = Dense(input_shape)(x_1)
_x_0 = multiply([x_0, a_0])
_x_1 = multiply([x_1, a_1])
_x = add([_x_0, x_1])

sig = Dense(1, activation='sigmoid', use_bias=use_bias)(_x)

rhyme2vec = Model(outputs=sig, inputs=[x_0, x_1])
print('=======Model Information=======' + '\n')
rhyme2vec.summary()
rhyme2vec.compile(optimizer='adagrad', loss='binary_crossentropy')
rhyme2vec.fit([x_train_0, x_train_1], y_train,
            shuffle=False,
            epochs=epochs,
            batch_size=batch_size1
            )

rank = rhyme2vec.predict([x_test_0, x_test_1])

rank = rank.reshape(test_num, 300)
rank_list = GR.get_rank_matrix(rank)
result_rhyme2vec = GR.get_result_by_ranks(rank_list, rec_k_list)

print('=======Result=======' + '\n')

print('C-Line result:' + '\n')
print(result_0)
print('\n' + 'Skip-Line result:' + '\n')
print(result_1)
print('\n' + 'DA result:' + '\n')
print(result_DA)
print('\n' + 'Rhyme2vec result:' + '\n')
print(result_rhyme2vec)
