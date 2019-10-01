# -*- coding: utf-8 -*-
"""
Created on Thu May  9 15:16:07 2019

@author: PC10
"""

from attention_decoder import AttentionDecoder
from keras.callbacks import EarlyStopping
from keras.layers import advanced_activations,Dropout, Concatenate ,Dense, LSTM,MaxPooling1D ,Conv1D, MaxPooling1D,Input, TimeDistributed, Flatten, Conv2D,Reshape,Permute, Flatten
from keras.models import Model, Sequential
from keras.utils import plot_model
import TimeSeriesUtils as TSU
#X[Tw,As,Af]  where As is the amount of sellers, Af is the amount of features and Tw is the window thats being looked into the past 
X_train, y_train, X_val, y_val = TSU.chunk_data_by_date(X[:,:,1:],100,90)
y_train, y_val = np.take(y_train,-1,axis = -1), np.take(y_val,-1,axis = -1)

X_train_teacher_forcing,X_val_teacher_forcing,X_train_no_teacher_forcing,X_val_no_teacher_forcing = TSU.teacher_forcing_generator(y_train,y_val,temporal_axis_output = -2, flatten = False)
y_train = y_train.reshape(y_train.shape[:-1])
y_val= y_val.reshape(y_val.shape[:-1])

encoder_input = Input(shape = X_train.shape[1:],name = 'encoder_input')
dummy_input = Input(shape = [X_train.shape[1], X_train.shape[2]],name = 'dummy_input')
decoder_input = Input(shape = (None,1),name = 'decoder_input')
permute_conv = Permute((2,3,1))
conv1 = TimeDistributed(Conv1D(1, kernel_size =  1),input_shape = (X_train.shape[1],X_train.shape[2],X_train.shape[3]))
flat = Reshape([X_train.shape[1],X_train.shape[2]*X_train.shape[3]])
concat = Concatenate()
dropout =  Dropout(0.2)
#conv2 = TimeDistributed(Conv1D(1, kernel_size =  1),input_shape = (X_train.shape[1],X_train.shape[2],X_train.shape[3]),name = 'tim_distributed_CNN_2')
permute = Permute((2,1,3))
reshape = Reshape((look_back_period,X_train.shape[2]))
encoderLSTM = LSTM(units = 12,return_state = True,return_sequences = True,name = 'enc_LSTM',dropout = 0.1)
decoderLSTM = LSTM(units = 12,return_state = True,return_sequences = True,name = 'dec_LSTM',dropout = 0.1)
reshape2 = Reshape((look_back_period,X_train.shape[2]))
dense_output = TimeDistributed(Dense(1),name = 'time_distirbuted_dense_output')
output_reshape = Reshape(y_train.shape[:-1])
## encoder
#conv1_out = conv1(encoder_input)
conv1_out = flat(encoder_input)
#reshape_out_1 = reshape(conv1_out)
encoder_in_concat = concat([conv1_out,dummy_input])

encoder_out =  encoderLSTM(encoder_in_concat)
encoder_states = encoder_out[1:]
encoder_out = encoder_out[0]

decoder_out = decoderLSTM(decoder_input, initial_state = encoder_states)
decoder_states = decoder_out[1:]
decoder_out = decoder_out[0]

output = dense_output(decoder_out)
#output = output_reshape(output)

model = Model([encoder_input,dummy_input,decoder_input],output)
model.summary()
plot_model(model,show_shapes = True, to_file='model.png')


model.compile(optimizer = 'adam',loss = 'MSE')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)
train_history = model.fit([X_train,X_d_t,X_train_no_teacher_forcing], y_train, batch_size = 128, epochs = 500,validation_data = ([X_val,X_d_v,X_val_no_teacher_forcing], y_val),callbacks = [es])

plt.clf()
plt.plot(train_history.history['loss'])
plt.plot(train_history.history['val_loss'])

pred = model.predict([X_val[-1:],X_train_no_teacher_forcing])

i=0
i+=1
seller = i
plt.clf()
plt.plot(range(len(y_val)),y_val[0,:,seller], color = 'darkorange')
plt.plot(range(pred.shape[1]),pred[0,:,seller], color = 'darkorange')


plt.plot(labels_test[:days],len(labels_test[:days])*[np.mean(pred[0,:,seller])],color = 'yellowgreen')
plt.plot(labels_test[:days],len(labels_test[:days])*[np.mean(y_val[0,:,seller])], color = 'r')





model.summary()

assert not np.isnan(y).any()
assert not np.isnan(X_train).any()

model.compile(optimizer = 'adam', loss = 'mse')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
train_history = model.fit(X_train,y_train,batch_size = 256,epochs = 500,validation_data = (X_val,y_val),callbacks = [es])

plt.clf()
plt.plot(train_history.history['loss'])
plt.plot(train_history.history['val_loss'])

X_train_modified = X_train[len(X_train)-1:len(X_train),:,:]

pred = model.predict(X_train_modified)
y_val[0].shape

[history[history.seller_id == seller].date.min() for seller in sellers]

i=0
i+=1
seller = i
plt.clf()
plt.plot(labels_test[:days],y_val[0,:,seller], color = 'darkorange')
plt.plot(labels_test[:days],len(labels_test[:days])*[np.mean(y_val[0,:,seller])], color = 'r')

plt.plot(labels_test[:days],pred[0,:,seller],color = 'g')
plt.plot(labels_test[:days],len(labels_test[:days])*[np.mean(pred[0,:,seller])],color = 'yellowgreen')


def predict(model,X, n_output):
    preds = []
    X.shape = X[len(X)-1:len(X),:,:]
    sample =  0
    while sample < len(X_modified):
        X_modified = X[sample,:-n_output,:].shape
        pred_array = 
        pred = model.predcit(X)[sample,:n_output,:]
        preds.append(pred)
        sample+=n_output