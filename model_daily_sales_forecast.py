# -*- coding: utf-8 -*-
"""
Created on Fri May 24 11:40:10 2019

@author: PC10
"""

from attention_decoder import AttentionDecoder
from keras.callbacks import EarlyStopping
from keras.layers import advanced_activations,Dropout ,Dense, LSTM,MaxPooling1D ,Conv1D, MaxPooling1D,Input, TimeDistributed, Flatten, Conv2D,Reshape,Permute, Flatten
from keras.models import Model, Sequential
from keras.utils import plot_model
import TimeSeriesUtils as TSU
#X[Tw,As,Af]  where As is the amount of sellers, Af is the amount of features and Tw is the window thats being looked into the past 

X.shape[X<0] = 0
pred_period = 30
look_back_period = 30
X_train, y_train, X_val, y_val = TSU.chunk_data_by_date(X,pred_period,look_back_period)
#y_train, y_val = np.take(y_train,-1,axis = -1), np.take(y_val,-1,axis = -1)
y_train,y_val  = np.take(y_train,seller_pos,axis = -2),np.take(y_val,seller_pos,axis = -2)
X_train,X_val  = np.take(X_train,seller_pos,axis = -2),np.take(X_val,seller_pos,axis = -2)
X_train_teacher_forcing,X_val_teacher_forcing,X_train_no_teacher_forcing,X_val_no_teacher_forcing = TSU.teacher_forcing_generator(y_train,y_val,temporal_axis_output = -2, flatten = False)
lattent_dim = 5




encoder_input = Input(shape = X_train.shape[1:])
decoder_input = Input(shape = (y_train.shape[1],1))
dropout =  Dropout(0.2)
#conv2 = TimeDistributed(Conv1D(1, kernel_size =  1),input_shape = (X_train.shape[1],X_train.shape[2],X_train.shape[3]),name = 'tim_distributed_CNN_2')
encoderLSTM = LSTM(units = lattent_dim,input_shape = (X_train.shape[1],X_train.shape[-1]),return_state = True,return_sequences = True,name = 'enc_LSTM',dropout = 0.2)
dense_states = Dense(1)
decoderLSTM = LSTM(units = 1,input_shape = (X_train.shape[1],X_train.shape[-1]),return_state = True,return_sequences = True,name = 'dec_LSTM',dropout = 0.2)
reshape2 = Reshape((y_train.shape[1],1))
dense_output = TimeDistributed(Dense(1),name = 'time_distirbuted_dense_output')
model.output_shape
## encoder
#conv2_out = conv2(conv1_out)

encoder_out =  encoderLSTM(encoder_input)
encoder_states = [dense_states(encoder_out[1]),dense_states(encoder_out[2])]
encoder_out = encoder_out[0]

decoder_out = decoderLSTM(decoder_input, initial_state = encoder_states)
decoder_states = decoder_out[1:]
decoder_out = decoder_out[0]

#decoder_out = reshape2(decoder_out)
#output = dense_output(decoder_out)
output = dense_output(decoder_out)

model = Model([encoder_input,decoder_input],output)
encoder = Model(encoder_input,encoder_out)
model.summary()
model.compile(optimizer = 'adam',loss = 'MSE')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30)
train_history = model.fit([X_train,X_train_no_teacher_forcing], y_train, batch_size = 128, epochs = 5000,validation_data = ([X_val,X_val_no_teacher_forcing], y_val),callbacks = [es])

naive_model = np.take(X_train,-1,axis  = -1).mean()
view = y_train.reshape(y_train.shape[0],y_train.shape[1])



plt.clf()
plt.plot(train_history.history['loss'])
plt.plot(train_history.history['val_loss'])

decoder_state_inputs = [Input(shape = (1,)),Input(shape = (1,))]
decoder_outputs_and_states = decoderLSTM(decoder_input, initial_state = decoder_state_inputs)
decoder_outputs = decoder_outputs_and_states[0]
decoder_states = decoder_outputs_and_states [1:]
outputs = dense_output(decoder_outputs)

decoder_predict_model = Model([decoder_input]+decoder_state_inputs,[outputs]+ decoder_states)

preds = TSU.enc_dec_predict(X_val[-1:],encoder,decoder_predict_model,pred_period,look_back_period)
pred = model.predict([X_val[-1:],X_train_no_teacher_forcing])
preds = model.predict([X_val[-1:],X_train_no_teacher_forcing])

plt.clf()
plt.plot(plot_dates[-pred_period:],y_val[-1].flatten(), color = 'b')
plt.plot(plot_dates[-pred_period:],pred.flatten(), color = 'darkorange')
plt.plot(plot_dates[-pred_period:],len(y_val[-1].flatten())*[y_val[-1].flatten().mean()], color = 'b')

plt.plot(plot_dates[-pred_period:],len(plot_dates[-pred_period:])*[pred.ravel()], color = 'darkorange')
plt.plot(plot_dates[-pred_period:],len(y_val)*[naive_model], color = 'g')
