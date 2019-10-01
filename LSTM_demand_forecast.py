# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 19:44:53 2019

@author: PC10
"""
from tsmetrics import tsmetrics

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import TimeSeriesUtils as TSU
groupped_data = groupped_data.assign(daily_revenues_per_seller = groupped_data['daily_revenues']/groupped_data['amount_of_sellers'])

print(groupped_data.columns)
dependent_variable  = ['daily_sales']
features =  ['month','year','daily_revenues_per_seller','daily_views','daily_views_per_seller','amount_of_sellers','median_price','daily_views_per_seller',dependent_variable[0]]
one_hot_indexes = [0,1]

init_date = groupped_data[groupped_data.daily_sales > 0].date.min()
X_dates = groupped_data[groupped_data.date >= init_date].date
view = groupped_data[groupped_data.date.isin(X_dates)][features]
view = view.fillna(method = 'backfill')
X = view.values

X = TSU.one_hot_append(X,[0,1])

min_max_scaler = MinMaxScaler()
min_values = [X[:,i].min() for i in range(X.shape[1])]
max_values = [X[:,i].max() for i in range(X.shape[1])]

X = min_max_scaler.fit_transform(X)

pred_period = 100
look_back_period = 90
X_train, y_train, X_val, y_val  = TSU.chunk_data_by_date(X,pred_period,look_back_period)
X_train_teacher_forcing,X_val_teacher_forcing,X_train_no_teacher_forcing,X_val_no_teacher_forcing = TSU.teacher_forcing_generator(y_train,y_val)
normalized_groupped_data=(groupped_data-groupped_data.min())/(groupped_data.max()-groupped_data.min())
normalized_groupped_data[['daily_sales','daily_views']].plot()

naive_model = np.average(np.take(X_train,-1,axis  = -1),axis =  1)

from keras.models import Sequential, Model
from keras.layers import Dense, Input
from keras.layers import LSTM
from keras.layers import Dropout


model_input = X_train
model_output= y_train
latent_dim = 3
period_to_average = 5
assert model_input.shape[1]%period_to_average ==0

#Define an input sequence and process it.
encoder_inputs = Input(shape=(model_input.shape[1],model_input.shape[2]),name  = 'encoder_input')
average_pooling = AveragePooling2D(pool_size= (period_to_average,1),strides = None)
average_pooling_outputs = Reshape((model_input.shape[1]//period_to_average,model_input.shape[2]))(average_pooling(Reshape((model_input.shape[1],model_input.shape[2],1))(encoder_inputs)))

encoder = LSTM(latent_dim,return_sequences = True ,return_state=True, name = 'encoder',dropout = 0.5)
encoder_outputs, state_h, state_c = encoder(average_pooling_outputs)
#conv_att = Conv2D(filters = 1, kernel_size = (model_input.shape[1]//period_to_average,latent_dim), strides = (1,1))
#conv_att_output = Reshape((model_input.shape[1]//period_to_average,latent_dim))(conv_att(Reshape((model_input.shape[1]//period_to_average,latent_dim,1))(encoder_outputs)))
#encoder = LSTM(latent_dim,return_sequences = True ,return_state=True, name = 'encoder_2',dropout = 0.5)
#encoder_outputs, state_h, state_c = encoder(conv_att_output)

# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None,1),name  = 'decoder_input') #Input(shape=(10,X_train.shape[2]),name = 'decoder_input')
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim,return_sequences = True,return_state = True,name = 'decoder',dropout = 0.5)
decoder_outputs,dec_state_h,dec_state_c= decoder_lstm(decoder_inputs, initial_state = encoder_states)

decoder_dense = TimeDistributed(Dense(1))
dense_decoder_outputs = decoder_dense(decoder_outputs)
model = Model([encoder_inputs,decoder_inputs], dense_decoder_outputs)
encoder = Model(encoder_inputs,encoder_states)
model.summary()
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)



model.compile(optimizer='adam', loss='mse')
history = model.fit([model_input,X_train_no_teacher_forcing],model_output,
          batch_size=512,
          epochs=400,
          callbacks = [es],
          validation_data = ([X_val,np.zeros((y_val.shape[0],y_val.shape[1],1))],y_val)
          )


plt.clf()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])


from keras.utils import plot_model
plot_model(model,show_shapes = True, to_file='model1.png')


###################
decoder_state_inputs = [Input(shape = (latent_dim,)),Input(shape = (latent_dim,))]

decoder_outputs_and_states = decoder_lstm(decoder_inputs, initial_state = decoder_state_inputs)
decoder_outputs = decoder_outputs_and_states[0]
decoder_states = decoder_outputs_and_states [1:]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_predict_model = Model([decoder_inputs]+decoder_state_inputs,[decoder_outputs]+ decoder_states)

plot_model(decoder_predict_model,show_shapes = True, to_file='decoder.png')

preds = TSU.enc_dec_predict(X_val[-1:],encoder,decoder_predict_model,pred_period,look_back_period)

total_preds = model.predict([X_train,X_train_no_teacher_forcing])[:,:,0]

train_preds = TSU.get_anti_diag(total_preds)
x_preds_newest = [train_preds[i][0]  for i in range(len(train_preds))]
x_preds_oldest = [train_preds[i][-1]  for i in range(len(train_preds))]
train_preds_avg = TSU.average_anti_diag(total_preds)


# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
min_value = groupped_data['daily_sales'].min()
max_value = groupped_data['daily_sales'].max()

preds = np.array(preds).flatten()*(max_value - min_value)+min_value
train_preds_avg = train_preds_avg.flatten()*(max_value - min_value)+min_value

y_val_limit = y_train[:,-1].flatten()
y_val_set = y_val_limit*(max_value - min_value)+min_value
y_train_transf = groupped_data['daily_sales'][groupped_data.date.isin(X_dates[look_back_period + pred_period:-pred_period])]

pyplot.clf()
pyplot.plot(groupped_data['date'],groupped_data['daily_sales'],color = 'b') # plot the entire  series
pyplot.plot(X_dates[look_back_period + 1*pred_period:-1*pred_period],y_val_set,color = 'darkorange') # highlights the valdiation set
pyplot.plot(X_dates[pred_period+look_back_period:-pred_period],train_preds_avg,color = 'y') # plots the training preds averaged
pyplot.plot(X_dates[len(X_dates)-1*pred_period:len(X_dates)-0*pred_period],pred_period*[y_val_set.mean()], color = 'r') #plot the mean of the pred_period set

pyplot.plot(X_dates[-1*pred_period:],preds.ravel(),color = 'r') # plot the  preds
pyplot.plot(X_dates[len(X_dates)-1*pred_period:len(X_dates)-0*pred_period],pred_period*[y_train_transf[-look_back_period:].mean()], color = 'darkorange') # plot the mean of the look_back_period set
pyplot.plot(X_dates[len(X_dates)-1*pred_period:len(X_dates)-0*pred_period],pred_period*[preds.mean()], color = 'r') #plot the mean of the pred set
pyplot.plot(X_dates[len(X_dates)-1*pred_period:len(X_dates)-0*pred_period],pred_period*[y_val_set.mean()], color = 'c') #plot the mean of the y_val set


####

i = 0
pyplot.clf()
pyplot.plot(range(i,len(x_preds[i])+i),x_preds[i])
pyplot.plot(range(len(y_val[i,i:i+pred_period])),y_val[i,i:i+pred_period])

i+=1

pyplot.clf()
pyplot.scatter(range(len(x_preds_newest)),x_preds_oldest,color = 'r')
pyplot.scatter(range(len(x_preds_newest)),x_preds_newest,color = 'g')
pyplot.scatter(range(len(x_preds_newest)-pred_period+1),train_preds_avg,color = 'b')
plot_dict ={}
for xe, ye in zip(range(len(train_preds)),train_preds):
    plot_dict[xe] = ye
fig, ax = pyplot.subplots()
ax.boxplot(plot_dict.values())
####



