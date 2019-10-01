# -*- coding: utf-8 -*-
"""
Created on Thu May  9 15:16:07 2019

@author: PC10
"""

from attention_decoder import AttentionDecoder
from keras.callbacks import EarlyStopping
from keras.constraints import nonneg
from keras.layers import advanced_activations,Dropout, Concatenate ,Dense ,LSTM,MaxPooling1D ,Conv1D, MaxPooling1D,Input, TimeDistributed, Flatten, Conv2D,Reshape,Permute, Flatten
from keras.models import Model, Sequential
from keras.utils import plot_model
import TimeSeriesUtils as TSU
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
#X[Tw,As,Af]  where As is the amount of sellers, Af is the amount of features and Tw is the window thats being looked into the past 
pred_period = 80
look_back_period = 30
look_back_period_naive = 10
future_rolling = 15
latent_dim = 6
dropout_rate = 0
scale_factor = 1

complete_view.columns
features = ['market_size_units','active_seller','position_median','relative_price','daily_sales_sum']
scaled_featues = []
dependent_variable = ['daily_sales_sum']

complete_view_scaled = complete_view.copy()
scaler= MinMaxScaler()
complete_view_scaled[scaled_featues] = complete_view_scaled[scaled_featues].fillna(0).apply(lambda x: (x-x.min())/(x.max()-x.min()))
complete_view_scaled = complete_view_scaled.assign(daily_sales_sum_future = complete_view_scaled['daily_sales_sum'].groupby(level = 'seller_id').apply(lambda x: x.loc[::-1].rolling(future_rolling, min_periods = 1).mean().loc[::-1])*scale_factor)
X= TSU.df_to_array(complete_view_scaled.fillna(0)[features])
X_dates = list(complete_view.index.levels[0])
X_train, y_train, X_val, y_val = TSU.chunk_data_by_date(X,pred_period,look_back_period)
y = TSU.df_to_array(complete_view_scaled.fillna(0)[dependent_variable])
_, y_train, _, y_val = TSU.chunk_data_by_date(y,pred_period,look_back_period)
complete_view_scaled = complete_view_scaled.assign(daily_sales_sum_future = complete_view_scaled['daily_sales_sum'].groupby(level = 'seller_id').apply(lambda x: x.loc[::-1].rolling(future_rolling, min_periods = 1).mean().loc[::-1])/scale_factor)

X.shape[0]-y_train.shape[0]-y_val.shape[0]
moving_averge_error = [abs(complete_view.loc[(slice(None),seller),'daily_sales_sum'].reset_index(level = 'seller_id').rolling(look_back_period_naive, min_periods = look_back_period_naive//5).mean().tshift(periods = -look_back_period_naive, freq = 'D').fillna(0)['daily_sales_sum'].values-complete_view.loc[(slice(None),seller),'daily_sales_sum'].reset_index(level = 'seller_id').fillna(0)['daily_sales_sum'].values).mean() for seller in sellers]
ewm_error = [abs(complete_view.loc[(slice(None),seller),'daily_sales_sum'].reset_index(level = 'seller_id').fillna(0).ewm(span= look_back_period_naive).mean().tshift(periods = -0, freq = 'D').fillna(0)['daily_sales_sum'].values-complete_view.loc[(slice(None),seller),'daily_sales_sum'].reset_index(level = 'seller_id').fillna(0)['daily_sales_sum'].values).mean() for seller in sellers]

np.array(ewm_error)/np.array(moving_averge_error)

plt.clf()
complete_view.loc[(slice(None),97556253),'daily_sales_sum'].reset_index(level = 'seller_id').fillna(0).rolling(look_back_period_naive, min_periods = look_back_period_naive//5).mean().tshift(periods = -0, freq = 'D').fillna(0)['daily_sales_sum'].plot()
complete_view.loc[(slice(None),97556253),'daily_sales_sum'].reset_index(level = 'seller_id').fillna(0).ewm(span= 360).mean().tshift(periods = -0, freq = 'D').fillna(0)['daily_sales_sum'].plot()
complete_view_scaled.loc[(slice(None),97556253),'daily_sales_sum_future'].reset_index(level = 'seller_id').fillna(0)['daily_sales_sum_future'].tshift(periods = -pred_period,freq = 'D').plot()

X_train_no_teacher_forcing,X_val_no_teacher_forcing= np.zeros([X_train.shape[0],pred_period,latent_dim]),np.zeros([X_val.shape[0],pred_period,latent_dim])
y_train, y_val = np.take(y_train,-1,axis = -1), np.take(y_val,-1,axis = -1)

encoder_input = Input(shape = X_train.shape[1:],name = 'encoder_input')
decoder_input = Input(shape = (None,latent_dim),name = 'decoder_input')
permute_conv = Permute((2,3,1))
conv1 = TimeDistributed(Conv1D(4, kernel_size =  1),input_shape = (X_train.shape[1],X_train.shape[2],X_train.shape[3]))
conv2 = TimeDistributed(Conv1D(2, kernel_size =  1),input_shape = (X_train.shape[1],X_train.shape[2],X_train.shape[3]))
conv3 = TimeDistributed(Conv1D(1, kernel_size =  1),input_shape = (X_train.shape[1],X_train.shape[2],X_train.shape[3]))
concat = Concatenate()
dropout =  Dropout(dropout_rate)
permute = Permute((2,1,3))
reshape = Reshape((look_back_period,X_train.shape[2]))
encoderLSTM = LSTM(units = latent_dim,return_state = True,return_sequences = True,name = 'enc_LSTM',dropout = dropout_rate)
encoderLSTM2 = LSTM(units = latent_dim,return_state = True,return_sequences = True,name = 'enc_LSTM2',dropout = dropout_rate)
decoderLSTM = LSTM(units = latent_dim,return_state = True,return_sequences = True,name = 'dec_LSTM',dropout = dropout_rate)
reshape2 = Reshape((look_back_period,X_train.shape[2]))
dense_output = TimeDistributed(Dense(X_train.shape[2], activation = 'relu', W_constraint=nonneg()),name = 'time_distirbuted_dense_output')
##
##
## encoder
conv1_out = conv1(encoder_input)
conv2_out = conv2(conv1_out)
conv3_out = conv3(conv2_out)

reshape_out_1 = reshape(conv3_out)

encoder_out =  encoderLSTM(reshape_out_1)
#encoder_out =  encoderLSTM2(encoder_out[0])
encoder_states = encoder_out[1:]
encoder_out = encoder_out[0]


decoder_out = decoderLSTM(decoder_input, initial_state = encoder_states)
decoder_states = decoder_out[1:]
decoder_out = decoder_out[0]
output = dense_output(decoder_out)


model = Model([encoder_input, decoder_input],output)
model.summary()
#plot_model(model,show_shapes = True, to_file='model.png')

model.compile(optimizer = 'adam',loss = 'MSE')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30,restore_best_weights = True)
train_history = model.fit([X_train,X_train_no_teacher_forcing], y_train, batch_size = 128, epochs = 50000,validation_data = ([X_val,X_val_no_teacher_forcing], y_val),callbacks = [es])

plt.clf()
plt.plot(train_history.history['loss'])
plt.plot(train_history.history['val_loss'])

enc_pred_model = Model(encoder_input,encoder_states)

dec_input_states = [Input(shape = (latent_dim,)),Input(shape = (latent_dim,))]

dec_outputs_and_states = decoderLSTM(decoder_input, initial_state = dec_input_states)
dec_outputs = dec_outputs_and_states[0]
dec_states = dec_outputs_and_states[1:]
output = dense_output(dec_outputs)
dec_pred_model = Model([decoder_input] + dec_input_states,[output]+ dec_states)

y_train,y_val = y_train/scale_factor,y_val/scale_factor
preds = TSU.enc_dec_predict(X_val,enc_pred_model,dec_pred_model,pred_period,latent_dim)/scale_factor
i=0
i+=1
seller = i
print(sellers[i])
train_preds = TSU.average_anti_diag(TSU.enc_dec_predict(X_train,enc_pred_model,dec_pred_model,pred_period,latent_dim)[:,:,seller])/scale_factor
seller_preds= TSU.average_anti_diag(preds[:,:,seller])
moving_average = complete_view.loc[(slice(None),sellers[seller]),'daily_sales_sum'].fillna(0).reset_index(level = 'seller_id').rolling(look_back_period_naive).mean()['daily_sales_sum'].shift(periods = 0,freq = 'D')
plt.clf()
#plt.plot(X_dates[-2*pred_period:-1*pred_period],y_val[0,:,seller], color = 'darkorange')
plt.plot(X_dates[-2*pred_period:][-future_rolling:],len(X_dates[-2*pred_period:][-future_rolling:])*[np.nanmean(TSU.average_anti_diag(y_val[:,:,seller])[-future_rolling:])], color = 'b',label = 'pred_period_average')
plt.plot(X_dates[-2*pred_period:],TSU.average_anti_diag(y_val[:,:,seller]), color = 'b',label = 'validation_data')
plt.plot(X_dates[-2*pred_period:],seller_preds,color = 'c',label = 'validation_set_pred')
plt.plot(X_dates[-2*pred_period:-1*pred_period],len(X_dates[-pred_period:])*[np.mean(preds[0,:,seller])],color = 'c',label = 'val_set_average')
plt.plot(X_dates[look_back_period:-1*pred_period],train_preds,color = 'yellowgreen')
plt.plot(X_dates[look_back_period:-2*pred_period],y_train[:,0,seller],color = 'darkgreen',label = 'train_set')
#plt.plot(X_dates[look_back_period:-2*pred_period],TSU.average_anti_diag(X_train[:,:,seller,-1]),color = 'b')
plt.plot(moving_average.index,moving_average,label = 'moving_average_model')
column = 'amount_of_ads'
#plt.plot(dates,complete_view_scaled.loc[(dates,sellers[seller]),column],label = column)
plt.legend()

rmse = {}
rmspe = {}
for seller in sellers:
    rmse[seller] = abs(np.nanmean(TSU.average_anti_diag(y_val[:,:,sellers.index(seller)]))- np.nanmean(TSU.average_anti_diag(preds[:,:,sellers.index(seller)])[-future_rolling:]))
    rmspe[seller] = rmse[seller]/np.nanmean(TSU.average_anti_diag(y_val[:,:,sellers.index(seller)]))
naive_score = {}
for seller in sellers:
    naive_score[seller] = moving_averge_error[sellers.index(seller)]/rmse[seller]

print(naive_score)