# -*- coding: utf-8 -*-
import dataset as dataset
import numpy as np
from keras.layers import Input, Embedding, Flatten, Dense, Concatenate, LeakyReLU, PReLU
from keras.models import Model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from keras.layers import BatchNormalization, Dropout
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import Adam

#----------------------------------------------------------------------------#
# Import train and test
#----------------------------------------------------------------------------#
(train_data, train_targets), (test_data, test_targets),(val_data, val_targets) , scaler = dataset.load_data()

#----------------------------------------------------------------------------#
# Input data
#----------------------------------------------------------------------------#
weekday_input = Input(shape=(1,))
hour_input = Input(shape=(1,))
cell_id_input = Input(shape=(1,))
sms_input = Input(shape=(1,))
internet_input = Input(shape=(1,))
#----------------------------------------------------------------------------#
# Embedding layers (categorical data)
#----------------------------------------------------------------------------#
embedding_dim = 10 
embedding_weekday = Embedding(input_dim=7, output_dim=embedding_dim)(weekday_input)
embedding_hour = Embedding(input_dim=24, output_dim=embedding_dim)(hour_input)
embedding_cell_id = Embedding(input_dim=10000, output_dim=embedding_dim)(cell_id_input)

#----------------------------------------------------------------------------#
# Flatten embedding data
#----------------------------------------------------------------------------#
flatten_weekday = Flatten()(embedding_weekday)
flatten_hour = Flatten()(embedding_hour)
flatten_cell_id = Flatten()(embedding_cell_id)

#----------------------------------------------------------------------------#
# Dense layers (continuous and normalized data) 
#----------------------------------------------------------------------------#
dense_sms = Dense(8, activation='relu')(sms_input)
dense_internet = Dense(8,activation='relu')(internet_input)

#----------------------------------------------------------------------------#
# Concatenate input layers
#----------------------------------------------------------------------------#
combined = Concatenate()([flatten_weekday, flatten_hour, flatten_cell_id, dense_sms, dense_internet])

#----------------------------------------------------------------------------#
# Dense layers (hidden)
#----------------------------------------------------------------------------#
dense1 = Dense(32, activation=PReLU())(combined)
dense1 = BatchNormalization()(dense1)

dense2 = Dense(8, activation=PReLU())(dense1)
dense2 = BatchNormalization()(dense2)

#----------------------------------------------------------------------------#
# Output layer
#----------------------------------------------------------------------------#
output_layer = Dense(1, activation='linear')(dense2)

#----------------------------------------------------------------------------#
# Model
#----------------------------------------------------------------------------#
# Define model
model = Model(inputs=[weekday_input, hour_input, cell_id_input, sms_input, internet_input], outputs=output_layer)

# Define optimizer
optimizer = Adam(learning_rate=0.00001)

# Compile model using MSE loss function
model.compile(loss='mean_squared_error', optimizer=optimizer)

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# Model fit
history = model.fit(
    [train_data[:, 1], train_data[:, 0], train_data[:, 2], train_data[:, 3], train_data[:, 4]],
    train_targets,
    validation_data=(
        
        [val_data[:, 1], val_data[:, 0], val_data[:, 2], val_data[:, 3],val_data[:, 4]],
        val_targets
    ),
    epochs=100,
    batch_size=128,
    callbacks = [early_stopping],
    verbose=1
)

# Model predictions
predictions = model.predict([test_data [:,1], test_data[:,0], test_data [:,2], test_data[:,3], test_data[:, 4]])

# Model loss
loss = model.evaluate([test_data [:,1], test_data[:,0], test_data [:,2], test_data[:,3], test_data[:, 4]], test_targets)

#----------------------------------------------------------------------------#
# Results
#----------------------------------------------------------------------------#
# MSE 
plt.figure()
plt.plot(history.history['loss'], label='Training MSE')
plt.plot(history.history['val_loss'], label='Validation MSE')
plt.title('Mean Squared Error bs = 128 (5000 cells) ')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()
plt.show()

# Predictions vs test data
predictions_original_scale = scaler.inverse_transform(predictions)
test_targets_original_scale = scaler.inverse_transform(test_targets)

plt.figure()
plt.plot(test_targets_original_scale, label='Real Values')
plt.plot(predictions_original_scale, label='Predictions')
plt.title('Real Values vs. Predictions bs = 128 (5000 cells)')
plt.legend()
plt.show()

# Predictions vs test data (ZOOM)
plt.figure()
plt.plot(test_targets_original_scale, label='Real Values')
plt.plot(predictions_original_scale, label='Predictions')
plt.title('Real Values vs. Predictions bs = 128 (5000 cells)')
plt.ylim(0, 1200)
plt.xlim(2000, 2100)
plt.legend()
plt.show()


# MSE results
mse_train = history.history['loss'][-1]
print(f'MSE en el conjunto de entrenamiento: {mse_train}')

mse_val = history.history['val_loss'][-1]
print(f'MSE en el conjunto de validación: {mse_val}')


rmse_test = np.sqrt(loss)
print(f'MSE en el conjunto de prueba: {loss}')
