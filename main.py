# dataframe, numpy
import pandas as pd
import numpy as np

# Scaler
from sklearn.preprocessing import MinMaxScaler

# Model, LSTM
from keras.models import Sequential
from keras.layers import Dense, LSTM

# graph
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# downloader file
import urllib.request

# common
import os
import time
import math
import sys
from os import path
from datetime import datetime

# source
# https://www.ecdc.europa.eu/en/publications-data/download-todays-data-geographic-distribution-covid-19-cases-worldwide
LINK_SRC = 'https://www.ecdc.europa.eu/sites/default/files/documents/COVID-19-geographic-disbtribution-worldwide.xlsx'
DOWNLOADED_SRC = 'COVID-19-geographic-disbtribution-worldwide.xlsx'
SHEET = 'COVID-19-geographic-disbtributi'

# Wide of trained sample
SAMPLE_TRAINED = 20
PERCENT_TRAINED = 70

# Respected field, CHANGE HERE !!!!!
DATE_FIELD = 'dateRep'
PREDICTED_FIELD = 'cases'
COUNTRY = 'United_Kingdom'

# MAIN PROGRAM
if __name__ == '__main__':
  country = COUNTRY
  if (len(sys.argv) > 1):
    country = ' '.join(sys.argv[1:])
    print('Using argv as country: ' + country)
  else:
    print('Using DEFAULT country: ' + country)
  
  print('\nReading data…\n')
  
  srcExcel = f'cov19-worldwide-{datetime.now().strftime("%Y-%m-%d")}.xls'
  # Try read Buffer File
  fileBuffExist = path.exists(srcExcel)
  if fileBuffExist:
    print(f'Reading data from local: {srcExcel}')
  else:
    try:
      print('Downloading…')
      link = LINK_SRC
      urllib.request.urlretrieve(link, srcExcel)
    except urllib.error.HTTPError as ex:
      print('Download FAILED')
      print(ex)

      print(f'\nUsing EMBEDDED SOURCE: {DOWNLOADED_SRC}')
      srcExcel = DOWNLOADED_SRC

  # Reading source file
  df = pd.read_excel(srcExcel, sheet_name=SHEET)
  df[DATE_FIELD] = pd.to_datetime(df[DATE_FIELD], dayfirst=True)

  # Create mask/filter based on country
  mask = df['countriesAndTerritories'] == country

  # Mask by country
  df = df.loc[mask]
  df = df.sort_values(by=DATE_FIELD)
  df = df.reset_index() # reset Index
  print(df.head())
  print(df.info())

  # prepare dataset and use only field defined value
  dataset = df.filter([PREDICTED_FIELD]).values

  # Create len of percentage training set
  trainingDataLen = math.ceil((len(dataset) * PERCENT_TRAINED) / 100)
  print('Size of trainingSet: ' + str(trainingDataLen))

  # Scale the dataset between 0 - 1
  scaler = MinMaxScaler(feature_range=(0, 1))
  scaledData = scaler.fit_transform(dataset)

  # Scaled trained data
  trainData = scaledData[:trainingDataLen , :]

  # Split into trained x and y
  xTrain = []
  yTrain = []
  for i in range(SAMPLE_TRAINED, len(trainData)):
    xTrain.append(trainData[i-SAMPLE_TRAINED:i , 0])
    yTrain.append(trainData[i , 0])

  # Convert trained x and y as numpy array
  xTrain, yTrain = np.array(xTrain), np.array(yTrain)
  print('x - y train shape: ' + str(xTrain.shape) + ' ' + str(yTrain.shape))

  # Reshape x trained data as 3 dimension array
  xTrain = np.reshape(xTrain, (xTrain.shape[0], xTrain.shape[1], 1))
  print('Expected x train shape: ' + str(xTrain.shape))
  print('')

  print('Processing the LSTM model...\n')

  # Build LSTM model
  model = Sequential()
  model.add(LSTM(10, input_shape=(xTrain.shape[1], 1)))
  model.add(Dense(1, activation='linear'))

  # Compile model
  model.compile(optimizer='adam', loss='mean_squared_error')

  # Train the model
  model.fit(xTrain, yTrain, shuffle=False, epochs=300)

  print('\nDone Processing the LSTM model...')

  # Prepare testing dataset
  testData = scaledData[trainingDataLen - SAMPLE_TRAINED: , :]

  # Create dataset test x and y
  xTest = []
  yTest = dataset[trainingDataLen: , :]
  for i in range(SAMPLE_TRAINED, len(testData)):
    xTest.append(testData[i - SAMPLE_TRAINED:i, 0])

  # Convert test set as numpy array
  xTest = np.array(xTest)

  # Reshape test set as 3 dimension array
  xTest = np.reshape(xTest, (xTest.shape[0], xTest.shape[1], 1))

  # Models predict price values
  predictions = model.predict(xTest)
  predictions = scaler.inverse_transform(predictions)

  # Get root mean square (RMSE)
  rmse = np.sqrt(np.mean(predictions - yTest) ** 2)
  print('\nRoot mean square (RMSE):' + str(rmse))

  # Add prediction for Plot
  train = df.loc[:trainingDataLen, [DATE_FIELD, PREDICTED_FIELD] ]
  valid = df.loc[trainingDataLen:, [DATE_FIELD, PREDICTED_FIELD] ]
  print('validLength: {}, predictionLength: {}'.format(len(valid), len(predictions)))

  # Create dataframe prediction
  dfPrediction = pd.DataFrame(predictions, columns = ['predictions'])

  # Reset the index
  valid = valid.reset_index()
  dfPrediction = dfPrediction.reset_index()

  # Merge valid data and prediction data
  valid = pd.concat([valid, dfPrediction], axis=1)

  # Visualize
  fig, ax = plt.subplots(num=f'{country} prediction {PREDICTED_FIELD}')
  plt.subplots_adjust(bottom=0.2)

  # Add graph info
  ax.set_title(f'With RMSE: {rmse:,.2f}')
  ax.set_xlabel(DATE_FIELD, fontsize=14)
  ax.set_ylabel(PREDICTED_FIELD, fontsize=14)
  ax.grid(linestyle='-', linewidth='0.5', color='gray')

  # plot trained data
  ax.plot(train[DATE_FIELD], train[PREDICTED_FIELD])

  # plot actual and predictions
  ax.plot(valid[DATE_FIELD], valid[[PREDICTED_FIELD, 'predictions']])

  # add legend
  ax.legend(['Train', 'Actual', 'Prediction'], loc='lower right')

  # finally show graph
  plt.show()

  print('')
  print('Exiting…')
  print('')