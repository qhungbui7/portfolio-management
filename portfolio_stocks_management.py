# -*- coding: utf-8 -*-
"""portfolio_stocks_management.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/12uBXsj6-fkfG5ZkLfu0kEOzRjzyQVfv5
"""

from google.colab import drive
drive.mount('ML',force_remount = True)

! nvidia-smi

import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
from tensorflow import keras
from collections import deque

import pandas as pd
from pandas_datareader import data as pdr
import datetime
import random
import seaborn as sns
import matplotlib.pyplot as plt

# resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
# tf.config.experimental_connect_to_cluster(resolver)
# # This is the TPU initialization code that has to be at the beginning.
# tf.tpu.experimental.initialize_tpu_system(resolver)
# print("All devices: ", tf.config.list_logical_devices('TPU'))

# strategy = tf.distribute.TPUStrategy(resolver)

!pip install stockstats

import stockstats
from stockstats import StockDataFrame as Sdf

NUM_SHARES = 20
NUM_STOCKS = 1
ACCOUNT = 1000 # 1000 Dollars
SIZE_ACTION_SPACE = ( 2 * NUM_SHARES + 1 ) ** NUM_STOCKS 
BATCH_SIZE = 256
EPOCHS_TRAIN = 200
# EPOCHS_VALIDATION = 100
EPSILON = 1.0
MAX_EPSILON = 1.0
MIN_EPSILON = 0.01
DECAY = 0.01
RANDOM_SEED = 5 
MIN_REPLAY = 1000
TRADE = np.array(range(-NUM_SHARES // 2,NUM_SHARES //2 + 1)).astype('float64')
NUM_USING_STOCKS = 1

"""Directory

Apple stock is used for illustrating
"""

AAPL_dir = '/content/ML/MyDrive/ML/PortfolioManagement/AAPL.csv' # Apple
AMD_dir = '/content/ML/MyDrive/ML/PortfolioManagement/AMD.csv'   # Advanced Micro Devices
AMZN_dir = '/content/ML/MyDrive/ML/PortfolioManagement/AMZN.csv' # Amazon
BABA_dir = '/content/ML/MyDrive/ML/PortfolioManagement/BABA.csv' # Alibaba
FB_dir = '/content/ML/MyDrive/ML/PortfolioManagement/FB.csv'     # Facebook
GOOG_dir = '/content/ML/MyDrive/ML/PortfolioManagement/GOOG.csv' # Google
INTC_dir = '/content/ML/MyDrive/ML/PortfolioManagement/INTC.csv' # Intel
MSFT_dir = '/content/ML/MyDrive/ML/PortfolioManagement/MSFT.csv' # Microsoft
NFLX_dir = '/content/ML/MyDrive/ML/PortfolioManagement/NFLX.csv' # Netflix
NVDA_dir = '/content/ML/MyDrive/ML/PortfolioManagement/NVDA.csv' # Nvidia
RACE_dir = '/content/ML/MyDrive/ML/PortfolioManagement/RACE.csv' # Ferrari
TSLA_dir = '/content/ML/MyDrive/ML/PortfolioManagement/TSLA.csv' # Tesla

"""### Choose some potential stocks for diversity properties.

I use Apple's, Amazone's, Google's, Microsoft's, Ferrari's stocks for training and the others for testing, some of which may will be a validation test.
"""

AAPL_raw = pd.read_csv(AAPL_dir) 
# AMD_raw = pd.read_csv(AMD_dir) 
AMZN_raw = pd.read_csv(AMZN_dir) 
# BABA_raw = pd.read_csv(BABA_dir) 
# FB_raw = pd.read_csv(FB_dir) 
GOOG_raw = pd.read_csv(GOOG_dir)  
MSFT_raw = pd.read_csv(MSFT_dir) 
NFLX_raw = pd.read_csv(NFLX_dir) # Test the model in this company 
# NVDA_raw = pd.read_csv(NVDA_dir) 
# RACE_raw = pd.read_csv(RACE_dir) 
INTC_raw = pd.read_csv(INTC_dir)
# TSLA_raw = pd.read_csv(TSLA_dir)

AAPL_raw

AAPL_raw.describe()

"""Amazone corp dataset is very suitable to be a scaler"""

AAPL_raw.info()

"""**Scale multiple datasets and use scaler of Apple stock is the main scaler**"""

AAPL_raw['Date'] = pd.to_datetime(AAPL_raw['Date'], format = '%Y-%m-%d %H:%M:%S') # string to datetime
AAPL_raw['Date'] = AAPL_raw['Date'].map(datetime.datetime.timestamp)
AMZN_raw['Date'] = pd.to_datetime(AMZN_raw['Date'], format = '%Y-%m-%d %H:%M:%S') # string to datetime
AMZN_raw['Date'] = AMZN_raw['Date'].map(datetime.datetime.timestamp)
GOOG_raw['Date'] = pd.to_datetime(GOOG_raw['Date'], format = '%Y-%m-%d %H:%M:%S') # string to datetime
GOOG_raw['Date'] = GOOG_raw['Date'].map(datetime.datetime.timestamp)
MSFT_raw['Date'] = pd.to_datetime(MSFT_raw['Date'], format = '%Y-%m-%d %H:%M:%S') # string to datetime
MSFT_raw['Date'] = MSFT_raw['Date'].map(datetime.datetime.timestamp)
INTC_raw['Date'] = pd.to_datetime(INTC_raw['Date'], format = '%Y-%m-%d %H:%M:%S') # string to datetime
INTC_raw['Date'] = INTC_raw['Date'].map(datetime.datetime.timestamp)

NFLX_raw['Date'] = pd.to_datetime(NFLX_raw['Date'], format = '%Y-%m-%d %H:%M:%S') # string to datetime
NFLX_raw['Date'] = NFLX_raw['Date'].map(datetime.datetime.timestamp)

"""### Experiment with demo of a correlation maps of Apple's stock """

stock = Sdf.retype(pd.read_csv(AAPL_dir))
macd = stock['macd']
rsi = stock['rsi_6']
cci = stock['cci']
adx = stock['adx']


cr = stock['cr']
kdjk = stock['kdjk']
wr = stock['wr_10']
dma = stock['dma']
trix = stock['trix']
vr = stock['vr']



macd.index = range(0,len(macd))
rsi.index = range(0,len(rsi))
cci.index = range(0,len(cci))
adx.index = range(0,len(adx))


cr.index = range(0,len(cr))
kdjk.index = range(0,len(kdjk))
wr.index = range(0,len(wr))
dma.index = range(0,len(dma))
trix = range(0,len(trix))
vr = range(0,len(vr))

AAPL_experiment = pd.DataFrame({
    # 'time' : AAPL_raw['Date'], 
    'open' : AAPL_raw['Open'],
    'high' : AAPL_raw['High'],
    'low' : AAPL_raw['Low'],
    'close' : AAPL_raw['Close'],
    'adjclose'  : AAPL_raw['Adj Close'], 
    'volume' : AAPL_raw['Volume'].astype('float64'),
    
    'cr' : cr,
    'kdjk' : kdjk,
    'wr' : wr,
    'dma' : dma,
    'trix' : trix, 
    'vr' : vr,

    'macd' : macd, 
    'rsi' : rsi,
    'cci' : cci,
    'adx' : adx

    
    })
AAPL_experiment['cr'].loc[0] = 0
AAPL_experiment['rsi'].loc[0] = 0 
AAPL_experiment['cci'].loc[0] = 0 
AAPL_experiment['adx'].loc[0] = 0 
AAPL_experiment

corrmat = AAPL_experiment.corr()
top_corr_features = corrmat.index
plt.figure(figsize = (11, 11))
sns_plot = sns.heatmap(AAPL_experiment[top_corr_features].corr(), annot=True, cmap="Blues");

"""As you can see, from the heatmap we can conclude that there are a couple of group of nearly linear features :

1. Open high low close adj close 
2. dima macd
3. rsi kdjk cci

Therefore, I decide to eliminate some feature such as dima, open, high, low, close ( trix and vr too since they seems useless ). In the other hand, I keep some 9 different features, namely adjclose, volume, cr, kdjk, wr, macd, rsi, cci, adx.Time series properties will be replaced by use batching method or add more input size in the neural network ( will edit this paragraph later )

#### Prepare datasets for training

Use this module to calculate some financial stock indexs. For more detail, visit https://pypi.org/project/stockstats/
"""

stock = Sdf.retype(pd.read_csv(AAPL_dir)) # Apple 
macd = stock['macd']
rsi = stock['rsi_6']
cci = stock['cci']
adx = stock['adx']

cr = stock['cr']
kdjk = stock['kdjk']
wr = stock['wr_10']

macd.index = range(0,len(macd))
rsi.index = range(0,len(rsi))
cci.index = range(0,len(cci))
adx.index = range(0,len(adx))
cr.index = range(0,len(cr))
kdjk.index = range(0,len(kdjk))
wr.index = range(0,len(wr))


AAPL = pd.DataFrame({
    'aclose'  : AAPL_raw['Adj Close'], 
    #'vol' : AAPL_raw['Volume'].astype('float64'),
    
    'cr' : cr,
    'kdjk' : kdjk,
    'wr' : wr,

    'macd' : macd, 
    'rsi' : rsi,
    'cci' : cci,
    'adx' : adx
})
AAPL = AAPL.replace([np.inf, -np.inf], np.nan).dropna(axis=0) # drop na and inf
AAPL.index = range(0,len(AAPL))
AAPL

stock = Sdf.retype(pd.read_csv(AMZN_dir)) # Amazone 
macd = stock['macd']
rsi = stock['rsi_6']
cci = stock['cci']
adx = stock['adx']

cr = stock['cr']
kdjk = stock['kdjk']
wr = stock['wr_10']

macd.index = range(0,len(macd))
rsi.index = range(0,len(rsi))
cci.index = range(0,len(cci))
adx.index = range(0,len(adx))
cr.index = range(0,len(cr))
kdjk.index = range(0,len(kdjk))
wr.index = range(0,len(wr))


AMZN = pd.DataFrame({
    'aclose'  : AMZN_raw['Adj Close'], 
    #'vol' : AMZN_raw['Volume'].astype('float64'),
    
    'cr' : cr,
    'kdjk' : kdjk,
    'wr' : wr,

    'macd' : macd, 
    'rsi' : rsi,
    'cci' : cci,
    'adx' : adx
})
AMZN = AMZN.replace([np.inf, -np.inf], np.nan).dropna(axis=0) # drop na and inf
AMZN.index = range(0,len(AMZN))

stock = Sdf.retype(pd.read_csv(GOOG_dir)) # Google
macd = stock['macd']
rsi = stock['rsi_6']
cci = stock['cci']
adx = stock['adx']

cr = stock['cr']
kdjk = stock['kdjk']
wr = stock['wr_10']

macd.index = range(0,len(macd))
rsi.index = range(0,len(rsi))
cci.index = range(0,len(cci))
adx.index = range(0,len(adx))
cr.index = range(0,len(cr))
kdjk.index = range(0,len(kdjk))
wr.index = range(0,len(wr))


GOOG = pd.DataFrame({
    'aclose'  : GOOG_raw['Adj Close'], 
    #'vol' : GOOG_raw['Volume'].astype('float64'),
    
    'cr' : cr,
    'kdjk' : kdjk,
    'wr' : wr,

    'macd' : macd, 
    'rsi' : rsi,
    'cci' : cci,
    'adx' : adx
})
GOOG = GOOG.replace([np.inf, -np.inf], np.nan).dropna(axis=0) # drop na and inf
GOOG.index = range(0,len(GOOG))

stock = Sdf.retype(pd.read_csv(MSFT_dir)) # Microsoft
macd = stock['macd']
rsi = stock['rsi_6']
cci = stock['cci']
adx = stock['adx']

cr = stock['cr']
kdjk = stock['kdjk']
wr = stock['wr_10']

macd.index = range(0,len(macd))
rsi.index = range(0,len(rsi))
cci.index = range(0,len(cci))
adx.index = range(0,len(adx))
cr.index = range(0,len(cr))
kdjk.index = range(0,len(kdjk))
wr.index = range(0,len(wr))


MSFT = pd.DataFrame({
    'aclose'  : MSFT_raw['Adj Close'], 
    #'vol' : MSFT_raw['Volume'].astype('float64'),
    'cr' : cr,
    'kdjk' : kdjk,
    'wr' : wr,

    'macd' : macd, 
    'rsi' : rsi,
    'cci' : cci,
    'adx' : adx
})
MSFT = MSFT.replace([np.inf, -np.inf], np.nan).dropna(axis=0) # drop na and inf
MSFT.index = range(0,len(MSFT))

stock = Sdf.retype(pd.read_csv(INTC_dir)) # Intel
macd = stock['macd']
rsi = stock['rsi_6']
cci = stock['cci']
adx = stock['adx']

cr = stock['cr']
kdjk = stock['kdjk']
wr = stock['wr_10']

macd.index = range(0,len(macd))
rsi.index = range(0,len(rsi))
cci.index = range(0,len(cci))
adx.index = range(0,len(adx))
cr.index = range(0,len(cr))
kdjk.index = range(0,len(kdjk))
wr.index = range(0,len(wr))


INTC = pd.DataFrame({
    'aclose'  : INTC_raw['Adj Close'], 
    #'vol' : INTC_raw['Volume'].astype('float64'),
    
    'cr' : cr,
    'kdjk' : kdjk,
    'wr' : wr,

    'macd' : macd, 
    'rsi' : rsi,
    'cci' : cci,
    'adx' : adx
})
INTC = INTC.replace([np.inf, -np.inf], np.nan).dropna(axis=0) # drop na and inf
INTC.index = range(0,len(INTC))

stock = Sdf.retype(pd.read_csv(NFLX_dir)) # Intel
macd = stock['macd']
rsi = stock['rsi_6']
cci = stock['cci']
adx = stock['adx']

cr = stock['cr']
kdjk = stock['kdjk']
wr = stock['wr_10']

macd.index = range(0,len(macd))
rsi.index = range(0,len(rsi))
cci.index = range(0,len(cci))
adx.index = range(0,len(adx))
cr.index = range(0,len(cr))
kdjk.index = range(0,len(kdjk))
wr.index = range(0,len(wr))


NFLX = pd.DataFrame({
    'aclose'  : NFLX_raw['Adj Close'], 
    #'vol' : NFLX_raw['Volume'].astype('float64'),
    
    'cr' : cr,
    'kdjk' : kdjk,
    'wr' : wr,

    'macd' : macd, 
    'rsi' : rsi,
    'cci' : cci,
    'adx' : adx
})
NFLX = NFLX.replace([np.inf, -np.inf], np.nan).dropna(axis=0) # drop na and inf
NFLX.index = range(0,len(NFLX))

"""Scale the data and the Standard Scaler of Amazone is the criteria of the other dataset """

scalarMean =  AMZN['aclose'].mean()
scalarStd = AMZN['aclose'].std()
scaled_account = (ACCOUNT - scalarMean) / scalarStd
scaled_account

cols = AAPL.columns

# All dataset is the same 
for col in cols:
  standardMean = AMZN[col].mean()
  standardStd = AMZN[col].std()
  AAPL[col] = AAPL[col].apply(lambda x : (x - standardMean) / standardStd)
AAPL

# All dataset is the same 
for col in cols:
  standardMean = AMZN[col].mean()
  standardStd = AMZN[col].std()
  GOOG[col] = GOOG[col].apply(lambda x : (x - standardMean) / standardStd)
GOOG

# All dataset is the same 
for col in cols:
  standardMean = AMZN[col].mean()
  standardStd = AMZN[col].std()
  MSFT[col] = MSFT[col].apply(lambda x : (x - standardMean) / standardStd)  
MSFT

# All dataset is the same 
for col in cols:
  standardMean = AMZN[col].mean()
  standardStd = AMZN[col].std()
  INTC[col] = INTC[col].apply(lambda x : (x - standardMean) / standardStd)  
INTC

# All dataset is the same 
for col in cols:
  standardMean = AMZN[col].mean()
  standardStd = AMZN[col].std()
  NFLX[col] = NFLX[col].apply(lambda x : (x - standardMean) / standardStd)
NFLX

# All dataset is the same 
for col in cols:
  standardMean = AMZN[col].mean()
  standardStd = AMZN[col].std()
  AMZN[col] = (AMZN[col]-standardMean) / standardStd  
AMZN

AAPL_dataset = tf.data.Dataset.from_tensor_slices(AAPL.values)
AAPL_dataset

for feat in AAPL_dataset.take(5) : 
  print('Cols : {}'.format(feat))

"""**Core**

Defined constant
"""

TRADE

def observe(balance, shares, data, index):
  price = data['aclose'][index]
  return np.array([balance, price, shares, data['macd'][index], data['rsi'][index], data['cci'][index], data['adx'][index], data['cr'][index], data['kdjk'][index], data['wr'][index]])

"""21 action : Sell, Hold, Buy an amount of shares"""

#@title Old version
# def agent(state_shape, action_shape):
#   learning_rate  = 0.001
#   model = keras.Sequential() ; 
#   initializer = tf.keras.initializers.Zeros() 
#   model.add(keras.layers.InputLayer(shape = (7,)))
#   model.add(keras.layers.Dense(24, activation = 'relu', kernel_initializer=initializer))
#   model.add(keras.layers.Dense(32,activation='relu', kernel_initializer=initializer))
#   model.add(keras.layers.Dense(21,activation='linear', kernel_initializer=initializer))
#   model.compile(loss=tf.keras.losses.Huber(),optimizer=tf.keras.optimizers.SGD(momentum=0.6, learning_rate=learning_rate), metrics = ['accuracy'])
#   return model

#@title Old version
# def train(replay_memory, model, done): #   replay_memory.append([observation, action, new_observation, reward, done])
#   learning_rate = 0.7
#   discount = 0.8
#   # print(observation)
#   print('Training')
#   if len(replay_memory) < MIN_REPLAY:
#     print('Not enough')
#     return 
  
#   mini_batch = random.sample(replay_memory, BATCH_SIZE) # dont need time order any more, random for avoiding overfitting

#   current_state = np.array([replay[0] for replay in mini_batch]) # 64 x 7  current state - all observation in batch
#   current_qslist =  model.predict(current_state) # 64 x 21  predict a list of Q_value of after action from each current_state
#   new_current_state = np.array([replay[2] for replay in mini_batch]) # 64 x 7 get the state after an action by using new_observation
#   future_qslist = model.predict(new_current_state) # 64 x 21 get the Q_values of future env for calculation of the formula above

#   X = np.array([observe(account_scaled,0, AAPL, 0)]) 
#   y = np.array([TRADE]) 
#   train = [] 

#   for index, (observation, action, new_observation, reward, done) in enumerate(mini_batch) : 
#     if not done : 
#       max_future_q = reward + discount * np.max(future_qslist[index]) # 1 x 1 calculate the optimal future value, base case of recursive formula above
#     else : 
#       max_future_q = reward # if done then there won't be a future value
#     current_qs = current_qslist[action] # 21 x 1 
#     current_qs[action] = current_qs[action] + learning_rate * (reward + discount * max_future_q) # 1 x 1 wish the network will return this function 
#     X = np.concatenate((X,np.array([observation])))
#     y = np.concatenate((y,np.array([current_qs])))
#   X = X[1:]
#   y = y[1:]

#   train = tf.data.Dataset.from_tensor_slices((X,y))

#   train = train.prefetch(2)
#   train = train.batch(BATCH_SIZE)
#   model.fit(train, batch_size = BATCH_SIZE, verbose = 1, shuffle = True)

def more_one_step(observation, action, net_worth): # out : observation : [balance, price, shares, macd, rsi, cci, adx]
  done = False
  new_observation = observation.copy()
  new_observation[0] = new_observation[0] - action * new_observation[1] # price, buy *action shares
  new_observation[2] += action
  reward =  (new_observation[0] + new_observation[1] * new_observation[2]) - net_worth # new net_worth - old net_worth
  net_worth = new_observation[0] + new_observation[2] * new_observation[1] # calculate new net_worth
  if (net_worth <= 0) :
     done = True # enable some credit 
  return new_observation, reward, new_observation[2], done, net_worth
  # cal the money time after time

huber_loss = keras.losses.Huber()
optimizer = keras.optimizers.Adam(learning_rate = 0.001)
# @tf.function
def Actor_Critic_Network(inputs_size, actions_size):
    init = keras.initializers.HeUniform()
    inputs_layer = keras.layers.Input(shape=(inputs_size,))
    layer_1 = keras.layers.Dense(128, activation='tanh', kernel_initializer = init)(inputs_layer)
    batch_norm_1 = keras.layers.BatchNormalization()(layer_1)
    layer_2 = keras.layers.Dense(256, activation='tanh', kernel_initializer = init)(batch_norm_1)
    batch_norm_2 = keras.layers.BatchNormalization()(layer_2)
    actions_layer = keras.layers.Dense(actions_size, activation='softmax')(batch_norm_2)
    critic_layer = keras.layers.Dense(1)(batch_norm_2)
    model = keras.Model(inputs = inputs_layer, outputs = [actions_layer,critic_layer]) 
    model.summary()
    return model

model = Actor_Critic_Network(10,21)
# @tf.function
def executionTraining(num, dataset):
  gamma = 0.9
  # tape.watch(model.trainable_variables)
  for epoch in range(EPOCHS_TRAIN):
    print('Dataset ', num, ' Epoch ' , epoch )
    net_worth = scaled_account 
    account = scaled_account
    done = False 
    row = 0
    shares = np.zeros((NUM_USING_STOCKS))
    value = [] 
    ret = [] 
    actions_logs = [] 
    critic_logs = []
    rewards = []
    epoch_reward  = 0  
    with tf.GradientTape() as tape : 
      while not done :
        for row in range(len(dataset)): # limit the number of steps too 
          observation_1 = observe(account,shares[0], AAPL, row) 
          input = tf.convert_to_tensor(observation_1)
          input = tf.expand_dims(input, 0) # batch dimension
          prob_actions_1, critic_1 = model(input)
          critic_logs.append(critic_1)
          probabilities_actions = prob_actions_1

          action_1 = np.random.choice(21, p = np.squeeze(probabilities_actions)) # wrong probabilities 
          actions_logs.append(tf.math.log(probabilities_actions[0][action_1]))

          new_observation_1, reward, shares[0], done, new_net_worth = more_one_step(observation_1,action_1,net_worth)
          account = new_observation_1[0]

          rewards.append(new_net_worth - net_worth) # greedy method


          net_worth = new_net_worth 

          if done: 
            break   
      discounted_sum = 0 
      for r in rewards[::-1] : 
        discounted_sum = r + gamma * discounted_sum
        ret.append(discounted_sum)
      history = zip(actions_logs,critic_logs,ret)     
      actor_loss = [] 
      critic_loss = []
      for log_prob, value, ret in history  : 
        diff = ret - value
        actor_loss.append(-log_prob * diff)  
        critic_loss.append(huber_loss(tf.expand_dims(value, 0), tf.expand_dims(ret, 0)))
      sum_loss = sum(actor_loss) + sum(critic_loss)
      grads = tape.gradient(sum_loss, model.trainable_variables)
      optimizer.apply_gradients(zip(grads,model.trainable_variables))

# strategy.run(executionTraining,args=())
executionTraining(0, AAPL)
executionTraining(1, GOOG)
# executionTraining(2, AMZN)
executionTraining(3, INTC)
executionTraining(4, MSFT)

model_dir = '/content/ML/MyDrive/ML/PortfolioManagement/model/model_200e_1.hdf5'

def saveModel(model, dir): 
  model.save(dir)
saveModel(model, model_dir)

model = keras.models.load_model(model_dir, compile = False)

print(scaled_account * scalarStd + scalarMean)
print(scaled_account)
print(scalarStd)
print(scalarMean)

def testingExecution(dataset): 
  gamma = 0.9 
  account = scaled_account
  net_worth = scaled_account
  net_worth_history = [net_worth]
  critic_logs = [] 
  shares = 0 
  shares_logs = []
  done = False
  while not done : 
    for row in range(len(dataset)) : 
      shares_logs.append(shares)
      observation = observe(account, shares, dataset, row) 
      inputs = observation.reshape([1,10])
      prob_actions, critic = model.predict(inputs)
      critic_logs.append(critic)
      # action = np.random.choice(21, p = prob_actions)
      action = np.argmax(prob_actions) # no exploration
      new_observation, reward, shares, done, new_net_worth = more_one_step(observation,action,net_worth) 
      net_worth_history.append(new_net_worth)
      net_worth = new_net_worth  

    if done or row == len(dataset)-1: 
      break 
  return np.array(net_worth_history), np.array(critic_logs)
v_net_worth, v_critic_logs = testingExecution(NFLX)

# Visualization the net_worth_history
v_net_worth = v_net_worth * scalarStd + scalarMean # reverse the scaler
print(v_net_worth)
def visualization_net_worth(v_net_worth):
  plt.plot( range(0,len(v_net_worth)),v_net_worth, label='Net worth')
  plt.xlabel('Time')
  plt.ylabel('Rate')
  plt.legend(loc='upper left')
visualization_net_worth(v_net_worth)

def visualization_shares(v_net_worth):
  plt.plot( range(0,len(v_net_worth)),v_net_worth, label='Net worth')
  plt.xlabel('Time')
  plt.ylabel('Rate')
  plt.legend(loc='upper left')
visualization_net_worth(v_net_worth)


# v_critic_logs = v_critic_logs.squeeze()
# # v_critic_logs = v_critic_logs * scalarStd + scalarMean
# r_AMZN = AMZN['aclose'].to_numpy() 
# # r_AMZN = r_AMZN * scalarStd + scalarMean
# print(v_critic_logs)
# print(r_AMZN)
# def visualization_critic_logs(v_critic_logs, adj): 
#   plt.plot(range(0,len(v_critic_logs)), v_critic_logs, label='Prediction')
#   plt.plot(range(0,len(v_critic_logs)), adj, label='Price')
#   # plt.plot(history.history['val_mean_absolute_error'], label = 'val_mean_absolute_error')
#   plt.xlabel('Time')
#   plt.ylabel('US Dollar')
#   # plt.ylim([0, 5])
#   plt.legend(loc='upper left')   
# visualization_critic_logs(v_critic_logs, r_AMZN )

"""https://arxiv.org/pdf/1602.01783.pdf


Policy gradient of mu = The mean of (sigma from initial state until the game is done of(Q value of each state * old agent choose action for each state ))

https://arxiv.org/pdf/1509.02971.pdf
"""

#@title Old version
# def main(): # normaly calculate and trace the action when do an action, need to normalize the money, scale the money by the time starting the process
  
#     tf.random.set_seed(RANDOM_SEED)
#     epsilon = EPSILON
#     decay=0.01
#     observation = observe(account_scaled,0, AAPL, 0) # in : balance, shares, data, index   out : np.array[balance, price, shares, macd, rsi, cci, adx]
#     model = agent(observation.shape,TRADE.shape[0])
#     target_model = agent(observation.shape,TRADE.shape[0])
#     target_model.set_weights(model.get_weights())
#     replay_memory = deque(maxlen=2000000)
#     steps_to_target = 0 
#     for epoch in range(EPOCHS_TRAIN):
#       observation = observe(account_scaled,0, AAPL, 0) 
#       balance = account_scaled
#       net_worth = balance
#       price = AAPL['adjclose'][0]
#       shares = 0 
#       done = False
#       j = 0 
#       print('Epoch : ',epoch)
#       while j < len(AAPL_dataset) and not done: 
#         steps_to_target += 1  
#         rand = np.random.rand() 
#         print(rand, epsilon)
#         observation = observe(observation[0],observation[2],AAPL,j) # update the price, get the balance, shares from previous step new observation 
#         # print(observation)
#         # print(epsilon, j)
#         if rand <= epsilon:
#           # explore
#           action = random.randrange(0,len(TRADE)) # random arg
#         else : 
#           # exploit
#           # print(observation)
#           obs = observation.reshape([1,len(observation)])
#           # print(obs.shape)
#           predicted = model.predict(obs).flatten()
#           action = np.argmax(predicted)
#         new_observation, reward, share, done, net_worth = more_one_step(observation, action, net_worth) # Do the action 
#         # print(usdollar(net_worth))
#         replay_memory.append([observation, action, new_observation, reward, done]) # Save the action
#         # print(len(replay_memory))
#         # print(net_worth) # wrong unit
#         if ( j % 64 == 0 or done):
#           train(replay_memory,model, done)
#         j+=1 
#         print(usdollar(net_worth))
#         observation = new_observation
#         if done:
#           if steps_to_target >= 200 :
#             print('Copy weights from main network to target network')
#             target_model.set_weights(model.get_weights())
#       epsilon = MIN_EPSILON + ( MAX_EPSILON - MIN_EPSILON) * np.exp(-decay * epoch)
# main()