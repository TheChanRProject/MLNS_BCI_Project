!pip install ann_visualizer
!pip install graphviz
import graphviz
import keras
from keras.models import Sequential
from keras.layers import Dense
from ann_visualizer.visualize import ann_viz
network = Sequential()

network.add(Dense(units=100,activation='relu',kernel_initializer='uniform',input_dim=189))
network.add(Dense(units=2, activation='relu', kernel_initializer='uniform'))
ann_viz(network, title="Our Neural Network")
