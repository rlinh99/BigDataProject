from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D



# lstm = Sequential()
# lstm.add(Embedding(vocabulary_size, 64, input_length=maxlen))
# lstm.add(Bidirectional(LSTM(64)))
# lstm.add(Dense(64, activation='relu'))
# lstm.add(Dense(11,activation='softmax'))
# lstm.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])