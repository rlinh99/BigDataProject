# from __future__ import print_function
# import numpy as np
# import spacy
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import validation
# from data_loader import train_data, test_data
# from keras.preprocessing.text import Tokenizer
# from keras.utils.np_utils  import to_categorical
# from sklearn.model_selection import train_test_split
#
# from keras.preprocessing import sequence
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Activation, GlobalMaxPooling1D, Conv1D, \
#     MaxPooling1D
#
# df = pd.concat([train_data, test_data])
# df.rename(columns={'recordId': 'id', 'drugName': 'drug_name', 'usefulCount': 'useful_count'}, inplace=True)
# df.dropna(inplace=True)
# df['review_len']=df['reviewComment'].apply(len)
# nlp = spacy.load('en_core_web_sm')
#
# df['token_review'] = df['reviewComment'].apply(nlp, disable=['parser','ner','tagger'])
# stop_words = nlp.Defaults.stop_words
# punctuations = '  ...!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\r\n\ufeff1\r\n\r\n&#039;'
# def feature_engineered_tokenized_review(tr):
#     test = [token.lemma_.lower() for token in tr if token.lemma_.lower() not in stop_words and token.lemma_.lower() not in punctuations and token.lemma_ not in ['-PRON-']]
#     return test
# df['fe_treview'] = df['token_review'].apply(feature_engineered_tokenized_review)
#
# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(df['fe_treview'])
# sequences = tokenizer.texts_to_sequences(df['fe_treview'])
#
# maxlen = 125
# x = sequence.pad_sequences(sequences, maxlen=maxlen)
#
# y_cat = to_categorical(df['rating'], num_classes=6)
# X = x
# y = y_cat
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# print(X_train)
# # model 1
# vocabulary_size = len(tokenizer.word_counts)
# # batch_size = 64
# # model1 = Sequential()
# # model1.add(Embedding(vocabulary_size +1, 64, input_length=maxlen))
# # model1.add(Bidirectional(LSTM(64)))
# # model1.add(Dense(64, activation='relu'))
# # model1.add(Dense(6,activation='softmax'))
# # model1.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
# # print('Train...')
# # model1.fit(X_train, y_train,epochs=10)
#
#
#
# max_features = vocabulary_size+1
# #maxlen = 100
# embedding_size = 128
#
# # Convolution
# kernel_size = 5
# filters = 64
# pool_size = 4
#
# # LSTM
# lstm_output_size = 70
#
# # Training
# batch_size = 30
# epochs = 15
# model3 = Sequential()
# model3.add(Embedding(max_features, embedding_size, input_length=maxlen))
# model3.add(Dropout(0.25))
# model3.add(Conv1D(filters,kernel_size, padding='valid',activation='relu',strides=1))
# model3.add(MaxPooling1D(pool_size=pool_size))
# model3.add(LSTM(lstm_output_size))
# model3.add(Dense(6))
# model3.add(Activation('sigmoid'))
#
# predicted = model3.predict(X_test, batch_size=batch_size)
# class_label = predicted.argmax(axis=-1)
# class_label2 = predicted.argmax(axis = 1)
# test_label = np.argmax(y_test, axis=-1)
# test_label2 = np.argmax(y_test, axis=1)
#
# print(class_label)
# print(class_label2)
# print(test_label)
# print(test_label2)
# a, b = validation.get_f1_score(test_label.ravel(), class_label.ravel())
# print(a)
# print(b)