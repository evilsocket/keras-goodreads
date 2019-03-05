import os
import pandas as pd 

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

data_split = 0.33
max_words  = 5000
embed_dim  = 128
lstm_out   = 196
batch_size = 32

corpus    = []
ratings   = []
to_eval   = []

data = pd.read_csv('./goodreads_library_export.csv')

for no, row in data.iterrows():
    book_id  = row['Book Id']
    rating   = row['My Rating']
    filename = os.path.join('data', '%s.txt' % book_id)

    if os.path.exists(filename):
        with open(filename) as fp:
            text = fp.read()
    
    if rating == 0:
        to_eval.append((row, text))
    else:
        corpus.append(text)
        ratings.append(rating)

tokenizer = Tokenizer(num_words=max_words, split=' ')
tokenizer.fit_on_texts(corpus)

X = tokenizer.texts_to_sequences(corpus)
X = pad_sequences(X)
Y = to_categorical(ratings)

model = Sequential()
model.add(Embedding(max_words, embed_dim, input_length = X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense( Y.shape[1], activation='softmax' ) )
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])

if not os.path.exists('model.h5'):
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = data_split)

    earlyStop = EarlyStopping(monitor = 'val_acc', min_delta=0.0000001, patience = 10, mode = 'auto')

    try:
        model.fit(X_train, Y_train, epochs=50, batch_size=batch_size, verbose = 2)
    except KeyboardInterrupt:
        pass

    model.save('model.h5')
else:
    model.load_weights('model.h5')

print("\n\n")

for data in to_eval:
    row, text = data
    x = tokenizer.texts_to_sequences([text])
    x = pad_sequences(x, maxlen=654)
    y = model.predict_classes(x)[0]
   
    left = 5 - y
    s = (y * '*') + (left * '-') 

    print("%s: %s" % (s, row['Title']))






