import random
import json
import pickle
import numpy as np
import nltk

from nltk.stem import WordNetLemmatizer  # to turn words to base words(lemmatizer)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())  # json to text

words = []
classes = []
documents = []
punctuations = ['?', '!', '.', ',']

for intent in intents['intents']:
    for j in intent['patterns']:
        word = nltk.word_tokenize(j)  # tokenize words, split them into individual words to compare
        words.extend(word)
        documents.append((word, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# lemmatizing words
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in punctuations]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# initializing training data
training = []
output_empty = [0] * len(classes)
for document in documents:
    bag = []
    pattern = document[0]
    pattern = [lemmatizer.lemmatize(word.lower()) for word in pattern]
    for word in words:
        bag.append(1) if word in pattern else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y), epochs=250, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

print("done")
