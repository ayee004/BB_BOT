import random
import json
import pickle
import numpy as np
import nltk

from nltk.stem import WordNetLemmatizer  # to turn words to base words(lemmatizer)
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())  # json to text

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

def clean_up_sentence(input):
    sw = nltk.word_tokenize(input)
    sw = [lemmatizer.lemmatize(word.lower()) for word in sw]
    return sw

def bag_of_words(input, words, show_details=True):
    s = clean_up_sentence(input)
    bag = [0]*len(words)
    for w in s:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return(np.array(bag))

def predict_class(input):
    bow = bag_of_words(input, words)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

print("The bot is now active")

while True:
    message = input("")
    ints = predict_class(message)
    res = getResponse(ints, intents)
    print(res)