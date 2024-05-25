import random
import json
import pickle
import numpy as np

import nltk
import tensorflow as ts


ERROR_THRESHOLD = .25

# Ensure the NLTK resources is here
nltk.download("popular", quiet=True)


class Chatbot:
    def __init__(self, model_loc: str, data_dir: str) -> None:
        self.intents = json.loads(open(data_dir+'/intents.json').read())
        self.words = pickle.load(open(data_dir+'/words.pkl', 'rb'))
        self.classes = pickle.load(open(data_dir+'/classes.pkl', 'rb'))

        self.model = ts.keras.models.load_model(model_loc)
        self.lemmatizer = nltk.WordNetLemmatizer()
    
    def __clean_up_sentence(self, sentence: str) -> list:
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [ self.lemmatizer.lemmatize(word) for word in sentence_words ]
        return sentence_words

    def __bag_of_words(self, sentence: str) -> np.array:
        sentence_words = self.__clean_up_sentence(sentence)
        bag = [0] * len(self.words)
        for w in sentence_words:
            for idx, word in enumerate(self.words):
                if word == w:
                    bag[idx] = 1
        return np.array(bag)
    
    def __predict_class(self, sentence: str) -> list:
        bow = self.__bag_of_words(sentence)
        predict = self.model.predict(np.array([bow]), verbose=0)[0]
        restults = [[i, r] for i, r in enumerate(predict) if r > ERROR_THRESHOLD]
        restults.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in restults:
            return_list.append({
                'intent': self.classes[r[0]],
                'probability': str(r[1])
            })
        return return_list
    
    def __get_response(self, intents_list: list, intents_json: list) -> str:
        tag = intents_list[0]['intent']
        list_of_intents = intents_json['intents']
        result = "Je n'ai pas compris..."
        for intent in list_of_intents:
            if intent['tag'] == tag:
                result = random.choice(intent['responses'])
        return result

    def request(self, message: str) -> str:
        ints = self.__predict_class(message)
        return self.__get_response(ints, self.intents)
