import random
import json
import pickle
import numpy as np
import logging

import nltk
import tensorflow as tf


nltk.download("popular", quiet=True)

logger = logging.getLogger(__name__)
logging.basicConfig(format='[%(asctime)s] %(levelname)s %(name)s: %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S', level=logging.INFO)

class ModelTrainer:
    def __init__(self, model_loc: str, data_dir: str,
                 epoch: int = 200, batch_size: int = 5, verbose: bool = False) -> None:
        with open(data_dir+'/intents.json', encoding='utf-8') as intent_reader:
            self.intents = json.load(intent_reader)
        self.words = []
        self.classes = []
        self.docs = []
        self.data_dir = data_dir
        self.model_loc = model_loc
        self.ignore_letter = ['?', '!', '.', ',', ';']
        self.lemmatizer = nltk.WordNetLemmatizer()
        self._epoch = epoch
        self._batch_size = batch_size
        self._verbose = verbose
    
    def prepare_data(self) -> tuple:
        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                tokens = nltk.word_tokenize(pattern)
                self.words.extend(tokens)
                self.docs.append((tokens, intent['tag']))

                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])
        
        self.words = [ self.lemmatizer.lemmatize(word.lower()) for word in self.words if word not in self.ignore_letter]

        self.words = sorted(set(self.words))
        self.classes = sorted(set(self.classes))
        
        self.save_data_pkl()
        
        training = []
        output_empty = [0] * len(self.classes)

        for doc in self.docs:
            bag = []
            word_patterns = doc[0]
            word_patterns = [ self.lemmatizer.lemmatize(word.lower()) for word in word_patterns ]
            for word in self.words:
                bag.append(1) if word in word_patterns else bag.append(0)
            
            output_row = list(output_empty)
            output_row[self.classes.index(doc[1])] = 1
            training.append([bag, output_row])

        random.shuffle(training)
        training = np.array(training, dtype=object)

        train_x = np.array(list(training[:, 0]))
        train_y = np.array(list(training[:, 1]))
        logger.info("Data prepered.")
        return train_x, train_y
        
        
    def save_data_pkl(self, custom_loc: str = None) -> None:
        if custom_loc is None: custom_loc = self.data_dir
        
        pickle.dump(self.words, open(custom_loc+'/words.pkl', 'wb'))
        pickle.dump(self.classes, open(custom_loc+'/classes.pkl', 'wb'))
        logger.info("Set of data saved on directory: "+custom_loc)


    def prepare_model(self, input_shape: tuple, output_shape: int) -> tf.keras.Sequential:
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=128, input_shape=input_shape, activation='relu'),
            tf.keras.layers.Dropout(rate=.5),
            tf.keras.layers.Dense(units=64, activation='relu'),
            tf.keras.layers.Dropout(rate=.5),
            tf.keras.layers.Dense(units=output_shape, activation='softmax')
        ])
        model.compile(optimizer=tf.optimizers.SGD(learning_rate=.01, decay=1e-6, momentum=.9, nesterov=True),
                    loss="categorical_crossentropy",
                    metrics=['accuracy'])

        logger.info("Model compiled.")
        if self._verbose: print(model.summary())
        return model
    

    def save_model(self, model: tf.keras.Sequential, custom_loc: str = None) -> None:
        if custom_loc is None: custom_loc = self.model_loc
        try:
            model.save(custom_loc)
        except:
            logger.error("Error on saving model. Maybe it's due to a wrong saving path.")
        logger.info("Model saved.")


    def start_training(self, model: tf.keras.Sequential,
                       x_train: np.array, y_train: np.array) -> tf.keras.Sequential:
        logger.info("Starting training the model...")
        model.fit(x_train, y_train, epochs=self._epoch, batch_size=self._batch_size, verbose=self._verbose)
        logger.info("The model has been trained.")
        return model
    
    