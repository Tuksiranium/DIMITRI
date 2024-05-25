#from model.chatbot import Chatbot

#chatbot = Chatbot('chat_models/chatbot_model_v1.h5', 'data')

#print(chatbot.request('Salut'))

#import logging

#logging.basicConfig(level=logging.INFO)
#logging.info('This is an info message')

from model.model_trainer import ModelTrainer

DATA_DIR = "data/"
MODEL_LOC = "chat_models/chatbot_model_v2.h5"
trainer = ModelTrainer(MODEL_LOC, DATA_DIR)
