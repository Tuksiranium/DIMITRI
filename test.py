from model.Chatbot import Chatbot

chatbot = Chatbot('chat_models/chatbot_model_v1.h5', 'data')

print(chatbot.request('Salut'))