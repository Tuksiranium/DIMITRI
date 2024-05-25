import tensorflow as tf
from model.model_trainer import ModelTrainer


DATA_DIR = "data/"
MODEL_LOC = "chat_models/chatbot_model_v2.h5"

trainer = ModelTrainer(MODEL_LOC, DATA_DIR, verbose=True)

train_x, train_y = trainer.prepare_data()

input_shape = (len(train_x[0]),)
output_shape = len(train_y[0])

model = trainer.prepare_model(input_shape, output_shape)
model = trainer.start_training(model, train_x, train_y)

trainer.save_model(model)