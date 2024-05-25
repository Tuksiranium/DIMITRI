import discord
import logging
import os
from dotenv import load_dotenv

from model.chatbot import Chatbot

logger = logging.getLogger('discord.client')

load_dotenv()
DISCORD_CLIENT_KEY = os.getenv('DISCORD_CLIENT_KEY')
intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)
chatbot = Chatbot('chat_models/chatbot_model_v1.h5', 'data')

@client.event
async def on_message(message):
    if message.author == client.user: return
    
    if message.content.startswith('$DIMITRI'):
        response = chatbot.request(message.content[9:])
        logger.info('[Resquest]: "'+message.content[9:]+'", [Response]: "'+response+'"')
        await message.channel.send(response)



print('CLient running...')
client.run(DISCORD_CLIENT_KEY)