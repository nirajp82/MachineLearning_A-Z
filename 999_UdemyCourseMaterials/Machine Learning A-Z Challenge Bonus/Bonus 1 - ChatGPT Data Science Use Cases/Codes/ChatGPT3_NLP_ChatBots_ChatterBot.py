# NLP and ChatBots (ChatterBot)

# Install ChatterBot and related libraries
!pip install chatterbot
!pip install chatterbot-corpus

# Import the libraries
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer

# Create the chatbot
chatbot = ChatBot('My Chatbot')

# Create the trainer
trainer = ChatterBotCorpusTrainer(chatbot)

# Train the chatbot using the past conversations data
trainer.train(
    'chatterbot.corpus.english.greetings',
    'chatterbot.corpus.english.conversations'
)

# Test the chatbot
while True:
  user_input = input("User: ")
  chatbot_response = chatbot.get_response(user_input)
  print("Chatbot: ", chatbot_response)

# Deploy the chatbot
chatbot.trainer.export_for_training('./my_export.json')
