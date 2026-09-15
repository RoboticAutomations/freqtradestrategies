import telebot
import json

TOKEN = 'token'
bot = telebot.TeleBot(TOKEN)

@bot.message_handler(func=lambda msg: 'New' in msg.text or 'Exited' in msg.text or 'Performance' in msg.text)
def log_message(message):
    log = {'user': message.from_user.id, 'message': message.text}
    
    # Load existing data
    try:
        with open('logs.json', 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []

    # Add new log to data
    data.append(log)

    # Write data back to file
    with open('logs.json', 'w') as f:
        json.dump(data, f, indent=4)

    bot.forward_message("chatid", message.chat.id, message.message_id)

bot.polling()
