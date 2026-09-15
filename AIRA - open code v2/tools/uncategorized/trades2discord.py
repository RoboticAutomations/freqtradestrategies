from telethon import TelegramClient, events
import requests
import json

api_id = 'your_api_id'
api_hash = 'your_api_hash'

client = TelegramClient('session_name', api_id, api_hash)

@client.on(events.NewMessage)
async def my_event_handler(event):
    if 'New' in event.raw_text or 'Exiting' in event.raw_text:
        data = {'content': event.raw_text}
        
        # Save to json file
        with open('logs.json', 'a') as f:
            json.dump(data, f)
            f.write('\n')
        
        # Post to Discord webhook
        webhook_url = 'your_webhook_url'
        response = requests.post(
            webhook_url, data=json.dumps(data),
            headers={'Content-Type': 'application/json'}
        )
        if response.status_code != 204:
            raise ValueError('Request to discord returned an error %s, the response is:\n%s' % (response.status_code, response.text))

client.start()
client.run_until_disconnected()
