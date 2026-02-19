
import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TOKEN:
    print("Error: TELEGRAM_TOKEN not found in .env")
    exit(1)

URL = f"https://api.telegram.org/bot{TOKEN}/getUpdates"

print(f"Polling for updates from bot (Token starts with {TOKEN[:5]}...)...")
print("Please send a message (e.g., /start) to your bot now.")

while True:
    try:
        resp = requests.get(URL, timeout=10)
        data = resp.json()
        
        if not data.get("ok"):
            print(f"Error: {data}")
            time.sleep(2)
            continue
            
        updates = data.get("result", [])
        if updates:
            last_update = updates[-1]
            chat = last_update.get("message", {}).get("chat", {})
            chat_id = chat.get("id")
            username = chat.get("username")
            
            if chat_id:
                print("\n" + "="*40)
                print(f"SUCCESS! Found Chat ID")
                print(f"User: @{username}")
                print(f"Chat ID: {chat_id}")
                print("="*40)
                print(f"\nPlease add this to your .env file:\nTELEGRAM_CHAT_ID={chat_id}")
                break
        
        time.sleep(2)
        
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(2)
