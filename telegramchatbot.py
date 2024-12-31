import requests
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

def initialize_chatbot_model(model_name="microsoft/DialoGPT-medium"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

def get_chatbot_response(tokenizer, model, user_input):
    inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    outputs = model.generate(inputs, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[:, inputs.shape[-1]:][0], skip_special_tokens=True)
    return response

def get_updates(bot_token, offset=None):
    url = f"https://api.telegram.org/bot{bot_token}/getUpdates"
    params = {"offset": offset, "timeout": 100}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()["result"]
    else:
        print(f"Error fetching updates: {response.status_code}")
        return []

def send_message(bot_token, chat_id, text):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    data = {"chat_id": chat_id, "text": text}
    response = requests.post(url, data=data)
    if response.status_code == 200:
        print("Message sent successfully!")
    else:
        print(f"Error sending message: {response.status_code}")
        print(response.text)

# Main function to process messages
def main():
    bot_token = "bot_token"                                          # Replace with your Telegram bot token
    model_name = "microsoft/DialoGPT-medium"  
    tokenizer, model = initialize_chatbot_model(model_name)
    print("Bot is running...")
    last_update_id = None

    while True:
        updates = get_updates(bot_token, offset=last_update_id)
        for update in updates:
            last_update_id = update["update_id"] + 1
            if "message" in update and "text" in update["message"]:
                chat_id = update["message"]["chat"]["id"]
                user_message = update["message"]["text"]
                print(f"Received message: {user_message}")

                response = get_chatbot_response(tokenizer, model, user_message)
                print(f"Generated response: {response}")

                send_message(bot_token, chat_id, response)

        time.sleep(1)
if __name__ == "__main__":
    main()
