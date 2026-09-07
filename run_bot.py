#!/usr/bin/env python
"""
Run the Telegram bot for the AI Identity System.
"""
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check for required environment variables
token = os.getenv("TELEGRAM_BOT_TOKEN")
if not token or token == "your_bot_token_here" or token.startswith("1234567890:"):
    print("❌ A real TELEGRAM_BOT_TOKEN is required in .env")
    print("Please add your Telegram bot token to .env:")
    print("TELEGRAM_BOT_TOKEN=<token from @BotFather>")
    sys.exit(1)

from app.bot.telegram_bot import TelegramBot

if __name__ == "__main__":
    bot = TelegramBot()
    bot.run()
