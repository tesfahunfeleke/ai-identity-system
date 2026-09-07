import asyncio
import os
import logging
import httpx
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.request import HTTPXRequest
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

logger = logging.getLogger(__name__)

# API Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_PROXY_URL = os.getenv("TELEGRAM_PROXY_URL") or None

# Conversation state
user_sessions = {}


class TelegramBot:
    """Telegram bot for the AI Identity System."""

    def __init__(self):
        self.token = BOT_TOKEN
        self.api_base = API_BASE_URL
        self.application = None

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command."""
        welcome_message = (
            "👋 *Welcome to your Digital Twin!*\n\n"
            "I'm your personal AI assistant that knows you deeply.\n\n"
            "Available commands:\n"
            "/chat <message> - Chat with me\n"
            "/memory <query> - Search your memories\n"
            "/summary - Get today's summary\n"
            "/insights - Get insights about you\n"
            "/suggestions - Get proactive suggestions\n"
            "/reset - Reset our conversation\n"
            "/help - Show this message\n\n"
            "Just type your message and I'll respond naturally! 🧠"
        )
        await update.message.reply_text(welcome_message, parse_mode="Markdown")

    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command."""
        help_message = (
            "🤖 *Digital Twin Commands*\n\n"
            "/chat <message> - Chat with me\n"
            "/memory <query> - Search your memories\n"
            "/summary - Get today's summary\n"
            "/insights - Get insights about you\n"
            "/suggestions - Get proactive suggestions\n"
            "/reset - Reset our conversation\n"
            "/help - Show this message\n\n"
            "Or just send any message to chat naturally! 💬"
        )
        await update.message.reply_text(help_message, parse_mode="Markdown")

    async def chat(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /chat command."""
        message = " ".join(context.args) if context.args else update.message.text
        
        if message.startswith("/chat"):
            message = message[5:].strip()
            
        if not message:
            await update.message.reply_text("Please provide a message. Example: /chat Hello!")
            return

        await self._send_chat_response(update, message)

    async def memory(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /memory command."""
        query = " ".join(context.args)
        
        if not query:
            await update.message.reply_text(
                "🔍 Please provide a search query. Example: /memory coffee"
            )
            return

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.api_base}/memory/search",
                    json={"query": query, "k": 3}
                )
                data = response.json()

                if data.get("results_count", 0) == 0:
                    await update.message.reply_text(
                        "🔍 No memories found for that query."
                    )
                    return

                results = data.get("results", [])
                message = f"🔍 *Found {len(results)} memories:*\n\n"
                
                for i, result in enumerate(results, 1):
                    text = result.get("text", "")[:200]
                    source = result.get("metadata", {}).get("source_file", "unknown")
                    similarity = result.get("similarity", 0)
                    message += f"{i}. *{source}* (similarity: {similarity:.2f})\n"
                    message += f"   {text}...\n\n"

                await update.message.reply_text(message, parse_mode="Markdown")

        except Exception as e:
            logger.error(f"Error searching memory: {e}")
            await update.message.reply_text("❌ Error searching memory. Please try again.")

    async def summary(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /summary command."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{self.api_base}/proactive/daily-summary")
                data = response.json()
                summary = data.get("summary", "No summary available.")

                await update.message.reply_text(
                    f"📝 *Daily Summary*\n\n{summary}",
                    parse_mode="Markdown"
                )

        except Exception as e:
            logger.error(f"Error getting summary: {e}")
            await update.message.reply_text("❌ Error getting summary. Please try again.")

    async def insights(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /insights command."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{self.api_base}/proactive/insights")
                data = response.json()

                patterns = data.get("patterns", {})
                suggestions = data.get("suggestions", [])

                message = "🧠 *Your Insights*\n\n"
                message += "*Patterns Detected:*\n"
                
                daily_routines = patterns.get("daily_routines", {})
                if daily_routines.get("morning"):
                    message += "  ☀️ Morning routine detected\n"
                if daily_routines.get("evening"):
                    message += "  🌙 Evening routine detected\n"
                
                weekly = patterns.get("weekly_habits", {})
                if weekly:
                    message += "  📅 Weekly habits detected\n"

                if suggestions:
                    message += "\n*Suggestions:*\n"
                    for suggestion in suggestions:
                        message += f"  💡 {suggestion}\n"

                if not daily_routines and not weekly and not suggestions:
                    message += "  Not enough data yet. Keep chatting and I'll learn more! 🧠"

                await update.message.reply_text(message, parse_mode="Markdown")

        except Exception as e:
            logger.error(f"Error getting insights: {e}")
            await update.message.reply_text("❌ Error getting insights. Please try again.")

    async def suggestions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /suggestions command."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{self.api_base}/proactive/suggestions")
                data = response.json()
                suggestions = data.get("suggestions", [])

                if not suggestions:
                    await update.message.reply_text(
                        "💡 No suggestions right now. Keep sharing your activities!"
                    )
                    return

                message = "💡 *Proactive Suggestions*\n\n"
                for i, suggestion in enumerate(suggestions, 1):
                    message += f"{i}. {suggestion}\n"

                await update.message.reply_text(message, parse_mode="Markdown")

        except Exception as e:
            logger.error(f"Error getting suggestions: {e}")
            await update.message.reply_text("❌ Error getting suggestions. Please try again.")

    async def reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /reset command."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(f"{self.api_base}/chat/reset")
                data = response.json()
                
                await update.message.reply_text(
                    "🔄 *Conversation reset!*\n\n"
                    "I've cleared our conversation history. Let's start fresh! 🧠",
                    parse_mode="Markdown"
                )

        except Exception as e:
            logger.error(f"Error resetting conversation: {e}")
            await update.message.reply_text("❌ Error resetting conversation.")

    async def _send_chat_response(self, update: Update, message: str):
        """Send chat request to the API and handle response."""
        try:
            # Send typing indicator
            await update.message.chat.send_action(action="typing")
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.api_base}/chat/",
                    json={"message": message}
                )
                data = response.json()
                
                chat_response = data.get("response", "No response.")
                mood = data.get("mood", "neutral")
                tone = data.get("tone", "warm")

                # Format the response with mood and tone
                formatted_response = (
                    f"{chat_response}\n\n"
                    f"*Mood:* {mood} | *Tone:* {tone}"
                )

                await update.message.reply_text(
                    formatted_response,
                    parse_mode="Markdown"
                )

        except httpx.TimeoutException:
            await update.message.reply_text(
                "⏰ Sorry, the request timed out. Please try again."
            )
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            await update.message.reply_text(
                "❌ Sorry, I'm having trouble responding. Please try again."
            )

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle regular text messages."""
        message = update.message.text
        
        if not message:
            return

        # Check if it's a command (starts with /)
        if message.startswith("/"):
            return

        # Handle as chat message
        await self._send_chat_response(update, message)

    def run(self):
        """Run the Telegram bot."""
        if not self.token or self.token == "your_bot_token_here" or self.token.startswith("1234567890:"):
            logger.error("A real TELEGRAM_BOT_TOKEN is required")
            print("❌ Replace TELEGRAM_BOT_TOKEN in .env with the token from @BotFather")
            return

        # Create application with longer Telegram API timeouts.
        request = HTTPXRequest(
            proxy=TELEGRAM_PROXY_URL,
            connect_timeout=60.0,
            read_timeout=60.0,
            write_timeout=60.0,
            pool_timeout=60.0,
        )
        self.application = (
            Application.builder()
            .token(self.token)
            .request(request)
            .get_updates_request(request)
            .build()
        )

        # run_polling() initializes and starts the application lifecycle.
        # Add command handlers
        self.application.add_handler(CommandHandler("start", self.start))
        self.application.add_handler(CommandHandler("help", self.help))
        self.application.add_handler(CommandHandler("chat", self.chat))
        self.application.add_handler(CommandHandler("memory", self.memory))
        self.application.add_handler(CommandHandler("summary", self.summary))
        self.application.add_handler(CommandHandler("insights", self.insights))
        self.application.add_handler(CommandHandler("suggestions", self.suggestions))
        self.application.add_handler(CommandHandler("reset", self.reset))

        # Add message handler for regular messages
        self.application.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message)
        )

        # Start the bot
        print("🚀 Starting Telegram bot...")
        print("🤖 Bot is running! Check your bot on Telegram.")
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            asyncio.set_event_loop(asyncio.new_event_loop())
        self.application.run_polling(allowed_updates=Update.ALL_TYPES)
