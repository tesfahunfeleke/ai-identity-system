import asyncio
import logging
from datetime import datetime

import httpx

logger = logging.getLogger(__name__)


class SchedulerService:
    """Handles scheduled tasks like daily check-ins."""

    def __init__(self, api_base: str = "http://localhost:8000"):
        self.api_base = api_base
        self.check_in_time = "09:00"
        self.last_check_in = None
        self.active = False

    async def send_daily_check_in(self):
        """Send a daily reflection prompt."""
        try:
            prompt = (
                "🌅 Good morning! I'm your digital twin.\n\n"
                "I'd love to hear about your day. Here are some prompts to get started:\n"
                "• What's one thing you're looking forward to today?\n"
                "• What's on your mind?\n"
                "• Any goals you're working on?\n\n"
                "You can reply here, or just say 'skip' if you're busy today."
            )
            async with httpx.AsyncClient() as client:
                logger.info("Daily check-in prepared")
                self.last_check_in = datetime.now()
                return {"status": "sent", "message": prompt}
        except Exception as exc:
            logger.error("Error sending daily check-in: %s", exc)
            return {"status": "error", "message": str(exc)}

    async def run(self):
        """Run the scheduler loop."""
        self.active = True
        logger.info("Scheduler started. Daily check-ins at 9:00 AM.")
        while self.active:
            now = datetime.now()
            if (
                now.hour == 9
                and now.minute == 0
                and (self.last_check_in is None or self.last_check_in.date() < now.date())
            ):
                await self.send_daily_check_in()
                await asyncio.sleep(60)
            await asyncio.sleep(60)

    def stop(self):
        """Stop the scheduler."""
        self.active = False
        logger.info("Scheduler stopped.")
