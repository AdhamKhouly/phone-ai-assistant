"""Discord transport. Thin by design: all decisions live in ControlService / CallManager."""
from __future__ import annotations

import asyncio
import logging
import signal

import discord
from discord import app_commands

from ..calls.manager import CallEvent, CallManager
from ..config import Settings
from ..control import ControlService
from .notifications import format_event

log = logging.getLogger(__name__)


class PhoneAssistantBot(discord.Client):
    def __init__(self, settings: Settings, manager: CallManager, control: ControlService) -> None:
        intents = discord.Intents.none()
        intents.guilds = True
        intents.guild_messages = True
        intents.message_content = True  # lets a plain `answer` message work (privileged intent)
        super().__init__(intents=intents, allowed_mentions=discord.AllowedMentions.none())
        self.settings = settings
        self.manager = manager
        self.control = control
        self.tree = app_commands.CommandTree(self)
        manager.subscribe(self.on_call_event)
        self._expiry_task: asyncio.Task | None = None
        self._backend_task: asyncio.Task | None = None

    # ------------------------------------------------------------------ lifecycle
    async def setup_hook(self) -> None:
        self._register_slash_commands()
        guild = discord.Object(id=self.settings.guild_id)
        await self.tree.sync(guild=guild)  # guild-only commands: instant, private
        self._expiry_task = asyncio.create_task(self._expiry_loop())
        backend = self.manager.backend
        if hasattr(backend, "run"):
            self._backend_task = asyncio.create_task(backend.run())
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self.close()))

    async def close(self) -> None:
        """Graceful shutdown: release calls, tell #calls, then disconnect."""
        if getattr(self, "_closing_started", False):
            return
        self._closing_started = True
        if self._expiry_task:
            self._expiry_task.cancel()
        if hasattr(self.manager.backend, "stop"):
            self.manager.backend.stop()
        try:
            released = await self.manager.shutdown()
            if self.is_ready():
                note = f" Released {len(released)} call(s); your phone handles them normally." if released else ""
                await self._post_calls("🔴 Phone assistant going offline. Calls will not be answered "
                                       "by the AI until it's back." + note)
        except Exception:
            log.exception("error during shutdown notice")
        log.info("shutting down")
        await super().close()

    async def on_ready(self) -> None:
        log.info("Discord connected as bot user id %s", self.user.id if self.user else "?")
        await self._post_calls("🟢 Phone assistant online (backend: "
                               f"{self.manager.backend.name}). Type `help` in #assistant.")

    async def _expiry_loop(self) -> None:
        while not self.is_closed():
            try:
                await self.manager.expire_stale()
                await self.manager.enforce_max_duration()
            except Exception:
                log.exception("expiry loop error")
            await asyncio.sleep(1)

    # ------------------------------------------------------------------ inbound
    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot or message.guild is None or message.guild.id != self.settings.guild_id:
            return
        reply = await self.control.handle(message.author.id, message.channel.id,
                                          message.content, message.created_at)
        if reply:
            await message.reply(reply, mention_author=False)

    def _register_slash_commands(self) -> None:
        guild = discord.Object(id=self.settings.guild_id)

        async def run(interaction: discord.Interaction, text: str) -> None:
            reply = await self.control.handle(interaction.user.id, interaction.channel_id or 0,
                                              text, interaction.created_at)
            await interaction.response.send_message(
                reply or "Use this in #assistant or #calls.", ephemeral=reply is None)

        @self.tree.command(name="answer", description="Let the AI answer the ringing call", guild=guild)
        @app_commands.describe(call="Call number, only needed if several are ringing")
        async def answer(interaction: discord.Interaction, call: int | None = None):
            await run(interaction, f"answer {call or ''}")

        @self.tree.command(name="ignore", description="Leave the ringing call alone", guild=guild)
        async def ignore(interaction: discord.Interaction, call: int | None = None):
            await run(interaction, f"ignore {call or ''}")

        @self.tree.command(name="hangup", description="End the AI-handled call", guild=guild)
        async def hangup(interaction: discord.Interaction, call: int | None = None):
            await run(interaction, f"hangup {call or ''}")

        @self.tree.command(name="call", description="Request an outgoing call", guild=guild)
        @app_commands.describe(target="Phone number or contact name", objective="What the call is for")
        async def call(interaction: discord.Interaction, target: str, objective: str | None = None):
            await run(interaction, f"call {target}" + (f": {objective}" if objective else ""))

        @self.tree.command(name="status", description="Assistant and call status", guild=guild)
        async def status(interaction: discord.Interaction):
            await run(interaction, "status")

        @self.tree.command(name="help", description="Show commands", guild=guild)
        async def help_(interaction: discord.Interaction):
            await run(interaction, "help")

    # ------------------------------------------------------------------ outbound
    async def on_call_event(self, event: CallEvent) -> None:
        if event.kind == "backend" and getattr(self, "_closing_started", False):
            return  # shutting down: the "going offline" notice already covers it
        text = format_event(event, self.manager.clock(), self.settings.answer_window_s,
                            self.manager.backend.can_answer)
        if text and event.kind == "backend" and event.detail == "disconnected":
            note = getattr(self.manager.backend, "status_note", None)
            if callable(note):
                text += f"\n({note()})"   # which of the three it is; the bot keeps no log file
        if text:
            await self._post_calls(text)
            log.info("posted '%s' to #calls%s", event.kind,
                     f" for call #{event.call.id}" if event.call else "")

    async def _post_calls(self, text: str) -> None:
        if not self.is_ready() and not self.is_closed():
            # The native DIGITS backend is readable within ~1 s, before Discord has finished
            # connecting; hold early notices (e.g. "signed out", an already-ringing call) until then.
            try:
                await asyncio.wait_for(self.wait_until_ready(), timeout=30)
            except asyncio.TimeoutError:
                log.warning("Discord not ready after 30s; a #calls notice was dropped")
                return
        channel = self.get_channel(self.settings.calls_channel_id)
        if channel is None:
            log.warning("calls channel not found; check CALLS_CHANNEL_ID and bot permissions")
            return
        try:
            await channel.send(text)  # type: ignore[union-attr]
        except discord.DiscordException:
            log.exception("failed to post to #calls")
