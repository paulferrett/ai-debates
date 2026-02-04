#!/usr/bin/env python3
"""
Debate Machine - Live Terminal UI

Split-screen view with real-time streaming as agents debate.
"""

import asyncio
import argparse
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, Container
from textual.widgets import Header, Footer, Static, RichLog, ProgressBar, Label
from textual.reactive import reactive
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown

import anthropic
import openai


# Import shared config from debate.py
from debate import (
    DebateConfig, ResolutionMode, Tone, MODEL_ALIASES,
    resolve_model, get_provider, TONE_INSTRUCTIONS,
    get_agent_prompt, get_judge_prompt, parse_response,
    create_debate_folder, write_round_doc, write_overview, write_outcome
)


class MultiProviderStreamClient:
    """Client that supports streaming from both providers"""

    def __init__(self):
        self.anthropic_client = anthropic.Anthropic()
        self.openai_client = openai.OpenAI()

    async def stream_message(self, model: str, max_tokens: int, messages: list, callback):
        """Stream a message, calling callback with each chunk"""
        provider = get_provider(model)

        if provider == "anthropic":
            with self.anthropic_client.messages.stream(
                model=model,
                max_tokens=max_tokens,
                messages=messages,
            ) as stream:
                full_text = ""
                for text in stream.text_stream:
                    full_text += text
                    await callback(text)
                return full_text

        elif provider == "openai":
            oai_messages = [{"role": m["role"], "content": m["content"]} for m in messages]

            stream = self.openai_client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=oai_messages,
                stream=True,
            )

            full_text = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    full_text += text
                    await callback(text)
            return full_text


class AgentPanel(Static):
    """Panel showing one agent's current state"""

    thinking = reactive(False)
    confidence = reactive(0.5)
    current_text = reactive("")

    def __init__(self, name: str, position: str, model: str, side: str, **kwargs):
        super().__init__(**kwargs)
        self.agent_name = name
        self.position = position
        self.model = model
        self.side = side  # "left" or "right"

    def compose(self) -> ComposeResult:
        yield Label(f"[bold]{self.agent_name}[/bold] ({self.position})", id=f"name-{self.side}")
        yield Label(f"[dim]{self.model}[/dim]", id=f"model-{self.side}")
        yield ProgressBar(total=100, show_eta=False, id=f"conf-{self.side}")
        yield Label("Confidence: --", id=f"conf-label-{self.side}")
        yield RichLog(id=f"log-{self.side}", wrap=True, markup=True, max_lines=500)

    def update_confidence(self, conf: float):
        self.confidence = conf
        bar = self.query_one(f"#conf-{self.side}", ProgressBar)
        bar.update(progress=conf * 100)
        label = self.query_one(f"#conf-label-{self.side}", Label)
        label.update(f"Confidence: {conf:.0%}")

    def set_thinking(self, thinking: bool):
        self.thinking = thinking
        label = self.query_one(f"#name-{self.side}", Label)
        if thinking:
            label.update(f"[bold yellow]{self.agent_name}[/bold yellow] ({self.position}) [blink]thinking...[/blink]")
        else:
            label.update(f"[bold]{self.agent_name}[/bold] ({self.position})")

    def append_text(self, text: str):
        log = self.query_one(f"#log-{self.side}", RichLog)
        log.write(text, scroll_end=True)

    def clear_log(self):
        log = self.query_one(f"#log-{self.side}", RichLog)
        log.clear()

    def set_round_header(self, round_num: int):
        log = self.query_one(f"#log-{self.side}", RichLog)
        log.write(f"\n[bold cyan]═══ Round {round_num} ═══[/bold cyan]\n")


class StatusBar(Static):
    """Status bar showing debate progress"""

    def compose(self) -> ComposeResult:
        yield Label("", id="status-text")

    def update_status(self, text: str):
        label = self.query_one("#status-text", Label)
        label.update(text)


class DebateApp(App):
    """Live debate TUI application"""

    CSS = """
    Screen {
        layout: grid;
        grid-size: 2 3;
        grid-rows: auto 1fr auto;
    }

    Header {
        column-span: 2;
    }

    AgentPanel {
        border: solid $primary;
        padding: 1;
        height: 100%;
    }

    #left-panel {
        border: solid $success;
    }

    #right-panel {
        border: solid $warning;
    }

    #name-left, #name-right {
        text-style: bold;
        margin-bottom: 1;
    }

    #model-left, #model-right {
        color: $text-muted;
        margin-bottom: 1;
    }

    ProgressBar {
        margin-bottom: 1;
    }

    RichLog {
        height: 100%;
        border: round $surface;
        padding: 0 1;
    }

    StatusBar {
        column-span: 2;
        height: 3;
        border: solid $surface;
        padding: 1;
    }

    Footer {
        column-span: 2;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("p", "pause", "Pause"),
    ]

    def __init__(self, config: DebateConfig):
        super().__init__()
        self.config = config
        self.client = MultiProviderStreamClient()
        self.paused = False
        self.rounds_data = []
        self.debate_path = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield AgentPanel(
            self.config.agent_a_name,
            self.config.agent_a_position,
            self.config.model_a,
            "left",
            id="left-panel"
        )
        yield AgentPanel(
            self.config.agent_b_name,
            self.config.agent_b_position,
            self.config.model_b,
            "right",
            id="right-panel"
        )
        yield StatusBar(id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        self.title = f"Debate: {self.config.topic[:50]}..."
        self.run_worker(self.run_debate())

    def action_pause(self) -> None:
        self.paused = not self.paused
        status = self.query_one("#status-bar", StatusBar)
        if self.paused:
            status.update_status("[yellow]PAUSED[/yellow] - Press 'p' to resume")
        else:
            status.update_status("Resumed...")

    async def run_debate(self):
        """Main debate loop"""
        left = self.query_one("#left-panel", AgentPanel)
        right = self.query_one("#right-panel", AgentPanel)
        status = self.query_one("#status-bar", StatusBar)

        # Create output folder
        self.debate_path = create_debate_folder(self.config)
        write_overview(self.debate_path, self.config)

        status.update_status(f"Output: {self.debate_path}")
        await asyncio.sleep(1)

        visible_history = ""
        last_a_argument = None
        last_b_argument = None

        for round_num in range(1, self.config.max_rounds + 1):
            # Wait if paused
            while self.paused:
                await asyncio.sleep(0.1)

            status.update_status(f"[bold]Round {round_num}/{self.config.max_rounds}[/bold]")

            left.set_round_header(round_num)
            right.set_round_header(round_num)

            # Agent A
            left.set_thinking(True)
            left.clear_log()
            left.set_round_header(round_num)

            text_a = ""
            async def callback_a(chunk):
                nonlocal text_a
                text_a += chunk
                left.append_text(chunk)

            prompt_a = get_agent_prompt(
                self.config.agent_a_name, self.config.agent_a_position, self.config.tone,
                self.config.topic, visible_history, last_b_argument,
                self.config.require_steelman, None
            )

            full_a = await self.client.stream_message(
                self.config.model_a, 4000,
                [{"role": "user", "content": prompt_a}],
                callback_a
            )

            thinking_a, short_a, long_a, conf_a, ready_a, synth_a = parse_response(full_a)
            left.set_thinking(False)
            left.update_confidence(conf_a)

            write_round_doc(self.debate_path, self.config.agent_a_name, self.config.agent_a_position,
                           round_num, thinking_a, short_a, long_a, conf_a)

            await asyncio.sleep(0.5)

            # Agent B
            right.set_thinking(True)
            right.clear_log()
            right.set_round_header(round_num)

            a_public = f"## Short Form\n{short_a}\n\n## Long Form\n{long_a}"

            text_b = ""
            async def callback_b(chunk):
                nonlocal text_b
                text_b += chunk
                right.append_text(chunk)

            prompt_b = get_agent_prompt(
                self.config.agent_b_name, self.config.agent_b_position, self.config.tone,
                self.config.topic, visible_history, a_public,
                self.config.require_steelman, None
            )

            full_b = await self.client.stream_message(
                self.config.model_b, 4000,
                [{"role": "user", "content": prompt_b}],
                callback_b
            )

            thinking_b, short_b, long_b, conf_b, ready_b, synth_b = parse_response(full_b)
            right.set_thinking(False)
            right.update_confidence(conf_b)

            write_round_doc(self.debate_path, self.config.agent_b_name, self.config.agent_b_position,
                           round_num, thinking_b, short_b, long_b, conf_b)

            # Update history
            b_public = f"## Short Form\n{short_b}\n\n## Long Form\n{long_b}"
            visible_history += f"\n\n### Round {round_num}\n\n"
            visible_history += f"**{self.config.agent_a_name}:**\n{a_public}\n\n"
            visible_history += f"**{self.config.agent_b_name}:**\n{b_public}\n\n"

            self.rounds_data.append({
                "round": round_num,
                "a_conf": conf_a,
                "b_conf": conf_b,
                "a_ready": ready_a,
                "b_ready": ready_b,
                "a_short": short_a,
                "b_short": short_b,
            })

            last_a_argument = a_public
            last_b_argument = b_public

            # Check consensus
            if self.config.resolution_mode == ResolutionMode.CONSENSUS:
                if ready_a and ready_b:
                    status.update_status("[green]✓ CONSENSUS REACHED[/green]")
                    synthesis = synth_a or synth_b or "Both ready to conclude"
                    write_outcome(self.debate_path, self.config, self.rounds_data, "consensus", None, synthesis)
                    await asyncio.sleep(3)
                    self.exit()
                    return

            await asyncio.sleep(1)

        # Max rounds - resolve
        status.update_status("Max rounds reached. Resolving...")

        if self.config.resolution_mode == ResolutionMode.JUDGE:
            status.update_status(f"[yellow]Calling judge ({self.config.model_judge})...[/yellow]")

            judge_text = ""
            async def callback_j(chunk):
                nonlocal judge_text
                judge_text += chunk

            judge_prompt = get_judge_prompt(self.config.topic, visible_history, None)
            full_judge = await self.client.stream_message(
                self.config.model_judge, 3000,
                [{"role": "user", "content": judge_prompt}],
                callback_j
            )

            winner = None
            if "advocate" in full_judge.lower():
                winner = self.config.agent_a_name
            elif "critic" in full_judge.lower():
                winner = self.config.agent_b_name

            status.update_status(f"[bold green]Winner: {winner or 'TIE'}[/bold green]")
            write_outcome(self.debate_path, self.config, self.rounds_data, "judge", winner,
                         "See judge analysis", full_judge)
        else:
            write_outcome(self.debate_path, self.config, self.rounds_data, "max_rounds", None,
                         f"No consensus after {self.config.max_rounds} rounds")
            status.update_status("[yellow]No consensus reached[/yellow]")

        status.update_status(f"[green]Complete![/green] Output: {self.debate_path}")
        await asyncio.sleep(5)
        self.exit()


def main():
    alias_help = "Model aliases: " + ", ".join(sorted(MODEL_ALIASES.keys()))

    parser = argparse.ArgumentParser(
        description="AI Debate Machine - Live Terminal UI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  %(prog)s "Should we use microservices?"
  %(prog)s "Tabs vs spaces" --model-a sonnet --model-b gpt-4o-mini -r 3

{alias_help}
        """
    )

    parser.add_argument("topic", help="The debate topic")
    parser.add_argument("--rounds", "-r", type=int, default=3, help="Max rounds (default: 3)")
    parser.add_argument("--mode", "-m", choices=["consensus", "judge"], default="judge")
    parser.add_argument("--tone", "-t",
                       choices=["truth_seeking", "academic", "adversarial", "collaborative", "socratic"],
                       default="truth_seeking")
    parser.add_argument("--model", default=None)
    parser.add_argument("--model-a", default="sonnet")
    parser.add_argument("--model-b", default="sonnet")
    parser.add_argument("--model-judge", default="sonnet")
    parser.add_argument("--agent-a", default="Advocate")
    parser.add_argument("--agent-b", default="Critic")
    parser.add_argument("--position-a", default="FOR")
    parser.add_argument("--position-b", default="AGAINST")
    parser.add_argument("--output", "-o", default="debates")

    args = parser.parse_args()

    if args.model:
        model_a = resolve_model(args.model)
        model_b = resolve_model(args.model)
        model_judge = resolve_model(args.model)
    else:
        model_a = resolve_model(args.model_a)
        model_b = resolve_model(args.model_b)
        model_judge = resolve_model(args.model_judge)

    config = DebateConfig(
        topic=args.topic,
        max_rounds=args.rounds,
        resolution_mode=ResolutionMode(args.mode),
        tone=Tone(args.tone),
        model_a=model_a,
        model_b=model_b,
        model_judge=model_judge,
        agent_a_name=args.agent_a,
        agent_b_name=args.agent_b,
        agent_a_position=args.position_a,
        agent_b_position=args.position_b,
        output_dir=args.output,
        ask_questions=False,  # No questions in live mode
    )

    app = DebateApp(config)
    app.run()


if __name__ == "__main__":
    main()
