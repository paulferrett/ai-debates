#!/usr/bin/env python3
"""
Debate Machine - Structured AI self-debate with human input

Two agents debate a topic, each maintaining separate thinking.
Agents can ask the human clarifying questions to gather facts/preferences.
Both sides genuinely want the right decision - adversarial only if needed.

Output structure:
- Each round: long-form reasoning + short-form priority headlines
- Outcome: scannable summary with just headlines from each round
"""

import argparse
import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional
from pathlib import Path

# Load .env file
from dotenv import load_dotenv
load_dotenv()

import anthropic
import openai


class ResolutionMode(Enum):
    CONSENSUS = "consensus"
    JUDGE = "judge"


class Tone(Enum):
    TRUTH_SEEKING = "truth_seeking"
    ACADEMIC = "academic"
    ADVERSARIAL = "adversarial"
    COLLABORATIVE = "collaborative"
    SOCRATIC = "socratic"


@dataclass
class DebateConfig:
    topic: str
    max_rounds: int = 5
    resolution_mode: ResolutionMode = ResolutionMode.CONSENSUS
    tone: Tone = Tone.TRUTH_SEEKING
    model_a: str = "claude-sonnet-4-20250514"
    model_b: str = "claude-sonnet-4-20250514"
    model_judge: str = "claude-sonnet-4-20250514"
    agent_a_name: str = "Advocate"
    agent_b_name: str = "Critic"
    agent_a_position: str = "FOR"
    agent_b_position: str = "AGAINST"
    require_steelman: bool = True
    output_dir: str = "debates"
    ask_questions: bool = True
    max_questions_per_side: int = 3


# Model aliases for convenience
MODEL_ALIASES = {
    # Anthropic
    "opus": "claude-opus-4-20250514",
    "opus-4": "claude-opus-4-20250514",
    "opus-4.5": "claude-opus-4-5-20251101",
    "opus-45": "claude-opus-4-5-20251101",
    "sonnet": "claude-sonnet-4-20250514",
    "sonnet-4": "claude-sonnet-4-20250514",
    "sonnet-5": "claude-sonnet-4-5-20250514",  # Placeholder - update when available
    "haiku": "claude-3-5-haiku-latest",
    "haiku-3.5": "claude-3-5-haiku-latest",
    # OpenAI
    "gpt-4": "gpt-4",
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-4-turbo": "gpt-4-turbo",
    "gpt-3.5": "gpt-3.5-turbo",
    "gpt-3.5-turbo": "gpt-3.5-turbo",
    "o1": "o1",
    "o1-mini": "o1-mini",
    "o3": "o3",
    "o3-mini": "o3-mini",
}


def resolve_model(model_spec: str) -> str:
    """Resolve model alias to full model ID"""
    return MODEL_ALIASES.get(model_spec.lower(), model_spec)


def get_provider(model: str) -> str:
    """Determine which provider to use based on model name"""
    if model.startswith("claude") or model.startswith("anthropic"):
        return "anthropic"
    elif model.startswith("gpt") or model.startswith("o1") or model.startswith("o3"):
        return "openai"
    else:
        # Default to anthropic for unknown models
        return "anthropic"


class MultiProviderClient:
    """Unified client for calling both Anthropic and OpenAI models"""

    def __init__(self):
        self.anthropic_client = anthropic.Anthropic()
        self.openai_client = openai.OpenAI()

    def create_message(self, model: str, max_tokens: int, messages: list, system: str = None) -> str:
        """Call the appropriate provider and return the response text"""
        provider = get_provider(model)

        if provider == "anthropic":
            kwargs = {
                "model": model,
                "max_tokens": max_tokens,
                "messages": messages,
            }
            if system:
                kwargs["system"] = system

            response = self.anthropic_client.messages.create(**kwargs)
            return response.content[0].text

        elif provider == "openai":
            # Convert to OpenAI format
            oai_messages = []
            if system:
                oai_messages.append({"role": "system", "content": system})
            for msg in messages:
                oai_messages.append({"role": msg["role"], "content": msg["content"]})

            response = self.openai_client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=oai_messages,
            )
            return response.choices[0].message.content

        else:
            raise ValueError(f"Unknown provider for model: {model}")


TONE_INSTRUCTIONS = {
    Tone.TRUTH_SEEKING: """You genuinely want to find the right answer, not just win.
Be respectful and assume good faith. Acknowledge good points. Change your mind if warranted.
Adversarial pushback is fine when someone is unreasonable, but default to collaboration.""",
    Tone.ACADEMIC: "Rigorous, evidence-based discourse. Be precise and charitable.",
    Tone.ADVERSARIAL: "Challenge every assumption. Look for flaws. Win, but stay professional.",
    Tone.COLLABORATIVE: "Work toward truth together. Build on good ideas. Seek synthesis.",
    Tone.SOCRATIC: "Ask probing questions. Expose assumptions. Guide toward understanding.",
}


def get_question_prompt(agent_name: str, position: str, topic: str, max_questions: int) -> str:
    return f"""You are {agent_name}, preparing to argue {position} on this topic:

TOPIC: {topic}

Before the debate, ask the human up to {max_questions} clarifying questions.
These should be OBJECTIVE questions to establish facts, constraints, or preferences.

Good: "What's the budget?" / "What's the timeline?" / "Who are the stakeholders?"
Bad: Leading questions, rhetorical questions, arguments disguised as questions

Both you and your opponent want to help the human make the RIGHT decision.
Your opponent will also ask questions - you'll both see all answers.

Provide questions as a numbered list, or respond "NO QUESTIONS" if none needed.

Questions:"""


def get_agent_prompt(
    agent_name: str,
    position: str,
    tone: Tone,
    topic: str,
    visible_history: str,
    opponent_last_argument: Optional[str],
    require_steelman: bool,
    human_qa: Optional[str] = None
) -> str:
    qa_section = ""
    if human_qa:
        qa_section = f"""
HUMAN'S ANSWERS:
{human_qa}
Use these to inform your arguments. Both sides have access to this.
"""

    return f"""You are {agent_name}, arguing {position} on the topic below.

TOPIC: {topic}

YOUR APPROACH: {TONE_INSTRUCTIONS[tone]}
{qa_section}
{"PREVIOUS ROUNDS (public arguments only):" + chr(10) + visible_history if visible_history else "This is the opening round."}

{f"OPPONENT'S LATEST ARGUMENT:{chr(10)}{opponent_last_argument}{chr(10)}{chr(10)}Respond to their points." if opponent_last_argument else "Present your opening position."}

Structure your response in these sections:

---

## PRIVATE THINKING
(Your internal reasoning - opponent will NOT see this)

Think through: strongest points, weaknesses, steelman of opponent, what they get right, where you might be wrong.

---

## SHORT FORM (Priority Headlines)

List your 3-5 most important points as priority-numbered headlines.
These should be scannable and self-contained.

Format:
1. **[Headline]** - One sentence summary
2. **[Headline]** - One sentence summary
3. **[Headline]** - One sentence summary

---

## LONG FORM (Full Argument)

### Main Claim
One clear sentence stating your position.

### Point 1: [Same headline as Short Form #1]
Full reasoning, evidence, and explanation for this point.

### Point 2: [Same headline as Short Form #2]
Full reasoning, evidence, and explanation for this point.

### Point 3: [Same headline as Short Form #3]
Full reasoning, evidence, and explanation for this point.

(Continue for each point in your Short Form list)

### Rebuttals
Responses to opponent's previous points (if any).

### Steelman
{f"State the STRONGEST version of your opponent's argument. Be generous." if require_steelman else ""}

### Concessions
Points you now agree with your opponent on.

---

## STATUS

**Confidence:** [0-100%] how confident in your position
**Ready to Conclude:** [YES/NO] - further debate unlikely to change positions?
**Proposed Synthesis:** (if ready) What resolution could both sides accept?"""


def get_judge_prompt(topic: str, full_debate: str, human_qa: Optional[str] = None) -> str:
    qa_section = ""
    if human_qa:
        qa_section = f"""
HUMAN'S CONTEXT:
{human_qa}
"""

    return f"""You are an impartial judge evaluating this debate.

TOPIC: {topic}
{qa_section}
FULL DEBATE:
{full_debate}

Evaluate based on: logical coherence, quality of reasoning, responsiveness, intellectual honesty, steelmanning quality, addressing the human's needs.

Write your judgment:

## Summary
Brief overview of the debate.

## Winner
Declare: ADVOCATE, CRITIC, or TIE

## Reasoning
Why this side won (or why it's a tie).

## Recommended Decision
What should the human actually do? Be practical and specific.

## Key Takeaways
Bullet points of the most valuable insights."""


def parse_response(text: str) -> tuple[str, str, str, float, bool, Optional[str]]:
    """Parse agent response into thinking, short form, and long form sections"""
    import re

    thinking = ""
    short_form = ""
    long_form = ""
    confidence = 0.5
    ready = False
    synthesis = None

    # Extract PRIVATE THINKING
    if "## PRIVATE THINKING" in text:
        parts = text.split("## PRIVATE THINKING", 1)
        if "## SHORT FORM" in parts[1]:
            thinking = parts[1].split("## SHORT FORM")[0].strip()
        else:
            thinking = parts[1][:1000].strip()

    # Extract SHORT FORM
    if "## SHORT FORM" in text:
        parts = text.split("## SHORT FORM", 1)
        if "## LONG FORM" in parts[1]:
            short_form = parts[1].split("## LONG FORM")[0].strip()
        else:
            short_form = parts[1][:1000].strip()
        # Clean up the header
        short_form = short_form.replace("(Priority Headlines)", "").strip()

    # Extract LONG FORM
    if "## LONG FORM" in text:
        parts = text.split("## LONG FORM", 1)
        if "## STATUS" in parts[1]:
            long_form = parts[1].split("## STATUS")[0].strip()
        else:
            long_form = parts[1].strip()
        # Clean up the header
        long_form = long_form.replace("(Full Argument)", "").strip()

    # Fallback if sections not found
    if not short_form and not long_form:
        long_form = text

    # Extract confidence
    for line in text.split("\n"):
        if "confidence" in line.lower():
            numbers = re.findall(r'(\d+)', line)
            if numbers:
                confidence = int(numbers[0]) / 100
                break

    # Check ready to conclude
    if "ready to conclude" in text.lower():
        section = text.lower().split("ready to conclude")[1][:50]
        ready = "yes" in section

    # Extract synthesis
    if "proposed synthesis" in text.lower():
        parts = text.split("Proposed Synthesis")
        if len(parts) > 1:
            synth = parts[1].strip()
            # Get text until next section or end
            synth = synth.split("##")[0].strip()
            synth = synth.lstrip(":").strip()
            if synth and len(synth) > 10:
                synthesis = synth[:500]

    return thinking, short_form, long_form, confidence, ready, synthesis


def create_debate_folder(config: DebateConfig) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_topic = "".join(c if c.isalnum() or c in " -_" else "_" for c in config.topic)[:50]
    folder_name = f"{timestamp}_{safe_topic}"

    base = Path(config.output_dir)
    debate_path = base / folder_name

    (debate_path / config.agent_a_name.lower()).mkdir(parents=True, exist_ok=True)
    (debate_path / config.agent_b_name.lower()).mkdir(parents=True, exist_ok=True)

    return debate_path


def write_round_doc(
    path: Path,
    agent_name: str,
    position: str,
    round_num: int,
    thinking: str,
    short_form: str,
    long_form: str,
    confidence: float
):
    """Write a round document with both short and long forms"""
    doc = f"""# {agent_name} - Round {round_num}
**Position:** {position}
**Confidence:** {confidence:.0%}

---

## Private Thinking
*(Hidden from opponent during debate)*

{thinking}

---

## Short Form (Priority Headlines)

{short_form}

---

## Long Form (Full Argument)

{long_form}
"""
    filename = path / agent_name.lower() / f"round_{round_num}.md"
    filename.write_text(doc)


def write_qa_doc(path: Path, questions_a: str, questions_b: str, answers: dict, config: DebateConfig):
    doc = f"""# Pre-Debate Questions & Answers

Both agents asked clarifying questions before the debate began.

---

## Questions from {config.agent_a_name} ({config.agent_a_position})

{questions_a}

---

## Questions from {config.agent_b_name} ({config.agent_b_position})

{questions_b}

---

## Human's Answers

"""
    for q, a in answers.items():
        doc += f"**Q: {q}**\n\n{a}\n\n"

    (path / "qa.md").write_text(doc)


def write_overview(path: Path, config: DebateConfig):
    model_info = f"- **{config.agent_a_name} Model:** `{config.model_a}`\n"
    model_info += f"- **{config.agent_b_name} Model:** `{config.model_b}`"
    if config.resolution_mode == ResolutionMode.JUDGE:
        model_info += f"\n- **Judge Model:** `{config.model_judge}`"

    doc = f"""# Debate Overview

**Topic:** {config.topic}

**Started:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Configuration
- **Max Rounds:** {config.max_rounds}
- **Resolution Mode:** {config.resolution_mode.value}
- **Tone:** {config.tone.value}
{model_info}

## Participants
- **{config.agent_a_name}:** {config.agent_a_position}
- **{config.agent_b_name}:** {config.agent_b_position}

---

*Documents:*
- `qa.md` - Pre-debate Q&A
- `{config.agent_a_name.lower()}/round_N.md` - {config.agent_a_name}'s arguments
- `{config.agent_b_name.lower()}/round_N.md` - {config.agent_b_name}'s arguments
- `outcome.md` - Final result
"""
    (path / "overview.md").write_text(doc)


def write_outcome(
    path: Path,
    config: DebateConfig,
    rounds_data: list,
    resolution: str,
    winner: Optional[str],
    synthesis: str,
    judge_analysis: Optional[str] = None
):
    """Write outcome with scannable round-by-round headlines"""

    # Build confidence trajectory
    trajectory = "| Round | " + config.agent_a_name + " | " + config.agent_b_name + " |\n"
    trajectory += "|-------|" + "-" * (len(config.agent_a_name) + 2) + "|" + "-" * (len(config.agent_b_name) + 2) + "|\n"
    for r in rounds_data:
        trajectory += f"| {r['round']} | {r['a_conf']:.0%} | {r['b_conf']:.0%} |\n"

    # Build round-by-round summary with short forms
    round_summary = ""
    for r in rounds_data:
        round_summary += f"### Round {r['round']}\n\n"
        round_summary += f"**{config.agent_a_name}** ({r['a_conf']:.0%} confident):\n"
        if r.get('a_short'):
            round_summary += r['a_short'] + "\n\n"
        else:
            round_summary += "*[No short form available]*\n\n"

        round_summary += f"**{config.agent_b_name}** ({r['b_conf']:.0%} confident):\n"
        if r.get('b_short'):
            round_summary += r['b_short'] + "\n\n"
        else:
            round_summary += "*[No short form available]*\n\n"

        round_summary += "---\n\n"

    doc = f"""# Debate Outcome

**Topic:** {config.topic}

**Resolution:** {resolution}

**Winner:** {winner or "No clear winner (synthesis reached)"}

---

## Confidence Trajectory

{trajectory}

---

## Round-by-Round Summary (Headlines Only)

{round_summary}

---

## Final Synthesis

{synthesis}

"""

    if judge_analysis:
        doc += f"""---

## Judge's Analysis

{judge_analysis}
"""

    model_info = f"{config.agent_a_name}: {config.model_a} | {config.agent_b_name}: {config.model_b}"
    if judge_analysis:
        model_info += f" | Judge: {config.model_judge}"

    doc += f"""
---

*Completed: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | {model_info} | Rounds: {len(rounds_data)}*
"""

    (path / "outcome.md").write_text(doc)


def gather_questions(client: MultiProviderClient, config: DebateConfig) -> tuple[str, str, dict, str]:
    print("\n--- Pre-Debate Question Phase ---\n")

    print(f"{config.agent_a_name} ({config.model_a}) is preparing questions...")
    questions_a = client.create_message(
        model=config.model_a,
        max_tokens=1000,
        messages=[{"role": "user", "content": get_question_prompt(
            config.agent_a_name, config.agent_a_position, config.topic, config.max_questions_per_side
        )}]
    ).strip()

    print(f"{config.agent_b_name} ({config.model_b}) is preparing questions...")
    questions_b = client.create_message(
        model=config.model_b,
        max_tokens=1000,
        messages=[{"role": "user", "content": get_question_prompt(
            config.agent_b_name, config.agent_b_position, config.topic, config.max_questions_per_side
        )}]
    ).strip()

    print(f"\n{'='*60}")
    print("QUESTIONS FOR YOU (the human decision-maker)")
    print(f"{'='*60}\n")

    all_questions = []

    if "NO QUESTIONS" not in questions_a.upper():
        print(f"From {config.agent_a_name} ({config.agent_a_position}):")
        print(questions_a)
        for line in questions_a.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                q = line.lstrip('0123456789.-) ').strip()
                if q and '?' in q:
                    all_questions.append((config.agent_a_name, q))
        print()

    if "NO QUESTIONS" not in questions_b.upper():
        print(f"From {config.agent_b_name} ({config.agent_b_position}):")
        print(questions_b)
        for line in questions_b.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                q = line.lstrip('0123456789.-) ').strip()
                if q and '?' in q:
                    all_questions.append((config.agent_b_name, q))
        print()

    answers = {}
    if all_questions:
        print(f"{'='*60}")
        print("Please answer (press Enter twice to skip):")
        print(f"{'='*60}\n")

        for i, (agent, q) in enumerate(all_questions, 1):
            print(f"[{i}/{len(all_questions)}] {q}")
            print(f"  (asked by {agent})")
            lines = []
            while True:
                line = input("> ")
                if line == "":
                    if lines:
                        break
                    else:
                        lines.append("[No answer provided]")
                        break
                lines.append(line)
            answers[q] = "\n".join(lines)
            print()

    qa_text = ""
    if answers:
        for q, a in answers.items():
            qa_text += f"Q: {q}\nA: {a}\n\n"

    return questions_a, questions_b, answers, qa_text


def run_debate(config: DebateConfig):
    client = MultiProviderClient()

    debate_path = create_debate_folder(config)
    write_overview(debate_path, config)

    print(f"\n{'='*60}")
    print(f"DEBATE: {config.topic}")
    print(f"Output: {debate_path}")
    print(f"{'='*60}")

    human_qa = None
    if config.ask_questions:
        questions_a, questions_b, answers, human_qa = gather_questions(client, config)
        write_qa_doc(debate_path, questions_a, questions_b, answers, config)

    print(f"\n{'='*60}")
    print("DEBATE BEGINS")
    print(f"{'='*60}\n")

    visible_history = ""
    rounds_data = []
    last_a_argument = None
    last_b_argument = None
    conf_a = 0.5
    conf_b = 0.5

    for round_num in range(1, config.max_rounds + 1):
        print(f"--- Round {round_num} ---")

        # Agent A
        print(f"  {config.agent_a_name} ({config.model_a}) thinking...")
        text_a = client.create_message(
            model=config.model_a,
            max_tokens=4000,
            messages=[{"role": "user", "content": get_agent_prompt(
                config.agent_a_name, config.agent_a_position, config.tone,
                config.topic, visible_history, last_b_argument,
                config.require_steelman, human_qa
            )}]
        )
        thinking_a, short_a, long_a, conf_a, ready_a, synth_a = parse_response(text_a)

        write_round_doc(debate_path, config.agent_a_name, config.agent_a_position,
                       round_num, thinking_a, short_a, long_a, conf_a)
        print(f"    Confidence: {conf_a:.0%}")

        # Agent B
        print(f"  {config.agent_b_name} ({config.model_b}) thinking...")
        # Build A's public argument (short + long form, no thinking)
        a_public = f"## Short Form\n{short_a}\n\n## Long Form\n{long_a}"

        text_b = client.create_message(
            model=config.model_b,
            max_tokens=4000,
            messages=[{"role": "user", "content": get_agent_prompt(
                config.agent_b_name, config.agent_b_position, config.tone,
                config.topic, visible_history, a_public,
                config.require_steelman, human_qa
            )}]
        )
        thinking_b, short_b, long_b, conf_b, ready_b, synth_b = parse_response(text_b)

        write_round_doc(debate_path, config.agent_b_name, config.agent_b_position,
                       round_num, thinking_b, short_b, long_b, conf_b)
        print(f"    Confidence: {conf_b:.0%}")

        # Update visible history with public arguments only
        visible_history += f"\n\n### Round {round_num}\n\n"
        visible_history += f"**{config.agent_a_name}:**\n{a_public}\n\n"
        b_public = f"## Short Form\n{short_b}\n\n## Long Form\n{long_b}"
        visible_history += f"**{config.agent_b_name}:**\n{b_public}\n\n"

        rounds_data.append({
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

        # Check for consensus
        if config.resolution_mode == ResolutionMode.CONSENSUS:
            if ready_a and ready_b:
                print(f"\n✓ Both agents ready to conclude at round {round_num}")
                synthesis = synth_a or synth_b or "Both agents ready to conclude"
                write_outcome(debate_path, config, rounds_data, "consensus", None, synthesis)
                print(f"\nOutput written to: {debate_path}")
                return

            if conf_a < 0.3 and conf_b < 0.3:
                print(f"\n✓ Both agents have low confidence - converging")
                write_outcome(debate_path, config, rounds_data, "convergence", None,
                            "Both positions weakened - consider middle ground")
                print(f"\nOutput written to: {debate_path}")
                return

    print(f"\nMax rounds ({config.max_rounds}) reached.")

    if config.resolution_mode == ResolutionMode.JUDGE:
        print(f"Calling judge ({config.model_judge})...")
        judge_text = client.create_message(
            model=config.model_judge,
            max_tokens=3000,
            messages=[{"role": "user", "content": get_judge_prompt(
                config.topic, visible_history, human_qa
            )}]
        )

        winner = None
        judge_lower = judge_text.lower()
        if "winner" in judge_lower:
            winner_section = judge_lower.split("winner")[1][:100]
            if "advocate" in winner_section:
                winner = config.agent_a_name
            elif "critic" in winner_section:
                winner = config.agent_b_name
            elif "tie" in winner_section:
                winner = "Tie"

        print(f"Judge decision: {winner or 'See analysis'}")
        write_outcome(debate_path, config, rounds_data, "judge", winner,
                     "See judge's analysis below", judge_text)
    else:
        write_outcome(debate_path, config, rounds_data, "max_rounds", None,
                     f"No consensus after {config.max_rounds} rounds.\n\n"
                     f"Final positions:\n"
                     f"- {config.agent_a_name}: {conf_a:.0%} confident\n"
                     f"- {config.agent_b_name}: {conf_b:.0%} confident")

    print(f"\nOutput written to: {debate_path}")
    print(f"  - overview.md")
    print(f"  - qa.md")
    print(f"  - {config.agent_a_name.lower()}/round_*.md")
    print(f"  - {config.agent_b_name.lower()}/round_*.md")
    print(f"  - outcome.md")


def main():
    # Build model alias help text
    alias_help = "Model aliases: " + ", ".join(sorted(MODEL_ALIASES.keys()))

    parser = argparse.ArgumentParser(
        description="AI Debate Machine - Structured self-debate with human input",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  %(prog)s "Should we use microservices or monoliths?"
  %(prog)s "Is TDD worth the overhead?" --rounds 3 --tone adversarial
  %(prog)s "Tabs vs spaces" --mode judge --no-questions

  # Model battles:
  %(prog)s "Best programming language" --model-a opus-45 --model-b sonnet
  %(prog)s "AI ethics" --model-a opus-4.5 --model-b gpt-4o --model-judge opus-45

{alias_help}
        """
    )

    parser.add_argument("topic", help="The debate topic or decision to make")
    parser.add_argument("--rounds", "-r", type=int, default=5, help="Max rounds (default: 5)")
    parser.add_argument("--mode", "-m", choices=["consensus", "judge"], default="consensus",
                       help="Resolution mode (default: consensus)")
    parser.add_argument("--tone", "-t",
                       choices=["truth_seeking", "academic", "adversarial", "collaborative", "socratic"],
                       default="truth_seeking", help="Debate tone (default: truth_seeking)")

    # Model arguments
    parser.add_argument("--model", default=None,
                       help="Model for all agents (shortcut for setting all three)")
    parser.add_argument("--model-a", default="sonnet",
                       help="Model for agent A (default: sonnet)")
    parser.add_argument("--model-b", default="sonnet",
                       help="Model for agent B (default: sonnet)")
    parser.add_argument("--model-judge", default="sonnet",
                       help="Model for judge (default: sonnet)")

    # Agent configuration
    parser.add_argument("--agent-a", default="Advocate", help="Name for agent A")
    parser.add_argument("--agent-b", default="Critic", help="Name for agent B")
    parser.add_argument("--position-a", default="FOR", help="Agent A's position")
    parser.add_argument("--position-b", default="AGAINST", help="Agent B's position")

    # Other options
    parser.add_argument("--no-steelman", action="store_true", help="Don't require steelmanning")
    parser.add_argument("--no-questions", action="store_true", help="Skip the question phase")
    parser.add_argument("--max-questions", type=int, default=3, help="Max questions per side")
    parser.add_argument("--output", "-o", default="debates", help="Output directory")

    args = parser.parse_args()

    # Resolve models - if --model is specified, use it for all
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
        require_steelman=not args.no_steelman,
        output_dir=args.output,
        ask_questions=not args.no_questions,
        max_questions_per_side=args.max_questions
    )

    run_debate(config)


if __name__ == "__main__":
    main()
