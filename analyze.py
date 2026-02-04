#!/usr/bin/env python3
"""
Debate Analyzer - Post-debate analysis and scoring

Analyzes completed debates to provide:
- Quality scores for each side
- Debate dynamics analysis
- Prompt/system recommendations for improvement
"""

import argparse
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import re

from dotenv import load_dotenv
load_dotenv()

import anthropic
import openai

from debate import get_provider, resolve_model


@dataclass
class ArgumentScores:
    logical_coherence: int  # 0-100
    evidence_quality: int
    responsiveness: int
    steelman_accuracy: int
    intellectual_honesty: int
    persuasiveness: int
    overall: int

    def to_dict(self):
        return {
            "logical_coherence": self.logical_coherence,
            "evidence_quality": self.evidence_quality,
            "responsiveness": self.responsiveness,
            "steelman_accuracy": self.steelman_accuracy,
            "intellectual_honesty": self.intellectual_honesty,
            "persuasiveness": self.persuasiveness,
            "overall": self.overall,
        }


@dataclass
class DebateAnalysis:
    agent_a_scores: ArgumentScores
    agent_b_scores: ArgumentScores
    dynamics: dict
    recommendations: list[str]
    prompt_suggestions: dict
    winner_analysis: str


ANALYSIS_PROMPT = """You are an expert debate analyst. Analyze this debate and provide detailed scoring and recommendations.

DEBATE TOPIC: {topic}

FULL DEBATE CONTENT:
{content}

Analyze the debate and respond with a JSON object (no markdown code blocks, just raw JSON):

{{
    "agent_a_scores": {{
        "logical_coherence": <0-100>,
        "evidence_quality": <0-100>,
        "responsiveness": <0-100>,
        "steelman_accuracy": <0-100>,
        "intellectual_honesty": <0-100>,
        "persuasiveness": <0-100>,
        "overall": <0-100>,
        "reasoning": "<brief explanation>"
    }},
    "agent_b_scores": {{
        "logical_coherence": <0-100>,
        "evidence_quality": <0-100>,
        "responsiveness": <0-100>,
        "steelman_accuracy": <0-100>,
        "intellectual_honesty": <0-100>,
        "persuasiveness": <0-100>,
        "overall": <0-100>,
        "reasoning": "<brief explanation>"
    }},
    "dynamics": {{
        "momentum_shifts": ["<description of key moments where momentum changed>"],
        "knockout_arguments": ["<arguments that significantly impacted the debate>"],
        "missed_opportunities": ["<points either side should have made but didn't>"],
        "logical_fallacies": ["<any fallacies detected, with attribution>"],
        "strawman_instances": ["<any strawmanning detected>"],
        "strongest_exchange": "<description of the most productive exchange>",
        "weakest_exchange": "<description of where debate quality dropped>"
    }},
    "winner_analysis": "<detailed explanation of who won and why, or why it's a tie>",
    "recommendations": [
        "<specific actionable recommendation for improving debate quality>",
        "<another recommendation>",
        "<etc>"
    ],
    "prompt_suggestions": {{
        "agent_a": [
            "<specific prompt adjustment to improve agent A's performance>",
            "<another suggestion>"
        ],
        "agent_b": [
            "<specific prompt adjustment to improve agent B's performance>",
            "<another suggestion>"
        ],
        "general": [
            "<system-wide prompt improvements>",
            "<another suggestion>"
        ]
    }}
}}

Be specific and actionable in your recommendations. Reference specific moments from the debate."""


def read_debate_content(debate_path: Path) -> tuple[str, str, list[dict]]:
    """Read all content from a debate folder"""
    topic = ""
    content_parts = []
    rounds_data = []

    # Read overview for topic
    overview_path = debate_path / "overview.md"
    if overview_path.exists():
        overview = overview_path.read_text()
        # Extract topic
        for line in overview.split("\n"):
            if line.startswith("**Topic:**"):
                topic = line.replace("**Topic:**", "").strip()
                break

    # Read Q&A if exists
    qa_path = debate_path / "qa.md"
    if qa_path.exists():
        content_parts.append("=== PRE-DEBATE Q&A ===\n")
        content_parts.append(qa_path.read_text())

    # Find agent folders
    agent_folders = [d for d in debate_path.iterdir() if d.is_dir()]

    # Determine round count
    max_round = 0
    for folder in agent_folders:
        for f in folder.glob("round_*.md"):
            round_num = int(f.stem.split("_")[1])
            max_round = max(max_round, round_num)

    # Read rounds in order
    for round_num in range(1, max_round + 1):
        content_parts.append(f"\n=== ROUND {round_num} ===\n")

        round_data = {"round": round_num}

        for folder in sorted(agent_folders):
            agent_name = folder.name
            round_file = folder / f"round_{round_num}.md"

            if round_file.exists():
                content_parts.append(f"\n--- {agent_name.upper()} ---\n")
                round_content = round_file.read_text()
                content_parts.append(round_content)

                # Extract confidence
                for line in round_content.split("\n"):
                    if "Confidence:" in line:
                        match = re.search(r'(\d+)', line)
                        if match:
                            conf = int(match.group(1)) / 100
                            round_data[f"{agent_name}_conf"] = conf

        rounds_data.append(round_data)

    # Read outcome
    outcome_path = debate_path / "outcome.md"
    if outcome_path.exists():
        content_parts.append("\n=== OUTCOME ===\n")
        content_parts.append(outcome_path.read_text())

    return topic, "\n".join(content_parts), rounds_data


def analyze_debate(debate_path: Path, model: str = "claude-sonnet-4-20250514") -> DebateAnalysis:
    """Analyze a completed debate"""

    topic, content, rounds_data = read_debate_content(debate_path)

    # Call LLM for analysis
    provider = get_provider(model)

    prompt = ANALYSIS_PROMPT.format(topic=topic, content=content[:50000])  # Limit content size

    if provider == "anthropic":
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=model,
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )
        result_text = response.content[0].text
    else:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=model,
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )
        result_text = response.choices[0].message.content

    # Parse JSON response
    try:
        # Clean up response if needed
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0]
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0]

        data = json.loads(result_text.strip())

        a_scores = ArgumentScores(
            logical_coherence=data["agent_a_scores"]["logical_coherence"],
            evidence_quality=data["agent_a_scores"]["evidence_quality"],
            responsiveness=data["agent_a_scores"]["responsiveness"],
            steelman_accuracy=data["agent_a_scores"]["steelman_accuracy"],
            intellectual_honesty=data["agent_a_scores"]["intellectual_honesty"],
            persuasiveness=data["agent_a_scores"]["persuasiveness"],
            overall=data["agent_a_scores"]["overall"],
        )

        b_scores = ArgumentScores(
            logical_coherence=data["agent_b_scores"]["logical_coherence"],
            evidence_quality=data["agent_b_scores"]["evidence_quality"],
            responsiveness=data["agent_b_scores"]["responsiveness"],
            steelman_accuracy=data["agent_b_scores"]["steelman_accuracy"],
            intellectual_honesty=data["agent_b_scores"]["intellectual_honesty"],
            persuasiveness=data["agent_b_scores"]["persuasiveness"],
            overall=data["agent_b_scores"]["overall"],
        )

        return DebateAnalysis(
            agent_a_scores=a_scores,
            agent_b_scores=b_scores,
            dynamics=data["dynamics"],
            recommendations=data["recommendations"],
            prompt_suggestions=data["prompt_suggestions"],
            winner_analysis=data["winner_analysis"],
        )

    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error parsing analysis: {e}")
        print(f"Raw response: {result_text[:1000]}")
        raise


def format_score_bar(score: int, width: int = 20) -> str:
    """Create a visual score bar"""
    filled = int(score / 100 * width)
    empty = width - filled

    if score >= 80:
        color = "green"
    elif score >= 60:
        color = "yellow"
    else:
        color = "red"

    return f"[{color}]{'█' * filled}{'░' * empty}[/{color}] {score}"


def write_analysis_report(analysis: DebateAnalysis, debate_path: Path, topic: str):
    """Write analysis report to debate folder"""

    report = f"""# Debate Analysis Report

**Topic:** {topic}
**Analyzed:** {Path(debate_path).name}

---

## Argument Quality Scores

### Agent A (Advocate)

| Metric | Score |
|--------|-------|
| Logical Coherence | {analysis.agent_a_scores.logical_coherence}/100 |
| Evidence Quality | {analysis.agent_a_scores.evidence_quality}/100 |
| Responsiveness | {analysis.agent_a_scores.responsiveness}/100 |
| Steelman Accuracy | {analysis.agent_a_scores.steelman_accuracy}/100 |
| Intellectual Honesty | {analysis.agent_a_scores.intellectual_honesty}/100 |
| Persuasiveness | {analysis.agent_a_scores.persuasiveness}/100 |
| **Overall** | **{analysis.agent_a_scores.overall}/100** |

### Agent B (Critic)

| Metric | Score |
|--------|-------|
| Logical Coherence | {analysis.agent_b_scores.logical_coherence}/100 |
| Evidence Quality | {analysis.agent_b_scores.evidence_quality}/100 |
| Responsiveness | {analysis.agent_b_scores.responsiveness}/100 |
| Steelman Accuracy | {analysis.agent_b_scores.steelman_accuracy}/100 |
| Intellectual Honesty | {analysis.agent_b_scores.intellectual_honesty}/100 |
| Persuasiveness | {analysis.agent_b_scores.persuasiveness}/100 |
| **Overall** | **{analysis.agent_b_scores.overall}/100** |

---

## Winner Analysis

{analysis.winner_analysis}

---

## Debate Dynamics

### Momentum Shifts
"""

    for shift in analysis.dynamics.get("momentum_shifts", []):
        report += f"- {shift}\n"

    report += """
### Knockout Arguments
"""
    for arg in analysis.dynamics.get("knockout_arguments", []):
        report += f"- {arg}\n"

    report += """
### Missed Opportunities
"""
    for opp in analysis.dynamics.get("missed_opportunities", []):
        report += f"- {opp}\n"

    report += """
### Logical Fallacies Detected
"""
    fallacies = analysis.dynamics.get("logical_fallacies", [])
    if fallacies:
        for f in fallacies:
            report += f"- {f}\n"
    else:
        report += "- None detected\n"

    report += """
### Strawman Instances
"""
    strawmen = analysis.dynamics.get("strawman_instances", [])
    if strawmen:
        for s in strawmen:
            report += f"- {s}\n"
    else:
        report += "- None detected\n"

    report += f"""
### Best Exchange
{analysis.dynamics.get("strongest_exchange", "N/A")}

### Weakest Exchange
{analysis.dynamics.get("weakest_exchange", "N/A")}

---

## Recommendations for Future Debates

"""
    for i, rec in enumerate(analysis.recommendations, 1):
        report += f"{i}. {rec}\n"

    report += """
---

## Prompt Adjustment Suggestions

### For Agent A (Advocate)
"""
    for sug in analysis.prompt_suggestions.get("agent_a", []):
        report += f"- {sug}\n"

    report += """
### For Agent B (Critic)
"""
    for sug in analysis.prompt_suggestions.get("agent_b", []):
        report += f"- {sug}\n"

    report += """
### System-Wide Improvements
"""
    for sug in analysis.prompt_suggestions.get("general", []):
        report += f"- {sug}\n"

    # Write report
    report_path = debate_path / "analysis.md"
    report_path.write_text(report)

    return report_path


def print_analysis(analysis: DebateAnalysis, topic: str):
    """Print analysis to console with formatting"""
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel

    console = Console()

    console.print(f"\n[bold]Debate Analysis: {topic}[/bold]\n")

    # Score comparison table
    table = Table(title="Argument Quality Scores")
    table.add_column("Metric", style="cyan")
    table.add_column("Agent A", justify="right")
    table.add_column("Agent B", justify="right")

    metrics = [
        ("Logical Coherence", "logical_coherence"),
        ("Evidence Quality", "evidence_quality"),
        ("Responsiveness", "responsiveness"),
        ("Steelman Accuracy", "steelman_accuracy"),
        ("Intellectual Honesty", "intellectual_honesty"),
        ("Persuasiveness", "persuasiveness"),
        ("Overall", "overall"),
    ]

    for label, key in metrics:
        a_score = getattr(analysis.agent_a_scores, key)
        b_score = getattr(analysis.agent_b_scores, key)

        a_color = "green" if a_score >= 70 else "yellow" if a_score >= 50 else "red"
        b_color = "green" if b_score >= 70 else "yellow" if b_score >= 50 else "red"

        if key == "overall":
            table.add_row(
                f"[bold]{label}[/bold]",
                f"[bold {a_color}]{a_score}[/bold {a_color}]",
                f"[bold {b_color}]{b_score}[/bold {b_color}]"
            )
        else:
            table.add_row(
                label,
                f"[{a_color}]{a_score}[/{a_color}]",
                f"[{b_color}]{b_score}[/{b_color}]"
            )

    console.print(table)
    console.print()

    # Winner analysis
    console.print(Panel(analysis.winner_analysis, title="Winner Analysis"))

    # Key recommendations
    console.print("\n[bold]Key Recommendations:[/bold]")
    for i, rec in enumerate(analysis.recommendations[:5], 1):
        console.print(f"  {i}. {rec}")

    # Prompt suggestions
    console.print("\n[bold]Prompt Suggestions:[/bold]")
    if analysis.prompt_suggestions.get("agent_a"):
        console.print("  [cyan]Agent A:[/cyan]")
        for sug in analysis.prompt_suggestions["agent_a"][:2]:
            console.print(f"    • {sug}")
    if analysis.prompt_suggestions.get("agent_b"):
        console.print("  [yellow]Agent B:[/yellow]")
        for sug in analysis.prompt_suggestions["agent_b"][:2]:
            console.print(f"    • {sug}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze a completed debate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s debates/20240101_120000_Topic_here
  %(prog)s debates/latest --model opus-45
        """
    )

    parser.add_argument("debate_path", help="Path to debate folder")
    parser.add_argument("--model", "-m", default="sonnet", help="Model for analysis (default: sonnet)")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")

    args = parser.parse_args()

    debate_path = Path(args.debate_path)

    # Handle "latest" shortcut
    if args.debate_path == "debates/latest" or args.debate_path == "latest":
        debates_dir = Path("debates")
        if debates_dir.exists():
            folders = sorted([d for d in debates_dir.iterdir() if d.is_dir()])
            if folders:
                debate_path = folders[-1]
            else:
                print("No debates found")
                return
        else:
            print("No debates directory found")
            return

    if not debate_path.exists():
        print(f"Debate folder not found: {debate_path}")
        return

    model = resolve_model(args.model)
    print(f"Analyzing {debate_path} with {model}...")

    topic, _, _ = read_debate_content(debate_path)
    analysis = analyze_debate(debate_path, model)

    if args.json:
        output = {
            "topic": topic,
            "agent_a_scores": analysis.agent_a_scores.to_dict(),
            "agent_b_scores": analysis.agent_b_scores.to_dict(),
            "dynamics": analysis.dynamics,
            "winner_analysis": analysis.winner_analysis,
            "recommendations": analysis.recommendations,
            "prompt_suggestions": analysis.prompt_suggestions,
        }
        print(json.dumps(output, indent=2))
    else:
        print_analysis(analysis, topic)

        report_path = write_analysis_report(analysis, debate_path, topic)
        print(f"\nFull report written to: {report_path}")


if __name__ == "__main__":
    main()
