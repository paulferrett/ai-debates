# AI Debate Machine

Structured AI self-debate where two agents argue a topic, each maintaining separate private thinking. Agents only see each other's public arguments—not internal reasoning. Supports cross-provider battles (Claude vs GPT).

## Features

- **Isolated thinking** - Each agent has private reasoning hidden from opponent
- **Structured output** - Short-form headlines + long-form arguments
- **Human input** - Pre-debate Q&A phase to provide context
- **Multi-provider** - Mix Claude and OpenAI models in same debate
- **Multiple interfaces** - CLI, live TUI, or programmatic
- **Post-debate analysis** - Scoring, dynamics analysis, prompt recommendations

## Installation

```bash
git clone https://github.com/paulferrett/ai-debates.git
cd ai-debates

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Add API keys
cp .env.example .env
# Edit .env with your keys
```

Create `.env`:
```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

## Usage

### Standard CLI

```bash
# Basic debate
./debate.py "Should we use microservices or monoliths?"

# Model battle: Claude vs GPT
./debate.py "Best approach to error handling" \
  --model-a sonnet \
  --model-b gpt-4o-mini \
  --rounds 3

# With judge
./debate.py "Tabs vs spaces" --mode judge

# Skip pre-debate questions
./debate.py "Is TDD worth it?" --no-questions

# Adversarial tone
./debate.py "Rust vs Go" --tone adversarial
```

### Live TUI

Split-screen terminal interface with real-time streaming:

```bash
./debate_live.py "Should AI have curiosity?" \
  --model-a sonnet \
  --model-b gpt-4o-mini
```

Controls:
- `q` - Quit
- `p` - Pause/Resume

### Post-Debate Analysis

Analyze completed debates for quality scores and recommendations:

```bash
# Analyze most recent debate
./analyze.py latest

# Analyze specific debate
./analyze.py debates/20240101_120000_Topic_here

# JSON output
./analyze.py latest --json
```

## Output Structure

Each debate creates a folder:

```
debates/20240101_120000_Should_we_use_microservices/
├── overview.md           # Config and setup
├── qa.md                 # Pre-debate questions & answers
├── advocate/
│   ├── round_1.md       # Private thinking + public argument
│   ├── round_2.md
│   └── ...
├── critic/
│   ├── round_1.md
│   ├── round_2.md
│   └── ...
├── outcome.md           # Round summaries + final synthesis
└── analysis.md          # Post-debate scoring (if analyzed)
```

## Model Aliases

| Alias | Model |
|-------|-------|
| `sonnet` | claude-sonnet-4-20250514 |
| `opus-45` | claude-opus-4-5-20251101 |
| `gpt-4o` | gpt-4o |
| `gpt-4o-mini` | gpt-4o-mini |
| `o1` | o1 |

## CLI Options

```
debate.py [topic] [options]

Options:
  --rounds, -r       Max debate rounds (default: 5)
  --mode, -m         consensus or judge (default: consensus)
  --tone, -t         truth_seeking, academic, adversarial,
                     collaborative, socratic (default: truth_seeking)
  --model            Model for all agents
  --model-a          Model for advocate (default: sonnet)
  --model-b          Model for critic (default: sonnet)
  --model-judge      Model for judge (default: sonnet)
  --no-questions     Skip pre-debate Q&A
  --no-steelman      Don't require steelmanning
  --output, -o       Output directory (default: debates)
```

## How It Works

1. **Question Phase** - Both agents ask clarifying questions, human answers
2. **Debate Rounds** - Each round:
   - Agent A presents argument (private thinking + public argument)
   - Agent B sees only A's public argument, responds
   - Both update confidence levels
3. **Resolution** - Either:
   - **Consensus**: Both agents mark "ready to conclude"
   - **Judge**: Third model evaluates and picks winner
4. **Analysis** (optional) - Score arguments, identify dynamics, suggest improvements

## Analysis Metrics

| Metric | Description |
|--------|-------------|
| Logical Coherence | Internal consistency of arguments |
| Evidence Quality | Strength of supporting points |
| Responsiveness | How well they address opponent |
| Steelman Accuracy | Fairness in representing opponent |
| Intellectual Honesty | Willingness to concede valid points |
| Persuasiveness | Overall convincingness |

## License

MIT
