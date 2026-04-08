# DebtCrush — Debt Payoff Optimisation Environment

> An OpenEnv reinforcement learning environment where agents learn to pay off
> multiple debts optimally by discovering the avalanche strategy through reward signal alone.

[![Space](https://img.shields.io/badge/🤗%20Space-Vishmeluck%2Fmy__env-blue)](https://huggingface.co/spaces/Vishmeluck/my_env)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-green)](https://github.com/meta-pytorch/OpenEnv)

---

## Overview

DebtCrush places an agent in charge of a household debt portfolio. Each month,
the agent allocates a fixed budget across multiple debts (credit cards, loans)
to minimise total interest paid and clear all debts as fast as possible.

The core challenge: the agent must **discover the avalanche strategy** (pay
highest-APR debt first) purely from reward signal — without being told it exists.

### Why this is interesting for RL

| Strategy | What it does | Total interest paid |
|---|---|---|
| Random | Splits payments randomly | ~$1,800 |
| Snowball | Pays smallest balance first | ~$900 |
| **Avalanche** | Pays highest APR first | **~$400** |
| Optimal agent | Should converge to avalanche | ~$350 |

The gap between random and optimal is large and fully measurable — making
reward signal dense and training meaningful.

---

## Environment

**Space URL:** `https://huggingface.co/spaces/Vishmeluck/my_env`  
**API endpoint:** `https://vishmeluck-my-env.hf.space`

### Interface

Every interaction follows the standard OpenEnv 3-method interface:

```python
env.reset()        # Start a new episode
env.step(action)   # Take one monthly payment action
env.state()        # Get episode metadata
```

### Action

```python
MyAction(
    extra_payments="280.0, 0.0",   # Comma-separated extra payment per debt
    task="t1"                       # Task difficulty: t1, t2, or t3
)
```

`extra_payments` are amounts **above the mandatory minimums**. Minimums are
applied automatically. The sum must not exceed the remaining budget after
minimums are paid — the environment auto-scales if exceeded.

### Observation

```python
MyObservation(
    balances=[2000.0, 3000.0],        # Current balance per debt ($)
    aprs=[0.24, 0.12],                # Annual percentage rate per debt
    min_payments=[40.0, 60.0],        # Mandatory minimum per debt ($)
    months_budget=400.0,              # Total monthly budget ($)
    total_interest_paid=0.0,          # Cumulative interest paid so far ($)
    month=0,                          # Current month (0-indexed)
    reward=0.0,                       # Reward this step
    done=False                        # Episode complete
)
```

---

## Tasks

Three tasks of increasing difficulty. All scores are deterministic and in [0.0, 1.0].

### T1 — Easy (`task="t1"`)

| Property | Value |
|---|---|
| Debts | 2 (Card A, Card B) |
| Balances | $2,000 @ 24% APR, $5,000 @ 12% APR |
| Monthly budget | $400 |
| Max steps | 60 months |

**Challenge:** Two debts with a clear APR difference. Agent must learn to
prioritise Card A (24% APR) over Card B (12% APR) despite Card A having a
lower balance.

---

### T2 — Medium (`task="t2"`)

| Property | Value |
|---|---|
| Debts | 3 (Card A, Card B, Loan) |
|Balances | $4,000 @ 24%, $2,000 @ 18%, $6,000 @ 8% |
| Monthly budget | $600 |
| Max steps | 60 months |

**Challenge:** Three debts with tighter budget. Agent must resist the temptation
to pay off the smaller Card B balance first and stay disciplined on high-APR debts.

---

### T3 — Hard (`task="t3"`)

| Property | Value |
|---|---|
| Debts | 5 (3 cards, 2 loans) |
| Balances | $5,000–$8,000 across 6%–27% APR range |
| Monthly budget | $1,000 |
| Max steps | 60 months |

**Challenge:** Five debts with wide APR spread and tight budget. Agent must
maintain a consistent avalanche ordering across a long horizon while
managing minimum payments on all debts simultaneously.

---

## Reward Design

### Per-step reward

```
r_step = − interest_this_month / initial_total_debt
```

Small negative signal each month proportional to interest paid. Encourages
the agent to minimise ongoing interest cost.

### Terminal reward (all debts cleared)

```
speed_bonus        = 1 − (months_taken / 60)
interest_efficiency = 1 − (total_interest_paid / initial_total_debt)

r_terminal = 0.6 × speed_bonus + 0.4 × interest_efficiency
```

### Timeout penalty (60 months without clearing debts)

```
r_timeout = −1.0
```

### Why this reward is interesting

- **Explore-exploit tension:** paying minimum on everything is safe but
  accumulates interest. Concentrating payments is risky but optimal.
- **Delayed gratification:** avalanche strategy sacrifices short-term
  balance reduction for long-term interest savings.
- **Anti-reward-hacking:** timeout penalty prevents the agent from
  ignoring debts; interest penalty prevents lazy minimum-only payments.

### Score formula

```
final_score = clamp(r_terminal, 0.0, 1.0)
```

All scores are deterministic given a fixed task and action sequence.

---

## Graders

| Metric | Formula | Weight |
|---|---|---|
| Speed bonus | `1 − months_taken / 60` | 60% |
| Interest efficiency | `1 − total_interest_paid / initial_debt` | 40% |

Both components are pure arithmetic — no LLM judge, no randomness.

---

## Quick Start

### Connect from Python

```python
import sys, os
sys.path.insert(0, os.path.abspath("OpenEnv/src"))

from my_env import MyAction, MyEnv

with MyEnv(base_url="https://vishmeluck-my-env.hf.space").sync() as env:
    result = env.reset()
    print("Balances:", result.observation.balances)

    # Avalanche: throw extra budget at highest-APR debt
    result = env.step(MyAction(extra_payments="280.0, 0.0", task="t1"))
    print("After month 1:", result.observation.balances)
    print("Reward:", result.reward)
```

### Install dependencies

```bash
pip install openenv-core
git clone --depth=1 https://github.com/meta-pytorch/OpenEnv.git
```

---

## Running Inference

### Required environment variables

| Variable | Description | Example |
|---|---|---|
| `HF_TOKEN` | HuggingFace API key | `hf_xxxx...` |
| `MODEL_NAME` | LLM model to use | `Qwen/Qwen2.5-72B-Instruct` |
| `API_BASE_URL` | LLM API endpoint | `https://router.huggingface.co/v1` |
| `ENV_URL` | DebtCrush Space URL | `https://vishmeluck-my-env.hf.space` |

### Run inference

```bash
# Set variables (Windows)
set HF_TOKEN=your_token_here
set MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
set API_BASE_URL=https://router.huggingface.co/v1
set ENV_URL=https://vishmeluck-my-env.hf.space

python inference.py
```

```bash
# Set variables (Linux/Mac)
export HF_TOKEN=your_token_here
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export API_BASE_URL=https://router.huggingface.co/v1
export ENV_URL=https://vishmeluck-my-env.hf.space

python inference.py
```

### Expected output format

```
[START] task=t1 env=debt_crush model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=280.00, 0.00 reward=-0.01 done=false error=null
[STEP] step=2 action=280.00, 0.00 reward=-0.01 done=false error=null
...
[END] success=true steps=24 score=0.82 rewards=-0.01,-0.01,...,0.82
[START] task=t2 env=debt_crush model=Qwen/Qwen2.5-72B-Instruct
...
[START] task=t3 env=debt_crush model=Qwen/Qwen2.5-72B-Instruct
...
```

---

## Project Structure

```
my_env/
├── inference.py              ← Inference script (entry point)
├── models.py                 ← Pydantic action/observation models
├── client.py                 ← OpenEnv client
├── README.md                 ← This file
├── openenv.yaml              ← OpenEnv configuration
├── pyproject.toml            ← Package metadata
└── server/
    ├── app.py                ← FastAPI server
    ├── my_env_environment.py ← Core environment logic
    ├── Dockerfile            ← Container definition
    └── requirements.txt      ← Server dependencies
```

---

## Environment Variables (server-side)

No server-side environment variables are required. The environment is
fully self-contained and stateless between episodes.

---

## Scaling

The environment supports concurrent WebSocket sessions
(`SUPPORTS_CONCURRENT_SESSIONS = True`). Each session gets its own
isolated environment instance.

| Deployment | Concurrent sessions |
|---|---|
| HF Spaces (free) | ~128 |
| Single Docker container | ~2,048 |
