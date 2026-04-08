"""
Inference Script — DebtCrush Environment
=========================================
Runs an LLM agent against the DebtCrush environment across all 3 tasks.
The agent must allocate monthly debt payments to minimise total interest paid.

Required environment variables:
    API_BASE_URL   LLM endpoint (default: HuggingFace router)
    MODEL_NAME     Model identifier
    HF_TOKEN       HuggingFace / API key
"""

import asyncio
import json
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

import sys
repo = os.path.abspath("OpenEnv")
for p in [repo, os.path.join(repo, "src")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from my_env import MyAction, MyEnv  # noqa: E402  (generated client)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
ENV_URL      = os.getenv("ENV_URL",      "https://vishmeluck-my-env.hf.space")
BENCHMARK    = "debt_crush"
MAX_STEPS    = 60       # hard cap (5 years of months)
TEMPERATURE  = 0.2      # low temp for consistent numeric output
MAX_TOKENS   = 100

TASKS = ["t1", "t2", "t3"]

# Number of debts per task — agent needs to know how many values to output
TASK_DEBT_COUNTS = {"t1": 2, "t2": 3, "t3": 5}

# ---------------------------------------------------------------------------
# Logging helpers (mandatory format)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# LLM prompt helpers
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You are a debt repayment agent. Each month you allocate extra payments
    across multiple debts to minimise total interest paid.

    Rules:
    - You will be told the number of debts, their balances, APRs, and monthly budget.
    - Minimum payments are handled automatically — you only allocate the EXTRA budget.
    - Respond with ONLY a JSON array of floats, one per debt, e.g.: [200.0, 0.0]
    - The sum of your values must not exceed the extra budget shown.
    - Focus extra payments on the highest APR debt first (avalanche strategy).
    - If a debt balance is 0, assign 0 to it.
""").strip()


def build_user_prompt(obs, extra_budget: float, n_debts: int) -> str:
    lines = []
    for i in range(n_debts):
        bal = obs.balances[i] if i < len(obs.balances) else 0.0
        apr = obs.aprs[i] if i < len(obs.aprs) else 0.0
        mp  = obs.min_payments[i] if i < len(obs.min_payments) else 0.0
        lines.append(f"  Debt {i}: balance=${bal:.2f}, APR={apr*100:.1f}%, min_payment=${mp:.2f}")

    debt_block = "\n".join(lines)
    return textwrap.dedent(f"""
        Month: {obs.month}
        Monthly budget: ${obs.months_budget:.2f}
        Extra budget available: ${extra_budget:.2f}
        Total interest paid so far: ${obs.total_interest_paid:.2f}

        Debts:
        {debt_block}

        Reply with a JSON array of {n_debts} floats summing to at most {extra_budget:.2f}.
    """).strip()


def get_agent_action(client: OpenAI, obs, n_debts: int) -> List[float]:
    """Ask the LLM for extra payment allocations. Falls back to avalanche if parsing fails."""
    # Compute extra budget = total budget - sum of minimums for active debts
    min_total = sum(
        min(obs.min_payments[i], obs.balances[i])
        for i in range(n_debts)
        if i < len(obs.balances) and obs.balances[i] > 0
    )
    extra_budget = max(0.0, obs.months_budget - min_total)

    user_prompt = build_user_prompt(obs, extra_budget, n_debts)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text = (completion.choices[0].message.content or "").strip()
        payments = json.loads(text)
        if isinstance(payments, list) and len(payments) == n_debts:
            return [max(0.0, float(p)) for p in payments]
    except Exception as e:
        print(f"[DEBUG] LLM parse error: {e} | raw: {text!r}", flush=True)

    # Fallback: pure avalanche — put everything on highest APR active debt
    result = [0.0] * n_debts
    best_i, best_apr = -1, -1.0
    for i in range(n_debts):
        if i < len(obs.balances) and obs.balances[i] > 0:
            if obs.aprs[i] > best_apr:
                best_apr = obs.aprs[i]
                best_i = i
    if best_i >= 0:
        result[best_i] = extra_budget
    return result


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

async def run_episode(task: str) -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    n_debts = TASK_DEBT_COUNTS[task]

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    async with MyEnv(base_url=ENV_URL) as env:
        try:
            # Reset with task selection via action field
            result = await env.reset()
            obs = result.observation

            for step in range(1, MAX_STEPS + 1):
                if result.done:
                    break

                extra_payments = get_agent_action(client, obs, n_debts)

                # Encode task in action (extra_payments as comma string per your models.py)
                action_str = ", ".join(f"{p:.2f}" for p in extra_payments)
                action = MyAction(extra_payments=action_str, task=task)

                result = await env.step(action)
                obs = result.observation

                reward     = result.reward or 0.0
                done       = result.done
                steps_taken = step
                rewards.append(reward)

                log_step(
                    step=step,
                    action=action_str,
                    reward=reward,
                    done=done,
                    error=None,
                )

                if done:
                    break

            # Score = clamp terminal reward to [0, 1]
            # Terminal reward from env already encodes speed + interest efficiency
            terminal = rewards[-1] if rewards else 0.0
            score = min(max(float(terminal), 0.0), 1.0)
            success = score >= 0.1

        except Exception as exc:
            print(f"[DEBUG] Episode error: {exc}", flush=True)

        finally:
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ---------------------------------------------------------------------------
# Main — run all 3 tasks
# ---------------------------------------------------------------------------

async def main() -> None:
    for task in TASKS:
        await run_episode(task)


if __name__ == "__main__":
    asyncio.run(main())
