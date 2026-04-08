# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
My Env Environment Implementation.

A simple test environment that echoes back messages sent to it.
Perfect for testing HTTP server infrastructure.
"""

from uuid import uuid4
from typing import List

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import MyAction, MyObservation
except ImportError:
    from models import MyAction, MyObservation


class MyEnvironment(Environment):
    """
    A simple echo environment that echoes back messages.

    This environment is designed for testing the HTTP server infrastructure.
    It maintains minimal state and simply echoes back whatever message it receives.

    Example:
        >>> env = MyEnvironment()
        >>> obs = env.reset()
        >>> print(obs.echoed_message)  # "My Env environment ready!"
        >>>
        >>> obs = env.step(MyAction(message="Hello"))
        >>> print(obs.echoed_message)  # "Hello"
        >>> print(obs.message_length)  # 5
    """
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    Max_Months = 60

    TASKS = {
        "t1": {
            "monthly_budget": 400.0,
            "debts":[
                {"name": "CardA", "balance": 2000.0,"apr": 0.24,"min_payment": 40.0},
                {"name": "CardB", "balance": 5000.0, "apr": 0.12, "min_payment": 60.0},
            ],
        },
        "t2" : {
            "monthly_budget": 600.0,
            "debts":[
                {"name": "CardA", "balance": 4000.0,"apr": 0.24,"min_payment": 80.0},
                {"name": "CardB", "balance": 2000.0, "apr": 0.12, "min_payment": 40.0},
                {"name": "Loan", "balance": 6000.0, "apr": 0.08, "min_payment": 100.0},
            ],
        },
        "t3" : {
            "monthly_budget": 1000.0,
            "debts":[
                {"name": "CardA", "balance": 5000.0,"apr": 0.27,"min_payment": 100.0},
                {"name": "CardB", "balance": 3000.0, "apr": 0.18, "min_payment": 60.0},
                {"name": "CardC", "balance": 1500.0, "apr": 0.15, "min_payment": 30.0},
                {"name": "LoanA", "balance": 4000.0, "apr": 0.07, "min_payment": 80.0},
                {"name": "LoanB", "balance": 8000.0, "apr": 0.09, "min_payment": 160.0},
            ],
        },
    }

    # Enable concurrent WebSocket sessions.
    # Set to True if your environment isolates state between instances.
    # When True, multiple WebSocket clients can connect simultaneously, each
    # getting their own environment instance (when using factory mode in app.py).
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, task: str = "t1"):
        """Initialize the my_env environment."""
        self.task = task if task in self.TASKS else "t1"
        self._state = State(episode_id = str(uuid4()), step_count= 0)
        self._balances: List[float] = []
        self._aprs: List[float] = []
        self._min_payments: List[float] = []
        self._monthly_budget: float = 0.0
        self._total_interest_paid: float = 0.0
        self._initial_total_debt: float = 0.0
        self._month: int = 0
        self.reset()

    def reset(self) -> MyObservation:
        """
        Reset the environment.

        Returns:
            MyObservation with a ready message
        """
        cfg = self.TASKS[self.task]
        self._balances = [d["balance"] for d in cfg["debts"]]
        self._aprs = [d["apr"] for d in cfg["debts"]]
        self._min_payments = [d["min_payment"] for d in cfg["debts"]]
        self._monthly_budget = cfg["monthly_budget"]
        self._total_interest_paid = 0.0
        self._initial_total_debt = sum(self._balances)
        self._month = 0
        self._state = State(episode_id=str(uuid4()), step_count=0)
 
        return self._make_observation(reward=0.0, done=False)

    def step(self, action: MyAction) -> MyObservation:  # type: ignore[override]
        """
        Execute a step in the environment by echoing the message.

        Args:
            action: MyAction containing the message to echo

        Returns:
            MyObservation with the echoed message and its length
        """
        n = len(self._balances)
        extra = list(action.extra_payments)

        if len(extra) < n:
            extra += [0.0] * (n - len(extra))
        extra = extra[:n]

        #Charge interest
        interest_this_month = 0.0
        for i in range(n):
            if self._balances[i] > 0:
                interest = self._balances[i] * (self._aprs[i] / 12)
                self._balances[i] += interest
                interest_this_month += interest
        self._total_interest_paid += interest_this_month

        #pay minimums
        minimum_spent = 0.0
        for i in range(n):
            if self._balances[i] > 0:
                pay = min(self._min_payments[i], self._balances[i])
                self._balances[i] -= pay
                minimum_spent += pay
        
        #Apply extra payments
        budget_for_extra = max(0.0, self._monthly_budget - minimum_spent)
        total_extra_requested = sum(max(0.0,e) for e in extra)

        #Auto scale if agent over-spends
        scale = 1.0
        if total_extra_requested > budget_for_extra and total_extra_requested > 0:
            scale = budget_for_extra / total_extra_requested
 
        for i in range(n):
            if self._balances[i] > 0 and extra[i] > 0:
                actual_extra = extra[i] * scale
                self._balances[i] = max(0.0, self._balances[i] - actual_extra)
 
        self._month += 1
        self._state.step_count = self._month

        #Terminate check
        all_paid = all(b <= 0.01 for b in self._balances)
        timed_out = self._month >= self.Max_Months
        done = all_paid or timed_out

        #Reward
        reward = -interest_this_month / self._initial_total_debt
 
        if done:
            if all_paid:
                # Speed bonus: finishing in fewer months is better
                speed_bonus = 1.0 - (self._month / self.MAX_MONTHS)
                # Interest efficiency: paying less total interest is better
                interest_efficiency = max(
                    0.0, 1.0 - (self._total_interest_paid / self._initial_total_debt)
                )
                reward += 0.6 * speed_bonus + 0.4 * interest_efficiency
            else:
                # Timed out without clearing debts — hard penalty
                reward -= 1.0
 
        return self._make_observation(reward=round(reward, 6), done=done)

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state

    def _make_observation(self, reward: float, done: bool) -> MyObservation:
        return  MyObservation(
        balances=[round(b, 2) for b in self._balances],
        aprs=self._aprs,
        min_payments=self._min_payments,
        months_budget=self._monthly_budget,
        total_interest_paid=round(self._total_interest_paid, 2),
        month=self._month,
        reward=reward,
        done=done,
    )
