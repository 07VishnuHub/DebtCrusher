# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the My Env Environment.

The my_env environment is a simple test environment that echoes back messages.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field
from typing import List


class MyAction(Action):
    """Action for the My Env environment - just a message to echo."""

    extra_payments: List[float] = Field(..., description="Extra payment per debt above minimum. Length must match number of debts.")
    task: str = Field(default="t1", description="Task difficulty: 't1'(easy), 't2'(medium), 't3'(hard)")


class MyObservation(Observation):
    """Observation from the My Env environment - the echoed message."""

    balances: List[float] = Field(default_factory=list, description="List of balances for each debt")
    aprs: List[float] = Field(default_factory=list, description="Annual Percentage rate per debt")
    min_payments: List[float] = Field(default_factory=list, description="Minimum payment per debt")
    months_budget: float = Field(default_factory=0.0, description="Total dollars available to spend on all debts this month")
    total_interest_paid: float = Field(default_factory=0.0, description="Cumulative interest paid so far across debts this month")
    month: int = Field(default_factory=0, description="Current month number (0-indexed)")
    
    
