from server.my_env_environment import MyEnvironment

env = MyEnvironment(task="t1")
obs = env.reset()
print("=== RESET ===")
print("Balances:", obs.balances)        # [2000.0, 3000.0]
print("Budget:", obs.months_budget)     # 400.0

# Month 1 — avalanche: throw everything at high-APR debt
action = type('A', (), {'extra_payments': [280.0, 0.0], 'task': 't1'})()
obs = env.step(action)
print("\n=== AFTER MONTH 1 ===")
print("Balances:", obs.balances)        # Card A shrinks fast
print("Reward:", obs.reward)
print("Done:", obs.done)