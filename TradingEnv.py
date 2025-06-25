import gym
import numpy as np
from gym import spaces

class TradingEnv(gym.Env):
    def __init__(self, df, initial_balance=1000):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.action_space = spaces.Discrete(3)  # 0 = hold, 1 = buy, 2 = sell
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.asset = 0
        self.current_step = 0
        self.trades = []
        return self._get_obs()

    def _get_obs(self):
        row = self.df.iloc[self.current_step]
        return np.array([
            self.balance,
            self.asset,
            row['close'],
            row['volume'],
            row['rsi'],
            row['sma']
        ], dtype=np.float32)

    def step(self, action):
        done = False
        price = self.df.iloc[self.current_step]['close']

        # Take action
        if action == 1 and self.balance >= price:  # buy
            self.asset += 1
            self.balance -= price
        elif action == 2 and self.asset > 0:  # sell
            self.asset -= 1
            self.balance += price

        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            done = True

        next_obs = self._get_obs()
        total_value = self.balance + self.asset * price
        self.reward = total_value - self.initial_balance  # net profit

        return next_obs, self.reward, done, {}

    def render(self):
        print(f"Step: {self.current_step} | Balance: {self.balance:.2f} | Assets: {self.asset} | Reward: {self.reward}")



import pandas as pd
import ta

np.random.seed(42)
n = 500
price = np.cumsum(np.random.randn(n)) + 100
volume = np.random.rand(n) * 100

df = pd.DataFrame({
    'close': price,
    'volume': volume,
})

df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
df['sma'] = ta.trend.SMAIndicator(df['close'], window=10).sma_indicator()
df = df.dropna().reset_index(drop=True)

from stable_baselines3 import PPO

env = TradingEnv(df)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50_000)

obs = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    env.render()

    