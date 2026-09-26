import asyncio
import json
import queue
import threading
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import DQN
import websockets

state_queue = queue.Queue(maxsize=1)
action_queue = queue.Queue(maxsize=1)


async def ws_handler(websocket):
  print('🟢 Scratch подключился!')
  try:
    async for message in websocket:
      data = json.loads(message)

      obs = np.array(data['obs'], dtype=np.float32)
      reward = float(data.get('reward', 0.0))

      # Надежная проверка done (работает с 0/1, "0"/"1", true/false)
      raw_done = data.get('done', False)
      if isinstance(raw_done, str):
        done = raw_done.lower() in ['true', '1']
      else:
        done = bool(raw_done)

      # 1. Передаем состояние в Gym
      state_queue.put((obs, reward, done))

      # 2. Ждем команду от алгоритма
      action_command = await asyncio.to_thread(action_queue.get)

      # Логирование отправки
      if action_command == 'RESET':
        print('🔄 [PYTHON] ---> Отправляем команду RESET в Scratch!')
      else:
        print(f'📤 [PYTHON] Отправка действия: {action_command}')

      # 3. Отправляем команду в Scratch
      await websocket.send(str(action_command))

  except websockets.exceptions.ConnectionClosed:
    print('🔴 Соединение закрыто.')
  except Exception as e:
    print(f'❌ Ошибка сервера: {e}')


async def run_server():
  async with websockets.serve(ws_handler, '0.0.0.0', 8765):
    print('🚀 Сервер запущен на ws://localhost:8765')
    await asyncio.Future()


def start_ws_server():
  asyncio.run(run_server())


class ScratchAppleEnv(gym.Env):

  def __init__(self):
    super().__init__()
    self.action_space = spaces.Discrete(3)
    self.observation_space = spaces.Box(
        low=np.array([-240.0, -180.0, -240.0, -180.0], dtype=np.float32),
        high=np.array([240.0, 180.0, 240.0, 180.0], dtype=np.float32),
        dtype=np.float32,
    )

  def reset(self, seed=None, options=None):
    super().reset(seed=seed)
    print('⚠️ [GYM] Раунд окончен! Вызываем reset() и слали RESET...')
    action_queue.put('RESET')

    # Ждем первого сообщения от Scratch после сброса
    obs, reward, done = state_queue.get()
    print('✅ [GYM] Игра успешно сброшена, продолжаем обучение!')
    return obs, {}

  def step(self, action):
    action_map = {0: 'LEFT', 1: 'NONE', 2: 'RIGHT'}
    action_command = action_map[action]

    action_queue.put(action_command)

    obs, reward, done = state_queue.get()
    return obs, reward, done, False, {}


async def ws_handler(websocket):
  print('🟢 Scratch подключился!')
  try:
    async for message in websocket:
      data = json.loads(message)

      obs = np.array(data['obs'], dtype=np.float32)
      reward = float(data.get('reward', 0.0))
      raw_done = data.get('done', 0)

      # Приведение к bool
      done = bool(raw_done) if not isinstance(raw_done, str) else raw_done in ['1', 'true', 'True']

      # 🔍 ДИАГНОСТИКА: Печатаем пришедшие данные
      print(f"📥 От Scratch -> Reward: {reward} | Done: {done} (исходное: {raw_done})")

      state_queue.put((obs, reward, done))

      action_command = await asyncio.to_thread(action_queue.get)

      if action_command == "RESET":
          print("🔄 [PYTHON] ---> ОТПРАВЛЯЕМ RESET B SCRATCH!")

      await websocket.send(str(action_command))

  except Exception as e:
    print(f'❌ Ошибка: {e}')

if __name__ == '__main__':
  ws_thread = threading.Thread(target=start_ws_server, daemon=True)
  ws_thread.start()

  env = ScratchAppleEnv()

  model = DQN(
      'MlpPolicy',
      env,
      learning_rate=1e-3,
      buffer_size=10000,
      learning_starts=50,
      batch_size=32,
      gamma=0.99,
      exploration_fraction=0.3,
      verbose=1,
  )

  print('🧠 Начинаем обучение...')
  model.learn(total_timesteps=15000)