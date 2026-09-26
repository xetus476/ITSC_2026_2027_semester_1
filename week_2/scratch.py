import asyncio
import json
import queue
import threading
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import DQN
import websockets

# -------------------------------------------------------------------
# 1. Очереди для синхронизации WebSocket (asyncio) и Gym (поток)
# -------------------------------------------------------------------
state_queue = queue.Queue(maxsize=1)
action_queue = queue.Queue(maxsize=1)


# -------------------------------------------------------------------
# 2. Обработчик WebSocket и запуск сервера
# -------------------------------------------------------------------
async def ws_handler(websocket):
  print('🟢 Scratch (TurboWarp) подключился к Python!')
  try:
    async for message in websocket:
      # А. Распаковываем данные от Scratch
      data = json.loads(message)
      obs = np.array(data['obs'], dtype=np.float32)
      reward = float(data['reward'])
      done = bool(data['done'])

      # Б. Отправляем состояние в среду Gymnasium
      state_queue.put((obs, reward, done))

      # В. Ждем решение (команду) от модели DQN без блокировки event loop
      action_command = await asyncio.to_thread(action_queue.get)

      # Г. Отправляем команду обратно в Scratch
      await websocket.send(json.dumps({'action': action_command}))

  except websockets.exceptions.ConnectionClosed:
    print('🔴 Соединение со Scratch закрыто.')


async def run_server():
  """Асинхронная функция запуска сервера"""
  async with websockets.serve(ws_handler, '0.0.0.0', 8765):
    print('🚀 Сервер успешно запущен на ws://localhost:8765')
    print('⏳ Запустите проект в TurboWarp (нажмите Зеленый флаг)...')
    await asyncio.Future()  # Бесконечно держим сервер активным


def start_ws_server():
  """Точка входа для фонового потока сервера"""
  asyncio.run(run_server())


# -------------------------------------------------------------------
# 3. Пользовательская среда Gymnasium для ловли яблок
# -------------------------------------------------------------------
class ScratchAppleEnv(gym.Env):

  def __init__(self):
    super().__init__()

    # 3 ДЕЙСТВИЯ: 0 = LEFT, 1 = NONE, 2 = RIGHT
    self.action_space = spaces.Discrete(3)

    # Вектор состояния: [x_player, y_player, x_apple, y_apple]
    self.observation_space = spaces.Box(
        low=np.array([-240.0, -180.0, -240.0, -180.0], dtype=np.float32),
        high=np.array([240.0, 180.0, 240.0, 180.0], dtype=np.float32),
        dtype=np.float32,
    )

  def reset(self, seed=None, options=None):
    super().reset(seed=seed)

    # Отправляем в Scratch команду перезапуска
    action_queue.put('RESET')

    # Ждем первичное состояние сцены после рестарта
    obs, reward, done = state_queue.get()
    return obs, {}

  def step(self, action):
    # Преобразуем индекс действия в текстовую команду
    action_map = {0: 'LEFT', 1: 'NONE', 2: 'RIGHT'}
    action_command = action_map[action]

    # Передаем действие в WebSocket
    action_queue.put(action_command)

    # Ждем результат от Scratch
    obs, reward, done = state_queue.get()

    return obs, reward, done, False, {}


# -------------------------------------------------------------------
# 4. Главная точка входа и обучение DQN
# -------------------------------------------------------------------
if __name__ == '__main__':
  # Запускаем WebSocket сервер в фоновом потоке
  ws_thread = threading.Thread(target=start_ws_server, daemon=True)
  ws_thread.start()

  # Инициализация среды Gymnasium
  env = ScratchAppleEnv()

  # Создание модели DQN
  model = DQN(
      'MlpPolicy',
      env,
      learning_rate=1e-3,
      buffer_size=10000,
      learning_starts=100,
      batch_size=32,
      gamma=0.99,
      exploration_fraction=0.3,
      verbose=1,
  )

  print('🧠 Начинаем обучение нейросети ловле яблок...')
  # Обучение модели
  model.learn(total_timesteps=15000)

  # Сохранение обученной модели
  model.save('dqn_apple_catcher')
  print('✅ Модель сохранена в файл dqn_apple_catcher.zip')