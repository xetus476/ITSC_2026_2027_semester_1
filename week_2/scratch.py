import asyncio
import json
import numpy as np
import websockets


# 1. Функция-обработчик сообщений
async def ws_handler(websocket):
  print('🟢 Scratch успешно подключился к Python!')
  async for message in websocket:
    try:
      # Парсим JSON от Scratch
      data = json.loads(message)
      raw_obs = data['obs']
      reward = float(data['reward'])
      done = bool(data['done'])
      obs = np.array(raw_obs, dtype=np.float32)

      print(
          f'Принято от Scratch -> Obs: {obs}, Reward: {reward}, Done: {done}'
      )

      # ⚠️ ОБЯЗАТЕЛЬНО: Отправляем ответ обратно в Scratch!
      # Пока нейросеть не подключена, отправляем команду "NONE" (ничего не делать)
      response = json.dumps({'action': 'NONE'})
      await websocket.send(response)

    except json.JSONDecodeError:
      print(f'❌ Некорректный JSON от Scratch: {message}')
    except Exception as e:
      print(f'❌ Ошибка: {e}')


# 2. Главная функция запуска сервера
async def main():
  # Запускаем сервер на порту 8765
  async with websockets.serve(ws_handler, '0.0.0.0', 8765):
    print('🚀 Сервер запущен на ws://localhost:8765')
    print('⏳ Ожидание подключения из TurboWarp...')
    await asyncio.Future()  # Держим сервер активным


# 3. Точка входа
if __name__ == '__main__':
  asyncio.run(main())