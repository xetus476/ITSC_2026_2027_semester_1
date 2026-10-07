import os
import time
import random
import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import gc

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options

from selenium.webdriver.common.action_chains import ActionChains
import re

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gc.collect()
torch.cuda.empty_cache()
print(torch.cuda.memory_allocated() / 1024**2, "MB")

class Boltzman:
    def __init__(self,actions: list):
        self.actions = actions
    def Generator_actions(self,Q_value, temperature):
        if random.random() < temperature:
            return np.random.choice(self.actions)
        else:
            return np.argmax(Q_value)

class CNN_DQN(nn.Module):
    def __init__(self,input_dim=(1, 84, 84),num_action:int = 4):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_dim[0],32,kernel_size=8,stride=8),
            nn.ReLU(),
            nn.Conv2d(32,64,kernel_size=4,stride=2),
            nn.GELU(),
            nn.Conv2d(64,64,kernel_size=3,stride=1)
        )

        conv_out_size = self._get_conv_out(input_dim)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size,256),
            nn.GELU(),
            nn.Linear(256,num_action)
        )

    def _get_conv_out(self, shape):
        o = self.cnn(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self,x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class Browser2048Env:
    def __init__(self,headless=False):
        options = Options()
        if headless:
            options.add_argument('--headless')
        options.add_argument('--window-size=1200,900')
        
        self.driver = webdriver.Chrome(options=options)
        self.driver.get("https://play2048.co/")
        time.sleep(2) # Ожидание загрузки страницы
        
        self.body = self.driver.find_element(By.TAG_NAME, 'body')
        
        # 0: Вверх, 1: Вниз, 2: Влево, 3: Вправо
        self.actions = [Keys.ARROW_UP, Keys.ARROW_DOWN, Keys.ARROW_LEFT, Keys.ARROW_RIGHT]
        self.last_score = 0

    def get_score(self):
        try:
            score = self.driver.execute_script("""
                const elements = [...document.querySelectorAll('*')];
    
                for (const el of elements) {
                    if ((el.innerText || '').trim() === 'SCORE') {
    
                        const parent = el.parentElement;
    
                        if (!parent) {
                            continue;
                        }
    
                        const numbers = parent.innerText.match(/\\d+/g);
    
                        if (numbers && numbers.length > 0) {
                            return parseInt(numbers[0], 10);
                        }
                    }
                }
    
                return null;
            """)
    
            if score is not None:
                return int(score)
    
        except Exception as e:
            print("SCORE ERROR:", e)
    
        return self.last_score

    def get_screenshot_state(self):
        self.driver.save_screenshot("temp_screenshot.png")
        img = Image.open("temp_screenshot.png").convert('L')

        try:
            board_element = self.driver.find_element(By.CLASS_NAME, 'game-container')
            location = board_element.location
            size = board_element.size

            left = int(location['x'])
            top = int(location['y'])
            right = int(location['x'] + size['width'])
            bottom = int(location['y'] + size['height'])

            # Защита от нулевых или некорректных размеров
            if right > left and bottom > top:
                img = img.crop((left, top, right, bottom))
        except Exception:
            pass # Если элемент не найден, берется полный скриншот
        
        board_resized = img.resize((84, 84))
        state_array = np.array(board_resized, dtype=np.float32) / 255.0
        return np.expand_dims(state_array, axis=0)

    def is_game_over(self):
        """Проверка окончания игры по тексту и классам Tailwind"""
        try:
            # Способ 1: Поиск элемента по точному/частичному тексту "Game Over"
            game_over_elements = self.driver.find_elements(
                By.XPATH, "//*[contains(text(), 'Game Over')]"
            )
            for elem in game_over_elements:
                if elem.is_displayed():
                    return True
        except Exception:
            pass
        return False



    def reset(self):
        try:
            # Способ 1: Пытаемся найти и кликнуть кнопку "Try again" или "New Game"
            # В верстке Tailwind кнопка перезапуска обычно имеет текст "Try again" или похожий
            retry_buttons = self.driver.find_elements(By.XPATH, "//*[contains(text(), 'Try again') or contains(text(), 'New Game') or contains(@class, 'restart')]")
            clicked = False
            for btn in retry_buttons:
                if btn.is_displayed():
                    btn.click()
                    clicked = True
                    break
            
            # Способ 2: Если кнопка не найдена, очищаем localStorage и обновляем страницу
            if not clicked:
                self.driver.execute_script("window.localStorage.clear();")
                self.driver.refresh()
                
        except Exception:
            # Запасной вариант на крайний случай
            try:
                self.driver.execute_script("window.localStorage.clear();")
                self.driver.refresh()
            except Exception:
                pass

        time.sleep(1)  # Пауза на перерисовку поля
        self.last_score = 0
        self.no_move_count = 0  # Сбрасываем счетчик тупиков
        return self.get_screenshot_state()

    def step(self, action_idx):
        try:
            state = self.get_screenshot_state()

            print("ACTION:", action_idx)

            ActionChains(self.driver).send_keys(
                self.actions[action_idx]
            ).perform()

            time.sleep(0.2)

            next_state = self.get_screenshot_state()

            # Получаем новый score
            current_score = self.get_score()

            # ВАЖНО:
            # last_score здесь ещё должен содержать score ДО действия
            score_diff = current_score - self.last_score

            done = self.is_game_over()

            print(
                f"STATE CHANGED: {not np.array_equal(state, next_state)}"
            )
            print(f"SCORE: {current_score}")
            print(f"LAST SCORE: {self.last_score}")
            print(f"SCORE DIFF: {score_diff}")

            # Reward
            if done:
                reward = -10.0

            elif np.array_equal(state, next_state):
                reward = -2.0

            elif score_diff > 0:
                reward = float(score_diff)

            else:
                reward = 0.0

            # ВОТ ЗДЕСЬ обновляем last_score
            self.last_score = current_score

            return next_state, reward, done

        except Exception as e:
            print(f"ОШИБКА ВНУТРИ STEP: {e}")
            raise

        
    
    def close(self):
        self.driver.quit()

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (
            torch.FloatTensor(np.array(state)),
            torch.LongTensor(action),
            torch.FloatTensor(reward),
            torch.FloatTensor(np.array(next_state)),
            torch.FloatTensor(done)
        )

    def __len__(self):
        return len(self.buffer)

def train():
    print(f"Используется устройство: {DEVICE}")

    BATCH_SIZE = 32
    GAMMA = 0.99
    EPS_START = 1.0
    EPS_END = 0.05
    EPS_DECAY = 0.995
    TARGET_UPDATE = 10
    LEARNING_RATE = 0.00025
    NUM_EPISODES = 50

    env = Browser2048Env(headless=False)

    policy_net = CNN_DQN().to(DEVICE)
    target_net = CNN_DQN().to(DEVICE)

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(),lr=LEARNING_RATE)

    memory = ReplayBuffer(capacity=5000)

    epsilon = EPS_START

    for episode in range(1, NUM_EPISODES + 1):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Выбор действия (Epsilon-Greedy)
            if random.random() < epsilon:
                action = random.randint(0, 3)
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    q_values = policy_net(state_tensor)
                    action = q_values.argmax(dim=1).item()

            next_state, reward, done = env.step(action)
            print(f"Шаг выполнен | Reward: {reward} | Done: {done}")
            # Выполнение действия в браузере
            #next_state, reward, done = env.step(action)
            total_reward += reward

            # Сохранение в буфер
            memory.push(state, action, reward, next_state, float(done))
            state = next_state

            # Обучение модели
            if len(memory) >= BATCH_SIZE:
                b_state, b_action, b_reward, b_next_state, b_done = memory.sample(BATCH_SIZE)

                b_state = b_state.to(DEVICE)
                b_action = b_action.unsqueeze(1).to(DEVICE)
                b_reward = b_reward.unsqueeze(1).to(DEVICE)
                b_next_state = b_next_state.to(DEVICE)
                b_done = b_done.unsqueeze(1).to(DEVICE)

                # Q(s, a)
                q_eval = policy_net(b_state).gather(1, b_action)

                # Max Q(s', a') из Target сети
                with torch.no_grad():
                    q_next = target_net(b_next_state).max(1)[0].unsqueeze(1)
                    q_target = b_reward + (1 - b_done) * GAMMA * q_next

                loss = nn.MSELoss()(q_eval, q_target)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Уменьшаем Epsilon
        epsilon = max(EPS_END, epsilon * EPS_DECAY)

        # Обновляем Target Network
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(f"Action: {action} | Score: {env.last_score} | Reward: {total_reward} | Done: {done}")

    print("Обучение завершено!")
    env.close()


if __name__ == "__main__":
    train()