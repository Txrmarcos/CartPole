import gymnasium as gym
import time
import math
import numpy as np

class Car:
    def __init__(self):
        # Inicializa os parâmetros do ambiente e do algoritmo de aprendizado
        self.MAX_CART_POSITION = 4.8  # Posição máxima do carrinho no ambiente
        self.MIN_CART_POSITION = 2.4  # Posição mínima para penalidade
        self.MAX_POLE_ANGLE = math.ceil(.418 * (180 / math.pi))  # Ângulo máximo do pêndulo em graus
        self.MIN_POLE_ANGLE = math.ceil(.2095 * (180 / math.pi))  # Ângulo mínimo relevante em graus
        self.bins = (6, 12, 6, 12)  # Número de divisões para discretizar cada variável do estado
        self.actions = 2  # Número de ações possíveis (esquerda ou direita)
        self.alpha = 0.1  # Taxa de aprendizado
        self.gamma = 0.99  # Fator de desconto
        self.epsilon = 1.0  # Taxa de exploração
        # Inicializa a tabela Q com zeros
        self.q_table = np.zeros(self.bins + (self.actions,)) 
        # Define os limites para discretização de cada dimensão do estado
        self.state_bins = [
            np.linspace(-4.8, 4.8, self.bins[0]),  # Posição do carrinho
            np.linspace(-4, 4, self.bins[1]),  # Velocidade do carrinho
            np.linspace(-0.418, 0.418, self.bins[0]),  # Ângulo do pêndulo
            np.linspace(-4, 4, self.bins[1])  # Velocidade angular
        ]

    def isInRoad(self, state):
        # Avalia a penalidade baseada no ângulo do pêndulo
        angle = state[2]  # Ângulo do pêndulo
        between = self.setRadian(angle)  # Converte para graus
        penalidade = -2  # Penalidade base

        # Ajusta a penalidade de acordo com o ângulo
        if between <= 12 and between > 8:
            penalidade *= 5
        elif between >= -12 and between < -8:
            penalidade *= 5
        elif between >= 7 and between < 3:
            penalidade *= 3
        elif between >= -7 and between < -3:
            penalidade *= 3
        elif between >= 2 and between < 0:
            penalidade *= 1
        elif between >= -2 and between < 0:
            penalidade *= 1
        else:
            penalidade = 0

        # Log de depuração
        print("estado do carrinho max é de 2.4 e o atual é:", state[0], "ou em radianos ->", self.setRadian(state[0]))

        return penalidade

    def outOfRoad(self, state):
        # Verifica se o carrinho saiu da pista e aplica penalidade severa se necessário
        distance = state[0]
        if distance < self.MIN_CART_POSITION:
            return -100    
        elif distance > self.MAX_CART_POSITION:
            return -100 
        else:
            print("entrou else")
            return 0

    def distanceOfLoss(self, x1, x2):
        # Calcula a diferença angular entre dois valores em graus
        print(f'x1: {x1} x2: {x2}')
        return self.setRadian(x1 - x2)

    def choose_action(self, state):
        # Escolhe uma ação usando uma política epsilon-greedy
        if np.random.random() < self.epsilon:  # Exploração
            return np.random.choice([0, 1])
        state = self.discretize_state(state)  # Discretiza o estado
        return np.argmax(self.q_table[state])  # Escolha da melhor ação com base na tabela Q

    def update_q_table(self, state, action, reward, next_state):
        # Atualiza a tabela Q usando a fórmula Q-Learning
        state = self.discretize_state(state)
        next_state = self.discretize_state(next_state)

        # Calcula o valor alvo e o erro temporal-diferença
        best_next_action = np.argmax(self.q_table[next_state])  
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error  # Atualização Q

    def setRadian(self, dec):
        # Converte um valor em radianos para graus inteiros com sinal preservado
        decin = abs(math.ceil(dec * (180 / math.pi)))
        if dec < 0:
            decin = -decin
        return decin

    def discretize_state(self, state):
        # Discretiza um estado contínuo em índices válidos para a tabela Q
        discretized_state = []
        for i, value in enumerate(state):
            # Converte o estado contínuo em um índice válido
            discretized_value = np.digitize(value, self.state_bins[i]) - 1  # Subtraia 1 para evitar índice fora do intervalo
            discretized_value = max(0, min(self.bins[i] - 1, discretized_value))  # Garante que o índice está dentro dos limites
            discretized_state.append(discretized_value)
        return tuple(discretized_state)

    def train(self, episodes=5000):
        # Treina o agente por um número especificado de episódios
        env = gym.make('CartPole-v1')  # Inicializa o ambiente
        for episode in range(episodes):
            state, _ = env.reset()  # Reinicia o ambiente
            done = False
            total_r = 0  # Soma total de recompensas no episódio

            while not done:
                action = self.choose_action(state)  # Escolhe uma ação
                next_state, reward, done, truncated, _ = env.step(action)  # Executa a ação
                
                # Penaliza se o episódio terminar antes de atingir 200 recompensas
                if done and total_r < 500:
                    reward1 = self.isInRoad(state)
                    reward2 = self.outOfRoad(state)
                    reward = min(reward1, reward2)

                # Atualiza a tabela Q
                self.update_q_table(state, action, reward, next_state)
                total_r += reward  # Atualiza a soma total de recompensas
                state = next_state  # Atualiza o estado atual

            print("episodio:", episode, "total de reward:", total_r)
            self.epsilon *= 0.99  # Diminui a taxa de exploração gradualmente
        env.close()
        np.save("q_table.npy", self.q_table)  # Salva a tabela Q em um arquivo
        print("Q-table salva no arquivo 'q_table.npy'.")

    def test(self, episodes=10):
    # Carrega a tabela Q do arquivo
        self.q_table = np.load("q_table.npy")
        print("Q-table carregada do arquivo 'q_table.npy'.")

        env = gym.make('CartPole-v1', render_mode="human")  # Inicializa o ambiente
        for episode in range(episodes):
            state, _ = env.reset()  # Reinicia o ambiente
            done = False
            total_r = 0  # Soma total de recompensas no episódio

            while not done:
                env.render()  # Renderiza o ambiente para visualização
                state = self.discretize_state(state)  # Discretiza o estado atual
                action = np.argmax(self.q_table[state])  # Escolhe a melhor ação com base na tabela Q
                next_state, reward, done, truncated, _ = env.step(action)  # Executa a ação
                total_r += reward  # Atualiza a soma total de recompensas
                state = next_state  # Atualiza o estado atual

            print(f"Episódio {episode + 1}: Total de recompensa: {total_r}")
        env.close()

agent = Car()
agent.test(episodes=5)

