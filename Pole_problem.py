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
        angle = state[2] 
        between = self.setRadian(angle)  
        penalidade = -2  

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

    def setAction(self, state):
        # Escolhe uma ação usando uma política epsilon-greedy
        if np.random.random() < self.epsilon:  
            return np.random.choice([0, 1])
        state = self.discretization(state)  # Discretiza o estado
        return np.argmax(self.q_table[state])  # Escolha da melhor ação com base na tabela Q

    def updateTable(self, state, action, reward, next_state):
        # Atualiza a tabela Q usando a fórmula Q-Learning
        state = self.discretization(state)
        next_state = self.discretization(next_state)

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

    def discretization(self, state):
        # Discretiza um estado contínuo em índices válidos para a tabela Q
        array = []
        for i, value in enumerate(state):
            # Converte o estado contínuo em um índice válido
            disc_val = np.digitize(value, self.state_bins[i]) - 1  # Subtraia 1 para evitar índice fora do intervalo
            disc_val = max(0, min(self.bins[i] - 1, disc_val))  # Garante que o índice está dentro dos limites
            array.append(disc_val)
        return tuple(array)

    def train(self, episodes=5000):
        # Treina o agente por um número especificado de episódios
        env = gym.make('CartPole-v1')  # Inicializa o ambiente
        for episode in range(episodes):
            state, _ = env.reset()  # Reinicia o ambiente
            done = False
            total_r = 0  # Soma total de recompensas no episódio

            while not done:
                action = self.setAction(state)  # Escolhe uma ação
                next_state, reward, done, truncated, _ = env.step(action)  # Executa a ação
                
                # Penaliza se o episódio terminar antes de atingir 500 recompensas
                if done and total_r < 500:
                    reward1 = self.isInRoad(state)
                    reward2 = self.outOfRoad(state)
                    reward = min(reward1, reward2)

                # Atualiza a tabela Q
                self.updateTable(state, action, reward, next_state)
                total_r += reward  # Atualiza a soma total de recompensas
                state = next_state  # Atualiza o estado atual

            print("episodio:", episode, "total de reward:", total_r)
            self.epsilon *= 0.99  # Diminui a taxa de exploração gradualmente
        env.close()
        np.save("q_table.npy", self.q_table)  # Salva a tabela Q em um arquivo

    def test(self, episodes=10):
    # Carrega a tabela Q do arquivo
        self.q_table = np.load("q_table.npy")

        env = gym.make('CartPole-v1', render_mode="human")  # Inicializa o ambiente
        for episode in range(episodes):
            state, _ = env.reset()  # Reinicia o ambiente
            done = False
            total_r = 0  # Soma total de recompensas no episódio

            while not done:
                env.render()  # Renderiza o ambiente para visualização
                state = self.discretization(state)  # Discretiza o estado atual
                action = np.argmax(self.q_table[state])  # Escolhe a melhor ação com base na tabela Q
                next_state, reward, done, truncated, _ = env.step(action)  # Executa a ação
                total_r += reward  # Atualiza a soma total de recompensas
                state = next_state  # Atualiza o estado atual

            print(f"Episódio {episode + 1}: Total de recompensa: {total_r}")
        env.close()

# Instancia o agente e inicia o treinamento
agent = Car()
agent.train()
