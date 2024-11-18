# Solução para o Problema Cartpole com Reinforcement Learning Clássico

Repositorio com a implementação: https://github.com/Txrmarcos/CartPole

## Introdução

Este relatório apresenta a solução para o problema Cartpole utilizando técnicas de Reinforcement Learning clássicas, como Q-learning e SARSA. O objetivo principal foi desenvolver uma política ótima para o ambiente **CartPole-v1** da biblioteca Gym, aplicando equações de otimalidade de Bellman, sem o uso de redes neurais.

---

## Descrição do Problema

O problema do Cartpole consiste em equilibrar um pêndulo invertido sobre um carrinho móvel. O objetivo é impedir que o pêndulo caia, aplicando forças que mantenham o equilíbrio central pelo maior tempo possível. A tarefa do agente é maximizar a recompensa acumulada ao longo do tempo.

---

## Modelagem como Processo de Decisão de Markov (MDP)

O problema foi modelado como um Processo de Decisão de Markov (MDP), com os seguintes componentes:

- **Estados:** Representam a posição do carrinho, velocidade, ângulo do pêndulo e velocidade angular.
- **Ações:** Aplicação de forças discretas para a esquerda ou para a direita.
- **Recompensa:** Recompensa de +1 para cada passo em que o pêndulo permaneça equilibrado.
- **Objetivo:** Maximizar a soma de recompensas futuras descontadas.

As equações de otimalidade de Bellman foram utilizadas para calcular as funções de valor associadas a cada estado, relacionando recompensas imediatas e estados futuros.

---

## Implementação do Algoritmo de RL

A implementação foi realizada em Python utilizando o ambiente **CartPole-v1** da biblioteca Gym. Usando o **Q-learning**, com discretização dos estados para lidar com a natureza contínua do ambiente.

### Discretização dos Estados

Os estados contínuos foram divididos em bins discretos utilizando a função `np.digitize`. Isso permitiu transformar o espaço contínuo em um conjunto finito de estados, compatível com os algoritmos de RL clássicos.

### Algoritmo de Q-learning

O algoritmo de Q-learning foi implementado com os seguintes passos:

1. **Inicialização:**

   - Q-tabela foi inicializada com zeros.
   - Definidos hiperparâmetros: taxa de aprendizado (α), fator de desconto (γ), e taxa de exploração (ε).
2. **Iterações:**

   - Para cada episódio:
     - Obter o estado inicial.
     - Selecionar uma ação usando a política ε-greedy.
     - Executar a ação e observar o próximo estado e a recompensa.
     - Atualizar a Q-tabela utilizando a equação de Bellman:
       `Q(s, a) = Q(s, a) + alpha * (reward + gamma * max(Q(s', a')) - Q(s, a)) `
3. **Política Ótima:**

   - Após o treinamento, a política ótima foi extraída selecionando as ações que maximizam o valor Q para cada estado.

---

## Resultados

O agente foi treinado por 5.000 episódios. Após o treinamento, ele foi capaz de manter o pêndulo equilibrado por uma média de 500 passos consecutivos, demonstrando o aprendizado de uma política ótima.

### Desempenho do Agente

- **Recompensa Média:** 500 por episódio no estado estacionário.
- **Taxa de Sucesso:** O agente conseguiu equilibrar o pêndulo consistentemente após o treinamento.

---

## Conclusão

A implementação do problema Cartpole utilizando Q-learning demonstrou que algoritmos clássicos de RL são capazes de resolver problemas de controle, como o pêndulo invertido. A utilização de equações de Bellman foi essencial para calcular as funções de valor e determinar políticas ótimas.
