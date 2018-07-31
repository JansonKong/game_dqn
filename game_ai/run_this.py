from game_env import Game
from game_net import DeepQNetwork
import numpy as np

train_steps = 4000000


def train_game():
    step = 0
    scores = []
    for episode in range(train_steps):
        if step%1000 == 0:
            print("step:", step)
        score = 0
        observation = game.init_map()
        print("初始地图：", observation)
        moves = []
        while(True):
            is_done = False
            action = RL.choose_action(observation, train=True)
            moves.append(action)
            print("action:",action)
            print("observation:",observation)
            # RL take action and get next observation and reward
            observation_, reward, is_done = game.move(action)
            if reward >= 0:
                score += reward

            print("observation_:",observation_, "reward:",reward, "is_done:",is_done)
            RL.store_transition(observation, action, reward, observation_)
            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if is_done:
                print("moves:", moves, "score:", score)
                scores.append(score)
                break
            step += 1
    import matplotlib.pyplot as plt
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Scores')
    plt.xlabel('training steps')
    plt.show()

if __name__ == "__main__":
    game = Game()
    # print(game.init_map())
    # print(game.init_map())
    RL = DeepQNetwork(n_actions=9,
                      map_w=12,
                      map_h=12,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=300,
                      memory_size=2000,
                      e_greedy_increment=0.2,
                      output_graph=False
                      )
    train_game()
    RL.plot_cost()
    # print(game.init_map())