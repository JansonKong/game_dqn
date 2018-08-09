from game_env import Game
from BrainDQN_Nature import BrainDQN
import time

def run_game():
    game = Game()
    RL = BrainDQN(actions=8)
    step = 0
    for episode in range(500):
        # initial observation
        observation = game.reset_map()
        # print(observation.shape)
        RL.setInitState(observation)
        while True:
            # RL choose action based on observation
            action = RL.getAction()
            # for i in range(6):
            #     print(game.states[:,:,i])
            # print("action:", action)
            # RL take action and get next observation and reward
            observation_, reward, done = game.step(action)
            # for i in range(6):
            #     print(game.states[:,:,i])
            # print("reward:",reward)
            # print("done:", done)
            RL.setPerception(observation_, action, reward, done)

            # break while loop when end of this episode
            if done:
                # print("done,reset")
                observation = game.reset_map()
                RL.setInitState(observation)

run_game()
