from .game_env import Game
from .BrainDQN_Nature import BrainDQN
import time

def run_game():
    game = Game()
    RL = BrainDQN(actions=8)
    step = 0
    for episode in range(500):
        # initial observation
        observation = game.reset()
        print("observation:", observation)
        while True:

            RL.setInitState(observation)

            # RL choose action based on observation
            action = RL.getAction()

            # RL take action and get next observation and reward
            observation_, reward, done = game.step(action)

            RL.setPerception(observation, action, reward, done)

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1

    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    run_game()
