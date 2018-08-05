from game_env import Game
import numpy as np

game = Game()
game.init_map()
print(game.map)
game.move(0)
game.move(0)
print(game.map)

new_map = game.transform_to_3dim(game.map)
print(new_map)
new_map = np.reshape(new_map, [6,6,4])
print(new_map)