"""
地图：[12,12,6]
[12,12,0]:自己的位置
[12,12,1]:敌人位置
[12,12,2]:墙的位置
[12,12,3]:已经走过的位置
[12,12,4]:目的点
[12,12,5]:未定义
"""

import numpy as np
np.random.seed(1)

class Game(object):
    def __init__(self):
        self.action_space = ['u', 'd', 'l', 'r', 'u_l', 'u_r', 'd_l', 'd_r']
        self.n_actions = len(self.action_space)
        self.map_x = 12
        self.map_y = 12

    # 随机初始化地图,wall_num为墙的数量
    def reset_map(self, wall_num=2):
        self.states = np.zeros([self.map_x, self.map_y, 6])

        # 自己目前的位置
        self.current_pos = self.get_random_pos()
        self.set_state(self.current_pos, 0, 1)

        # 敌人的位置
        self.enemy_pos = self.get_random_pos()
        while not self.is_empty(self.enemy_pos[0], self.enemy_pos[1]):
            self.enemy_pos = self.get_random_pos()
        self.set_state(self.enemy_pos, 1, 1)

        # 设置墙的位置
        self.walls_pos = []
        for i in range(wall_num):
            wall_pos = self.get_random_pos()
            while not self.is_empty(wall_pos[0], wall_pos[1]):
                wall_pos = self.get_random_pos()
            self.walls_pos.append(wall_pos)
            self.set_state(wall_pos, 2, 1)

        # 设置终点
        self.target_pos = self.get_random_pos()
        while not self.is_empty(self.target_pos[0], self.target_pos[1]):
            self.target_pos = self.get_random_pos()
        self.set_state(self.target_pos, 4, 1)

    def step(self, action):
        reward, done = 0, False
        x = self.current_pos[0]
        y = self.current_pos[1]

        # 自己的位置置空
        self.states[x][y][0] = 0

        # 已经走过的位置更新
        self.states[x][y][3] = 1



    # 设置地图信息状态为state: 1（有东西）0 （空）dim: 第三维
    def set_state(self, pos, dim, state):
        x = pos[0]
        y = pos[1]
        self.states[x][y][dim] = state

    def is_empty(self, x, y):
        for z in range(self.states.shape[2]):
            if self.states[x][y][z] != 0:
                return False
        return True

    def get_random_pos(self):
        pos_x = np.random.randint(0, self.map_x)
        pos_y = np.random.randint(0, self.map_y)
        return [pos_x, pos_y]


game = Game()
game.reset_map()

for i in range(6):
    print(game.states[:,:,i])
# print(game.states)