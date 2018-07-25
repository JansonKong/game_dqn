# game_dqn
DQN 游戏AI

代码：
"""
物品：
空地：0
自己：1
敌人：2
墙：3
剑：4
盾牌：5
已经走过：6

动作：
'u':0
'd':1
'l':2
'r':3
'u_l':4
'u_r':5
'd_l':6
'd_r':7
'stop':8

反馈定义：
越界：-100
走到墙：-100
走到敌人：-100
走到剑：+100
走到盾牌：+100
走到已经走过的：-100
走到敌人上下左右的格子：+100
走到空地：0
当停下来时，是起始点：-100
当停下来时，不是起始点：
    空地：0
    周围有一扇墙：+10
    周围有两扇墙：+20
    周围有三扇墙：+30
    周围有四扇墙：+40

终止条件定义：
撞到墙，已经走过的格子，敌人的格子
自己选择停止


"""

import numpy as np
np.random.seed(1)

class Game(object):
    def __init__(self):
        self.action_space = ['u', 'd', 'l', 'r', 'u_l', 'u_r', 'd_l', 'd_r', 'stop']
        self.n_actions = len(self.action_space)
        self.map_h = 12  # grid height
        self.map_w = 12  # grid width
        self.map = np.zeros([12, 12])

    # 随机初始化地图,wall_num为墙的数量
    def init_map(self, wall_num=2):
        # 设置我的位置
        my_pos_x, my_pos_y = self.get_random_pos()
        self.map[my_pos_x,my_pos_y] = 1

        # 设置敌人位置
        enemy_pos_x, enemy_pos_y = self.get_random_pos()
        while self.map[enemy_pos_x, enemy_pos_y] != 0:
            enemy_pos_x, enemy_pos_y = self.get_random_pos()
        self.map[enemy_pos_x, enemy_pos_y] = 2

        # 设置墙的位置
        for i in range(wall_num):
            wall_pos_x, wall_pos_y = self.get_random_pos()
            while self.map[wall_pos_x, wall_pos_y] != 0:
                wall_pos_x, wall_pos_y = self.get_random_pos()
            self.map[wall_pos_x, wall_pos_y] = 3

        # 设置剑的位置
        sword_pos_x, sword_pos_y = self.get_random_pos()
        while self.map[sword_pos_x, sword_pos_y] != 0:
            sword_pos_x, sword_pos_y = self.get_random_pos()
        self.map[sword_pos_x, sword_pos_y] = 4

        # 设置盾的位置
        shield_pos_x, shield_pos_y = self.get_random_pos()
        while self.map[shield_pos_x, shield_pos_y] != 0:
            shield_pos_x, shield_pos_y = self.get_random_pos()
        self.map[shield_pos_x, shield_pos_y] = 5

        # 设置已经走过的格子
        walked_pos_x, walked_pos_y = self.get_random_pos()
        while self.map[walked_pos_x, walked_pos_y] != 0:
            walked_pos_x, walked_pos_y = self.get_random_pos()
        self.map[walked_pos_x, walked_pos_y] = 6

    def set_map_pos(self, pos_x, pos_y, goods):
        self.map[pos_x, pos_y] = goods

    def get_random_pos(self):
        pos_x = np.random.randint(0, self.map_h)
        pos_y = np.random.randint(0, self.map_w)
        return pos_x, pos_y

    # 获取我的位置
    def get_my_pos(self):
        for i in range(self.map_h):
            for j in range(self.map_w):
                if self.map[i][j] == 1:
                    return i, j
        return None

    # 获取敌人位置
    def get_enemy_pos(self):
        for i in range(self.map_h):
            for j in range(self.map_w):
                if self.map[i][j] == 2:
                    return i, j
        return None

    def step(self, action):
        my_pos_x, my_pos_y = self.get_my_pos()
game = Game()
game.init_map(5)
print(game.map)

