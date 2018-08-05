"""
物品：
    空地：0
    自己：1
    敌人：2
    墙：3
    剑：4
    盾牌：5
    已经走过：6
    毒圈：7

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
走到毒圈：-100 # -50
当停下来时，是起始点：-100
当停下来时，不是起始点：
    空地：0
    周围有一扇墙：+10
    周围有两扇墙：+20
    周围有三扇墙：+30
    周围有四扇墙：+40

终止条件定义：
    撞到墙，走进毒圈，已经走过的格子，敌人的格子
    自己选择停止

"""

import numpy as np
np.random.seed(1)


class Game(object):
    def __init__(self):
        self.action_space = ['u', 'd', 'l', 'r', 'u_l', 'u_r', 'd_l', 'd_r', 'stop']
        self.n_actions = len(self.action_space)
        self.map_h = 6     # grid height
        self.map_w = 6     # grid
        self.poison_circle = 0
        self.map = np.zeros([12, 12])
        self.start_x = 0
        self.start_y = 0

    # 随机初始化地图,wall_num为墙的数量
    def init_map(self, wall_num=2, poison_circle = 0):
        self.map = np.zeros([self.map_h, self.map_w])
        self.poison_circle = poison_circle

        # 设置毒圈
        for i in range(self.map_h):
            for j in range(poison_circle):
                self.map[i][j] = 7
                self.map[i][self.map_w - 1 - j] = 7

        for i in range(self.map_w):
            for j in range(poison_circle):
                self.map[j][i] = 7
                self.map[self.map_h - 1 - j][i] = 7

        # 设置我的位置
        my_pos_x, my_pos_y = self.get_random_pos()
        while self.map[my_pos_x, my_pos_y] != 0:
            my_pos_x, my_pos_y = self.get_random_pos()
        self.map[my_pos_x,my_pos_y] = 1
        self.start_x = my_pos_x
        self.start_y = my_pos_y

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

        # # 设置剑的位置
        # sword_pos_x, sword_pos_y = self.get_random_pos()
        # while self.map[sword_pos_x, sword_pos_y] != 0:
        #     sword_pos_x, sword_pos_y = self.get_random_pos()
        # self.map[sword_pos_x, sword_pos_y] = 4
        #
        # # 设置盾的位置
        # shield_pos_x, shield_pos_y = self.get_random_pos()
        # while self.map[shield_pos_x, shield_pos_y] != 0:
        #     shield_pos_x, shield_pos_y = self.get_random_pos()
        # self.map[shield_pos_x, shield_pos_y] = 5

        # # 设置已经走过的格子
        # walked_pos_x, walked_pos_y = self.get_random_pos()
        # while self.map[walked_pos_x, walked_pos_y] != 0:
        #     walked_pos_x, walked_pos_y = self.get_random_pos()
        # self.map[walked_pos_x, walked_pos_y] = 6
        return self.map.copy()

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

    """
    反馈定义：
    越界：-100
    走到墙：-100
    走到敌人：-100
    走到剑：+100
    走到盾牌：+100
    走到已经走过的：-100
    走到空地：
        能攻击敌人：+100
        不能攻击敌人：0
    走到毒圈：-100
    当停下来时，是起始点：-100
    当停下来时，不是起始点：
        周围有一扇墙：+10
        周围有两扇墙：+20
        周围有三扇墙：+30
        周围有四扇墙：+40

    终止条件定义：
        撞到墙，走进毒圈，已经走过的格子，敌人的格子
        自己选择停止
    """
    def move(self, action):
        my_pos_x, my_pos_y = self.get_my_pos()
        self.set_map_pos(my_pos_x, my_pos_y, 6)
        reward = 0
        is_done = False

        # 我的位置更新
        if action == 0:
            my_pos_x -= 1
        elif action == 1:
            my_pos_x += 1
        elif action == 2:
            my_pos_y -= 1
        elif action == 3:
            my_pos_y += 1
        elif action == 4:
            my_pos_x -= 1
            my_pos_y -= 1
        elif action == 5:
            my_pos_x -= 1
            my_pos_y += 1
        elif action == 6:
            my_pos_x += 1
            my_pos_y -= 1
        elif action == 7:
            my_pos_y += 1
            my_pos_x += 1
        elif action == 8:   # 自己停下来 停止
            is_done = True
            if my_pos_x == self.start_x and my_pos_y == self.start_y:   # 刚开始就停下来
                reward = -100
            else:
                # reward = 10 * self.get_not_killed_num_around(my_pos_x, my_pos_y)    # 不是停在起始点
                reward = 0
            return self.map.copy(), reward, is_done

        # 判断是否出界或者到毒圈，停止，reward = -100
        if self.is_out_poison(my_pos_x, my_pos_y):
            is_done = True
            reward = -100
            return self.map.copy(), reward, is_done

        # 走到已经走过的格子，敌人的格子
        if self.map[my_pos_x, my_pos_y] == 6 or self.map[my_pos_x,my_pos_y] == 2 or self.map[my_pos_x, my_pos_y] == 3:
            is_done = True
            reward = -100
            return self.map.copy(), reward, is_done

        # 走到剑，盾
        print(my_pos_x,my_pos_y)
        if self.map[my_pos_x, my_pos_y] == 4 or self.map[my_pos_x, my_pos_y] == 5:
            is_done = False
            reward = 100
            self.map[my_pos_x, my_pos_y] = 1
            return self.map.copy(), reward, is_done

        # 走到空地
        if self.map[my_pos_x, my_pos_y] == 0:
            if self.can_kill(my_pos_x, my_pos_y):
                is_done = False
                reward = 100
            else:
                is_done = False
                reward = 0
            self.map[my_pos_x, my_pos_y] = 1
            return self.map.copy(), reward, is_done

        # return self.map, reward, is_done

    # 判断是否越界或者进到毒圈
    def is_out_poison(self, my_pos_x, my_pos_y):
        if (my_pos_x < self.poison_circle) or (my_pos_x > self.map_w - 1 - self.poison_circle) or (my_pos_y < self.poison_circle) or (my_pos_y > self.map_h - 1 -self.poison_circle):
            return True
        else:
            return False

    # 判断点是否合法
    def is_out(self, my_pos_x, my_pos_y):
        if(my_pos_x >= 0 and my_pos_x < self.map_h and my_pos_y >= 0 and my_pos_y < self.map_w):
            return False
        else:
            return True

    # 计算该空地是否能攻击敌人
    def can_kill(self, pos_x, pos_y):
        can_kill = False
        up_pos_y = pos_y
        up_pos_x = pos_x + 1

        down_pos_y = pos_y
        down_pos_x = pos_x - 1

        left_pos_x = pos_x
        left_pos_y = pos_y - 1

        right_pos_x = pos_x
        right_pos_y = pos_y + 1

        if (not self.is_out(up_pos_x, up_pos_y)) and self.map[up_pos_x][up_pos_y] == 2:
            can_kill = True
        if (not self.is_out(down_pos_x, down_pos_y)) and self.map[down_pos_x][down_pos_y] == 2:
            can_kill = True
        if (not self.is_out(left_pos_x, left_pos_y)) and self.map[left_pos_x][left_pos_y] == 2:
            can_kill = True
        if (not self.is_out(right_pos_x, right_pos_y)) and self.map[right_pos_x, right_pos_y] == 2:
            can_kill = True

        return can_kill

    # 转换为三维向量（4,map_h,map_w)(第一维：自己的位置，敌人的位置，墙的位置，已经走过的位置）
    def transform_to_3dim(self, old_map):
        new_map = np.zeros([4, self.map_h, self.map_w])
        for i in range(self.map_h):
            for j in range(self.map_w):
                if old_map[i][j] != 0:
                    if old_map[i][j] == 1:
                        new_map[0][i][j] = 1
                    elif old_map[i][j] == 2:
                        new_map[1][i][j] = 1
                    elif old_map[i][j] == 3:
                        new_map[2][i][j] = 1
                    elif old_map[i][j] == 6:
                        new_map[3][i][j] = 1
                    else:
                        print("transform_to_3dim error: 不是自己，敌人，墙，已经走过的,空地")
        return new_map

    # 计算位置上下左右不可攻击点的数量 毒气 墙 越界
    def get_not_killed_num_around(self, pos_x, pos_y):
        not_killed_count = 0
        up_pos_y = pos_y + 1
        up_pos_x = pos_x

        down_pos_y = pos_y - 1
        down_pos_x = pos_x

        left_pos_x = pos_x - 1
        left_pos_y = pos_y

        right_pos_x = pos_x + 1
        right_pos_y = pos_y

        if self.is_out_poison(up_pos_x, up_pos_y) or self.map[up_pos_x][up_pos_y] == 3:
            not_killed_count += 1
        if self.is_out_poison(down_pos_x, down_pos_y) or self.map[down_pos_x][down_pos_y] == 3:
            not_killed_count += 1
        if self.is_out_poison(left_pos_x, left_pos_y) or self.map[left_pos_x][left_pos_y]  == 3:
            not_killed_count += 1
        if self.is_out_poison(right_pos_x, right_pos_y) or self.map[right_pos_x, right_pos_y]  ==3:
            not_killed_count += 1

        return not_killed_count

# game = Game()
# game.init_map(5, 2)
# print(game.map)
# print(game.can_kill(6, 8))
# print(game.can_kill(5, 9))
# print(game.can_kill(3, 3))
#
# print(game.get_not_killed_num_around(2,2))
# print(game.get_not_killed_num_around(3,3))
# print(game.get_not_killed_num_around(4,4))
# print(game.get_not_killed_num_around(5,5))
# print(game.get_not_killed_num_around(8,9))
# print(game.is_out_poison(0, 0))
# print(game.is_out_poison(0, 1))
# print(game.is_out_poison(1, 0))
# print(game.is_out_poison(-1, 0))
# print(game.is_out_poison(1, 1))
# print(game.is_out_poison(11, 0))
# print(game.is_out_poison(0, 11))
# print(game.is_out_poison(5, 5))
# print(game.is_out_poison(11, 11))
# print(game.is_out_poison(2, 0))
# print(game.is_out_poison(0, 10))
#
# map, reward, is_done = game.move(2)
# print("map:", map)
# print("reward:", reward)
# print("is_done:", is_done)
# map, reward, is_done = game.move(0)
# print("map:", map)
# print("reward:", reward)
# print("is_done:", is_done)
# map, reward, is_done = game.move(4)
# print("map:", map)
# print("reward:", reward)
# print("is_done:", is_done)
# map, reward, is_done = game.move(6)
# print("map:", map)
# print("reward:", reward)
# print("is_done:", is_done)
# map, reward, is_done = game.move(7)
# print("map:", map)
# print("reward:", reward)
# print("is_done:", is_done)
# map, reward, is_done = game.move(1)
# print("map:", map)
# print("reward:", reward)
# print("is_done:", is_done)
# map, reward, is_done = game.move(6)
# print("map:", map)
# print("reward:", reward)
# print("is_done:", is_done)
# # map, reward, is_done = game.move(3)
# # print("map:", map)
# # print("reward:", reward)
# # print("is_done:", is_done)