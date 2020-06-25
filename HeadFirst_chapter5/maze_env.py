import pygame, time, random
import sys
import math, copy
from pygame.locals import *

SCREEN_WIDTH = 550
SCREEN_HEIGTH = 550
BG_COLOR = pygame.Color(255,255,255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
BLAKE = (0,0,0)
TEXT_COLOR = pygame.Color(255, 0, 0)

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3


class Maze():
    def __init__(self):
        pygame.display.init()
        self.window = pygame.display.set_mode([SCREEN_WIDTH, SCREEN_HEIGTH])
        pygame.display.set_caption("Maze Game")
        self.robot = []


        self.state_space = self.get_state_space()  # 其中key为状态编号，其值为坐标
        self.n_state = len(self.state_space)
        self.action_space = [0, 1, 2, 3]
        self.n_action = len(self.action_space)

        self.trap_space = {}
        self.treasure_space = {}
        self.terminate_space = {}

        self.transition = {}
        self.get_transition()  # 获得转移函数

        self.current_state = None  # 当前状态

        self.trap_space[3] = (300, 0)
        self.trap_space[8] = (300, 100)
        self.trap_space[10] = (0, 200)
        self.trap_space[11] = (100, 200)
        self.trap_space[22] = (200, 400)
        self.trap_space[23] = (300, 400)
        self.trap_space[24] = (400, 400)

        self.treasure_space[14] = (450, 250)

        self.get_terminate_space()

        self.robot_image = pygame.image.load('img/robot.jpg')
        self.treasure_image = pygame.image.load('img/treasure.jpg')


    def reset(self):
        # self.current_state = random.choice(list(self.state_space.keys()))
        # while self.current_state in self.terminate_space:
        #     self.current_state = random.choice(list(self.state_space.keys()))
        self.current_state = 0
        return self.current_state


    def render(self):
        time.sleep(0.05)
        self.window.fill(WHITE)
        self.getEvent()

        # 绘画横向线段
        grid_length = 6
        for j in range(0, grid_length):
            pygame.draw.line(self.window, BLAKE, (0, j * 100), (500, j * 100), 2)

        # 绘画纵向线段
        for i in range(0, grid_length):
            pygame.draw.line(self.window, BLAKE, (i * 100, 0), (i * 100, 500), 2)

        # 绘制宝藏
        rect = self.treasure_image.get_rect()
        rect.left = 410
        rect.top = 205
        self.window.blit(self.treasure_image, rect)

        # 绘制陷阱
        self.createTrap()

        # 绘制robot

        pygame.draw.circle(self.window, RED, (self.state_space[self.current_state][0],
                                              self.state_space[self.current_state][1]), 50)

        pygame.display.flip()



    def step(self, action):
        # 系统当前状态
        key = (self.current_state, action)

        if key in self.transition:
            next_state = self.transition[key]
        else:
            next_state = self.current_state
        # self.state = observation

        done = False

        # next_index = (next_state[0], next_state[1])
        if next_state in self.trap_space:
            r = -1
            done = True
        elif next_state in self.treasure_space:
            r = 1
            done = True
        else:
            r = 0

        return next_state, r, done


    def getTextSurface(self, text):
        pygame.font.init()
        # 查看所有可用字体
        # print(pygame.font.get_fonts())

        #获取字体Font对象
        font = pygame.font.SysFont('kaiti', 30)

        # 绘制文字信息
        textSurface = font.render(text, True, TEXT_COLOR)

        return textSurface



    def createTrap(self):
        pygame.draw.rect(self.window, BLAKE, Rect((300, 0), (100, 100)))
        pygame.draw.rect(self.window, BLAKE, Rect((300, 100), (100, 100)))
        pygame.draw.rect(self.window, BLAKE, Rect((0, 200), (100, 100)))
        pygame.draw.rect(self.window, BLAKE, Rect((100, 200), (100, 100)))
        pygame.draw.rect(self.window, BLAKE, Rect((200, 400), (100, 100)))
        pygame.draw.rect(self.window, BLAKE, Rect((300, 400), (100, 100)))
        pygame.draw.rect(self.window, BLAKE, Rect((400, 400), (100, 100)))




    def get_transition(self):
        self.transition[(0, DOWN)] = 5; self.transition[(0, RIGHT)] = 1
        self.transition[(1, DOWN)] = 6; self.transition[(1, LEFT)] = 0; self.transition[(1, RIGHT)] = 2
        self.transition[(2, DOWN)] = 7; self.transition[(2, LEFT)] = 1; self.transition[(2, RIGHT)] = 3
        self.transition[(4, DOWN)] = 9; self.transition[(4, LEFT)] = 3

        self.transition[(5, UP)] = 0; self.transition[(5, DOWN)] = 10; self.transition[(5, RIGHT)] = 6
        self.transition[(6, UP)] = 1; self.transition[(6, DOWN)] = 11; self.transition[(6, LEFT)] = 5; self.transition[(6, RIGHT)] = 7
        self.transition[(7, UP)] = 2; self.transition[(7, DOWN)] = 12; self.transition[(7, LEFT)] = 6; self.transition[(7, RIGHT)] = 8
        self.transition[(9, UP)] = 4; self.transition[(9, DOWN)] = 14; self.transition[(9, LEFT)] = 8

        self.transition[(12, UP)] = 7; self.transition[(12, DOWN)] = 17; self.transition[(12, LEFT)] = 11; self.transition[(12, RIGHT)] = 13
        self.transition[(13, UP)] = 8; self.transition[(13, DOWN)] = 18; self.transition[(13, LEFT)] = 12; self.transition[(13, RIGHT)] = 14

        self.transition[(15, UP)] = 10; self.transition[(15, DOWN)] = 20; self.transition[(15, RIGHT)] = 16
        self.transition[(16, UP)] = 11; self.transition[(16, DOWN)] = 21; self.transition[(16, LEFT)] = 15; self.transition[(16, RIGHT)] = 17
        self.transition[(17, UP)] = 12; self.transition[(17, DOWN)] = 22; self.transition[(17, LEFT)] = 16; self.transition[(17, RIGHT)] = 18
        self.transition[(18, UP)] = 13; self.transition[(18, DOWN)] = 23; self.transition[(18, LEFT)] = 17; self.transition[(18, RIGHT)] = 19
        self.transition[(19, UP)] = 14; self.transition[(19, DOWN)] = 24; self.transition[(19, LEFT)] = 18

        self.transition[(20, UP)] = 15; self.transition[(20, RIGHT)] = 21
        self.transition[(21, UP)] = 16; self.transition[(21, LEFT)] = 20; self.transition[(21, RIGHT)] = 22

    def get_terminate_space(self):
        self.terminate_space[3] = (300, 0)
        self.terminate_space[8] = (300, 100)
        self.terminate_space[10] = (0, 200)
        self.terminate_space[11] = (100, 200)
        self.terminate_space[22] = (200, 400)
        self.terminate_space[23] = (300, 400)
        self.terminate_space[24] = (400, 400)

        self.terminate_space[14] = (450, 250)


    def get_state_space(self):
        #从左到右从上到下，依次使用在网格中的坐标标号
        states = {}
        #------------以下代码使用网格坐标来表示状态，键值为行列坐标，其值为真实的坐标值-----------------
        i = 0
        for top in range(50, 550, 100):
            for left in range(50, 550, 100):
                states[i] = (left, top)
                i += 1
        return states



    def getEvent(self):
        #获取所有事件
        eventList = pygame.event.get()
        for event in eventList:
            if event.type == pygame.QUIT:
                exit()









