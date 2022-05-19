from turtle import pos
import pygame
from config import *

class Button:
    def __init__(self, pos_x, pos_y, width, height, screen):
        self.pos = self.pos_x, self.pos_y = pos_x, pos_y
        self.size = self.width, self.height = width, height
        self.screen = screen

    def render(self):
        if self.is_mouse_over():
            pygame.draw.rect(self.screen, ADDITION_COLOR, [self.pos_x, self.pos_y, self.width, self.height])
        else:
            pygame.draw.rect(self.screen, OUTLINER_COLOR, [self.pos_x, self.pos_y, self.width, self.height])

    def is_mouse_over(self):
        mouse = pygame.mouse.get_pos()
        if self.pos_x <= mouse[0] <= self.pos_x + self.width and self.pos_y <= mouse[1] <= self.pos_y + self.height:
            return True
        else:
            return False