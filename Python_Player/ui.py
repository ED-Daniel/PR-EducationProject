from turtle import pos
import pygame
from config import *

class Text:
    def __init__(self, text, pos, font_family, font_size, screen, color=(0, 0, 0)):
        self.text = text
        self.font_family = font_family
        self.font_size = font_size
        self.font = pygame.font.SysFont(font_family, font_size)
        self.pos = self.pos_x, self.pos_y = pos
        self.screen = screen
        self.color = color
        self.text_surface = self.font.render(self.text, False, self.color)

    def render(self, dynamic=False):
        if dynamic:
            white_surface = pygame.Surface((self.text_surface.get_width(), self.text_surface.get_height()))
            white_surface.fill(BACKGROUND_COLOR)
            self.screen.blit(white_surface, self.pos)
        self.text_surface = self.font.render(self.text, False, self.color)
        self.screen.blit(self.text_surface, self.pos)

    def change_text(self, text):
        self.text = text
    
    def center(self):
        self.pos = (self.pos_x + (self.screen.get_width() - self.text_surface.get_width()) / 2,
                    self.pos_y + (self.screen.get_height() - self.text_surface.get_height()) / 2)
    
    def center_with_size(self, surface_size):
        self.pos = (self.pos_x + (surface_size[0] - self.text_surface.get_width()) / 2,
                    self.pos_y + (surface_size[1] - self.text_surface.get_height()) / 2)


class Button:
    def __init__(self, pos_x, pos_y, width, height, screen):
        self.pos = self.pos_x, self.pos_y = pos_x, pos_y
        self.size = self.width, self.height = width, height
        self.screen = screen
        self.text = None

    def render(self):
        if self.is_mouse_over():
            pygame.draw.rect(self.screen, ADDITION_COLOR, [self.pos_x, self.pos_y, self.width, self.height])
        else:
            pygame.draw.rect(self.screen, OUTLINER_COLOR, [self.pos_x, self.pos_y, self.width, self.height])
        if self.text is not None:
            self.text.render()

    def is_mouse_over(self):
        mouse = pygame.mouse.get_pos()
        if self.pos_x <= mouse[0] <= self.pos_x + self.width and self.pos_y <= mouse[1] <= self.pos_y + self.height:
            return True
        else:
            return False

    def add_text(self, text):
        self.text = Text(text, self.pos, 'Poppins', 40, self.screen, BACKGROUND_COLOR)
        self.text.center_with_size(self.size)