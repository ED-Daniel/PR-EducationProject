from concurrent.futures import thread
import threading
import pygame
import clipPreview
from pygame.locals import *
from moviepy.editor import *
from config import *
from clipPreview import PreviewThread, preview
from ui import Button
from utils import prompt_file


class App:
    def __init__(self):
        self._running = True
        self._display_surf = None
        self.size = self.width, self.height = 1200, 700
 
    def on_init(self):
        pygame.init()
        self._display_surf = pygame.display.set_mode(self.size, pygame.HWSURFACE | pygame.DOUBLEBUF)
        self._running = True
        self._display_surf.fill(BACKGROUND_COLOR)

        pygame.display.set_caption("Video Player")

        self.load_button = Button(50, 500, 200, 75, self._display_surf)
        self.play_button = Button(50, 600, 200, 75, self._display_surf)

        self.pause = False

        return True
 
    def on_event(self, event):
        if event.type == pygame.QUIT:
            self._running = False

        if event.type == pygame.MOUSEBUTTONDOWN and self.play_button.is_mouse_over():
            self.pause = not self.pause
            if self.pause:
                clipPreview.pause_event = threading.Event()
            else:
                clipPreview.pause_event.set()
                
        if event.type == pygame.MOUSEBUTTONDOWN and self.load_button.is_mouse_over():
            self.clip_file_name = prompt_file()
            if self.clip_file_name is not None and self.clip_file_name != "":
                clip = VideoFileClip(self.clip_file_name).resize(width=self.width - 300)
                videoSurface = pygame.Surface(clip.size)
                self.p_thread = PreviewThread(clip, videoSurface, self._display_surf, (50, 10), fps=24)
                self.p_thread.start()
            else:
                print("Error")

    def on_loop(self):
        pass

    def on_render(self):
        self.play_button.render()
        self.load_button.render()
        pygame.display.update()

    def on_cleanup(self):
        pygame.quit()
 
    def on_execute(self):
        if self.on_init() == False:
            self._running = False
 
        while self._running:
            for event in pygame.event.get():
                self.on_event(event)

            self.on_loop()
            self.on_render()
        self.on_cleanup()
 
if __name__ == "__main__" :
    theApp = App()
    theApp.on_execute()