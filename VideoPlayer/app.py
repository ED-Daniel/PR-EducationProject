import pygame
from pygame.locals import *
from moviepy.editor import *
from clipPreview import preview
 
class App:
    def __init__(self):
        self._running = True
        self._display_surf = None
        self.size = self.width, self.height = 1200, 700
 
    def on_init(self):
        pygame.init()
        self._display_surf = pygame.display.set_mode(self.size, pygame.HWSURFACE | pygame.DOUBLEBUF)
        self._running = True
        self._display_surf.fill((255, 255, 255))
        clip = VideoFileClip("Test2.mp4").resize(width=self.width)
        videoSurface = pygame.Surface(clip.size)
        preview(clip, videoSurface, self._display_surf, (50, 50), fps=24)
 
    def on_event(self, event):
        if event.type == pygame.QUIT:
            self._running = False

    def on_loop(self):
        pass

    def on_render(self):
        self._display_surf.fill((255, 255, 255))

    def on_cleanup(self):
        pygame.quit()
 
    def on_execute(self):
        if self.on_init() == False:
            self._running = False
 
        while( self._running ):
            for event in pygame.event.get():
                self.on_event(event)
            self.on_loop()
            self.on_render()
        self.on_cleanup()
 
if __name__ == "__main__" :
    theApp = App()
    theApp.on_execute()