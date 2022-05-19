import pygame
from pygame.locals import *
from moviepy.editor import *

from Python_Player.main import FONTS
from clipPreview import preview
import cv2
from PySide2 import QtCore, QtWidgets
import qimage2ndarray
from Python_Player import main, utils
class App:
    cameraOn = False
    def __init__(self, width = 800, height = 600, fps = 30):
        self._running = True
        self._display_surf = None
        self.size = self.width, self.height = 1200, 700
        self.camera_capture = cv2.VideoCapture(cv2.CAP_DSHOW)
        self.fps = fps
        self.frameTimer = QtCore.QTimer()

 
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
        self.camera_capture.release()
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
    def putIndexesOnScreen(self, frame = main.camera):
        tiredness = main.findTiredRatio(main.BLINKS_IN_MINUTE)
        frame = utils.textWithBackground(frame, f'Tiredness: {tiredness}', FONTS, 1.0, (30, 50), bgOpacity=0.9,
                                             textThickness=2)
        amazement = main.amazeCounter()

"""

    def cameraSettings(self, fps):
        self.camera_capture.set(3, self.width)
        self.camera_capture.set(3, self.height)
        self.frameTimer.timeout.connect(self.captureCameraStream)
        self.frameTimer.start(int(1000 // fps))
    def captureCameraStream(self):
        if self.cameraOn:
            ret, frame = self.camera_capture.read()
            frame = cv2.flip(frame, 1)
        if not ret:
            return False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (self.size), interpolation=cv2.INTER_AREA)
        
        frame here is ready for usage as an numpy array for the future analysis
        
        # image = qimage2ndarray.array2qimage(frame)
        
        use image for performing images
        

        return frame
"""


if __name__ == "__main__" :
    theApp = App()
    theApp.on_execute()