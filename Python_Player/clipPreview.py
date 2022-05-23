from asyncio import sleep
from concurrent.futures import thread
from audioPreview import previewAudio
import threading
import time

import numpy as np
import pygame as pg

from moviepy.decorators import (
    convert_masks_to_RGB,
    requires_duration,
)
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip

pause_event = None
terminate_thread = False


def imdisplay(imarray, globalSurface, screen=None, position=(0, 0)):
    a = pg.surfarray.make_surface(imarray.swapaxes(0, 1))
    if screen is None:
        screen = pg.display.set_mode(imarray.shape[:2][::-1])
    screen.blit(a, (0, 0))
    globalSurface.blit(screen, position)
    pg.display.flip()

"""
Displays the clip in a window, at the given frames per second (of movie)
rate. It will avoid that the clip be played faster than normal, but it
cannot avoid the clip to be played slower than normal if the computations
are complex. In this case, try reducing the ``fps``.
Parameters
----------
clip : clip to play
surface : video surface
globalSurface : global window surface (you know like video is in small box inside main window, so
    this main window is globalSurface)
position : tuple (x, y) - position of "surface" (video surface) on global surface
fps : int, optional
  Number of frames per seconds in the displayed video.
audio : bool, optional
  ``True`` (default) if you want the clip's audio be played during
  the preview.
audio_fps : int, optional
  The frames per second to use when generating the audio sound.
audio_buffersize : int, optional
  The sized of the buffer used generating the audio sound.
audio_nbytes : int, optional
  The number of bytes used generating the audio sound.
fullscreen : bool, optional
  ``True`` if you want the preview to be displayed fullscreen.
Examples
--------
>>> from moviepy.editor import *
>>>
>>> clip = VideoFileClip("media/chaplin.mp4")
>>> clip.preview(fps=10, audio=False)
"""

@requires_duration
@convert_masks_to_RGB
def preview(
    clip,
    surface,
    globalSurface,
    position=(0, 0),
    fps=15,
    audio=True,
    audio_fps=22050,
    audio_buffersize=3000,
    audio_nbytes=2,
):
    # compute and splash the first image
    screen = surface

    audio = audio and (clip.audio is not None)

    if audio:
        video_flag = threading.Event()
        audio_flag = threading.Event()

        audiothread = threading.Thread(
            target=previewAudio,
            args=(clip.audio, audio_fps, audio_buffersize, audio_nbytes, audio_flag, video_flag),
        )
        audiothread.start()

    img = clip.get_frame(0)
    imdisplay(img, globalSurface, screen, position)

    if audio:
        video_flag.set()
        audio_flag.wait()

    result = []

    t0 = time.time()
    t1 = time.time()
    t2 = time.time()

    trigger = True

    for t in np.arange(1.0 / fps, clip.duration - 0.001, 1.0 / fps):
        test_start = time.time()

        global terminate_thread
        if terminate_thread:
            break

        global pause_event
        if pause_event is not None:
            pause_event.wait()
            
        img = clip.get_frame(t)

        # Maybe it is important but I suppose no
        # for event in pg.event.get():
        #     if event.type == pg.QUIT or (
        #         event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE
        #     ):
        #         if audio:
        #             video_flag.clear()
        #         print("Interrupt")
        #         pg.quit()
        #         return result

        if trigger:
            t1 = time.time()
        else:
            t2 = time.time()

        diff = abs(t2 - t1)
        if (diff > 0.15):
            t0 = t2 - t if t2 > t1 else t1 - t
            print('VIDEO TIME CORRECTION')

        if trigger:
            time.sleep(max(0, t - (t1 - t0)))
        else:
            time.sleep(max(0, t - (t2 - t0)))
        
        imdisplay(img, globalSurface, screen, position)
        trigger = not trigger

        test_end = time.time()
        # print(test_end - test_start)


class PreviewThread(threading.Thread):
    def __init__(
        self, 
        clip,
        surface,
        globalSurface,
        position=(0, 0),
        fps=15,
        audio=True,
        audio_fps=22050,
        audio_buffersize=3000,
        audio_nbytes=2
    ):
        threading.Thread.__init__(self)
        self.clip = clip
        self.surface = surface
        self.globalSurface = globalSurface
        self.position = position
        self.fps = fps
        self.audio = audio
        self.audio_fps = audio_fps
        self.audio_buffersize = audio_buffersize
        self.audio_nbytes = audio_nbytes
    
    def run(self):
        preview(
            self.clip,
            self.surface,
            self.globalSurface,
            self.position,
            self.fps,
            self.audio,
            self.audio_fps,
            self.audio_buffersize,
            self.audio_nbytes
        )

