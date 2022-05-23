"""Audio preview functions for MoviePy editor."""

import time

import numpy as np
import pygame as pg

from moviepy.decorators import requires_duration

pause_event = None
terminate_thread = False


@requires_duration
def previewAudio(
    clip, fps=22050, buffersize=4000, nbytes=2, audio_flag=None, video_flag=None
):
    pg.mixer.quit()

    pg.mixer.init(fps, -8 * nbytes, clip.nchannels, 1024)
    totalsize = int(fps * clip.duration)
    pospos = np.array(list(range(0, totalsize, buffersize)) + [totalsize])
    timings = (1.0 / fps) * np.arange(pospos[0], pospos[1])
    sndarray = clip.to_soundarray(timings, nbytes=nbytes, quantize=True)
    chunk = pg.sndarray.make_sound(sndarray)

    if (audio_flag is not None) and (video_flag is not None):
        audio_flag.set()
        video_flag.wait()

    t1 = time.time()
    t2 = time.time()

    trigger = True

    channel = chunk.play()
    for i in range(1, len(pospos) - 1):
        global terminate_thread
        if terminate_thread:
            break

        global pause_event
        if pause_event is not None:
            pause_event.wait()

        timings = (1.0 / fps) * np.arange(pospos[i], pospos[i + 1])
        sndarray = clip.to_soundarray(timings, nbytes=nbytes, quantize=True)
        chunk = pg.sndarray.make_sound(sndarray)

        while channel.get_queue():
            time.sleep(0.003)
            if pause_event is not None:
                pause_event.wait()
            if video_flag is not None:
                if not video_flag.is_set():
                    channel.stop()
                    del channel
                    return
        
        if trigger:
            t1 = time.time()
        else:
            t2 = time.time()

        diff = abs(t2 - t1)
        if (diff > 0.25):
            time.sleep(0.2)
            print('AUDIO time correction')
        
        trigger = not trigger

        channel.queue(chunk)