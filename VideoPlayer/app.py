from concurrent.futures import thread
import threading
import pygame
from requests import delete
import clipPreview
import audioPreview
from pygame.locals import *
from moviepy.editor import *
from config import *
from clipPreview import PreviewThread, preview
from ui import Button, Text
from utils import prompt_file

class App:
    cameraOn = False
    def __init__(self, width = 800, height = 600, fps = 30):
        self._running = True
        self._display_surf = None
        self.size = self.width, self.height = 1200, 700
        self.fps = fps

 
    def on_init(self):
        pygame.init()
        self._display_surf = pygame.display.set_mode(self.size, pygame.HWSURFACE | pygame.DOUBLEBUF)
        self._running = True
        self._display_surf.fill(BACKGROUND_COLOR)

        pygame.display.set_caption("Video Player")

        self.load_button = Button(self.width - 250, self.height - 589, 200, 75, self._display_surf)
        self.play_button = Button(self.width - 250, self.height - 464, 200, 75, self._display_surf)
        self.load_button.add_text("Open file")
        self.play_button.add_text("Pause")

        detector_font_size = 25
        text_padding = 33
        counter_padding = 5
        font_family = 'Poppins'

        self.happiness_text = Text("Hapinnes", (self.width - 250, self.height - 356), font_family, detector_font_size, self._display_surf, OUTLINER_COLOR)
        self.happiness_counter = Text("62%", (self.width - 250, self.height - (356 - self.happiness_text.text_surface.get_height() - counter_padding)), font_family, detector_font_size, self._display_surf, OUTLINER_COLOR)

        self.engagement_text = Text("Engagement", (self.width - 250, self.height - (356 - self.happiness_text.text_surface.get_height() - text_padding)), font_family, detector_font_size, self._display_surf, OUTLINER_COLOR)
        prev_position = 356 - self.happiness_text.text_surface.get_height() - text_padding
        self.engagement_counter = Text("62%", (self.width - 250, self.height - (prev_position - self.engagement_text.text_surface.get_height() - counter_padding)), font_family, detector_font_size, self._display_surf, OUTLINER_COLOR)

        self.tiredness_text = Text("Tiredness", (self.width - 250, self.height - (prev_position - self.engagement_text.text_surface.get_height() - text_padding)), font_family, detector_font_size, self._display_surf, OUTLINER_COLOR)
        prev_position = prev_position - self.engagement_text.text_surface.get_height() - text_padding
        self.tiredness_counter = Text("62%", (self.width - 250, self.height - (prev_position - self.tiredness_text.text_surface.get_height() - counter_padding)), font_family, detector_font_size, self._display_surf, OUTLINER_COLOR)

        self.astonishment_text = Text("Astonishment", (self.width - 250, self.height - (prev_position - self.tiredness_text.text_surface.get_height() - text_padding)), font_family, detector_font_size, self._display_surf, OUTLINER_COLOR)
        prev_position = prev_position - self.tiredness_text.text_surface.get_height() - text_padding
        self.astonishment_counter = Text("62%", (self.width - 250, self.height - (prev_position - self.astonishment_text.text_surface.get_height() - counter_padding)), font_family, detector_font_size, self._display_surf, OUTLINER_COLOR)

        self.lost_attention_text = Text("Lost Attention", (self.width - 250, self.height - (prev_position - self.astonishment_text.text_surface.get_height() - text_padding)), font_family, detector_font_size, self._display_surf, OUTLINER_COLOR)
        prev_position = prev_position - self.astonishment_text.text_surface.get_height() - text_padding
        self.lost_attention_counter = Text("62%", (self.width - 250, self.height - (prev_position - self.lost_attention_text.text_surface.get_height() - counter_padding)), font_family, detector_font_size, self._display_surf, OUTLINER_COLOR)

        self.p_thread = None
        self.pause = False

        return True

    def update_text(self, happiness, engagement, tiredness, astonishment, lost_attention):
        self.happiness_counter.change_text(happiness)
        self.engagement_counter.change_text(engagement)
        self.tiredness_counter.change_text(tiredness)
        self.astonishment_counter.change_text(astonishment)
        self.lost_attention_counter.change_text(lost_attention)

    def on_event(self, event):
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            self._running = False

        if event.type == pygame.MOUSEBUTTONDOWN and self.play_button.is_mouse_over():
            self.pause = not self.pause
            if self.pause:
                clipPreview.pause_event = threading.Event()
                audioPreview.pause_event = threading.Event()
                self.play_button.add_text("Play")
            else:
                clipPreview.pause_event.set()
                audioPreview.pause_event.set()
                self.play_button.add_text("Pause")
                
        if event.type == pygame.MOUSEBUTTONDOWN and self.load_button.is_mouse_over():
            self.clip_file_name = prompt_file()
            if self.clip_file_name is not None and self.clip_file_name != "" and (self.p_thread is None or not self.p_thread.is_alive()):
                clip = VideoFileClip(self.clip_file_name).resize(width=self.width - 350)
                self.videoSurface = pygame.Surface(clip.size)
                self.p_thread = PreviewThread(clip, self.videoSurface, self._display_surf, (50, (self.height - clip.size[1]) // 2), fps=24)
                self.p_thread.start()
            else:
                print("Error")

    def on_loop(self):
        pass

    def on_render(self):
        self.play_button.render()
        self.load_button.render()

        self.happiness_text.render()
        self.engagement_text.render()
        self.tiredness_text.render()
        self.astonishment_text.render()
        self.lost_attention_text.render()
        self.happiness_counter.render()
        self.engagement_counter.render()
        self.tiredness_counter.render()
        self.astonishment_counter.render()
        self.lost_attention_counter.render()

        pygame.display.update()

    def on_cleanup(self):
        if clipPreview.pause_event is not None:
            clipPreview.pause_event.set()
        if audioPreview.pause_event is not None:
            audioPreview.pause_event.set()

        clipPreview.terminate_thread = True
        audioPreview.terminate_thread = True
        
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