import tkinter
import tkinter.filedialog

def prompt_file():
        """Create a Tk file dialog and cleanup when finished"""
        top = tkinter.Tk()
        top.withdraw()  # hide window
        file_name = tkinter.filedialog.askopenfilename(parent=top, filetypes=[("Video Files", ".mp4")])
        top.destroy()
        return file_name