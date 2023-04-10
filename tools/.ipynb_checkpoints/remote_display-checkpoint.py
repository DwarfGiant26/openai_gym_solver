import matplotlib.pyplot as plt
from IPython import display as ipythondisplay
from typing import List
import time

def show_frame(frame) -> None:
    '''
        Show frame as plot that get refreshed when the next plot needed to be displayed
        
        frame: 3d array representing pixel intensity in frames
    '''
    # set up frame in plot
    plt.imshow(frame)
    plt.axis('off')
    
    # clear output and display image
    ipythondisplay.clear_output(wait=True)
    ipythondisplay.display(plt.gcf())
    
def show_frames(frames: List, fps: float = 60) -> None:
    '''
        Show list of frames as video with specified fps
        
        frames: List of frames
    '''
    frame_duration = 1 / fps
    
    for frame in frames:
        show_frame(frame)
        time.sleep(frame_duration)
        
    ipythondisplay.clear_output(wait=True)
    