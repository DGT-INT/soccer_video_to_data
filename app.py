import matplotlib.pyplot as plt
import numpy as np
from shiny.express import ui, input, render

ui.h1("Footage Data Extraction", class_="text-center")

ui.input_file(
    id= "original_video",
    label= "Select a soccer video to analyze",
    multiple= False,
    accept= [".mp4"],
    button_label= "Click here to upload a video",
    placeholder= "Default file TBD")

