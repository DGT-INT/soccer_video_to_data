# shiny app test

from shiny import ui, render, App
import shutil
import os

BASE_DIR = os.path.dirname(__file__)
# Ensure the process cwd is the app directory so the server serves BASE_DIR/www
os.chdir(BASE_DIR)
UPLOAD_DIR = os.path.join(BASE_DIR, "www", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app_ui = ui.page_fluid(
    ui.h1("Smart Soccer Insights", class_="text-center"),
    
    ui.input_file(
        id= "original_video",
        label= "Select a soccer video to analyze",
        multiple= False,
        accept= [".mp4"],
        button_label= "Click here to upload a video",
        placeholder= "Default file TBD"
    ),

    ui.input_checkbox(
        id= "original_video_checkbox",
        label= "Display original video?",
        value= True
    ),

    ui.output_ui("uploaded_video"),

    ui.output_ui("test_vid")
)


def server(input, output, session):
    @output
    @render.ui
    def uploaded_video():
        files = input.original_video()
        if not files:
            return ui.HTML("<p>No video uploaded yet.</p>")
        file = files[0]
        temp_path = file["datapath"]
        filename = file["name"]
        save_path = os.path.join(UPLOAD_DIR, filename)
        # DEBUG PRINTS
        print("\n=== DEBUG INFO ===")
        print("__file__:", __file__)
        print("App base dir:", BASE_DIR)
        print("Working directory:", os.getcwd())
        print("Temp path:", temp_path)
        print("Saving to:", save_path)
        print("File exists in www/uploads?:", os.path.exists(save_path))
        print("=================\n")
        shutil.copy(temp_path, save_path)
        video_url = f"/uploads/{filename}"
        return ui.HTML(f"""
                       <video width="640" controls>
                       <source src="{video_url}" type="{file['type']}">
                       Your browser does not support the video tag.
                       </video>
                       """)
    

    def test_video_ui():
        filename = "sample_video.mp4"
        video_url = f"/uploads/{filename}"

        return ui.HTML(f"""
                       <video width="640" controls>
                       <source src="{video_url}" type="video/mp4">
                       Your browser does not support the video tag.
                       </video>
                       """)

app = App(app_ui, server)
