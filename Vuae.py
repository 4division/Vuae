import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from tkinter import PhotoImage
import platform
import cv2
import numpy as np

# Import functions or classes from the main script
from process import upscale_and_enhance_video, create_images_from_video, add_audio_to_video, clean_temp_images, upscale_button_click
temp_upscaled_images_path = "temp_upscaled_images"  # Replace with the actual path
outscale_value = 2

# Define global variables

output_image_folder = "temp_images"

root = None
cap = None

# Function to update the preview
def update_preview():
    global cap, root
    try:
        if cap is not None:
            ret, frame = cap.read()
            if ret:
                # Apply the selected effects to the frame (same as in upscale_and_enhance_video)
                frame = cv2.resize(frame, (400, 300))
                if sharpen_intensity_scale.get() > 0:
                    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
                    frame = cv2.filter2D(frame, -1, kernel)
                if denoise_strength_scale.get() > 0:
                    frame = cv2.fastNlMeansDenoisingColored(frame, None, denoise_strength_scale.get(), 10, 7, 21)

                # Show the frame in the OpenCV window
                cv2.imshow("Preview", frame)

            # Schedule the next update after 10 ms (adjust as needed)
            root.after(10, update_preview)
    except Exception as e:
        print("An error occurred during preview:", str(e))

def start_preview():
    global cap
    if cap is None:
        # Start capturing video
        cap = cv2.VideoCapture(input_path_var.get())
        cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)  # Create a resizable window
        update_preview()  # Start the preview update
    else:
        # Release video capture and close the OpenCV window
        cap.release()
        cv2.destroyAllWindows()
        cap = None

def run_gui():
    global input_path_var, output_path_var, output_image_folder, scale_factor_entry, sharpen_intensity_scale, denoise_strength_scale, realesrgan_checkbox, root

    try:
        # Create the main GUI window
        root = tk.Tk()
        root.title("Vuae - Video Upscaler and Enhancer")
        # Fixed issue with platform dependency
        if platform.system() == 'Windows':
            # Your Windows-specific code here
            root.iconbitmap('favicon.ico')

        style = ttk.Style()

        # Set the theme to "clam" or any other built-in theme
        style.theme_use("clam")  # You can change "clam" to other available themes

        # Set dark mode theme
        root.tk_setPalette(background='#FFFFFF', foreground='#1e1e1e')

        # Set larger window size
        root.geometry("800x600")

        # Load the background image
        bg_image = PhotoImage(file="background.png")  # Replace "background.png" with your image file

        # Create a Label widget to display the background image
        background_label = tk.Label(root, image=bg_image)
        background_label.place(relwidth=1, relheight=1)

        # Buttons and forms on the left
        form_frame = tk.Frame(root)
        form_frame.pack(side="left", padx=20, pady=10)

        # Create a Label widget for the video preview
        preview_label = tk.Label(form_frame)
        preview_label.pack(side="right", padx=20, pady=10)

        # Input video path File Dialog button
        input_path_label = tk.Label(form_frame, text="Select Input Video:", fg='#1e1e1e', bg='white')
        input_path_label.pack()
        input_path_var = tk.StringVar()
        input_path_entry = tk.Entry(form_frame, textvariable=input_path_var, state='readonly')
        input_path_entry.pack()

        def browse_input_path():
            file_path = filedialog.askopenfilename(title="Select Input Video File", filetypes=[("Video Files", "*.mp4")])
            if file_path:
                input_path_var.set(file_path)

        input_browse_button = tk.Button(form_frame, text="Browse", command=browse_input_path)
        input_browse_button.pack()

        # Output video path File Dialog button
        output_path_label = tk.Label(form_frame, text="Select Output Video:", fg='#1e1e1e', bg='white')
        output_path_label.pack()
        output_path_var = tk.StringVar()
        output_path_entry = tk.Entry(form_frame, textvariable=output_path_var, state='readonly')
        output_path_entry.pack()

        def browse_output_path():
            file_path = filedialog.asksaveasfilename(title="Save Output Video As", filetypes=[("Video Files", "*.mp4")])
            if file_path:
                output_path_var.set(file_path)

        output_browse_button = tk.Button(form_frame, text="Browse", command=browse_output_path)
        output_browse_button.pack()

        # Scale factor label and entry
        scale_factor_label = tk.Label(form_frame, text="Scale Factor:", fg='#1e1e1e', bg='white')
        scale_factor_label.pack()
        scale_factor_entry = tk.Entry(form_frame)
        scale_factor_entry.pack()

        # Sharpening intensity slider
        sharpen_intensity_label = tk.Label(form_frame, text="Sharpening Intensity", fg='#1e1e1e', bg='white')
        sharpen_intensity_label.pack(anchor="w")
        sharpen_intensity_scale = ttk.Scale(form_frame, from_=0, to=10, orient="horizontal")
        sharpen_intensity_scale.set(0)  # Default value
        sharpen_intensity_scale.pack(fill="x")

        # Denoise strength slider
        denoise_strength_label = tk.Label(form_frame, text="Denoise Strength", fg='#1e1e1e', bg='white')
        denoise_strength_label.pack(anchor="w")
        denoise_strength_scale = ttk.Scale(form_frame, from_=0, to=10, orient="horizontal")
        denoise_strength_scale.set(0)  # Default value
        denoise_strength_scale.pack(fill="x")

        # Upscale button
        upscale_button = tk.Button(form_frame, text="Upscale and Enhance Video", command=lambda: upscale_button_click(input_path_var, output_path_var, output_image_folder, scale_factor_entry, sharpen_intensity_scale, denoise_strength_scale, realesrgan_checkbox, temp_upscaled_images_path, outscale_value ))
        upscale_button.pack()

        # Checkbox for esrgan
        realesrgan_checkbox = tk.BooleanVar()
        realesrgan_checkbox.set(False)  # Default to disabled
        realesrgan_checkbox_button = tk.Checkbutton(form_frame, text="Enable RealESRGAN Upscaling", variable=realesrgan_checkbox)
        realesrgan_checkbox_button.pack()

      

        # Create a "Preview" button
        preview_button = tk.Button(form_frame, text="Preview", command=start_preview)
        preview_button.pack()

        # Close the terminal when the GUI window is closed
        root.protocol("WM_DELETE_WINDOW", root.quit)

        # Start the GUI main loop
        root.mainloop()

    except Exception as e:
        print("An error occurred:", str(e))

if __name__ == "__main__":
      run_gui()
