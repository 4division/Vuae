import cv2
import numpy as np
from tqdm import tqdm
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
import os
from inference_realesrgan import main


print("Vuae - Video Upscaler and Enhancer developed by 4-division")
cv2.ocl.setUseOpenCL(True)
cap = None
script_dir = os.path.dirname(__file__)
relative_script_path = os.path.join(script_dir, "inference_realesrgan.py")
global input_path_var, output_path_var, output_image_folder, scale_factor_entry, sharpen_intensity_scale, denoise_strength_scale, realesrgan_checkbox, root
temp_upscaled_images_path = "temp_upscaled_images"
outscale_value = 2







def upscale_with_realesrgan(temp_images, output_path, outscale, realesrgan_options):
    try:
        # Call the modified function from the imported script
        main(
            model_name=realesrgan_options["model_name"],
            input_path=temp_images,
            output_path=output_path,
            outscale=outscale,
              # Set to True based on your original code
        )

        print("RealESRGAN upscaling complete. Output saved as", output_path)

    except Exception as e:
        print("An error occurred during RealESRGAN upscaling:", str(e))
        


def upscale_and_enhance_video(input_path, output_path, temp_upscaled_images_path, scale_factor, sharpen_intensity, denoise_strength, outscale_value=2, realesrgan_options=None):
    
    try:
        if realesrgan_options is not None:
            # Apply RealESRGAN upscaling
            
            upscale_with_realesrgan(temp_upscaled_images_path, outscale_value, realesrgan_options)

            # Load the upscaled image using OpenCV
            upscaled_image = cv2.imread(temp_upscaled_images_path)

            # Initialize the video writer with the upscaled image dimensions
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 30.0, (upscaled_image.shape[1], upscaled_image.shape[0]))

            # Release the temporary image
            cv2.destroyAllWindows()

        else:
            if scale_factor is None or scale_factor <= 0:
                raise ValueError("Scale factor must be specified and greater than 0 when RealESRGAN is disabled.")

            else:
            # Open the input video file
                cap = cv2.VideoCapture(input_path)

                # Get the original video's frame width and height
                frame_width = int(cap.get(3))
                frame_height = int(cap.get(4))

                # Calculate the new frame dimensions after upscaling
                new_width = int(frame_width * scale_factor)
                new_height = int(frame_height * scale_factor)

                # Define the codec for video
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, 30.0, (new_width, new_height))

                # Calculate the total number of frames in the video
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                # Create a tqdm progress bar
                progress_bar = tqdm(total=total_frames, desc="Processing Frames", unit="frame")

            # Loop through the frames of the input video
        while True:
            ret, frame = cap.read()

            # Break the loop if we have reached the end of the video
            if not ret:
                break

            # Apply sharpening with user-defined intensity
            if sharpen_intensity > 0:
                kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
                frame = cv2.filter2D(frame, -1, kernel)

            # Apply deinterlacing with user-defined strength
            
            
            # Resize the frame to the new dimensions
            resized_frame = cv2.resize(frame, (new_width, new_height))

            # Write the resized frame to the output video
            out.write(resized_frame)

            # Update the progress bar
            progress_bar.update(1)

        # Close the progress bar
        progress_bar.close()

        # Release video objects
        cap.release()
        out.release()

        print("Video processing complete. Temporary video saved as", output_path)

    except Exception as e:
        print("An error occurred:", str(e))



from tqdm import tqdm

def create_images_from_video(input_video_path, output_image_folder):
    try:
        # Open the input video file
        cap = cv2.VideoCapture(input_video_path)
        frame_number = 0

        # Calculate the total number of frames in the video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create a tqdm progress bar
        progress_bar = tqdm(total=total_frames, desc="Creating Images", unit="frame")

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # Save the frame as an image in the output image folder
            image_filename = f"frame_{frame_number:04d}.png"
            image_path = os.path.join(output_image_folder, image_filename)
            cv2.imwrite(image_path, frame)

            frame_number += 1

            # Update the progress bar
            progress_bar.update(1)

        cap.release()
        progress_bar.close()

        print("Images created from video frames. Images saved in", output_image_folder)

    except Exception as e:
        print("An error occurred during image creation:", str(e))

# Example usage:

output_image_folder = "temp_images"  # Replace with the folder where you want to save the images

# Create the image folder if it doesn't exist
os.makedirs(output_image_folder, exist_ok=True)







def upscale_button_click(input_path_var, output_path_var,  output_image_folder, scale_factor_entry, sharpen_intensity_scale, denoise_strength_scale, realesrgan_checkbox, temp_upscaled_images_path, outscale_value):
   
    
   
    input_video_path = input_path_var.get()
    create_images_from_video(input_video_path, output_image_folder)
    temp_video_path = "temp_video.mp4"  # Temporary video file
    temp_compiledvideo_path = "temp2.mp4"
    output_video_path = output_path_var.get()  # Use the specified output path
    scale_factor_str = scale_factor_entry.get()  # Get the scale factor as a string
    sharpen_intensity = sharpen_intensity_scale.get()
    denoise_strength = denoise_strength_scale.get()

    # Check if RealESRGAN upscaling is enabled
    use_realesrgan = realesrgan_checkbox.get()

   


    # Determine the number of threads based on the user's choice
   

    # Define RealESRGAN options
    realesrgan_options = None
    if use_realesrgan:
        realesrgan_options = {
        "model_name": "realesr-general-x4v3",  # You can change the model name as needed
        "suffix": "out",
        "ext": "auto"
    }
        

    # Define scale_factor outside the if-else block with a default value of 1
    scale_factor = 1

    try:
        if not use_realesrgan:
            # If RealESRGAN is not enabled, parse the scale factor
            scale_factor = float(scale_factor_str)  # Convert the string to a float

        # Get the selected outscale value from the dropdown menu
        outscale_value = "2"

        if use_realesrgan:
            # Use a temporary path for the RealESRGAN upscaled video
            temp_upscaled_images_path = "temp_upscaled_images"
            os.makedirs(temp_upscaled_images_path, exist_ok=True)
            upscale_with_realesrgan(output_image_folder, temp_upscaled_images_path, outscale_value, realesrgan_options)

            # Compile the upscaled images back into a video using OpenCV
            compile_images_to_video(temp_upscaled_images_path, temp_compiledvideo_path)
            add_audio_to_video(input_video_path, temp_compiledvideo_path, output_video_path)

        if os.path.exists(temp_compiledvideo_path):
            os.remove(temp_compiledvideo_path)
            print("Temporary compiled video deleted:", temp_compiledvideo_path)
        else:
            
            upscale_and_enhance_video(input_video_path, output_video_path, temp_video_path, scale_factor, sharpen_intensity, denoise_strength)
            print("Normal Video Upscaling Complete. Errors below are for AI Upscaling.")

        # Add audio to the upscaled video and save it to the final output path
            add_audio_to_video(input_video_path, temp_video_path, output_video_path)
        
            

    except ValueError as ve:
        print("ValueError:", str(ve))
    except Exception as e:
        print("An error occurred:", str(e))

          # Clean up: Remove the temporary images
    clean_temp_images(output_image_folder)

# Function to clean up temporary images in the given folder
def clean_temp_images(folder_path):
    try:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print("Temporary images in", folder_path, "have been deleted.")
    except Exception as e:
        print("An error occurred while cleaning up temporary images:", str(e))

def compile_images_to_video(temp_upscaled_images_path, temp_compiledvideo_path):
    try:
        # Get a list of image file names in the directory
        image_files = sorted([os.path.join(temp_upscaled_images_path, img) for img in os.listdir(temp_upscaled_images_path) if img.endswith(('.jpg', '.jpeg', '.png'))])

        if not image_files:
            raise Exception("No upscaled images found in the directory.")

        # Read the first image to get dimensions
        first_image = cv2.imread(image_files[0])
        height, width, layers = first_image.shape

        # Define the codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_compiledvideo_path, fourcc, 30.0, (width, height))

        # Create a tqdm progress bar
        progress_bar = tqdm(total=len(image_files), desc="Compiling Video", unit="frame")

        # Write the images to the video
        for image_file in image_files:
            frame = cv2.imread(image_file)
            out.write(frame)
            progress_bar.update(1)

        # Close the progress bar
        progress_bar.close()

        # Release the video writer
        out.release()

        # Clean up: Remove the temporary images
        for image_file in image_files:
            os.remove(image_file)

        print("Images compiled into a video. Output saved as", temp_compiledvideo_path)

    except Exception as e:
        print("An error occurred during image compilation:", str(e))

     




def add_audio_to_video(input_video_path, temp_video_path, output_video_path):
    try:
        # Load the processed video without audio using moviepy
        video_clip = VideoFileClip(temp_video_path)

        # Load the original audio from the input video using moviepy
        audio_clip = AudioFileClip(input_video_path)

        # Set the audio of the video clip to the loaded audio clip
        video_clip = video_clip.set_audio(audio_clip)

        # Write the final video with audio
        video_clip.write_videofile(output_video_path, codec='libx264')

        print("Audio added to the video. Output saved as", output_video_path)

    except Exception as e:
        print("An error occurred:", str(e))





    



