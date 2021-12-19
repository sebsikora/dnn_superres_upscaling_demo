# Upscaling video with Python OpenCV
<br/>
<br/>

(C) 2021 Seb Sikora, published under the [MIT License](https://opensource.org/licenses/MIT)

[seb.nf.sikora@protonmail.com](mailto:seb.nf.sikora@protonmail.com)	
<br/>
<br/>

## Get using deep-learning based methods in your own Python projects with OpenCV dnn_superres
<br/>

[Super Resolution (SR)](https://blog.paperspace.com/image-super-resolution/) image upscaling via deep-learning based approaches can acheive really impressive results compared to naive methods.

It's really easy to leverage this power in your own projects using the [OpenCV dnn_superres module](https://docs.opencv.org/4.x/d5/d29/tutorial_dnn_superres_upscale_image_single.html), all you need to get started is to install the [OpenCV-contrib modules](https://pypi.org/project/opencv-contrib-python/) and download a [pre-trained](https://github.com/Saafke/FSRCNN_Tensorflow/tree/master/models)[ model](https://github.com/Saafke/EDSR_Tensorflow/tree/master/models). 

Xavier Weber has a great walk-through of the process of installing the modules and upscaling a single image [here](https://towardsdatascience.com/deep-learning-based-super-resolution-with-opencv-4fd736678066).
<br/>
<br/>

Below, we're going to run-through how to apply the same approach to upscaling video files.
<br/>
<br/>

## Upscaling a video file in Python using dnn_superres
<br/>

Using some of the other features of OpenCV, we can use the same techniques demonstrated above to upscale video files! 

To demonstrate this, let's start with some gorgeous freely available videos provided by [Ajaya Bist](https://www.pexels.com/video/close-up-view-of-a-parrot-4982608/) and [Erkan Avanoğlu](https://www.pexels.com/video/little-bird-inside-a-house-5761115/) over at [pexels.com](https://www.pexels.com/)

The original video dimensions are 1280x720, so first using ffmpeg we'll downscale both videos to 200x112 as shown at 100% scaling below. We will only retain the first ten seconds of the videos for the purpose of this demonstration.

```
user@home:~/dnn_superres/ffmpeg -t 10 -i video_1_1280x720.mp4 -vf scale=200:-2 -preset slow -crf 18 video_1_200x112.mp4
```

https://user-images.githubusercontent.com/18697847/146688988-b0cd3c35-f33b-4cbf-861c-862dd1003e22.mp4

https://user-images.githubusercontent.com/18697847/146691799-747a308f-d41a-434b-9e5e-1b147dcd2319.mp4

<br/>
<br/>

Now let's take a look at the code!

We're going to be using the OpenCV [VideoCapture](https://docs.opencv.org/4.5.4/d8/dfe/classcv_1_1VideoCapture.html#ac4107fb146a762454a8a87715d9b7c96) class to open our low-res source video and iterate through it frame-by-frame, the OpenCV contrib [dnn_superres interface](https://docs.opencv.org/4.x/d5/d29/tutorial_dnn_superres_upscale_image_single.html) to upscale each frame, and the OpenCV [VideoWriter](https://docs.opencv.org/4.5.4/dd/d9e/classcv_1_1VideoWriter.html#ac3478f6257454209fa99249cc03a5c59) class to create an output container and fill it with our upscaled frames.

Lastly, we will [use ffmpeg](https://superuser.com/questions/277642/how-to-merge-audio-and-video-file-in-ffmpeg) to mux the upscaled output video with the audio from the low-res source video.

```python
import cv2                                                # VideoCapture, VideoWriter, resize
from cv2 import dnn_superres                              # dnn_superres interface
import subprocess                                         # Needed to run ffmpeg to mux old audio & new video...
import os                                                 # ...(see at the end below)

scale_factor = 4   # Set upscaling factor here

in_file = './videos/original/video_1_200x112.mp4'        # Path to input video file (low res)
temp_file = in_file[:-4] + '_temp.mp4'                    # Path to temporary output video file (no audio)
out_file = in_file[:-4] + '_final.mp4'                    # Path to final output video file (with audio)

# ------ dnn_superres setup -------------------------------
# (no different from upscaling a single image)
# Create an SR object
sr = dnn_superres.DnnSuperResImpl_create()

# Read the desired pre-trained model
sr.readModel("./pre-trained-models/FSRCNN_x4.pb")
# (For details and pre-trained models see
# https://github.com/opencv/opencv_contrib/tree/master/modules/dnn_superres#models)

# Set the desired model and scale to get correct pre- and post-processing
# Other options for first argument "edsr", "espcn", "lapsrn"
sr.setModel("fsrcnn", scale_factor)
# ---------------------------------------------------------

# Create videocapture object with the path to the input video file as argument
cap = cv2.VideoCapture(in_file)

# Determine the video framerate, length in frames, frame height and width
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) * scale_factor    # Appropriately scale the height
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) * scale_factor      # and width for the upscaled output video

# Create a VideoWriter object that we will use to create the temporary output video file
# If we installed OpenCV via eg pip, H264 encoding is not included (https://stackoverflow.com/a/55598602)
# If we want the final upscaled output in H264 format we need to convert after-the-fact using ffmpeg
# Here we will use mp4v format for now
out = cv2.VideoWriter(temp_file, cv2.VideoWriter_fourcc('m','p','4','v'), fps, (width, height))

# Now we run the upscaling loop one frame-by-frame
i = 0
while cap.isOpened():
    status, frame = cap.read()    # Read the 'next frame' from the input video file
    
    if not status:                # If we have reached the end of the input video file stop the loop
        break
    
    result = sr.upsample(frame)   # Upscale the frame, just like upscaling a single image
    
    out.write(result)             # Write the upscaled frame to the temporary output video file
    
    # It can take a *long* time so nice to have some indication of progress
    i += 1
    print('Frame ' + str(i) + '/' + str(total_frames) + ' processed.')

# Close the input and temporary output video files
cap.release()
out.release()

# Using our upscaled frames, we have built an output video file, but it doesn't have any audio!
# We can use ffmpeg (again!) to create a final output file by muxing together the newly created
# temporary output file and the audio from the original input file.
print('Mux-ing audio...')

# Construct the cli command (info on doing this here - https://superuser.com/questions/277642/how-to-merge-audio-and-video-file-in-ffmpeg)
command = "ffmpeg -i {temp_file} -i {in_file} -c copy -map 0:v:0 -map 1:a:0 -shortest {out_file}".format(temp_file = temp_file, in_file = in_file, out_file = out_file)

# Run the command
subprocess.call(command,shell=True)

# Tidy up by removing the temporary output file
os.remove(temp_file)

```

We'll also create some counter-examples upscaled via more-traditional approaches.
<br/>
<br/>

First we can use the same code as above, modified to use the OpenCV [resize](https://docs.opencv.org/4.5.4/da/d54/group__imgproc__transform.html#ga47a974309e9102f5f08231edc7e7529d) function to upscale each frame, by default via linear interpolation. 

```python
# We use the same code as above, but we substitute this

result = cv2.resize(frame, (frame.shape[1] * scale_factor, frame.shape[0] * scale_factor))

# In place of this

result = sr.upsample(frame)
```
<br/>

Lastly we can use [ffmpeg](https://write.corbpie.com/a-guide-to-upscaling-or-downscaling-video-with-ffmpeg/), which by default uses bicubic interpolation.

```
user@home:~/dnn_superres/ffmpeg -i video_1_200x112.mp4 -vf scale=800:-2 -preset slow -crf 18 video_1_200x112_ffmpeg_x4.mp4
```
<br/>

So, to recap: We will upscale both videos x4 to 800x450 via the approaches outlined above and compare:
* cv2.resize (linear interpolation)
* ffmpeg (bicubic interpolation)
* dnn_superres with the FSRCNN model
* dnn_superres with the EDSR model
<br/>
<br/>

Let's take a look at the results for video_1:


https://user-images.githubusercontent.com/18697847/146691961-2013b167-7e68-4025-87b1-7b36b19c6e4f.mp4

https://user-images.githubusercontent.com/18697847/146691878-d7951e52-a7b4-4a23-9270-bec33a71e3a2.mp4

https://user-images.githubusercontent.com/18697847/146691968-454009ce-76ee-4505-9f8c-e2cc051e90e1.mp4

https://user-images.githubusercontent.com/18697847/146691996-9938906f-448c-4d1b-8dc0-3c6ed9e1200e.mp4

https://user-images.githubusercontent.com/18697847/146691906-14fa32e8-54f0-4721-bef0-6fa8ae41ab42.mp4

<br/>
<br/>

And for video_2:

https://user-images.githubusercontent.com/18697847/146692011-31c789f3-8eaa-45eb-83f8-3e0d89ed3e86.mp4

https://user-images.githubusercontent.com/18697847/146692055-f2fbb39c-8b47-4743-ad8c-f79b3c592655.mp4

https://user-images.githubusercontent.com/18697847/146692058-95f7c454-6c7a-4353-9148-ed3a70671bac.mp4

https://user-images.githubusercontent.com/18697847/146692064-f63a9107-491e-4248-9b8c-1ef6b18e6415.mp4

https://user-images.githubusercontent.com/18697847/146692071-963249d1-8847-482a-9fad-0eba38a1c62f.mp4

<br/>
<br/>

Blah blah blah...

https://user-images.githubusercontent.com/18697847/146692184-5dd55c74-dc6f-45fe-8a68-738fa366a611.mp4

https://user-images.githubusercontent.com/18697847/146692185-17c06663-38f7-41b0-992e-0df76cb1b3b9.mp4

<br/>
<br/>

Blah blah blah...

https://user-images.githubusercontent.com/18697847/146692198-4762b7d1-f771-4d16-a4b5-22a429fcf56e.mp4

https://user-images.githubusercontent.com/18697847/146692200-ecfe2967-7279-4fc6-b81a-86a576bfded7.mp4

