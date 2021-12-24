![banner](banner.png)
# Upscaling video with Python OpenCV dnn_superres
<br/>

(C) 2021 Seb Sikora, published under the [MIT License](https://opensource.org/licenses/MIT)

[seb.nf.sikora@protonmail.com](mailto:seb.nf.sikora@protonmail.com)	
<br/>
<br/>

## The Premise
<br/>

[Super Resolution (SR)](https://blog.paperspace.com/image-super-resolution/) image upscaling via deep-learning based approaches can acheive really impressive results compared to naive methods.

It's really easy to leverage this power in your own projects using the [OpenCV dnn_superres module](https://docs.opencv.org/4.x/d5/d29/tutorial_dnn_superres_upscale_image_single.html), all you need to get started is to install the [OpenCV-contrib modules](https://pypi.org/project/opencv-contrib-python/) and download a [pre-trained model](https://github.com/opencv/opencv_contrib/tree/master/modules/dnn_superres). 

Xavier Weber has a great walk-through of the process of installing the module and upscaling a single image [here](https://towardsdatascience.com/deep-learning-based-super-resolution-with-opencv-4fd736678066).
<br/>
<br/>

Below, we're going to run-through how to apply the same approach to upscaling video files.
<br/>
<br/>

## The Requirements

### Videos

To demonstrate this, let's start with some gorgeous freely available videos provided by [Ajaya Bist](https://www.pexels.com/video/close-up-view-of-a-parrot-4982608/), [Erkan Avanoğlu](https://www.pexels.com/video/little-bird-inside-a-house-5761115/), [Anna Bondarenko](https://www.pexels.com/video/blue-butterfly-sitting-on-a-hand-5757715/), [Bogdan Krupin](https://www.pexels.com/video/a-trippy-light-show-on-a-brick-wall-with-windows-10469592/) & [Assad Tanoli](https://www.pexels.com/video/pomegranate-stacked-at-fruit-stall-5731603/) over at [pexels.com](https://www.pexels.com/).

The original video dimensions are 1280x720, so first using ffmpeg we'll downscale both videos to 200x112 as shown at 100% scaling below. We will only retain the first ten seconds of the videos for the purpose of this demonstration.
<br/>
<br/>

```
user@home:~/dnn_superres/ffmpeg -t 10 -i video_1_1280x720.mp4 -vf scale=200:-2 -preset slow -crf 18 video_1_200x112.mp4
```
<br/>

https://user-images.githubusercontent.com/18697847/146688988-b0cd3c35-f33b-4cbf-861c-862dd1003e22.mp4

https://user-images.githubusercontent.com/18697847/146691799-747a308f-d41a-434b-9e5e-1b147dcd2319.mp4

https://user-images.githubusercontent.com/18697847/147356128-4e4866f7-7150-4c1e-bdd5-35d80e327346.mp4

https://user-images.githubusercontent.com/18697847/147356136-5ddc8373-d5a6-44a5-bfc0-98571998167d.mp4

https://user-images.githubusercontent.com/18697847/147356140-84e2d115-a7f5-42e2-be0c-98bd3ec94e7a.mp4

<br/>

### OpenCV dnn_superres contrib module

Blah blah blah...
<br/>

### Pre-trained Models

We need to download a pre-trained model that the dnn_superres module will use to upscale our videos. See [here](https://github.com/opencv/opencv_contrib/tree/master/modules/dnn_superres#models) for information on the available models and download links.

Four types of pre-trained model are available, varying in performance, available scaling factors and computational cost. Broadly, the 'fastest & least-performant' is [FSRCNN](), while the 'slowest and most-performant' is [EDSR](), with [LapSRN]() falling somewhere in between. Scaling factors of x2, x3 & x4 are provided by both FSRCNN and EDSR, while x2, x4 & x8 are provided by LapSRN.
<br/>

## The Code
<br/>

Now let's take a look at the code:
<br/>

We're going to be using the OpenCV [VideoCapture](https://docs.opencv.org/4.5.4/d8/dfe/classcv_1_1VideoCapture.html#ac4107fb146a762454a8a87715d9b7c96) class to open our low-res source video and iterate through it frame-by-frame, the OpenCV contrib [dnn_superres interface](https://docs.opencv.org/4.x/d5/d29/tutorial_dnn_superres_upscale_image_single.html) to upscale each frame, and the OpenCV [VideoWriter](https://docs.opencv.org/4.5.4/dd/d9e/classcv_1_1VideoWriter.html#ac3478f6257454209fa99249cc03a5c59) class to create an output container and fill it with our upscaled frames.

Lastly, we will [use ffmpeg](https://superuser.com/questions/277642/how-to-merge-audio-and-video-file-in-ffmpeg) to mux the upscaled output video with the audio from the low-res source video.

```python
import cv2                                                # VideoCapture, VideoWriter, resize
from cv2 import dnn_superres                              # dnn_superres interface
import subprocess                                         # Needed to run ffmpeg to mux old audio & new video...
import os                                                 # ...(see at the end below)

scale_factor = 4   # Set upscaling factor here

in_file = './videos/original/video_1_200x112.mp4'         # Path to input video file (low res)
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
<br/>

We'll also create some counter-examples upscaled via more-traditional approaches.
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

## The Results
Let's take a look at the results.
<br/>
<br/>

### Video_1:

https://user-images.githubusercontent.com/18697847/147270245-b51d1d8a-2bb6-4890-b50c-6c50c18cfbc4.mp4

https://user-images.githubusercontent.com/18697847/147270253-5abafe7b-8499-462d-9ae0-00f5c9f4ef9e.mp4

https://user-images.githubusercontent.com/18697847/147270261-f68148c3-442a-4968-aa6f-fc041eca483d.mp4

https://user-images.githubusercontent.com/18697847/147270266-f69970bd-c145-4169-a998-a66d301d9d81.mp4

https://user-images.githubusercontent.com/18697847/147270270-b0428cad-73a5-4fd9-a18b-48efcbb12d60.mp4

<br/>

### Video_2:

https://user-images.githubusercontent.com/18697847/147270322-deb0ad4b-0002-4b4a-94ee-cb6e88448d2d.mp4

https://user-images.githubusercontent.com/18697847/147270329-8b70b39c-bcac-413d-b484-7e6a3442ff0a.mp4

https://user-images.githubusercontent.com/18697847/147270334-1f1afd47-40a7-4a6e-982a-b20eff60ec1a.mp4

https://user-images.githubusercontent.com/18697847/147270346-0f7be659-29bc-445c-b414-bf2026560a10.mp4

https://user-images.githubusercontent.com/18697847/147270349-bcbcdd4a-c072-41d8-bd85-300262bf6e5b.mp4

<br/>

### Video_3:

https://user-images.githubusercontent.com/18697847/147270400-f4a4e1ec-4279-4b94-9eb5-2c7e8df3c9b7.mp4

https://user-images.githubusercontent.com/18697847/147270405-2d57268e-bd57-4b18-9fcb-238149d0cd7f.mp4

https://user-images.githubusercontent.com/18697847/147270407-6cb4c045-da8f-47a0-9453-dc2e043b83b2.mp4

https://user-images.githubusercontent.com/18697847/147270415-6f38c918-3b32-4eb6-be22-72d24043ca30.mp4

https://user-images.githubusercontent.com/18697847/147270418-f8cb317d-a3fc-437c-b412-71ddcca526eb.mp4

<br/>

### Video_4:

https://user-images.githubusercontent.com/18697847/147270457-3dd954c9-1af6-48d8-83ba-8d784c485424.mp4

https://user-images.githubusercontent.com/18697847/147270471-0f1f0869-8c0c-4c38-acbd-80ce3f7de52f.mp4

https://user-images.githubusercontent.com/18697847/147270480-9929ff1a-e228-43fa-906a-f707e160f089.mp4

https://user-images.githubusercontent.com/18697847/147270486-e65516f1-db67-4ba8-b272-c5b446ed21b6.mp4

https://user-images.githubusercontent.com/18697847/147270498-74bfb0ed-6f33-4098-a5a2-fe9f28825715.mp4

<br/>

### Video_5:

https://user-images.githubusercontent.com/18697847/147270537-8fd346f4-2637-4d67-8f86-313f6ac2e8d0.mp4

https://user-images.githubusercontent.com/18697847/147270544-075f6f49-38af-457a-98aa-547452223a3e.mp4

https://user-images.githubusercontent.com/18697847/147270556-17ff5771-afb9-407c-b4d1-2fc96b07e697.mp4

https://user-images.githubusercontent.com/18697847/147270568-0916093e-f2dd-4ed5-816b-8ab29b001253.mp4

https://user-images.githubusercontent.com/18697847/147270572-a2bca884-38bd-4cb9-be76-84cb179ff09e.mp4

<br/>


Blah blah blah...

### Video_1:

https://user-images.githubusercontent.com/18697847/147296979-f1e3bfc6-9aae-4a03-9616-3b6abcf3028f.mp4

https://user-images.githubusercontent.com/18697847/147296984-0ebdeaf8-e610-476d-b7be-8d0712fb5d9f.mp4

### Video_2:

https://user-images.githubusercontent.com/18697847/147297022-b2b84752-86c5-4135-8908-b9e033b27ee3.mp4

https://user-images.githubusercontent.com/18697847/147297026-972db4d6-dcd9-4fec-b4ec-099f7e38638d.mp4

### Video_3:

https://user-images.githubusercontent.com/18697847/147297079-5449edd3-f0da-43ec-beed-a116a4c34bb4.mp4

https://user-images.githubusercontent.com/18697847/147297091-22443d3b-bc45-4943-b86e-9d016c4f29fb.mp4

### Video_4:

https://user-images.githubusercontent.com/18697847/147297113-a7f9c651-a3cd-4737-ad2a-215dd7bb3853.mp4

https://user-images.githubusercontent.com/18697847/147297116-8fe00b6d-3d5b-4106-9ec3-c2db08797453.mp4

### Video_5:

https://user-images.githubusercontent.com/18697847/147297133-f1910dbc-1de4-489a-b9ae-c022ae3016b4.mp4

https://user-images.githubusercontent.com/18697847/147297139-121ae5ab-746d-47dd-b336-b137f0caa91d.mp4

