
import pyrealsense2 as rs
import numpy as np
import cv2
import torch
from show_example import showbbox
from train import get_model_instance_segmentation, get_transform
import transforms as T
import torchvision


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def eval_show(img):
    transform1 = T.Compose([
        T.ToTensor(),  # range [0, 255] -> [0.0,1.0]
    ]
    )
    num_class = 2
    model = get_model_instance_segmentation(num_class)
    model.load_state_dict(torch.load("test.pth"))
    model.eval()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)
    xx, _ = transform1(img, 0)
    xx = showbbox(model, xx)
    return xx

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device1 = pipeline_profile.get_device()
device_product_line = str(device1.get_info(rs.camera_info.product_line))

config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        image = eval_show(color_image)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        # if depth_colormap_dim != color_colormap_dim:
        #     resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
        #     images = np.hstack((resized_color_image, depth_colormap))
        # else:
        #     images = np.hstack((color_image, depth_colormap))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_NORMAL)
        cv2.imshow('RealSense', image)
        cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:

    # Stop streaming
    pipeline.stop()
