import os
import random
import tempfile
import traceback
from urllib.request import urlopen
import shutil

import cv2 as cv
import numpy as np
import streamlit as st

from models import *
import configs
# from strings import *


import gc  # garbage collection

def open_img_path_url(url_or_file, source_type, source_path=None, resize=False):
    img, mask = [], []

    if source_type == "path":
        if source_path is None:
            source_path = dataPath
        img = cv.imread(os.path.join(source_path, url_or_file))

    elif source_type == "url":
        resp = urlopen(url_or_file)
        img = np.asarray(bytearray(resp.read()), dtype="uint8")
        img = cv.imdecode(img, cv.IMREAD_COLOR)

    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    if not resize:
        return img

    else:
        img_h = img.shape[0]
        ratio = target_h / img_h
        r_img = cv.resize(img, None, fx=ratio, fy=ratio)

        try:
            r_img_w = r_img.shape[1]
            left_edge = target_w // 2 - r_img_w // 2

            mask = np.zeros((target_h, target_w, 3), dtype="uint8")
            mask[:, left_edge : left_edge + r_img_w] = r_img

            return img, mask

        except Exception:
            return img, r_img
#
#
# def display_tracker_options():
#     display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
#     is_display_tracker = True if display_tracker == 'Yes' else False
#     if is_display_tracker:
#         tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
#         return is_display_tracker, tracker_type
#     return is_display_tracker, None
#
#
# def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None):
#     """
#     Display the detected objects on a video frame using the YOLOv8 model.
#
#     Args:
#     - conf (float): Confidence threshold for object detection.
#     - model (YoloV8): A YOLOv8 object detection model.
#     - st_frame (Streamlit object): A Streamlit object to display the detected video.
#     - image (numpy array): A numpy array representing the video frame.
#     - is_display_tracking (bool): A flag indicating whether to display object tracking (default=None).
#
#     Returns:
#     None
#     """
#
#     # Resize the image to a standard size
#     image = cv2.resize(image, (720, int(720*(9/16))))
#
#     # Display object tracking, if specified
#     if is_display_tracking:
#         res = model.track(image, conf=conf, persist=True, tracker=tracker)
#     else:
#         # Predict the objects in the image using the YOLOv8 model
#         res = model.predict(image, conf=conf)
#
#     # # Plot the detected objects on the video frame
#     res_plotted = res[0].plot()
#     st_frame.image(res_plotted,
#                    caption='Detected Video',
#                    channels="BGR",
#                    use_column_width=True
#                    )
#
#
# def play_stored_video(conf, model):
#     """
#     Plays a stored video file. Tracks and detects objects in real-time using the YOLOv8 object detection model.
#
#     Parameters:
#         conf: Confidence of YOLOv8 model.
#         model: An instance of the `YOLOv8` class containing the YOLOv8 model.
#
#     Returns:
#         None
#
#     Raises:
#         None
#     """
#     source_vid = st.sidebar.selectbox(
#         "Choose a video...", configs.VIDEOS_DICT.keys())
#
#     is_display_tracker, tracker = display_tracker_options()
#
#     with open(configs.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
#         video_bytes = video_file.read()
#     if video_bytes:
#         st.video(video_bytes)
#
#     if st.sidebar.button('Detect Video Objects'):
#         try:
#             vid_cap = cv2.VideoCapture(
#                 str(configs.VIDEOS_DICT.get(source_vid)))
#             st_frame = st.empty()
#             while (vid_cap.isOpened()):
#                 success, image = vid_cap.read()
#                 if success:
#                     _display_detected_frames(conf,
#                                              model,
#                                              st_frame,
#                                              image,
#                                              is_display_tracker,
#                                              tracker
#                                              )
#                 else:
#                     vid_cap.release()
#                     break
#         except Exception as e:
#             st.sidebar.error("Error loading video: " + str(e))