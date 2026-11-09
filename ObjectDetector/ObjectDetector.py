from pathlib import Path
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
from general.utils import save_video_path

import cv2
import numpy as np
from ultralytics import YOLO, RTDETR
import supervision as sv

# Add the "ObjectDetector" directory to sys.path
main_path = Path(__file__).parent
sys.path.insert(0, str(main_path))

from ObjectDetector.utils.wrappers import boxmot_to_supervision, supervision_to_boxmot
from boxmot import BoTSORT

project_path = main_path.parent
sys.path.insert(0, str(project_path))

class ObjectDetector:
    def __init__(self, model_selection="yolov9c.pt"):
        # Define the main path of the project
        self.MAIN_PATH = Path(".")
        # Path for the module
        self.MODULE_PATH = self.MAIN_PATH / "ObjectDetector"
        path_to_check = Path(__file__).parent
        print(path_to_check)
        # OBJECT DETECTOR INITIALIZER
        self.WEIGTH_PATH = self.MODULE_PATH / "weights"
        if not self.WEIGTH_PATH.exists():
            raise Exception(f"The path '{self.WEIGTH_PATH.as_posix()}' is not recognized.")
        
        if model_selection == "yolov9c.pt":
            print(f"Using weights from '{self.WEIGTH_PATH / model_selection}'")
            self.object_detector_model = YOLO(self.WEIGTH_PATH / model_selection)
        elif model_selection == "yolov9e.pt":
            print(f"Using weights from '{self.WEIGTH_PATH / model_selection}'")
            self.object_detector_model = YOLO(self.WEIGTH_PATH / model_selection)
        elif model_selection == "rtdetr-l.pt":
            print(f"Using weights from '{self.WEIGTH_PATH / model_selection}'")
            self.object_detector_model = RTDETR(self.WEIGTH_PATH / model_selection)
        else:
            raise Exception(f"Model '{model_selection}' is not implemented.")
    
        
    def save_tracking(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"Error opening video file. No such file at '{video_path}'.")
        
        frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

        # Define the data path 
        self.DATA_PATH = self.MAIN_PATH / "data"
        if not self.DATA_PATH.exists():
            self.DATA_PATH.mkdir()
            raise Exception(f"The path '{self.DATA_PATH.as_posix()}' is not recognized.")
        save_video_path(self.DATA_PATH, video_path)

        video_name = Path(video_path).stem
        print(video_name)
        video_data_path = self.DATA_PATH / video_name
        if not video_data_path.exists():
            print(f"Creating data folder for video as '{video_data_path.as_posix()}'.")
            video_data_path.mkdir()

        # Utility to store Object Detector output data
        csv_sink = sv.CSVSink(video_data_path / "objects_detections_unprocessed.csv")
        
        # Annotators
        thickness = sv.calculate_optimal_line_thickness(resolution_wh=(frame_width,frame_height))
        text_scale = sv.calculate_optimal_text_scale(resolution_wh=(frame_width,frame_height))

        # Initialize bbox annotator
        bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=thickness)
        # Initialize label annotator 
        label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness)

        # Get total number of frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        print(f"The video has {total_frames} frames at {fps} fps.")
        # Progress bar
        progress_bar = tqdm(total=total_frames)
     
        track_buffer_S = 10
        frame_rate=30
        tracker_weight = self.WEIGTH_PATH / "clip_market1501.pt"
        tracker = BoTSORT(model_weights= tracker_weight, #Path('clip_market1501.pt'), # which ReID model to use
                        device="cuda:0",
                        fp16=False,
                        with_reid=True,
                        per_class=False,
                        frame_rate=frame_rate,
                        track_high_thresh=0.4, 
                        track_low_thresh=0.2,
                        new_track_thresh=0.5,
                        match_thresh=0.8,
                        track_buffer=track_buffer_S*frame_rate,
                        appearance_thresh=0.15,
                        proximity_thresh=0.99,
                        fuse_first_associate=False
                        )
        
        selected_classes = [0, 1, 2, 3, 5, 6, 7]
        message = [f"{self.object_detector_model.names[class_id]}" 
                   for class_id in selected_classes]
        print(f"Detecting the following classes {message}.")
        cap.release()

        frame_generator = select_less_blurry_Laplacian(video_path, number_of_frames=1)

        with csv_sink as sink:
            for frame, frame_number in frame_generator:
                annotated_frame = frame.copy()

                # Pass frame to object detector and obtain the detections
                result = self.object_detector_model.predict(frame, classes=selected_classes, imgsz=(960,544), augment=False, verbose=False)[0]
                detections = sv.Detections.from_ultralytics(result)

                detections = detections[np.isin(detections.class_id, selected_classes)]
                # Supervision detection to boxmot [x1, y1, x2, y2, conf, cls]
                dets = supervision_to_boxmot(detections)
                # Update the tracker
                dets = tracker.update(dets, annotated_frame)
                # Boxmot detections to supervision
                detections = boxmot_to_supervision(dets, detections, self.object_detector_model)
                # Filter detections to keep only those within the frame boundaries
                valid_indices = (
                        (detections.xyxy[:, 0] >= 0) & (detections.xyxy[:, 1] >= 0) &
                        (detections.xyxy[:, 2] <= frame_width) & (detections.xyxy[:, 3] <= frame_height) &
                        (detections.xyxy[:, 0] <= frame_width) & (detections.xyxy[:, 1] <= frame_height) &
                        (detections.xyxy[:, 2] >= 0) & (detections.xyxy[:, 3] >= 0)
                    )
                detections = detections[valid_indices]
                if detections.tracker_id is not None:
                    sink.append(detections, custom_data={"Frame": frame_number})
                    # Generate the labels
                    labels = [f"#{int(track_id)} {self.object_detector_model.names[int(class_id)]} {confidence:0.2f}"
                              for _, _, confidence, class_id, track_id, data in detections]
                    # Annotate the labels
                    annotated_frame = label_annotator.annotate(
                        scene=annotated_frame, detections=detections, labels=labels
                    )
                    # Annotate the bounding box
                    annotated_frame = bounding_box_annotator.annotate(
                        scene=annotated_frame, detections=detections
                    )
                annotated_frame = cv2.resize(annotated_frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
                cv2.imshow("Object Detection", annotated_frame)
                if cv2.waitKey(1) == ord("q"):
                    break
                progress_bar.update()
        cv2.destroyAllWindows()


def select_less_blurry_Laplacian(input_video_path, number_of_frames, debug=False):
    """
    Loads frames from a video, computes their blurriness, and yields chunks of frames.

    Args:
        input_video_path (str): Path to the input video file.
        number_of_frames (int): Number of frames to process in each chunk.
        debug (bool, optional): If True, displays all frames and the selected least blurry frame. Defaults to False.

    Yields:
        list of numpy.ndarray: A chunk of frames (size = number_of_frames).
    """
    cap = cv2.VideoCapture(input_video_path)
    total_frame_index = 0
    while True:
        frames = []
        for _ in range(number_of_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            total_frame_index += 1
        

        if not frames:
            cap.release()
            cv2.destroyAllWindows()
            break
        
        frame_blur_index = total_frame_index-number_of_frames
        least_blurry_frame = frames[0]

        if number_of_frames > 1:
            # Compute blurriness for each frame
            blur_scores = []
            for frame in frames:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blur_score = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
                blur_scores.append(blur_score)

            # Find the index of the least blurry frame
            least_blurry_index = blur_scores.index(max(blur_scores))
            least_blurry_frame = frames[least_blurry_index]
            frame_blur_index = total_frame_index-number_of_frames+least_blurry_index

            if debug:
                plt.figure(figsize=(10, 6))
                for i, frame in enumerate(frames):
                    print(blur_scores[i])
                    plt.subplot(2, len(frames), i + 1)
                    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    plt.title(f"Frame {i}")
                    plt.axis("off")

                plt.subplot(2, len(frames), len(frames) + 1)
                plt.imshow(cv2.cvtColor(least_blurry_frame, cv2.COLOR_BGR2RGB))
                plt.title(f"Least Blurry Frame {least_blurry_index} {frame_blur_index}")
                plt.axis("off")

                plt.show()

        yield least_blurry_frame, frame_blur_index
