import cv2
import numpy as np
import os
from tkinter.filedialog import askopenfilename
from pathlib import Path
import pandas as pd

from LaneDetector.ufldDetector.ultrafastLaneDetectorV2 import UltrafastLaneDetectorV2
from LaneDetector.ufldDetector.utils.type import LaneModelType
from LaneDetector.ufldDetector.utils.visualization import line_fit

def check_data_folder(file_name):
    DATA_PATH = Path(f"./data")
    if not DATA_PATH.exists():
        raise Exception(f"The path '{DATA_PATH.as_posix()}' is not recognized.")
    video_name = Path(file_name).stem
    DATA_PATH = Path(DATA_PATH / video_name)
    if not DATA_PATH.exists():
        print(f"Creating data folder in {DATA_PATH.as_posix()}")
        DATA_PATH.mkdir()

    return DATA_PATH

def update_dataframe(dataframe, frame_number, coords):
    new_row = {"Frame Number": int(frame_number)}
    for index, coordinates in enumerate(coords):
        column_name = f"Lane_{index}_Coordinates"
        coordinates_str = ';'.join([f'{x},{y}' for x, y in coordinates])
        new_row[column_name] = coordinates_str

    extended_df = pd.DataFrame([new_row])
    dataframe = pd.concat([dataframe, extended_df], ignore_index=True)
    return dataframe

def save_video_path(data_path, video_path):
    file_path = data_path.as_posix() + "/" + "video_path.csv"
    df = pd.DataFrame({"video_path": video_path}, index=[0])
    df.to_csv(file_path, index=False)

def lane_detection(video_path, LINE_POINT=1200):
    video_name = os.path.basename(video_path)
    print(f"Selected file path: {video_path}")
    print(f"Selected file: {video_name}")
    data_path = check_data_folder(video_name)

    lane_config = {
        "model_path": "./LaneDetector/models/tusimple_res18.pth",
        "model_config": "./LaneDetector/configs/tusimple_res18.py",
        "model_type": LaneModelType.UFLDV2_TUSIMPLE
    }

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Error opening video file. No such file at '{video_path}'.")    
    save_video_path(data_path, video_path)

    if "UFLDV2" in lane_config["model_type"].name:
        UltrafastLaneDetectorV2.set_defaults(lane_config)
        laneDetector = UltrafastLaneDetectorV2(save_coords=False, video_name=video_name)
    elif "UFLD" in lane_config["model_type"].name:
        raise NotImplementedError
    
    df = pd.DataFrame()
    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_cpy = frame.copy()
        cropped_frame = frame_cpy[:LINE_POINT, :]
        output_img, coords = laneDetector.detect_lanes(cropped_frame)
        df = update_dataframe(df, frame_number, coords)

        frame_cpy[:LINE_POINT, :] = output_img[:LINE_POINT, :] 
        output_img = cv2.resize(frame_cpy, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        cv2.imshow("Detected lanes", output_img)
        key = cv2.waitKey(1)

        if key == ord('q'):
            break
        elif key == ord("p"):
            cv2.waitKey(0)
        frame_number += 1

    file_path = data_path.as_posix() + "/" + "lane_detection_unprocessed.csv"
    df.to_csv(file_path, index=False)
    cap.release()
    cv2.destroyAllWindows()

def lane_detection_image(image_path, LINE_POINT=1200):
    image_name = os.path.basename(image_path)
    stem_name = Path(image_path).stem
    data_path = check_data_folder(stem_name)

    lane_config = {
        "model_path": "./LaneDetector/models/tusimple_res18.pth",
        "model_config": "./LaneDetector/configs/tusimple_res18.py",
        "model_type": LaneModelType.UFLDV2_TUSIMPLE
    }

    UltrafastLaneDetectorV2.set_defaults(lane_config)
    laneDetector = UltrafastLaneDetectorV2(save_coords=False, video_name=stem_name)

    frame = cv2.imread(image_path)
    if frame is None:
        raise Exception(f"Could not read image file at '{image_path}'.")

    frame_cpy = frame.copy()
    crop_height = min(LINE_POINT, frame.shape[0])
    cropped_frame = frame_cpy[:crop_height, :]

    _, coords = laneDetector.detect_lanes(cropped_frame)

    y_vals = [crop_height - 10, int(crop_height * 0.75), int(crop_height * 0.5)]
    ego_lanes = []
    
    # Index 1 = Left-Ego, Index 2 = Right-Ego
    for idx in [1, 2]:
        if idx < len(coords) and len(coords[idx]) >= 2:
            _, fitted_coords = line_fit(coords[idx], y_vals=y_vals, compute_r2=False)
            if fitted_coords and not any(np.isnan(p[0]) for p in fitted_coords):
                ego_lanes.append((idx, fitted_coords))
            else:
                ego_lanes.append((idx, coords[idx]))
        elif idx < len(coords) and len(coords[idx]) > 0:
            ego_lanes.append((idx, coords[idx]))
        else:
            ego_lanes.append((idx, []))

    annotated_crop = cropped_frame.copy()
    colors = {1: (0, 255, 0), 2: (255, 0, 0)}
    for idx, lane_coords in ego_lanes:
        if len(lane_coords) > 1:
            pts = np.array(lane_coords, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_crop, [pts], isClosed=False, color=colors[idx], thickness=4)
            for pt in lane_coords:
                cv2.circle(annotated_crop, (int(pt[0]), int(pt[1])), 5, colors[idx], -1)

    frame_cpy[:crop_height, :] = annotated_crop

    row_data = {"Image_Path": image_path, "Frame Number": 0}
    for idx, lane_coords in ego_lanes:
        label = "Ego_Left_Lane" if idx == 1 else "Ego_Right_Lane"
        row_data[label] = ";".join([f"{x},{y}" for x, y in lane_coords])

    df = pd.DataFrame([row_data])
    output_csv = data_path / "lane_detection_image_processed.csv"
    df.to_csv(output_csv, index=False)

    return frame_cpy, output_csv

if __name__ == "__main__":
    video_path = askopenfilename()
    LINE_POINT = 1220
    lane_detection(video_path, LINE_POINT=LINE_POINT)
