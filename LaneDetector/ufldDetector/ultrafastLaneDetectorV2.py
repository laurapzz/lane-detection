import torch
import cv2
import torch
import scipy.special
from scipy.spatial.distance import cdist
import torchvision.transforms as transforms
from PIL import Image
from enum import Enum
from typing import Tuple
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import copy
import os
import csv

try:
    from LaneDetector.ufldDetector.utils.type import LaneModelType, lane_colors
    from LaneDetector.ufldDetector.core import LaneDetectBase
    from LaneDetector.ufldDetector.utils.common import merge_config, get_model
except:
    from .utils.type import LaneModelType, lane_colors
    from .core import LaneDetectBase
    from .utils.common import merge_config, get_model

class ModelConfig():
    def __init__(self, model_path, model_config, model_type):
        if model_type == LaneModelType.UFLDV2_TUSIMPLE:
            _, self.cfg = merge_config([model_config, "--test_model", model_path])
        else:
            raise NotImplementedError

class UltrafastLaneDetectorV2(LaneDetectBase):
    _defaults = {
        "model_path": "models/tusimple_res18.pth",
        "model_config": "configs/tusimple_res18.py",
        "model_type": LaneModelType.UFLDV2_TUSIMPLE
    }

    def __init__(self, model_path: str = None, model_config: str = None, model_type: LaneModelType = None, 
                 save_coords: bool = False, video_name: str = None):
        LaneDetectBase.__init__(self)
        if None not in [model_path, model_config, model_type]:
            self.model_path, self.model_config, self.model_type = model_path, model_config, model_type

        if self.model_type not in [LaneModelType.UFLDV2_TUSIMPLE]:
            raise Exception("UltrafastLaneDetectorV2 can't use %s type." % self.model_type.name)
        
        self.cfg = ModelConfig(self.model_path, self.model_config, self.model_type).cfg
        self.cfg = self.cfg._cfg_dict
        self.batch_size = 1
        assert self.cfg.backbone in ['18','34','50','101','152','50next','101next','50wide','101wide']
        if self.cfg.dataset == "Tusimple":
            self.cls_num_per_lane = 56
        elif self.cfg.dataset == "CULane":
            self.cls_num_per_lane = 18
        else:
            raise NotImplementedError
        
        self.save_coords = save_coords
        self.frame_number = 0
        if video_name is not None:
            self.video_name = os.path.splitext(video_name)[0]
            
        self.model = self._initialize_model()
        self.img_transform = self._initialize_image_transform()

    def _initialize_model(self):
        cfg = self.cfg
        net = get_model(cfg)

        try:
            if torch.backends.mps.is_built():
                net = net.to("mps")
                state_dict = torch.load(cfg.test_model, map_location='mps')['model']
            else:
                net = net.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                state_dict = torch.load(cfg.test_model, map_location='cuda')['model']
        except:
            state_dict = torch.load(cfg.test_model, map_location='cpu')['model']
        
        compatible_state_dict = {}
        for k, v in state_dict.items():
            if 'module.' in k:
                compatible_state_dict[k[7:]] = v
            else:
                compatible_state_dict[k] = v

        net.load_state_dict(compatible_state_dict, strict=False)
        net.eval()
        return net
    
    def _initialize_image_transform(self):
        img_transforms = transforms.Compose([
            transforms.Resize((int(self.cfg.train_height / self.cfg.crop_ratio), self.cfg.train_width)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        return img_transforms
    
    def detect_lanes(self, image):
        input_tensor = self.prepare_input(image)
        output = self.inference(input_tensor)
        self.coords = self.pred2coords(output)
        visualization_img = self.draw_coords(image, self.coords, self.cfg)
        return visualization_img, self.coords
    
    def prepare_input(self, image):
        self.img_h, self.img_w = image.shape[0], image.shape[1]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(image)
        input_img = self.img_transform(img_pil)
        input_img = input_img[:, -self.cfg.train_height:, :]
        input_tensor = input_img[None, ...]

        if not torch.backends.mps.is_built():
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            input_tensor = input_tensor.to(device)

        return input_tensor
    
    def inference(self, input_tensor):
        with torch.no_grad():
            output = self.model(input_tensor)
        return output
    
    def pred2coords(self, pred, local_width=1):
        exist_row_prob = pred['exist_row'].softmax(1).cpu()
        exist_col_prob = pred['exist_col'].softmax(1).cpu()
        row_anchor = self.cfg.row_anchor
        col_anchor = self.cfg.col_anchor

        original_image_height = self.img_h
        original_image_width = self.img_w

        batch_size, num_grid_row, num_cls_row, num_lane_row = pred['loc_row'].shape
        batch_size, num_grid_col, num_cls_col, num_lane_col = pred['loc_col'].shape

        max_indices_row = pred['loc_row'].argmax(1).cpu()
        valid_row = pred['exist_row'].argmax(1).cpu()

        max_indices_col = pred['loc_col'].argmax(1).cpu()
        valid_col = pred['exist_col'].argmax(1).cpu()

        pred['loc_row'] = pred['loc_row'].cpu()
        pred['loc_col'] = pred['loc_col'].cpu()

        coords = []
        row_lane_idx = [0, 1, 2, 3]
        col_lane_idx = [0, 1, 2, 3]

        for i in row_lane_idx:
            tmp = []
            if valid_row[0, :, i].sum() > num_cls_row / 2:
                for k in range(valid_row.shape[1]):
                    if valid_row[0, k, i]:
                        all_ind = torch.tensor(list(range(max(0, max_indices_row[0, k, i] - local_width), min(num_grid_row - 1, max_indices_row[0, k, i] + local_width) + 1)))
                        out_tmp = (pred['loc_row'][0, all_ind, k, i].softmax(0) * all_ind.float()).sum() + 0.5
                        out_tmp = out_tmp / (num_grid_row - 1) * original_image_width
                        if exist_row_prob[0, 1, k, i] > 0.95:
                            tmp.append((int(out_tmp), int(row_anchor[k] * original_image_height)))
            coords.append(tmp)

        for i in col_lane_idx:
            tmp = []
            if valid_col[0, :, i].sum() > num_cls_col / 4:
                for k in range(valid_col.shape[1]):
                    if valid_col[0, k, i]:
                        all_ind = torch.tensor(list(range(max(0, max_indices_col[0, k, i] - local_width), min(num_grid_col - 1, max_indices_col[0, k, i] + local_width) + 1)))
                        out_tmp = (pred['loc_col'][0, all_ind, k, i].softmax(0) * all_ind.float()).sum() + 0.5
                        out_tmp = out_tmp / (num_grid_col - 1) * original_image_height
                        if exist_col_prob[0, 1, k, i] > 0.95:
                            tmp.append((int(col_anchor[k] * original_image_width), int(out_tmp)))
            coords.append(tmp)

        return coords

    def draw_coords(self, image, coords, cfg):
        im0 = cv2.resize(image, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA)

        for lane_num, lane_coords in enumerate(coords):
            for coord in lane_coords:
                font = cv2.FONT_HERSHEY_SIMPLEX 
                cv2.putText(im0, f"{lane_num}", coord, font, 0.5, lane_colors[lane_num], 2)

        if self.save_coords:
            self.save_lane_lines_csv(coords)
        return im0

    def save_lane_lines_csv(self, coords):
        csv_directory = "data/lane_detection_csv"
        self.frame_number += 1
        csv_file_name = f"ld_{self.video_name}.csv"
        csv_file_path = os.path.join(csv_directory, csv_file_name)

        if not os.path.exists(csv_directory): 
            os.makedirs(csv_directory)

        write_mode = "a"
        if self.frame_number == 1:
            write_mode = "w"

        with open(csv_file_path, write_mode, newline='') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=";", quoting=csv.QUOTE_MINIMAL)
            if write_mode == "w":
                csv_writer.writerow(["Frame Number", "Lane Number", "Starting Point", "Number of Points", "Point Coordinates"])
            for lane_num, lane_coords in enumerate(coords):
                if lane_num in [0, 1]:
                    standardized_coords = self.standardize_coords(lane_coords, 50)
                    points_str = ';'.join([f'{x},{y}' for x, y in standardized_coords])
                    csv_writer.writerow([self.frame_number, lane_num, 0, len(lane_coords), points_str])

    def standardize_coords(self, lane_coords, max_length=50):
        coords_array = np.zeros((max_length, 2))
        num_coords = min(len(lane_coords), max_length)
        if num_coords > 1:
            coords_array[:num_coords, :] = lane_coords[:num_coords]
        return coords_array.tolist()
