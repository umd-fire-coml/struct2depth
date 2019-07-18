import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
import keras.backend

# Root directory of the project
ROOT_DIR = "Mask_RCNN_master"
# Import Mask RCNN
import Mask_RCNN_master.mrcnn.utils as utils
import Mask_RCNN_master.mrcnn.model as modellib
from Mask_RCNN_master.mrcnn import visualize
# Import COCO config
import Mask_RCNN_master.samples.coco.coco as coco

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']
NON_MOVING_CLASSES = [10, 11, 12, 13, 14]
MAX_COLORS = 50    
    
class SegGen(object):
    
    def __init__(self, seq_length):
        # Directory to save logs and trained model
        self.MODEL_DIR = os.path.join(ROOT_DIR, "logs")
        self.colors = visualize.random_colors(MAX_COLORS)
        self.seq_length = seq_length
        
        # Local path to trained weights file
        self.COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
        # Download COCO trained weights from Releases if needed
        if not os.path.exists(self.COCO_MODEL_PATH):
            utils.download_trained_weights(self.COCO_MODEL_PATH)
            
        class _InferenceConfig(coco.CocoConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = self.seq_length

        self.config = _InferenceConfig()

        K = keras.backend.backend()
        if K=='tensorflow':
            keras.backend.set_image_dim_ordering('tf')
            

        # Create model object in inference mode.
        self.model = modellib.MaskRCNN(mode="inference", model_dir=self.MODEL_DIR, config=self.config)

        # Load weights trained on MS-COCO
        self.model.load_weights(self.COCO_MODEL_PATH, by_name=True)
        
    def IOU(self,box1, box2):
        x_overlap = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]));
        y_overlap = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]));
        interArea = x_overlap * y_overlap;
        unionArea = (box1[3] - box1[1]) * (box1[2] - box1[0]) + (box2[3] - box2[1]) * (box2[2] - box2[0]) - interArea
        return interArea / unionArea
        
        
    def generate_segmentation(self, images):
        """images is an array of images: shape (seq_length, H, W, 3). Order will be current, -1, -2"""
        if (len(images) != self.seq_length):
            print("Wrong length of images - it must match the sequence length used to instantiate the class")
            return
        
        color_map = {}
        segs = []
        results = self.model.detect(images, verbose=1)
        
        for k in range(len(results)):
            
            class_ids = results[k]['class_ids']
            boxes = results[k]['rois']
            masks = results[k]['masks']
            masks = masks.astype(int)

            for i in range(len(class_ids)):
                if class_ids[i] in NON_MOVING_CLASSES:
                    masks[:,:,i] = np.zeros((len(masks), len(masks[0])))


            seg = np.zeros((len(masks), len(masks[0]), 3))
            
            #This is the base image
            if k == 0:
                for i in range(len(class_ids)):
                    seg = visualize.apply_mask(seg, masks[:,:,i], self.colors[i])
                    color_map[i] = boxes[i]
            else:
                for i in range(len(class_ids)):
                    
                    color = -1
                    for j in range(len(color_map)):
                        if self.IOU(color_map[j], boxes[i]) >= 0.5:
                            color = self.colors[j]
                            color_map[j] = boxes[i]
                            
                    if color == -1:
                        color_map[len(color_map)] = boxes[i]
                        color = self.colors[len(color_map)]
                    
                    seg = visualize.apply_mask(seg, masks[:,:,i], color)
                    
                    
            segs.append(seg)
        
        return segs
