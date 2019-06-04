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
from Mask_RCNN_master.mrcnn import utils
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
    
    
class SegGen(object):
    
    def __init__(self):
        # Directory to save logs and trained model
        self.MODEL_DIR = os.path.join(ROOT_DIR, "logs")
        
        # Local path to trained weights file
        self.COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
        # Download COCO trained weights from Releases if needed
        if not os.path.exists(self.COCO_MODEL_PATH):
            utils.download_trained_weights(self.COCO_MODEL_PATH)
            
        class _InferenceConfig(coco.CocoConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

        self.config = _InferenceConfig()

        K = keras.backend.backend()
        if K=='tensorflow':
            keras.backend.set_image_dim_ordering('tf')


        # Create model object in inference mode.
        self.model = modellib.MaskRCNN(mode="inference", model_dir=self.MODEL_DIR, config=self.config)

        # Load weights trained on MS-COCO
        self.model.load_weights(self.COCO_MODEL_PATH, by_name=True)
        
        
    def generate_segmentation(self, image):
        results = self.model.detect([image], verbose=1)
        
        r = results[0]
        class_ids = r['class_ids']
        masks = r['masks']
        masks = masks.astype(int)

        for i in range(len(class_ids)):
            if class_ids[i] in NON_MOVING_CLASSES:
                masks[:,:,i] = np.zeros((len(masks), len(masks[0])))

        colors = visualize.random_colors(len(class_ids))
        seg = np.zeros((len(masks), len(masks[0]), 3), dtype=np.uint32)
        for i in range(len(class_ids)):
            seg = visualize.apply_mask(seg, masks[:,:,i], colors[i])
        
        return seg