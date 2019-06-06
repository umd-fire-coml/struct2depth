import os
import cv2
import glob
import numpy as np
import seg
import tensorflow as tf
import sys

class PreProcessor():
    def __init__(self,input_dir, output_dir):
        # Define global constant
        self.SEQ_LENGTH = 3
        self.WIDTH = 416
        self.HEIGHT = 128

        INPUT_DIR:str = "/home/ubuntu/kitti-3d-detection-unzipped/training/"
        OUTPUT_DIR:str = "/home/ubuntu/big_img_output"
        self.PREV_DIR = INPUT_DIR + "prev_2/"
        self.NUM_PREV_FRAM = 2;

        print("Default output directory is " + OUTPUT_DIR)
        
        # define default input and output directory
        if input_dir is None:
            self.input_dir = INPUT_DIR
        else:
            self.input_dir = input_dir
        if output_dir is None:
            self.output_dir = OUTPUT_DIR
        else:
            self.output_dir = output_dir
            
        # if input directory not found, return error.    
        if not os.path.isdir(self.input_dir):
            print("Incorrect Input Directory: "+ self.input_dir)
            return FileNotFoundError("X incorrect input directory")
        # if output directory is not found, create a new directory
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

        if not self.output_dir.endswith("/"):
            self.output_dir += "/"
        
        
    def data_gen(self):
        # get a list of png imag
        curr_img_list: list = glob.glob(self.input_dir+"image_2/*.png")
        curr_img_list = sorted(curr_img_list)

        for i in range(len(curr_img_list)): #len(curr_img_list)
            # creating segmentation object
            segger = seg.SegGen(self.SEQ_LENGTH) 
            # create a list that will contains all frames in resized form (np array)
            frame_list = [] 
            # creating a large image
            large_img = np.zeros(shape=(self.HEIGHT, self.WIDTH * self.SEQ_LENGTH, 3))
            large_mask = np.zeros(shape=(self.HEIGHT, self.WIDTH * self.SEQ_LENGTH, 3))

            seq : str = curr_img_list[i].split("/")[-1].split(".")[0]  # get serial number of the img
            print("processing img #" + seq)
            curr_img = cv2.imread(curr_img_list[i])  # convert current img into np array
            frame_list.append(curr_img)
            curr_img = cv2.resize(curr_img, (self.WIDTH, self.HEIGHT)) # resize img
            
            large_img[:, self.WIDTH * self.NUM_PREV_FRAM: self.WIDTH * (self.NUM_PREV_FRAM+1)] = curr_img  # put current img into large img
            for num_of_fram in range(self.NUM_PREV_FRAM):
                prev_img_dir = self.PREV_DIR + seq + "_0" + str(num_of_fram+1) + ".png"  # creat directory for previous frame
                prev_img = cv2.imread(prev_img_dir)  # convert previous frame into np array
                frame_list.append(prev_img)
                prev_img = cv2.resize(prev_img, (self.WIDTH, self.HEIGHT))
                large_img[:, (self.NUM_PREV_FRAM - num_of_fram-1) * self.WIDTH: (self.NUM_PREV_FRAM - num_of_fram) * self.WIDTH] = prev_img
            # output big img
            cv2.imwrite(self.output_dir + seq + ".png", large_img)
    
            # get segmentation for all frames here
            seg_list = segger.generate_segmentation(frame_list)
            #resize current frame mask
            seg_list[0] = cv2.resize(seg_list[0], (self.WIDTH, self.HEIGHT))
            # set mask for current frame
            large_mask[:, self.WIDTH * self.NUM_PREV_FRAM: self.WIDTH * (self.NUM_PREV_FRAM+1)] = seg_list[0]
            # set mask for previous frame
            for j in range(self.NUM_PREV_FRAM):
                mask = cv2.resize(seg_list[j+1], (self.WIDTH, self.HEIGHT))
                large_mask[:, (self.NUM_PREV_FRAM - j-1) * self.WIDTH: (self.NUM_PREV_FRAM - j) * self.WIDTH] = mask
            
            # output big segmask
            cv2.imwrite(self.output_dir+ seq + "-seg.png", large_mask)
        
        print("Large img and large seg generation is done")

        return
    
if __name__ == '__main__':
    input_dir_sys = None
    output_dir_sys = None
    
    if len(sys.argv) > 1:
        input_dir_sys = sys.argv[1]
    if len(sys.argv) > 2:
        output_dir_sys = sys.argv[2]

    print("*** starting segmentation generation ***")
    pre_processor = PreProcessor(input_dir_sys, output_dir_sys)
    pre_processor.data_gen()
    print("*** end of segmentation generation ***" )





