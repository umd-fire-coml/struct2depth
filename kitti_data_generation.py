import os
import cv2
import glob
import numpy as np

# Define global constant
SEQ_LENGTH = 3
WIDTH = 416
HEIGHT = 128

INPUT_DIR:str = "home/kitti-3d-detection-unzipped/training"
OUTPUT_DIR:str = "home/kitti-3d-detection-unzipped/training/output/"
PREV_DIR = INPUT_DIR + "/prev_2/"
NUM_Pre_FRAM = 2;


def data_gen(input_dir: str, output_dir: str):
    # define default input and output directory
    if input_dir is None:
        input_dir = INPUT_DIR
    if output_dir is None:
        output_dir = OUTPUT_DIR

    # if input directory not found, return error.
    if not os.path.isdir(input_dir):
        return FileNotFoundError("X incorrect input directory")
    # if output directory is not found, create a new directory
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    if not output_dir.endswith("/"):
        output_dir += "/"

    # get a list of png imag
    curr_img_list: list = glob.glob(input_dir+"/image_2/*.png")

    for i in range(len(curr_img_list)):
        # creating a large image
        large_img = np.zeros(shape=(HEIGHT, WIDTH * SEQ_LENGTH, 3))
        large_mask = np.zeros(shape=(HEIGHT, WIDTH * SEQ_LENGTH, 3))

        seq : str = curr_img_list[i].split("_")[0]  # get serial number of the img
        curr_img = cv2.imread(curr_img_list[i])  # convert current img into np array
        curr_img = cv2.resize(curr_img, (WIDTH, HEIGHT)) # resize img
        large_img[:, 0:WIDTH] = curr_img  # put current img into large img
        large_mask[:, 0:WIDTH] = MASK

        # creat mask for current frame

        for num_of_fram in range(NUM_Pre_FRAM):
            prev_img_dir = PREV_DIR + seq + "_0" + num_of_fram + ".png"  # creat directory for previous frame
            prev_img = cv2.imread(prev_img_dir)  # convert previous frame into np array
            prev_img = cv2.resize(prev_img, (WIDTH, HEIGHT))
            large_img[:, (num_of_fram+1) * WIDTH: (num_of_fram + 2) * WIDTH] = prev_img

            # create_mask for previous frame
            mask
            large_img[:, (num_of_fram + 1) * WIDTH: (num_of_fram + 2) * WIDTH] = mask

        # output big img
        cv2.imwrite(output_dir + seq + "_big_img.png", large_img)
        cv2.imwrite(output_dir+ seq + "big_mask.png", large_mask)

    return
