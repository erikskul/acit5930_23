#!/usr/bin/env python

import cv2
import numpy as np
import argparse
import csv
import os
import sys

# progressbar adapted from https://stackoverflow.com/questions/3160699/python-progress-bar
def progressbar(it, prefix="", size=60, out=sys.stdout):
    count = len(it)
    def show(j):
        x = int(size*j/count)
        print(f"{prefix}[{u'█'*x}{('.'*(size-x))}] {j}/{count}", end='\r', file=out, flush=True)
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    print("\n", flush=True, file=out)

def elastic_distortion(image, alpha, sigma):
    '''
    Parameters:
    - image: input image
    - alpha: scale of distortion
    - sigma: smoothness of distortion

    Returns:
    - distorted image
    '''

    # create a meshgrid of the image
    h, w = image.shape[:2]
    y, x = np.mgrid[:h, :w]

    # apply random displacements
    dx = alpha * np.random.randn(h, w)
    dy = alpha * np.random.randn(h, w)
    dx_smooth = cv2.GaussianBlur(dx, (0, 0), sigma)
    dy_smooth = cv2.GaussianBlur(dy, (0, 0), sigma)
    x_new = x + dx_smooth.astype(np.int32)
    y_new = y + dy_smooth.astype(np.int32)

    # make the new coordinates stay within the image and combine them into a single matrix
    x_new = np.clip(x_new, 0, w-1)
    y_new = np.clip(y_new, 0, h-1)
    xy_map = np.dstack((x_new, y_new)).astype(np.float32)

    # apply elastic distortion to the image
    distorted_image = cv2.remap(image, xy_map, None, cv2.INTER_LINEAR)

    # smooth the entire image
    distorted_image = cv2.GaussianBlur(distorted_image, (3,3), 0)

    return distorted_image

annotation_path = os.path.expanduser('~')+'/scout_ws/synth_dataset/annotations.csv'
image_dir = os.path.expanduser('~')+'/scout_ws/synth_dataset/images'
image_num = 1

if not os.path.exists(image_dir):
    os.makedirs(image_dir)

# take argument from command line for the number of images to generate
argParser = argparse.ArgumentParser()
argParser.add_argument('-n', '--num_images', type=int, default=10000, help='Number of images to generate')
args = argParser.parse_args()

print(f"Generating {args.num_images} images")

#for i in range(args.num_images):
for i in progressbar(range(args.num_images), "Generating: ", 40):
    
    # create a 32x32 one-channel image with every pixel having the value 0
    image = np.zeros((32, 32, 1), dtype=np.uint8)

    # number of features to add
    num_features_range = (3, 7)
    num_features = np.random.randint(*num_features_range)
    
    max_pixelvalue = 255

    for j in range(num_features):
        
        # randomly choose a radius for the feature
        feature_radius_range = (2,5)
        feature_radius = np.random.randint(*feature_radius_range)
        # randomly choose a location in the entire image to place the feature
        x = np.random.randint(feature_radius, image.shape[0] - feature_radius)
        y = np.random.randint(feature_radius, image.shape[1] - feature_radius)

        # create a feature and add it
        heat_spot = np.zeros((feature_radius*2+1, feature_radius*2+1), dtype=np.uint8) 
        cv2.circle(heat_spot, (feature_radius, feature_radius), feature_radius, (np.random.randint(max_pixelvalue//2, max_pixelvalue),), -1)
        x1, y1 = max(x-feature_radius, 0), max(y-feature_radius, 0)
        x2, y2 = min(x+feature_radius+1, image.shape[0]), min(y+feature_radius+1, image.shape[1])
        w, h = x2-x1, y2-y1
        image[x1:x2, y1:y2] += cv2.resize(heat_spot, (w, h))[:, :, np.newaxis]

        # add some noise
        image += np.random.randint(0, 10, image.shape, dtype=np.uint8)

        # could later add extra spots to only one section of the image, to ensure one section always has a lot higher sum of pixel values

    # add noise to the blurred image
    noise = np.zeros(image.shape, dtype=np.int16)
    cv2.randn(noise, 0, 64) # mean=0, standard deviation=64
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # apply gaussian blur
    image = cv2.GaussianBlur(image, (9, 9), 0)

    # add elastic distortion to the entire image
    # edit here: maximum displacement of pixels and grid size (how often to displace)
    max_disp = 3
    grid_size = 10

    # transformation matrix with random displacements
    h, w = image.shape[:2]
    dx = np.random.rand(grid_size, grid_size)
    dy = np.random.rand(grid_size, grid_size)
    dx = cv2.resize(dx, (w, h)).astype(np.float32) * 2 * max_disp - max_disp
    dy = cv2.resize(dy, (w, h)).astype(np.float32) * 2 * max_disp - max_disp
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + dx).astype(np.float32)
    map_y = (grid_y + dy).astype(np.float32)

    # apply elastic distortion to the image
    # can be commented out for no distortion. removes black spots, but features are more uniform
    # the second line is a similar distortion but where pixels are replaced instead of being moved (no black spots, but not sure if the distortion is good)
    image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
    #image = elastic_distortion(image, alpha=100, sigma=10)

    # split image into equally wide 3 vertical sections, left middle and right
    width = image.shape[1] // 3
    left = image[:, 1:width+1]
    middle = image[:, width+1:-width-1]
    right = image[:, -width-1:-1]

    # find the sum of the pixel values in each strip
    left_sum = np.sum(left)
    middle_sum = np.sum(middle)
    right_sum = np.sum(right)

    # data to be saved
    image_name = str(image_num) + '.png'
    pixel_values = [left_sum, middle_sum, right_sum]
    actions = ['left', 'middle', 'right']

    # pick a random action as an index
    action = np.random.randint(0, len(actions))

    # write the image and annotation to a file
    cv2.imwrite(f'{image_dir}/{image_num}.png', image)
    with open(annotation_path, 'a', newline='') as annotations_file:
        annotations_writer = csv.writer(annotations_file)
        annotations_writer.writerow([image_name, actions[action], pixel_values[action], "///", pixel_values[0], pixel_values[1], pixel_values[2]])
    
    # increment the image number
    image_num += 1

    # display the image
    # resized_image = cv2.resize(image, (128, 128))
    # cv2.imshow('image', resized_image)
    # cv2.waitKey(0)