import io
import os
import scipy.misc
import numpy as np
import six
import time
import glob
from IPython.display import display

from six import BytesIO

import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont, ImageTransform

import tensorflow as tf

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
#Load model
tf.keras.backend.clear_session()
model = tf.saved_model.load("C:/Users/HP/Desktop/CMNDData/export_model/saved_model")

import cv2
def run_inference_for_single_image(model, image):
  
	image = np.asarray(image)
	input_tensor = tf.convert_to_tensor(image)
	input_tensor = input_tensor[tf.newaxis,...]

	model_fn = model.signatures['serving_default']
	output_dict = model_fn(input_tensor)

	num_detections = int(output_dict.pop('num_detections'))
	output_dict = {key:value[0, :num_detections].numpy() 
	             for key,value in output_dict.items()}
	output_dict['num_detections'] = num_detections
	output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

	if 'detection_masks' in output_dict:
		detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
		          output_dict['detection_masks'], output_dict['detection_boxes'],
		           image.shape[0], image.shape[1])      
		detection_masks_reframed = tf.cast(detection_masks_reframed < 0.5,
		                                   tf.uint8)
		output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

	return output_dict

def load_image_into_numpy_array(path):
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def Count_box(detection_scores):
    return len([count for count in detection_scores if count >= 0.5])

def center_box(coord):
    return (coord[0]+coord[2])/2, (coord[1]+coord[3])/2

category_index = label_map_util.create_category_index_from_labelmap("C:/Users/HP/Desktop/CMNDData/Models/research/label_map.txt", use_display_name=True)

image_path = r'C:/Users/HP/Desktop/CMNDData/Dataset/1.png'
image_np = load_image_into_numpy_array(image_path)
print("Done load image ")
image_np = cv2.resize(image_np, dsize=None, fx=1, fy=1)
output_dict = run_inference_for_single_image(model, image_np)
print("Done inference")
vis_util.visualize_boxes_and_labels_on_image_array(
    image_np,
    output_dict['detection_boxes'],
    output_dict['detection_classes'],
    output_dict['detection_scores'],
    category_index,
    instance_masks=output_dict.get('detection_masks_reframed', None),
    use_normalized_coordinates=True,
    line_thickness=1)
print("Done draw on image ")
Imagee = Image.fromarray(image_np)
width, height = Imagee.size
Imagee.show()
if Count_box(output_dict['detection_scores'][:5]) == 3:
    box_list = []
    for i in range(3):
        ymin, xmin, ymax, xmax = float(output_dict['detection_boxes'][i][0])*height, float(output_dict['detection_boxes'][i][1])*width, float(output_dict['detection_boxes'][i][2])*height, float(output_dict['detection_boxes'][i][3])*width
        box_list.append((output_dict['detection_classes'][i],center_box((xmin, ymin, xmax, ymax))))

    # Xac Dinh Goc
    for box in box_list:
        if (box[0] == 2):
            topRight = box[1]
        elif (box[0] == 1):
            topLeft = box[1]
        else:
            botRight = box[1]

    print(topRight, topLeft, botRight)
    vector = [topRight[0]-topLeft[0], topRight[1]-topLeft[1]]
    print(vector)
    bot_left = [botRight[0]-vector[0], botRight[1]-vector[1]]
    
    import math
    new_width = int(math.sqrt((topRight[0]-topLeft[0])**2+(topRight[1]-topLeft[1])**2))
    new_height = int(math.sqrt((topRight[0]-botRight[0])**2+(topRight[1]-botRight[1])**2))
    transform=[topLeft[0], topLeft[1], bot_left[0], bot_left[1], botRight[0], botRight[1], topRight[0], topRight[1]]
    Imagee.transform((new_width,new_height), ImageTransform.QuadTransform(transform)).show()