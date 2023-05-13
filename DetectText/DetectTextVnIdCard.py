import numpy as np
from PIL import Image
from LoadModel import LoadCheckPoint
from Detect4Corner import Detect4CornerImage

def Count_box(detection_scores):
	return len([score for score in detection_scores if score >= 0.4])

def non_max_suppression_fast(boxes, labels, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []

	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
	#
	# initialize the list of picked indexes
	pick = []
	# grab the coordinates of the bounding boxes
	x1 = boxes[:, 1]
	y1 = boxes[:, 0]
	x2 = boxes[:, 3]
	y2 = boxes[:, 2]

	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(y2)

	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)

		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])

		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)

		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]

		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

	# return only the bounding boxes that were picked using the
	# integer data type
	final_labels = [labels[idx] for idx in pick]
	final_boxes = boxes[pick].astype("float")
	return final_boxes, final_labels

def detecTextVnIdCard(img):
	# detect 4 goc
	img = Detect4CornerImage.detect4corner(img)

	# load checkpoint
	configs_path = "./DetectText/pipeline.config"
	checkpoint_path = "./DetectText/output_model/ckpt-31"
	label_map_path = "./DetectText/label_map.txt"

	# predict
	detector = LoadCheckPoint.Detector(configs_path, checkpoint_path, label_map_path)
	image, original_image, output_dict = detector.predict(img)

	# Remove noise box
	score = Count_box(output_dict['detection_scores'])
	output_dict['detection_scores'], output_dict['detection_classes'], output_dict['detection_boxes'] = output_dict['detection_scores'][:score], output_dict['detection_classes'][:score], output_dict['detection_boxes'][:score]
	output_dict['detection_boxes'], output_dict['detection_classes'] = non_max_suppression_fast(output_dict['detection_boxes'], output_dict['detection_classes'], 0.969)

	# get coord
	data = [[], [], [], [], []]
	for i, detection_class in enumerate(output_dict['detection_classes']):
		if detection_class == 1:
			data[1].append(output_dict['detection_boxes'][i])
		elif detection_class == 2:
			data[0].append(output_dict['detection_boxes'][i])
		elif detection_class == 3:
			data[2].append(output_dict['detection_boxes'][i])
		elif detection_class == 4:
			data[3].append(output_dict['detection_boxes'][i])
		elif detection_class == 5:
			data[4].append(output_dict['detection_boxes'][i])

	return original_image, data