import cv2
from PIL import Image
from Transformer_OCR import vietOCR
from DetectText import DetectTextVnIdCard

def sort_box(coord):
	line1 = []
	line2 = []
	center = (max(coord , key=lambda k: [k[0]])[0]+min(coord , key=lambda k: [k[0]])[0])/2

	for l in coord:
		if l[0] < center:
			line1.append(l)
		else:
			line2.append(l)

	return sorted(line1 , key=lambda k: [k[1]])+sorted(line2 , key=lambda k: [k[1]])

def cropImg(img, coord):
	height, width = img.shape
	space = 5
	return img[int(coord[0]*height-space):int(coord[2]*height+space), int(coord[1]*width-space):int(coord[3]*width+space)]

def vietnam_id_card(img):
	# get box
	original_image, box = DetectTextVnIdCard.detecTextVnIdCard(img)

	# Img to text
	text_box = [[], [], [], [], []]
	count = 0
	gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
	print(box)
	for infos in box:
		print(infos)
		infos = sort_box(infos)
		for info in infos:
			text_img = Image.fromarray(cropImg(gray, info))
			text_box[count].append(vietOCR.convert_img_to_text(text_img))
		count += 1

	#option 1: return string like gg api
	return ' '.join(map(str, [r for result in text_box for r in result]))
	#option 2: return one of list
	return print([' '.join(map(str, result)) for result in text_box])


img_file = 'C:/Users/HP/Desktop/CMND_Test/baocuong.jpg'
print(vietnam_id_card(Image.open(img_file)))