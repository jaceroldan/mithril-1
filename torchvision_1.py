from torchvision.models import detection
import numpy as np
import argparse
import json
import torch
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str, required=True, help="path to the input image")
ap.add_argument("-m", "--model", type=str, default="frcnn-resnet",
	choices=['frcnn-resnet', 'frcnn-mobilenet', 'retinanet'],
	help='name of the object detection model')
ap.add_argument("-l", "--labels", type=str, default='coco-labels-2014_2017.txt',
	help="path to file containing list of categories in COCO dataset")
ap.add_argument('-c', '--confidence', type=float, default=0.5,
	help='minimum probability to filter weak detections')
args = vars(ap.parse_args())

DEVICE = torch.device('cuda' if torch.cuda.is_available() else "cpu")

CLASSES = open(args['labels'], 'r').read().split('\n')
print(CLASSES)
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

MODELS = {
	'frcnn-resnet': detection.fasterrcnn_resnet50_fpn,
	'frcnn-mobilenet': detection.fasterrcnn_mobilenet_v3_large_320_fpn,
	'retinanet': detection.retinanet_resnet50_fpn,
}

model = MODELS[args["model"]](
	pretrained=True,
	progress=True,
	num_classes=91,
	pretrained_backbone=True).to(DEVICE)
model.eval()

image = cv2.imread(args['image'])
orig = image.copy()

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image.transpose((2,0,1))

image = np.expand_dims(image, axis=0)
image = image / 255.0
image = torch.FloatTensor(image)

image = image.to(DEVICE)
detections = model(image)[0]
print(len(CLASSES))

for i in range(0, len(detections["boxes"])):
	confidence = detections['scores'][i]

	if confidence > args['confidence']:
		idx = int(detections['labels'][i])
<<<<<<< HEAD
=======
		print(detections)
>>>>>>> aa0ab2b1a4661bb8c84da19854f05d95d17dfafc
		box = detections['boxes'][i].detach().cpu().numpy()
		(startX, startY, endX, endY) = box.astype('int')

		print(idx)
<<<<<<< HEAD
		label = "{}: {:.2f}%".format(CLASSES[idx-1], confidence * 100)
=======
		label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
>>>>>>> aa0ab2b1a4661bb8c84da19854f05d95d17dfafc
		print("[INFO] {}".format(label))

		cv2.rectangle(orig, (startX, startY), (endX, endY), COLORS[idx], 2)
		y = startY - 15 if startY - 15 > 15 else startY + 15
		cv2.putText(orig, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

<<<<<<< HEAD
=======
torch.save(model, './models/model.pt')

>>>>>>> aa0ab2b1a4661bb8c84da19854f05d95d17dfafc
cv2.imshow("Output", orig)
cv2.imwrite('test.png', orig)
cv2.waitKey(0)

