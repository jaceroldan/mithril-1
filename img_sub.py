import rclpy # Python library for ROS 2
from rclpy.node import Node # Handles the creation of nodes
from sensor_msgs.msg import Image # Image is the message type
import cv2 # OpenCV library
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
import torch
import torchvision.transforms as transforms
from torchvision.models import detection
import numpy as np
import argparse
import json
 
ap = argparse.ArgumentParser()

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
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

MODELS = {
	'frcnn-resnet': detection.fasterrcnn_resnet50_fpn,
	'frcnn-mobilenet': detection.fasterrcnn_mobilenet_v3_large_320_fpn,
	'retinanet': detection.retinanet_resnet50_fpn,
}

model = MODELS['frcnn-mobilenet'](
	pretrained=True,
	progress=True,
	num_classes=91,
	pretrained_backbone=True).to(DEVICE)
model.eval()

class ImageSubscriber(Node):
  """
  Create an ImageSubscriber class, which is a subclass of the Node class.
  """
  def __init__(self):
    """
    Class constructor to set up the node
    """
    # Initiate the Node class's constructor and give it a name
    super().__init__('image_subscriber')
      
    # Create the subscriber. This subscriber will receive an Image
    # from the video_frames topic. The queue size is 10 messages.
    self.subscription = self.create_subscription(
      Image, 
      '/D455_1/color/image_raw', 
      self.listener_callback, 
      10)
    self.subscription # prevent unused variable warning
      
    # Used to convert between ROS and OpenCV images
    self.br = CvBridge()
   
  def listener_callback(self, data):
    """
    Callback function.
    """
    # Display the message on the console
    self.get_logger().info('Receiving video frame')
 
    # Convert ROS Image message to OpenCV image
    current_frame = self.br.imgmsg_to_cv2(data,"bgr8")
    image = current_frame
    image_tensor = transforms.ToTensor()(current_frame)
    orig = image.copy()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.transpose((2,0,1))
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    image = torch.FloatTensor(image)
    image = image.to(DEVICE)
    detections = model(image)[0]
    print('here')
    print('detections: ', len(detections))
    for i in range(0, len(detections["boxes"])):
    	confidence = detections['scores'][i]
    	
    	if confidence > args['confidence']:

    		idx = int(detections['labels'][i])
    		box = detections['boxes'][i].detach().cpu().numpy()
    		(startX, startY, endX, endY) = box.astype('int')
    		print(idx)
    		# if(CLASSES[idx-1] != 'person'): continue
    		label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
    		print("[INFO] {}".format(label))
    		cv2.rectangle(current_frame, (startX, startY), (endX, endY), COLORS[idx], 2)
    		y = startY - 15 if startY - 15 > 15 else startY + 15
    		cv2.putText(current_frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
    		# if(CLASSES[idx-1] == 'person'): break;

    # 	# Display image
    
    # cv2.imshow("Output", orig)
    cv2.imshow("input", current_frame)
    
    cv2.waitKey(1)
  
def main(args=None):
  
  # Initialize the rclpy library
  rclpy.init(args=args)
  
  # Create the node
  image_subscriber = ImageSubscriber()
  
  # Spin the node so the callback function is called.
  rclpy.spin(image_subscriber)
  
  # Destroy the node explicitly
  # (optional - otherwise it will be done automatically
  # when the garbage collector destroys the node object)
  image_subscriber.destroy_node()
  
  # Shutdown the ROS client library for Python
  rclpy.shutdown()
  
if __name__ == '__main__':
  main()
