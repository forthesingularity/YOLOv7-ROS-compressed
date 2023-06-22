#!/usr/bin/python3

import string
import numpy as np
import torch, cv2, os, rospy
import time
from torchvision import transforms
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts
from utils.ros import create_humans_detection_msg
from sensor_msgs.msg import Image
from yolov7_ros.msg import HumansStamped, Human, KeyboardInput # KeyboardInput message 객체 불러옴: 근데 경로를 확실히 알아야 함 
from cv_bridge import CvBridge
from torchvision.transforms import ToTensor
from pose_class_MLP import pose_class_MLP

class YoloV7_HPE:   # pretrained_MLP 추가됨
    def __init__(self, weights: string ="../weights/yolov7-w6-pose.pt", pretrained_MLP: string ="../weights/pose_class_MLP.pt", device: str = "cuda"):
        rospy.loginfo("class YoloV7_HPE")
        #PARAMS TBD
        self.device = device
        self.weigths = torch.load(weights, map_location=device)
        self.model = self.weigths['model']
        rospy.loginfo(pretrained_MLP)
        rospy.loginfo("model")
        self.pretrained_MLP=pose_class_MLP()
        self.pretrained_MLP.load_state_dict(torch.load(pretrained_MLP))
        
        rospy.loginfo("pretrained_MLP_weight loaded")
        _ = self.model.float().eval()
        _ = self.pretrained_MLP.eval()  # MLP를 eval 모드로
        if torch.cuda.is_available():
            self.model.half().to(device)
            self.pretrained_MLP.to(device)
           
    

class Yolov7_HPEPublisher:
    def __init__(self, visualize: bool, device: string, queue_size: int, img_topic: str = "/my_camera_topic", weights: str = "../weights/yolov7-w6-pose.pt",
                pretrained_MLP: str = "../weights/pose_class_MLP.pt", out_img_topic: str ="yolov7_hpe", skeleton_keypoints_out_topic: str = "yolov7_hpe_skeletons", 
                keyboard_input_topic: str = "keyop/teleop"): # string이 file 주소에 해당?
        """
        :param weights: path/to/yolo_weights.pt
        :param img_topic: name of the image topic to listen to
        :param out_img_topic: topic for visualization will be published under the
            namespace '/yolov7')
        :param skeleton_keypoints_out_topic: intersection over union threshold will be published under the
            namespace '/yolov7')
        :param device: device to do inference on (e.g., 'cuda' or 'cpu')
        :param queue_size: queue size for publishers
        :visualize: flag to enable publishing the detections visualized in the image
        """
        self.tensorize = ToTensor()
        self.model = YoloV7_HPE( weights = weights, pretrained_MLP = pretrained_MLP, device = device)
        self.bridge = CvBridge()
        self.avg_time = 0
        self.cnt = 0
        
        #Subscribe to Image
        self.img_subscriber = rospy.Subscriber(img_topic, Image, self.process_img_msg)
        
        #Visualization Publisher
        self.out_img_topic = out_img_topic + "visualization" if out_img_topic.endswith("/") else out_img_topic + "/visualization"
        self.visualization_publisher = rospy.Publisher(self.out_img_topic, Image, queue_size=queue_size) if visualize else None
        
        #Keypoints Publisher
        #self.skeleton_keypoints_out_topic = skeleton_keypoints_out_topic + "visualization" if out_img_topic.endswith("/") else skeleton_keypoints_out_topic + "/visualization"
        self.keyboard_input_topic = keyboard_input_topic
        self.skeleton_detection_publisher = rospy.Publisher(skeleton_keypoints_out_topic, HumansStamped, queue_size=queue_size)
        #@@@ for MLP @@@
        # keyboard input publisher를 새로 선언, keyboard input topic에 publish 해준다. 
        self.keyboard_input_publisher = rospy.Publisher(keyboard_input_topic, KeyboardInput, queue_size=queue_size)

        rospy.loginfo("YOLOv7 initialization complete. Ready to start inference! Best regard CILAB")

    def process_img_msg(self, image: Image):
        start_time = time.time()
        # keyboard inputs list for processing in MLP
        ki = [0, 32, 67, 101, 68, 100, 66, 65]
        # action name list based on keyboard inputs
        action = ['idle','space', 'right', 'enable', 'left', 'disable', 'down', 'up']
        """ callback function for publisher """
        #rospy.loginfo("subscribe")
        image = self.bridge.imgmsg_to_cv2(image, "bgr8")    
        image = letterbox(image, 960, stride=64, auto=True)[0]
        image_ = image.copy()
        image = transforms.ToTensor()(image)
        image = torch.tensor(np.array([image.numpy()]))
        
        #start_time = time.time()
        if torch.cuda.is_available():
            image = image.half().to(device)
        with torch.no_grad():
            output, _ = self.model.model(image)
        #end_time = time.time()    
        
        output = non_max_suppression_kpt(output, 0.25, 0.65, nc=self.model.model.yaml['nc'], nkpt=self.model.model.yaml['nkpt'], kpt_label=True)
        output = output_to_keypoint(output)
        nimg = image[0].permute(1, 2, 0) * 255
        nimg = nimg.cpu().numpy().astype(np.uint8)
        nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
        for idx in range(output.shape[0]):
            plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)
        #######################################################################
        # *** nm ***
        # new_output=torch.Tensor(output[0, 7:]).unsqueeze(0)
        new_output=torch.Tensor(output[0, 7:]).to(device)
        new_output=new_output - new_output.min()
        new_output=new_output / new_output.max()
        #print(new_output)
        #print(new_output.size())
        # *** nm ***
        # new_output=torch.nn.functional.normalize(new_output * 10, p=2.0).to(device)
        y = self.model.pretrained_MLP(new_output)    # keyboard input is a MLP output
        index = int(torch.argmax(y).cpu().item()) # index for keyboard input list
        keyboard_input = ki[index]  # keyboard input list로부터 keypoard_input 값을 얻어냄
        action_name = action[index] # action name list로부터 action_name 값을 얻어냄

        #Publishing Keypoints
        if (len(output) > 0):
            keypoints_array_msg = create_humans_detection_msg(output)
            keyboard_input_msg = KeyboardInput()    # KeyboardInput 객체 초기화
            keyboard_input_msg.pressedKey = keyboard_input  # pressedKey에 keyboard_input 값을 넣어줌
            self.keyboard_input_publisher.publish(keyboard_input_msg) # publish 명령 수행
            # 콘솔 창에 keyboard input 값과 action name 띄워주기 
            #rospy.loginfo(f'Published: keyboard input = {keyboard_input}, action name = {action_name}')
            self.skeleton_detection_publisher.publish(keypoints_array_msg)
            #@@@ for MLP @@@
        end_time = time.time()
        self.cnt += 1
        self.avg_time = (self.avg_time * (0.9) + (end_time - start_time)*0.1)
        if (self.cnt % 100 == 0 ):
            print("Hz:{} time:{}".format(1 / self.avg_time,self.avg_time*1000))

        #Publishing Visualization if Required
        if self.visualization_publisher:
            vis_msg = self.bridge.cv2_to_imgmsg(nimg)
            self.visualization_publisher.publish(vis_msg)
        #rospy.loginfo("publish")


if __name__ == "__main__":
    rospy.init_node("yolov7_human_pose")

    ns = rospy.get_name() + "/"

    weights = rospy.get_param(ns + "weights")
    pretrained_MLP = rospy.get_param(ns + "pretrained_MLP") # pretrained_MLP weight 경로  
    img_topic = rospy.get_param(ns + "img_topic")
    out_img_topic = rospy.get_param(ns + "out_img_topic")
    # 질문: topic을 새로 선언을 해줘야 하나? 아니면 skeleton keypoints topic을 그대로 써도 되나?
    skeleton_keypoints_out_topic = rospy.get_param(ns + "skeleton_keypoints_out_topic")
    keyboard_input_topic = rospy.get_param(ns+"keyboard_input_topic")
    queue_size = rospy.get_param(ns + "queue_size")
    visualize = rospy.get_param(ns + "visualize")
    device = rospy.get_param(ns + "device")

    # some sanity checks
    if not os.path.isfile(weights):
        raise FileExistsError("Weights not found.")

    if not ("cuda" in device or "cpu" in device):
        raise ValueError("Check your device.") 
        
        

    
    publisher = Yolov7_HPEPublisher(
        img_topic=img_topic,
        out_img_topic=out_img_topic, 
        skeleton_keypoints_out_topic = skeleton_keypoints_out_topic,
        keyboard_input_topic = keyboard_input_topic,
        weights=weights,
        pretrained_MLP = pretrained_MLP, # pretrained_MLP weight 경로   
        device=device,
        visualize=visualize,
        queue_size=queue_size
    )
    
    
    
    rospy.spin()









