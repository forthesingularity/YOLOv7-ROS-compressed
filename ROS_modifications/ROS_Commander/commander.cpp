#include "ros/ros.h"
#include "commander/KeyboardInput.h"
#include "commander/HumansStamped.h"
#define MSG msg->humans[0].skeleton_2d
#define CROSS (right_hand<left_hand)
#define ABOVE ((head_y<left_hand_y)&&(head_y<right_hand_y))
#define LEFT ((left_shoulder_y<left_hand_y) && (right_hand_y < right_elbow_y))
#define RIGHT  ((left_hand_y<left_elbow_y) && (right_elbow-right_shoulder>0.1))

int g_mem=0;	// 0:blank 1:above 2:back 3:left 4:right 

class SubscribeAndPublish
{
public:
  SubscribeAndPublish()
  { 
    sub_ = n_.subscribe("yolov7/yolov7_hpe_skeletons", 1, &SubscribeAndPublish::msgCallback, this);
    pub_ = n_.advertise<commander::KeyboardInput>("keyop/teleop", 1);
   
  }

void msgCallback(const commander::HumansStamped::ConstPtr& msg)
{
    h=MSG.keypoints[6].kp[0]-MSG.keypoints[7].kp[0];
    v=(MSG.keypoints[6].kp[1]-MSG.keypoints[12].kp[1])/2;

    left_hand=MSG.keypoints[11].kp[0]/h;
    right_hand=MSG.keypoints[10].kp[0]/h;
    left_elbow=MSG.keypoints[9].kp[0]/h;
    right_elbow=MSG.keypoints[8].kp[0]/h;
    right_shoulder=MSG.keypoints[6].kp[0]/h;

    left_hand_y=MSG.keypoints[11].kp[1]/v;
    right_hand_y=MSG.keypoints[10].kp[1]/v;
    left_elbow_y=MSG.keypoints[9].kp[1]/v;
    right_elbow_y=MSG.keypoints[8].kp[1]/v;
    left_shoulder_y=MSG.keypoints[7].kp[1]/v;
    right_shoulder_y=MSG.keypoints[6].kp[1]/v;
    head_y=MSG.keypoints[1].kp[1]/v;

    /*ROS_INFO(">>>>Published"); */

    commander::KeyboardInput move;
    if (ABOVE) 		{move.pressedKey=101; g_mem=0; /*ROS_INFO("[Enabled]"); */}	//enable
    else if (CROSS ) 	{move.pressedKey=100; g_mem=0; /*ROS_INFO("[Disabled]");*/}	//disable
    else if (LEFT)		{move.pressedKey=32; g_mem=0; /*ROS_INFO("[Reseted]");*/}	//reset
    else if (RIGHT){
      if ( abs(right_hand - right_elbow) > abs(right_hand_y - right_elbow_y)  )
        {
        if ( right_elbow < right_hand )		
        	{move.pressedKey=68; g_mem=4; /*ROS_INFO("[Right->LEFT]");*/}	//right 
        else 				
        	{move.pressedKey=67; g_mem=3; /*ROS_INFO("[Left->LEFT]");*/}	//left   
        }
     else
        {
        if ( right_elbow_y < right_hand_y )	
        	{move.pressedKey=65; g_mem=1;/* ROS_INFO("[Up]");*/}	//up 
        else 				
        	{move.pressedKey=66; g_mem=2;/* ROS_INFO("[Down]");*/}	//down
     }
    }
   
    pub_.publish(move); 
}

private: 
  ros::NodeHandle n_; 
  ros::Publisher pub_;
  ros::Subscriber sub_;
  float left_hand;
  float right_hand;
  float left_elbow;
  float right_elbow;
  float right_shoulder;
  float left_shoulder_y;
  float right_shoulder_y;
  float left_hand_y;
  float right_hand_y;
  float left_elbow_y;
  float right_elbow_y;
  float head_y;
  float h;
  float v;
};


int main(int argc, char **argv) 
{
ROS_INFO(">>>>Ready"); 
ros::init(argc, argv, "commander"); 
SubscribeAndPublish SAPObject;
ros::spin();
return 0;
}



