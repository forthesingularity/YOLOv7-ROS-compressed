https://hub.docker.com/r/sarubito2020/noetic_cuda
docker pull sarubito2020/noetic_cuda
    CUDA 11.2.0
    cuDNN 8.1.1.33
    python 3.8
    ROS Noetic

#Append this code on your ~/.bashrc

export ROS_MASTER_URI=http://(Host IP):11311
export ROS_IP=172.24.193.18 #(Slave IP)
