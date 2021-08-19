#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <ros/package.h>
#include "std_msgs/String.h"

#include <termio.h>

sensor_msgs::ImagePtr im_msg;
cv::Mat image[6];

int getch()
{
  static struct termios oldt, newt;
  tcgetattr( STDIN_FILENO, &oldt);           // save old settings
  newt = oldt;
  newt.c_lflag &= ~(ICANON);                 // disable buffering
  tcsetattr( STDIN_FILENO, TCSANOW, &newt);  // apply new settings

  int c = getchar();  // read character (non-blocking)

  tcsetattr( STDIN_FILENO, TCSANOW, &oldt);  // restore old settings
  return c;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "gazebo_signalling_panel_controller");
    ros::NodeHandle nh1;
    ros::NodeHandle nh2("~"); // private namespace

    // define from parameters
    std::string topic_monitor1;
    nh2.param<std::string>("topic_monitor1", topic_monitor1, "/monitor1/image1");
    
    std::string topic_monitor2;
    nh2.param<std::string>("topic_monitor2", topic_monitor2, "/monitor2/image2");

    std::string source_package;
    nh2.param<std::string>("source_package", source_package, "simspace");
    
    std::string source_folder;
    nh2.param<std::string>("source_folder", source_folder, "semaphores_pics");      

    std::string name_pic0;
    nh2.param<std::string>("name_pic0", name_pic0, "left");      

    std::string name_pic1;
    nh2.param<std::string>("name_pic1", name_pic1, "right"); 

    std::string name_pic2;
    nh2.param<std::string>("name_pic2", name_pic2, "up"); 

    std::string name_pic3;
    nh2.param<std::string>("name_pic3", name_pic3, "stop"); 

    std::string name_pic4;
    nh2.param<std::string>("name_pic4", name_pic4, "parking"); 

    std::string name_pic5;
    nh2.param<std::string>("name_pic5", name_pic5, "chess");

    int default_pic;
    nh2.param<int>("default_pic", default_pic, 3); 

    // define image transport and publishers 
    image_transport::ImageTransport it1(nh1);
    image_transport::Publisher pub1 = it1.advertise(topic_monitor1, 1);

    image_transport::ImageTransport it2(nh1);
    image_transport::Publisher pub2 = it2.advertise(topic_monitor2, 1);

    std::string path = ros::package::getPath(source_package);

    printf("%s",path.c_str());
    std::cout << "/" + source_folder + "/" << std::endl;

    // load images
    image[0] = cv::imread(path + "/" + source_folder + "/" + name_pic0 + ".png", cv::IMREAD_COLOR);
    image[1] = cv::imread(path + "/" + source_folder + "/" + name_pic1 + ".png", cv::IMREAD_COLOR);
    image[2] = cv::imread(path + "/" + source_folder + "/" + name_pic2 + ".png", cv::IMREAD_COLOR);
    image[3] = cv::imread(path + "/" + source_folder + "/" + name_pic3 + ".png", cv::IMREAD_COLOR);
    image[4] = cv::imread(path + "/" + source_folder + "/" + name_pic4 + ".png", cv::IMREAD_COLOR);
    image[5] = cv::imread(path + "/" + source_folder + "/" + name_pic5 + ".png", cv::IMREAD_COLOR);

    if (image[0].empty() || image[1].empty() || image[2].empty() || image[3].empty() || image[4].empty() || image[5].empty())
    {
        printf("Error on reading images.\n");
        return 0;
    }

    // publish default image
    im_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image[default_pic]).toImageMsg();

    // main loop
    ros::Rate loop_rate(10);

    while (nh1.ok()) {
        system("clear");
        printf("Choose the semaphore [Press <Esc> to exit]:\n");
        printf("[0] - "); printf("%s\n", name_pic0.c_str());
        printf("[1] - "); printf("%s\n", name_pic1.c_str());
        printf("[2] - "); printf("%s\n", name_pic2.c_str());
        printf("[3] - "); printf("%s\n", name_pic3.c_str());
        printf("[4] - "); printf("%s\n", name_pic4.c_str());
        printf("[5] - "); printf("%s\n", name_pic5.c_str());

        int opt = getch();
        switch (opt) {
        case '0':
            im_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image[0]).toImageMsg();
            break;
        case '1':
            im_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image[1]).toImageMsg();
            break;
        case '2':
            im_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image[2]).toImageMsg();
            break;
        case '3':
            im_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image[3]).toImageMsg();
            break;
        case '4':
            im_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image[4]).toImageMsg();
            break;
        case '5':
            im_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", image[5]).toImageMsg();
            break;
        case 27:
            exit(0);
            break;
        default:
            break;
        }

        pub1.publish(im_msg);
        pub2.publish(im_msg);
        ros::spinOnce();
        loop_rate.sleep();
    }
    printf("FINISH\n");
}
