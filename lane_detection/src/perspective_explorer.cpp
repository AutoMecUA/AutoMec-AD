/*
/*
/* Based on work available at: https://gist.github.com/anujonthemove/7b35b7c1e05f01dd11d74d94784c1e58
/* by inaciose
/*
*/

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

cv::RNG rng(12345); // countour

#define PI 3.1415926

int frameWidth = 640;
int frameHeight = 480;

int alpha_ = 90, beta_ = 90, gamma_ = 90;
int f_ = 500, dist_ = 500;

void findBiggerShape(cv::Mat img, std::string winn) {
  // bof: countour
  // https://stackoverflow.com/questions/64130631/how-to-use-opencv-and-python-to-find-corners-of-a-trapezoid-similar-to-finding-c
  // reduce noise and keep image sharp
  cv::Mat imgsrc_filtered;
  cv::bilateralFilter(img, imgsrc_filtered, 5, 175, 175);
  
  //finds edges on image
  cv::Mat imgsrc_edges;
  cv::Canny(imgsrc_filtered, imgsrc_edges, 75, 200);

  //gets contours (outlines) for shapes and sorts from largest area to smallest area
  std::vector<std::vector<cv::Point> > contours;
  std::vector<cv::Vec4i> hierarchy; 
  cv::findContours(imgsrc_edges, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

  cv::Mat drawing = cv::Mat::zeros( imgsrc_filtered.size(), CV_8UC3 );

  /*
  for( size_t i = 0; i< contours.size(); i++ )
  {
      cv::Scalar color = cv::Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );
      cv::drawContours( drawing, contours, (int)i, color, 2, cv::LINE_8, hierarchy, 0 );
  }
  imshow( "Contours", drawing );
  */

  // find max closed perim 
  size_t idx_contour_selected = -1;
  double area, maxarea = 0;
  for( size_t i = 0; i< contours.size(); i++ )
  {
    area = contourArea( contours[i],false);
    if(area >= maxarea) { 
        maxarea = area;
        idx_contour_selected = (int)i;
    }
  }

  // show max area
  if(idx_contour_selected >= 0) {
    // set draw color
    cv::Scalar color = cv::Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );
    // create image and draw on it
    cv::Mat drawing = cv::Mat::zeros( imgsrc_filtered.size(), CV_8UC3 );
    cv::drawContours( drawing, contours, (int)idx_contour_selected, color, 1, cv::LINE_8, hierarchy, 0 );
    imshow( winn, drawing );
  
  
  
    double perim = cv::arcLength(contours[(int)idx_contour_selected], true);
    // setting the precision
    double epsilon = 0.02*perim;
    // approximating the contour with a polygon
    //std::vector<std::vector<cv::Point> > approxCorners;
    std::vector<std::vector<cv::Point> > approxCorners( contours.size() );
    cv::approxPolyDP(contours[(int)idx_contour_selected], approxCorners[0], epsilon, true);
    // check how many vertices has the approximate polygon
    //std::cout << approxCorners[0].size() << std::endl;
    if(approxCorners[0].size()>=4) {
      for (int j = 0; j < approxCorners[0].size() ; j++) {
        std::cout << approxCorners[0][j].x << ", " <<  approxCorners[0][j].y << std::endl;
      }
      std::cout << "----" << std::endl;
    }

  
  
  }
}


void imageSubCallback(const sensor_msgs::ImageConstPtr& msg)
{
  cv::Mat imgsrc;
  cv::Mat imgdst;
  try
  {
    imgsrc = cv_bridge::toCvShare(msg, "bgr8")->image;
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
  }
  
  cv::resize(imgsrc, imgsrc, cv::Size(frameWidth, frameHeight));

  double focalLength, dist, alpha, beta, gamma; 

  alpha =((double)alpha_ -90) * PI/180;
  beta =((double)beta_ -90) * PI/180;
  gamma =((double)gamma_ -90) * PI/180;
  focalLength = (double)f_;
  dist = (double)dist_;

  cv::Size image_size = imgsrc.size();
  double w = (double)image_size.width, h = (double)image_size.height;

  // Projecion matrix 2D -> 3D
  cv::Mat A1 = (cv::Mat_<float>(4, 3)<< 
    1, 0, -w/2,
    0, 1, -h/2,
    0, 0, 0,
    0, 0, 1 );

  // Rotation matrices Rx, Ry, Rz
  cv::Mat RX = (cv::Mat_<float>(4, 4) << 
    1, 0, 0, 0,
    0, cos(alpha), -sin(alpha), 0,
    0, sin(alpha), cos(alpha), 0,
    0, 0, 0, 1 );

  cv::Mat RY = (cv::Mat_<float>(4, 4) << 
    cos(beta), 0, -sin(beta), 0,
    0, 1, 0, 0,
    sin(beta), 0, cos(beta), 0,
    0, 0, 0, 1	);

  cv::Mat RZ = (cv::Mat_<float>(4, 4) << 
    cos(gamma), -sin(gamma), 0, 0,
    sin(gamma), cos(gamma), 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1	);

  // R - rotation matrix
  cv::Mat R = RX * RY * RZ;

  // T - translation matrix
  cv::Mat T = (cv::Mat_<float>(4, 4) << 
    1, 0, 0, 0,  
    0, 1, 0, 0,  
    0, 0, 1, dist,  
    0, 0, 0, 1); 
  
  // K - intrinsic matrix 
  cv::Mat K = (cv::Mat_<float>(3, 4) << 
    focalLength, 0, w/2, 0,
    0, focalLength, h/2, 0,
    0, 0, 1, 0
    ); 

  cv::Mat transformationMat = K * (T * (R * A1));

  cv::warpPerspective(imgsrc, imgdst, transformationMat, image_size, cv::INTER_CUBIC | cv::WARP_INVERSE_MAP);

  findBiggerShape(imgsrc, "ss");
  findBiggerShape(imgdst, "rs");

  cv::imshow("Source", imgsrc);
  cv::imshow("Result", imgdst);
  cv::waitKey(30);

}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "perspective_explorer");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

  std::string sub_topic;
  //pnh.param<std::string>("sub_topic", sub_topic, "usb_cam/image_raw"); 
  pnh.param<std::string>("sub_topic", sub_topic, "/ackermann_vehicle/camera/rgb/image_raw"); 

  pnh.param<int>("width", frameWidth, 640); 
  pnh.param<int>("height", frameHeight, 480); 

  cv::namedWindow("Result", 1);

	cv::createTrackbar("Pitch", "Result", &alpha_, 180);
	cv::createTrackbar("Roll", "Result", &beta_, 180);
	cv::createTrackbar("Yaw", "Result", &gamma_, 180);
	cv::createTrackbar("f", "Result", &f_, 2000);
	cv::createTrackbar("Distance", "Result", &dist_, 2000);

  image_transport::ImageTransport it(nh);
  image_transport::Subscriber sub = it.subscribe(sub_topic, 1, imageSubCallback);
  ros::spin();

  cv::destroyWindow("Result");
}