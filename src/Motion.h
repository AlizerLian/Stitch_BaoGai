#pragma once
#ifndef __Motion_h__
#define __Motion_h__
#include <vector>
#include <filesystem>
#include <opencv2/xfeatures2d.hpp>
#include <condition_variable>
#include "opencv2/opencv_modules.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/timelapsers.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"
using namespace std;
using namespace cv;
class Motion
{
public:
	string yaml_path = "../stereo_calib1.yaml";
	Mat camera_matrix_left, dist_left;                               //左相机内参，畸变
	Mat camera_matrix_right, dist_right;                             //右相机内参，畸变
	Mat R, T;														 //旋转，平移矩阵
	string video_address1, video_address2;                           //视频地址
	int camera_num;													 //相机数量
	string SeamFinder_Type, Exposure_Type, Warper_Type,Blender_Type; //拼接缝查找器，曝光器，投影器，融合器类型	
	bool Is_Thread,Seam_Upgrade,Expourse_Upgrade,Is_Log;             //是否 多线程，拼接缝更新，曝光器更新，打印调试信息
	int Seam_Num, Exposure_Num;										 //拼接缝，曝光器更新帧数间隔
	int Seam_time, Exposure_time;                                    //拼接缝，曝光器更新时间间隔 
	Motion();
};
#endif