#pragma once
#ifndef __Motion_h__
#define __Motion_h__
#include <vector>
#include <filesystem>
#include <opencv2/features2d.hpp>
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
	Mat camera_matrix_left, dist_left;                               //������ڲΣ�����
	Mat camera_matrix_right, dist_right;                             //������ڲΣ�����
	Mat R, T;														 //��ת��ƽ�ƾ���
	string video_address1, video_address2;                           //��Ƶ��ַ
	int camera_num;													 //�������
	string SeamFinder_Type, Exposure_Type, Warper_Type,Blender_Type; //ƴ�ӷ���������ع�����ͶӰ�����ں�������	
	bool Is_Thread,Seam_Upgrade,Expourse_Upgrade,Is_Log;             //�Ƿ� ���̣߳�ƴ�ӷ���£��ع������£���ӡ������Ϣ
	int Seam_Num, Exposure_Num;										 //ƴ�ӷ죬�ع�������֡�����
	int Seam_time, Exposure_time;                                    //ƴ�ӷ죬�ع�������ʱ���� 
	Motion();
};
#endif