#ifndef ESTSEAM_H
#define ESTSEAM_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>
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
#include <filesystem>

using namespace std;
using namespace cv;
using namespace cv::detail;

class Est_seamm {
public:
	// 辅助变量
	mutex mtx_esS;
	condition_variable cond;
	bool bootflag_seam = false; //拼接缝boot标识
	int id = 0;

	// 输入
	vector<UMat>myimg{ UMat(),UMat() }; //读取warped_float图像
	vector<UMat>mymask{ UMat(),UMat() }; //读取warp后掩码 ---> find后存储拼接掩码
	vector<Point>corner; //读取角点
	vector<UMat>find_masks{ UMat(),UMat() }; //找到拼接缝以后的掩码

   // 函数
	void update_img(vector<UMat>images_warped_f_in, vector<Point>corners_in, vector<UMat>masks_warped_in); //读取信息
	void init_seamfinder(); //初始化搜索器并搜索拼接缝
	void save_left_mask(); //保存掩码
	void save_right_mask();
	void save_left_img();
	void save_right_img();
};

#endif // !ESTSEAM_H
