#ifndef ESTSEAM_H
#define ESTSEAM_H

#include <opencv2/opencv.hpp>
#include <opencv2/stitching/detail/seam_finders.hpp>
#include <vector>
#include <iostream>

using namespace std;
using namespace cv;
using namespace cv::detail;

class Est_seamm {
public:
	// 辅助变量
	mutex mtx_esS;
	condition_variable cond;
	bool bootflag_seam = false; //拼接缝boot标识

	// 输入
	vector<UMat>myimg{ UMat(),UMat() }; //读取warped_float图像
	vector<UMat>mymask{ UMat(),UMat() }; //读取warp后掩码 ---> find后存储拼接掩码
	vector<Point>corner; //读取角点

   // 函数
	void update_img(vector<UMat>images_warped_f_in, vector<Point>corners_in, vector<UMat>masks_warped_in); //读取信息
	void init_seamfinder(); //初始化搜索器并搜索拼接缝
};

#endif // !ESTSEAM_H
