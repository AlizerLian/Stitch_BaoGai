#ifndef BLENDIMG_H
#define BLENDIMG_H

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>
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

class graph_blender {
public:
	// 辅助变量
	bool try_cuda = false;

	// 输入
	vector<Mat>images_warped{ Mat(),Mat() }; //融合图像
	vector<Mat>masks_warped{ Mat(),Mat() }; //融合掩码
	vector<Point>corners{ Point(),Point() }; //图像角点
	vector<Size>sizes{ Size(),Size() }; //图像大小

	// 输出
	Mat result, result_mask; //融合后图像及掩码
	int id = 0;
	// 函数
	void update_source(vector<Mat>img_warped_s_in, vector<Mat>mask_warped_in, vector<Point>corner_in, vector<Size>size_in); //更新source
	void graph_blend(); //融合并保存
	void save_img();
private:
	// 中间变量
	int n = 0; //计数
	int blend_type = Blender::MULTI_BAND; //融合器类型
	float blend_strength = 5; //detail移植参数，初始化融合器

	// 函数
	void try_blend(); //进行融合
	void draw_result(); //保存结果
};

#endif // !BLENDIMG_H