#ifndef EST_EXPOSURE_H
#define EST_EXPOSURE_H

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
#include <condition_variable>

using namespace cv;
using namespace cv::detail;
using namespace std;

class Est_exposure {
// 第一轮调用，初始化变量
public:
	// 辅助变量
	mutex mtx_esE;
	condition_variable cond;
	bool bootflag_exposure = false; //第一轮boot标识
	Ptr<ExposureCompensator> compensator; //曝光器
	Ptr<WarperCreator> warper_creator = makePtr<cv::SphericalWarper>(); //两轮都用到
	Ptr<detail::RotationWarper> warper; //两轮都用到，接收creater
	
	// 接收输入
	vector<detail::CameraParams>cams{
		detail::CameraParams(),
		detail::CameraParams()
	};
	float warped_imagescale; //接收cam->focal外部计算
	double work_scale; //接收feature线程，与图像同步

	// 输出
	vector<UMat>masks_warped{ UMat(),UMat() }; //接收seam_finder进行更新
	vector<UMat>masks_warped_f{ UMat(),UMat() }; //送入seam_finder
	vector<UMat>images_warped_f{ UMat(),UMat() }; //送入seam_finder---拼接缝搜索用图
	vector<Point>mycorner{ Point(),Point() }; //协助warp--->mask_warped --- 拼接缝搜索用角点
private:
	int id = 0;
	// 接收输入
	std::vector<cv::Mat>images{ Mat(),Mat() }; //读外部输入image

	// 中间变量
	double seam_work_aspect = 1;
	double seam_megapix = 0.1;
	double compose_megapix = 0.6;
	double compose_scale = 1;
	double compose_work_aspect = 1; //从detail前移植，数值未动
	Mat empty_mat; // 格式化compensator，目前未用到
	vector<Mat>masks_have{ Mat(),Mat() }; //接收输入Mat类型mask
	vector<UMat>masks{ UMat(),UMat() }; //生成初始掩码---曝光器feed掩码
	std::vector<cv::UMat>images_warped{ UMat(),UMat() }; //曝光器feed图像
// -----------------------------------------------------------------------------------
// 第二轮调用，应用曝光器
public:
	// 输出
	bool bootflag_exposure_done = false; //第二轮boot标识
	vector<Mat>img_warped_s{ Mat(),Mat() }; //blend使用，曝光后img
	vector<Mat>mask_warped_blend{ Mat(),Mat() }; //blend使用，配套变形后掩码
	vector<Size>sizes{ Size(),Size() }; //存储compose_scale
	vector<Point>corners; //存储compose角点
private:
	// 中间变量
	bool is_compose_scale_set = false; //一次调用变为true，确定compose_scale
	BlocksCompensator* bcompensator; //曝光器指针
	int expos_comp_type = ExposureCompensator::GAIN_BLOCKS; //曝光器类型
	int expos_comp_nr_feeds = 1;
	int expos_comp_nr_filtering = 2;
	int expos_comp_block_size = 32; //初始化曝光器参数，直接从detail移植（以上三个）
// -------------------------------------------------------------------------------------
// 函数部分
public:
	void init_all(); //创建并初始化曝光器
	void exposure_compensator_update(); //更新曝光器
	void get_cams(vector<detail::CameraParams>input_cams,float input_warped_scale); //读相机参数
	void get_images(const Mat& input_image1, const Mat& input_image2,double workscale); //读图像
	void get_rwc_images(vector<Mat>rwc_imagesf_in, vector<Mat>rwc_masks_in, vector<Point>Corners_in,vector<Mat>rwc_images_in); //读rwc图像
	void warp_compensate_img(); //应用曝光器，接收返回掩码后才能正常工作
	void update_seamed_warpedmask(vector<UMat>warpedmasks_in); //接收seam_finder返回掩码
	void exposure_compensator_update_withrwc();
private:
	void get_feed(); //获取曝光器初始化参数
	void init_compensator(); //初始化曝光器
	void fill_compensator(); //填充曝光器，生成增益矩阵
	void save_leftwarpedmask_in(); //保存接收掩码
	void save_rightwarpedmask_in();
};


#endif // !EST_EXPOSURE_H
