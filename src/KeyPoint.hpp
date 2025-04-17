#ifndef KeyPoint_h
#define KeyPoint_h
#include "Camera_input.hpp"
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


#define LOG_Flag 0
#define CONST_CAMERA 1
#define LOGLN(msg) std::cout << msg << std::endl
class Find_Feature {
	//input
private:
	Mat src;
	//output
public:
	detail::ImageFeatures feature;
	condition_variable cond;
	mutex mtx_feature;
	float work_scale;
	bool boot_flag = false;
private:
	Mat dst;
	//params
private:
	mutex mtx_dst;
	float scale = 0.3;
	int id = 0;

public:
	//input
	cv::Mat get_dst() { return dst.clone(); };
	//process
	void find(const cv::Mat& img);
	void draw_points();
	//output
	detail::ImageFeatures get_feature() {// 深拷贝 feature
		cv::detail::ImageFeatures feature_copy;
		feature_copy.keypoints = feature.keypoints;
		feature_copy.descriptors = feature.descriptors.clone();
		feature_copy.img_idx = feature.img_idx;
		feature_copy.img_size = feature.img_size;
		return feature_copy;
	};




};

class Match_Features {
public:
	//input
	vector<detail::ImageFeatures> features;
	//output
	mutex mtx_cameras;
	vector<detail::MatchesInfo> pairwise_matches;
	vector<detail::CameraParams>cameras;
	float warped_image_scale;
	bool boot_flag = false;
	condition_variable cond;
	//params
	bool try_cuda = false;
	float match_conf = 0.5f;		//越大，点匹配越严厉
	float conf_thresh = 0.5f;		//不要太高，否则非线性优化无法收敛
public:
	//input
	void update_features_imgs(vector<detail::ImageFeatures> features, const vector<cv::Mat>& imgs);		//拷贝用于匹配的特征和图片
	//process
	void match();		//匹配特征点
	void est_params();		//估计相机参数（K、R、T）
	void show_matches();		//匹配结果可视化
	//output
	void log_matchinfo();
	void copy_cameras(vector<detail::CameraParams>& cameras_dst, vector<detail::CameraParams>& cameras_src);		//更新相机参数
	void copy_camera(detail::CameraParams& camera_dst, detail::CameraParams& camera_src);
private:
	mutex mtx_match;
	vector<cv::Mat> imgs;
	vector<detail::CameraParams> cameras_temp = {
		detail::CameraParams(),
		detail::CameraParams()
	};
	const size_t max_size = 100;
	int id = 0;
};


//后续在考虑封装成队列
/*
class CameraParamsQueue {
public:
	explicit CameraParamsQueue(size_t max_size);

	void push(const std::vector<detail::CameraParams>& cameras);
	bool pop(std::vector<detail::CameraParams>& cameras);
	size_t size() const;
	bool empty() const;

private:
	std::vector<detail::CameraParams> deepCopyCameras(const std::vector<detail::CameraParams>& cameras);

	std::queue<std::vector<detail::CameraParams>> queue_;
	size_t max_size_;
	mutable std::mutex mutex_;
	std::condition_variable cond_;
};*/

#endif 