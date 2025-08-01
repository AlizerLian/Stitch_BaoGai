#include "KeyPoint.hpp"
#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
namespace fs = std::filesystem;
void Find_Feature::find(const cv::Mat& img) {
	mtx_dst.lock();
	img.copyTo(src);
	mtx_dst.unlock();
	mtx_feature.lock();
	work_scale = min(1.0, sqrt(1 * 1e6 / src.size().area()));
	resize(src, dst, Size(), work_scale, work_scale, INTER_LINEAR_EXACT);
	//resize(src, dst, cv::Size(), scale, scale);
	//cv::cvtColor(dst, dst, cv::COLOR_BGR2GRAY);

	// 特征检测器
	Ptr<Feature2D> finder = cv::SIFT::create();//SIFT ORB
	detail::computeImageFeatures(finder, dst, feature);
	boot_flag = true;
	mtx_feature.unlock();
	cond.notify_all();


#if LOG_Flag
	LOGLN("Features in image: " << feature.keypoints.size());
#endif
	//draw_points(dst1, dst2);
}


void Find_Feature::draw_points() {
	cv::Mat img_with_keypoints;

	cv::drawKeypoints(
		dst,
		feature.keypoints,
		img_with_keypoints,
		cv::Scalar::all(-1),
		cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS
	);

	fs::path output_dir = "out";
	string filename = output_dir.string() + "/" + to_string(id++) + ".jpg";
	if (!imwrite(filename, img_with_keypoints)) {
		cerr << "保存失败: " << filename << endl;
	}
	cout << "已保存: " << filename << endl;
}




void Match_Features::show_matches() {
	const cv::detail::MatchesInfo& matches_info = pairwise_matches[0];
	// 创建输出图像
	std::vector<cv::KeyPoint> keypoints1 = features[0].keypoints;
	std::vector<cv::KeyPoint> keypoints2 = features[1].keypoints;
	cv::Mat match_img;
	cv::drawMatches(imgs[0], keypoints1, imgs[1], keypoints2, matches_info.matches, match_img);
	imshow("匹配结果", match_img);
	waitKey(10);
}

void Match_Features::log_matchinfo() {
	int total_matches = 0;
	for (size_t i = 0; i < pairwise_matches.size(); ++i)
	{
		int match_count = pairwise_matches[i].num_inliers;
		total_matches += match_count;
		LOGLN("Match_conf:" << pairwise_matches[i].confidence);
	}

	LOGLN("Match_size:" << pairwise_matches.size());
	cout << "Total matches: " << total_matches << endl;
}
void Match_Features::match() {
	pairwise_matches.clear();
	Ptr<detail::FeaturesMatcher> matcher;
	matcher = makePtr<detail::BestOf2NearestMatcher>(try_cuda, match_conf);
	(*matcher)(features, pairwise_matches);
	matcher->collectGarbage();
	//log_matchinfo();

	//show_matches();
	//对match后处理



}

void Match_Features::est_params() {
#if 1 //设置内参矩阵

	Mat t1 = Mat::zeros(3, 1, CV_32F);
	cv::Mat t2 = (cv::Mat_<double>(1, 3) <<
		6.3859914308691131e+01, -3.7415922677768045e+00, -6.5973789741118081e+00);
	cv::Mat R1 = cv::Mat::eye(3, 3, CV_32F);
	cv::Mat R2 = (cv::Mat_<double>(3, 3) <<
		8.2510274226364999e-01, -4.5659780865110876e-02, 5.6313466339797869e-01,
		2.6196966646010574e-02, 9.9874885351293430e-01, 4.2596320793550899e-02,
		-5.6437503811521694e-01, -2.0393921102851750e-02, 8.2526656562258605e-01);
	R2.convertTo(R2, CV_32F);
	t2.convertTo(t2, CV_32F);
	cameras_temp[0].focal = 3.1889300079940074e+02f;
	cameras_temp[0].ppx = 3.4891769313814382e+02f;
	cameras_temp[0].ppy = 2.6863376773389831e+02f;
	cameras_temp[0].aspect = 3.0870658821125119e+02f / 3.1889300079940074e+02f;
	cameras_temp[0].t = t1.clone();
	cameras_temp[0].R = R1.clone();

	cameras_temp[1].aspect = 2.9892883000791056e+02 / 2.9708375475985832e+02;
	cameras_temp[1].focal = 2.9708375475985832e+02;
	cameras_temp[1].ppx = 2.7990386288660073e+02f;
	cameras_temp[1].ppy = 2.3938920186161772e+02f;
	cameras_temp[1].t = t2.clone();
	cameras_temp[1].R = R2.clone();
	/*
	Mat K1 = cameras_temp[0].K();
	Mat K2 = cameras_temp[1].K();
	K1.convertTo(K1, CV_32F);
	K2.convertTo(K2, CV_32F);
	Mat H = pairwise_matches[2].H;
	H.convertTo(H, CV_32F);
	Mat R2 = K1.inv() * H * K2;
	cameras_temp[1].R = R1 * R2;*/

	/*
		Ptr<detail::BundleAdjusterBase> adjuster;
		adjuster = makePtr<detail::BundleAdjusterRay>();
		adjuster->setConfThresh(conf_thresh);
		Mat_<uchar> refine_mask = Mat::zeros(3, 3, CV_8U);
		refine_mask(0, 0) = 1;
		refine_mask(0, 1) = 1;
		refine_mask(0, 2) = 1;
		refine_mask(1, 1) = 1;
		refine_mask(1, 2) = 1;
		adjuster->setRefinementMask(refine_mask);
		if (!(*adjuster)(features, pairwise_matches, cameras_temp))
		{
			cout << "Camera parameters adjusting failed.\n";
			return;
		}*/

		// waveCorrect 
	vector<Mat> rmats;
	for (size_t i = 0; i < cameras_temp.size(); ++i)
		rmats.push_back(cameras_temp[i].R.clone());
	detail::waveCorrect(rmats, detail::WAVE_CORRECT_HORIZ);
	for (size_t i = 0; i < cameras_temp.size(); ++i)
		cameras_temp[i].R = rmats[i];

	// update
	unique_lock<mutex> lock(mtx_cameras);
	copy_cameras(cameras, cameras_temp);
	warped_image_scale = cameras[0].focal * 0.5f + cameras[1].focal * 0.5f;
	boot_flag = true;
	lock.unlock();
	cond.notify_all();
	return;
#else
	if (features[0].keypoints.size() < 4 || features[1].keypoints.size() < 4) {
		LOGLN("特征点数不足，无法估计相机参数");
		return;
	}
	if (pairwise_matches.size() < 4) {
		LOGLN("匹配对数不足，无法估计相机参数");
		return;
	}
	std::vector<int> indices = leaveBiggestComponent(features, pairwise_matches, 0.40f);
	if (indices.size() != 2) {
		std::cout << "无法拼接，两张图片不属于同一个连通组件。" << std::endl;
		return;
	}

	//estimate camera params

	/**/Ptr<detail::Estimator> estimator;
	estimator = makePtr<detail::HomographyBasedEstimator>();

	if (!(*estimator)(features, pairwise_matches, cameras_temp)) {
		cout << "Homography estimation failed.\n";
		return;
	}
	for (size_t i = 0; i < cameras_temp.size(); ++i) {
		Mat R;
		cameras_temp[i].R.convertTo(R, CV_32F);
		cameras_temp[i].R = R;
		LOGLN(cameras_temp[i].K());
		//LOGLN("Initial camera intrinsics #" << i << ":-------\nK:\n" << cameras[i].K() << "\nR:\n" << cameras[i].R);
	}


	//非线性优化，注意收敛
	Ptr<detail::BundleAdjusterBase> adjuster;
	adjuster = makePtr<detail::BundleAdjusterRay>();
	adjuster->setConfThresh(conf_thresh);
	Mat_<uchar> refine_mask = Mat::zeros(3, 3, CV_8U);
	//
	refine_mask(0, 0) = 1;
	refine_mask(0, 1) = 1;
	refine_mask(0, 2) = 1;
	refine_mask(1, 1) = 1;
	refine_mask(1, 2) = 1;
	adjuster->setRefinementMask(refine_mask);
	if (!(*adjuster)(features, pairwise_matches, cameras_temp))
	{
		cout << "Camera parameters adjusting failed.\n";
		return;
	}
	/*
	LOGLN("R1" << cameras_temp[0].R);
	LOGLN("K1" << cameras_temp[0].K());
	LOGLN("R2" << cameras_temp[1].R);
	LOGLN("K2" << cameras_temp[1].K());
	*/

	// waveCorrect 
	vector<Mat> rmats;
	for (size_t i = 0; i < cameras_temp.size(); ++i)
		rmats.push_back(cameras_temp[i].R.clone());
	detail::waveCorrect(rmats, detail::WAVE_CORRECT_HORIZ);
	for (size_t i = 0; i < cameras_temp.size(); ++i)
		cameras_temp[i].R = rmats[i];


	// update
	unique_lock<mutex> lock(mtx_cameras);
	copy_cameras(cameras, cameras_temp);
	warped_image_scale = cameras[0].focal * 0.5f + cameras[1].focal * 0.5f;
	boot_flag = true;
	id++;
	lock.unlock();
	cond.notify_all();
#endif

}

void Match_Features::update_features_imgs(vector<detail::ImageFeatures> org_features, const vector<cv::Mat>& org_mgs) {
	features.clear();
	imgs.clear();

	for (auto& feature : org_features) {
		detail::ImageFeatures feature_copy;
		feature_copy.img_idx = feature.img_idx;
		feature_copy.img_size = feature.img_size;
		feature_copy.keypoints = feature.keypoints;
		feature_copy.descriptors = feature.descriptors.clone();
		features.push_back(feature_copy);
	}
	for (auto& img : org_mgs) {
		imgs.push_back(img.clone());
	}

}

void Match_Features::copy_cameras(vector<detail::CameraParams>& cameras_dst, vector<detail::CameraParams>& cameras_src) {
	cameras_dst.clear();
	for (auto& camera : cameras_src) {
		detail::CameraParams camera_tmp;
		camera_tmp.aspect = camera.aspect;
		camera_tmp.focal = camera.focal;
		camera_tmp.ppx = camera.ppx;
		camera_tmp.ppy = camera.ppy;
		camera_tmp.R = camera.R.clone();
		camera_tmp.t = camera.t.clone();
		cameras_dst.push_back(camera_tmp);
	}
}
void Match_Features::copy_camera(detail::CameraParams& camera_dst, detail::CameraParams& camera_src) {
	camera_dst.aspect = camera_src.aspect;
	camera_dst.focal = camera_src.focal;
	camera_dst.ppx = camera_src.ppx;
	camera_dst.ppy = camera_src.ppy;
	camera_dst.R = camera_src.R.clone();
	camera_dst.t = camera_src.t.clone();
}