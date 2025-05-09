#ifndef ESTSEAM_H
#define ESTSEAM_H

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

class Est_seamm {
public:
	// ��������
	mutex mtx_esS;
	condition_variable cond;
	bool bootflag_seam = false; //ƴ�ӷ�boot��ʶ
	int id = 0;

	// ����
	vector<UMat>myimg{ UMat(),UMat() }; //��ȡwarped_floatͼ��
	vector<UMat>mymask{ UMat(),UMat() }; //��ȡwarp������ ---> find��洢ƴ������
	vector<Point>corner; //��ȡ�ǵ�
	vector<UMat>find_masks{ UMat(),UMat() }; //�ҵ�ƴ�ӷ��Ժ������

   // ����
	void update_img(vector<UMat>images_warped_f_in, vector<Point>corners_in, vector<UMat>masks_warped_in); //��ȡ��Ϣ
	void init_seamfinder(); //��ʼ��������������ƴ�ӷ�
	void save_left_mask(); //��������
	void save_right_mask();
	void save_left_img();
	void save_right_img();
};

#endif // !ESTSEAM_H
