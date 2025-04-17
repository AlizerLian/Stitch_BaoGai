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
	// ��������
	mutex mtx_esS;
	condition_variable cond;
	bool bootflag_seam = false; //ƴ�ӷ�boot��ʶ

	// ����
	vector<UMat>myimg{ UMat(),UMat() }; //��ȡwarped_floatͼ��
	vector<UMat>mymask{ UMat(),UMat() }; //��ȡwarp������ ---> find��洢ƴ������
	vector<Point>corner; //��ȡ�ǵ�

   // ����
	void update_img(vector<UMat>images_warped_f_in, vector<Point>corners_in, vector<UMat>masks_warped_in); //��ȡ��Ϣ
	void init_seamfinder(); //��ʼ��������������ƴ�ӷ�
};

#endif // !ESTSEAM_H
