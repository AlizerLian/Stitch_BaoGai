#include "Motion.h"
#include <opencv2/core/ocl.hpp>
using namespace cv;
using namespace std;

Motion::Motion()
{
	FileStorage fs(yaml_path, FileStorage::READ);
	fs["camera_matrix_left"]		>> camera_matrix_left;	
	fs["dist_left"]					>> dist_left;
	fs["camera_matrix_right"]		>> camera_matrix_right;
	fs["dist_right"]				>> dist_right;
	fs["R"]							>> R;
	fs["T"]							>> T;

	fs["video_address"]["address1"] >> video_address1;
	fs["video_address"]["address2"] >> video_address2;

	fs["camera_num"]				>> camera_num;

	fs["Is_Thread"]                 >> Is_Thread;
	fs["Seam_Upgrade"]				>> Seam_Upgrade;
	fs["Expourse_Upgrade"]			>> Expourse_Upgrade;

	fs["SeamFinder_Type"]			>> SeamFinder_Type;
	fs["Exposure_Type"]				>> Exposure_Type;
	fs["Warper_Type"]				>> Warper_Type;
	fs["Blender_Type"]				>> Blender_Type;

	fs["Seam_Num"]					>> Seam_Num;
	fs["Exposure_Num"]				>> Exposure_Num;
	fs["Seam_time"]					>> Seam_time;
	fs["Exposure_time"]				>> Exposure_time;

}