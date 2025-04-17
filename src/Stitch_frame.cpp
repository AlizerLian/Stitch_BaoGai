#include "Stitch_frame.hpp"

void graph_blender::update_source(vector<Mat>img_warped_s_in, vector<Mat>mask_warped_in, vector<Point>corner_in, vector<Size>size_in) {
	for (uint8_t i = 0;i < 2;++i) {
		images_warped[i] = img_warped_s_in[i].clone();
		masks_warped[i] = mask_warped_in[i].clone();
		corners[i] = corner_in[i];
		//LOGLN(corners[i]);
		//LOGLN(images_warped[i].size());
		sizes[i] = size_in[i];
	}
}

void graph_blender::graph_blend() {
	try_blend();
	id++;
	save_img();
#if 0 //imshow	
	Mat show;
	result.convertTo(show, CV_8U);
	imshow("img", show);
	waitKey(20);
#endif

}

void graph_blender::try_blend() {
	Ptr<Blender>blender;
	blender = Blender::createDefault(blend_type, try_cuda);
	Size dst_sz = resultRoi(corners, sizes).size();
	float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;
	if (blend_width < 1.f) {
		blender = Blender::createDefault(Blender::NO, try_cuda);
	}
	else if(blend_type == Blender::MULTI_BAND){
		MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(blender.get());
		if (mb) {
			mb->setNumBands(static_cast<int>(ceil(log(blend_width) / log(2.)) - 1.));
		}
		else {
			throw std::runtime_error("Blender is not of type MultiBandBlender");
		}
	}
	else if (blend_type == Blender::FEATHER) {
		FeatherBlender* fb = dynamic_cast<FeatherBlender*>(blender.get());
		fb->setSharpness(1.f / blend_width);
	}
	blender->prepare(corners, sizes);
	for (uint8_t j = 0;j < 2;++j) {
		if (j < images_warped.size() && j < masks_warped.size() && j < corners.size()) {
			if (images_warped[j].empty() || masks_warped[j].empty()) {
				cout << "cuo wu,shu ru de tu xiang shi kong de" << endl;
			}
			try {
				blender->feed(images_warped[j], masks_warped[j], corners[j]);
			}
			catch (const cv::Exception& e) {
				std::cerr << "OpenCV Exception caught: " << e.what() << std::endl;
			}
		}
		else {
			throw std::out_of_range("Index out of range for images_warped, masks_warped, or corners");
		}
	}
	blender->blend(result, result_mask);
	// crop ²ÎÊý¹Ì¶¨
	int left_border = 10;    // ×ó±ßºÚ±ß¿í¶È
	int right_border = 20;   // ÓÒ±ßºÚ±ß¿í¶È
	int top_border = 75;     // ÉÏ±ßºÚ±ß¿í¶È
	int bottom_border = 80;  // ÏÂ±ßºÚ±ß¿í¶È
	int height = result.rows;
	int width = result.cols;
	// ÉèÖÃ²Ã¼ôÇøÓò
	cv::Rect cropRect(left_border, top_border, width - left_border - right_border, height - top_border - bottom_border);
	// ²Ã¼ôÍ¼Ïñ
	result = result(cropRect);
}

void graph_blender::draw_result() {
	string outpath = "result.jpg";
	imwrite(outpath, result);
}

void graph_blender::save_img() {
	std::string base_dir = "../out_lab";
	std::filesystem::create_directories(base_dir);
	std::stringstream ss;
	ss << base_dir <<"/" << id << ".jpg";
	std::string filename = ss.str();
	cv::imwrite(filename, result);
}