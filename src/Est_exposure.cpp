#include "Est_exposure.hpp"

void Est_exposure::init_all() {
	compensator = ExposureCompensator::createDefault(expos_comp_type);
	init_compensator();
}

void Est_exposure::get_cams(vector<detail::CameraParams>input_cams, float input_warped_scale) {
	for (uint8_t i = 0;i < 2;++i) {
		cams[i].ppx = input_cams[i].ppx;
		cams[i].ppy = input_cams[i].ppy;
		cams[i].R = input_cams[i].R.clone();
		cams[i].t = input_cams[i].t.clone();
		cams[i].aspect = input_cams[i].aspect;
		cams[i].focal = input_cams[i].focal;
	}
	warped_imagescale = input_warped_scale;
}

void Est_exposure::update_seamed_warpedmask(vector<UMat>warpedmasks_in) {
	for (uint8_t i = 0;i < 2;++i) {
		masks_warped[i] = warpedmasks_in[i].clone();
	}
}

void Est_exposure::get_images(const Mat& input_image1, const Mat& input_image2, double workscale) {
	images.clear();
	std::vector<Mat> emptyGains;
	compensator->setMatGains(emptyGains);
	images.push_back(input_image1.clone());
	images.push_back(input_image2.clone());
	work_scale = workscale;
}

void Est_exposure::exposure_compensator_update() {
	get_feed();
	init_compensator();
	fill_compensator();
	bootflag_exposure = true;
	cond.notify_all();
}

void Est_exposure::exposure_compensator_update_withrwc() {
	init_compensator();
	fill_compensator();
	bootflag_exposure = true;
	cond.notify_all();
}

void Est_exposure::get_rwc_images(vector<Mat>rwc_imagesf_in, vector<Mat>rwc_masks_in, vector<Point>Corners_in, vector<Mat>rwc_images_in) {
	for (uint8_t i = 0;i < 2;++i) {
		images[i] = rwc_images_in[i].clone();
		mycorner[i] = Corners_in[i];
		images[i].copyTo(images_warped[i]);
		rwc_masks_in[i].copyTo(masks_warped_f[i]);
		masks_warped_f[i].convertTo(masks_warped_f[i], CV_8U);
		rwc_imagesf_in[i].copyTo(images_warped_f[i]);
	}
}

void Est_exposure::get_feed() {
	for (uint8_t i = 0;i < 2;++i) {
		masks[i].create(images[i].size(), CV_8U);
		masks[i].setTo(Scalar::all(255));
	}

	Ptr<detail::RotationWarper> warper;
	double seam_scale = min(1.0, sqrt(seam_megapix * 1e-6 / images[0].size().area()));
	warper = warper_creator->create(static_cast<float>(warped_imagescale * seam_work_aspect));
	for (uint8_t i = 0;i < 2;++i) {
		Mat_<float>K;
		cams[i].K().convertTo(K, CV_32F);
		float swa = (float)seam_work_aspect;
		K(0, 0) *= swa;K(0, 2) *= swa;
		K(1, 1) *= swa; K(1, 2) *= swa;
		mycorner[i] = warper->warp(images[i], K, cams[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
		sizes[i] = images_warped[i].size();
		warper->warp(masks[i], K, cams[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped_f[i]);
	}
	for (uint8_t i = 0;i < 2;++i) {
		images_warped[i].convertTo(images_warped_f[i], CV_32F);
	}
}

void Est_exposure::fill_compensator() {
	compensator->feed(mycorner, images_warped, masks_warped_f);
	bootflag_exposure = true;
}

void Est_exposure::init_compensator() {
	bcompensator = dynamic_cast<BlocksCompensator*>(compensator.get());
	bcompensator->setNrFeeds(expos_comp_nr_feeds);
	bcompensator->setNrGainsFilteringIterations(expos_comp_nr_filtering);
	bcompensator->setBlockSize(expos_comp_block_size, expos_comp_block_size);
}

void Est_exposure::warp_compensate_img() {
	Mat fullimg;Mat img;Mat img_warped;Mat mask_warped;Mat mask;
	Mat seam_mask;Mat dilate_mask;
	for (uint8_t i = 0;i < 2;++i) {
		fullimg = images[i].clone();
		if (!is_compose_scale_set) {
			corners.clear();
			sizes.clear();
			compose_scale = min(1.0, sqrt(compose_megapix * 1e6 / fullimg.size().area()));
			is_compose_scale_set = true;
			compose_work_aspect = compose_scale / work_scale;
			warped_imagescale *= static_cast<float>(compose_work_aspect);
			warper = warper_creator->create(warped_imagescale);
			for (uint8_t ii = 0;ii < 2;++ii) {
				cams[ii].focal *= compose_work_aspect;
				cams[ii].ppx *= compose_work_aspect;
				cams[ii].ppy *= compose_work_aspect;
				Size sz = images[ii].size();
				if (abs(compose_scale - 1) > 1e-1) {
					sz.width = cvRound(images[ii].size().width * compose_scale);
					sz.height = cvRound(images[ii].size().height * compose_scale);
				}
				Mat K;
				cams[ii].K().convertTo(K, CV_32F);
				Rect roi = warper->warpRoi(sz, K, cams[ii].R);
				corners.push_back(roi.tl());
				sizes.push_back(roi.size());
			}
		}
		if (abs(compose_scale - 1) > 1e-1) {
			if (compose_scale <= 0) {
				std::cout << "+++++++++++++++++++++++Invalid scale factor+++++++++++++++++++++++++" << std::endl;
			}
			if (fullimg.empty()) {
				std::cout << "++++++++++++++++++++++++Input image is empty+++++++++++++++++++++++++" << std::endl;
			}
			try {
				Size size = fullimg.size();
				size.height = size.height * compose_scale;
				size.width = size.width * compose_scale;
				resize(fullimg, img, size, INTER_LINEAR);
			}
			catch (const cv::Exception& e) {
				cout << compose_scale << endl;
				std::cerr << "OpenCV Exception caught: " << e.what() << std::endl;
			}
		}
		else {
			img = fullimg;
		}
		fullimg.release();
		Size img_size = img.size();
		Mat K;
		cams[i].K().convertTo(K, CV_32F);
		warper->warp(img, K, cams[i].R, INTER_LINEAR, BORDER_REFLECT, img_warped);
		mask.create(img_size, CV_8U);
		mask.setTo(Scalar::all(255));
		warper->warp(mask, K, cams[i].R, INTER_NEAREST, BORDER_CONSTANT, mask_warped);
		compensator->apply(i, corners[i], img_warped, mask_warped);
		img_warped.convertTo(img_warped_s[i], CV_16S);
		img_warped.release();
		img.release();
		mask.release();
		masks_warped[i].convertTo(masks_warped[i], CV_8U);
		dilate(masks_warped[i], dilate_mask, Mat());
		resize(dilate_mask, seam_mask, mask_warped.size(), 0, 0, INTER_LINEAR_EXACT);
		mask_warped_blend[i] = seam_mask & mask_warped;
	}
	bootflag_exposure_done = true;
	cond.notify_all();
}