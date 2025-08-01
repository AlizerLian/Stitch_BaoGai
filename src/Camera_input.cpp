#include "Camera_input.hpp"

void Video_Capture::org_push(const cv::Mat& frame) {
    //unique_lock<mutex> lock(mtx_org);
    mtx_org.lock();
    //×èÈû
    //cond.wait(lock, [this] { return org_queue.size() < max_size; });
    //·Ç×èÈû
    if (org_queue.size() >= org_max_size) {
        org_queue.pop();
    }
    org_queue.push(frame.clone());  // Éî¿½±´·ÀÖ¹Êý¾Ý¾ºÕù
    mtx_org.unlock();
    cond.notify_one();
}

bool Video_Capture::org_pop(Mat& frame) {
    unique_lock<mutex> lock(mtx_org);
    cond.wait_for(lock, 1s, [this] { return !org_queue.empty() || exit_flag; });

    if (org_queue.empty() || exit_flag)
        return false;

    frame = org_queue.front();
    org_queue.pop();
    cond.notify_one();
    return true;
}

void Video_Capture::rwc_push(const cv::Mat& frame) {
    unique_lock<mutex> lock(mtx_rwc);
    //×èÈû
    //cond.wait(lock, [this] { return rwc_queue.size() < max_size; });
    //·Ç×èÈû
    if (rwc_queue.size() >= rwc_max_size) {
        rwc_queue.pop();
    }
    rwc_queue.push(frame.clone());  // Éî¿½±´·ÀÖ¹Êý¾Ý¾ºÕù
    cond.notify_one();
}

bool Video_Capture::rwc_pop(Mat& frame) {
    unique_lock<mutex> lock(mtx_rwc);
    cond.wait_for(lock, 1s, [this] { return !rwc_queue.empty() || exit_flag; });

    if (rwc_queue.empty() || exit_flag)
        return false;

    frame = rwc_queue.front();
    rwc_queue.pop();
    cond.notify_one();
    return true;
}

void Video_Capture::rwc_org_push(const rwc_org& frame) {
    unique_lock<mutex> lock(mtx_rwc);
    //×èÈû
    //cond.wait(lock, [this] { return rwc_queue.size() < max_size; });
    //·Ç×èÈû
    if (rwc_org_queue.size() >= rwc_max_size) {
        rwc_queue.pop();
    }
    rwc_org_queue.push(ro_clone(frame));  // Éî¿½±´·ÀÖ¹Êý¾Ý¾ºÕù
    cond.notify_one();
}

bool Video_Capture::rwc_org_pop(rwc_org& frame) {
    unique_lock<mutex> lock(mtx_rwc);
    cond.wait_for(lock, 1s, [this] { return !rwc_org_queue.empty() || exit_flag; });

    if (rwc_org_queue.empty() || exit_flag)
        return false;

    frame = ro_clone(rwc_org_queue.front());  // Éî¿½±´
    rwc_org_queue.pop();
    cond.notify_one();
    return true;
}

void Video_Capture::warp(const Mat& frame, detail::CameraParams camera, float warped_image_scale) {
    try {
        /*
        LOGLN(camera.K());
        LOGLN(camera.R);
        LOGLN(warped_image_scale);
        */
        Mat mask;
        Mat img = frame.clone();
        resize(img, img, Size(0, 0), 0.5, 0.5);
        Size size = img.size();
        // Prepare images masks
        mask.create(img.size(), CV_8U);
        mask.setTo(Scalar::all(255));
        // Warp images and their masks
        Ptr<detail::RotationWarper> warper;
        warper = warper_creator->create(static_cast<float>(0.5 * warped_image_scale * seam_work_aspect));
        Mat_<float> K;
        camera.K().convertTo(K, CV_32F);
        float swa = static_cast<float>(seam_work_aspect);
        K(0, 0) *= swa;
        K(0, 2) *= swa;
        K(1, 1) *= swa;
        K(1, 2) *= swa;
        mtx_corner.lock();
        //Mat H = K * camera.R * K.inv();
        //LOGLN(H);
        corners = warper->warp(img, K, camera.R, INTER_LINEAR, BORDER_REFLECT, img_warped);
        warper->warp(mask, K, camera.R, INTER_NEAREST, BORDER_CONSTANT, mask_warped);
        //LOGLN(corners);
        mtx_corner.unlock();
        //LOGLN(img_warped.size());
        Mat img_warped_f;
        img_warped.convertTo(img_warped_f, CV_32F);
        rwc_org this_one;
        this_one.org = frame;
        this_one.rwc = img_warped_f;
        rwc_push(img_warped_f);
        rwc_org_push(this_one);
    }
    catch (const cv::Exception& e) {
        std::cerr << "-----OpenCV Error: " << e.what() << std::endl;
    }
    catch (...) {
        std::cerr << "------Unknown Error occurred." << std::endl;
    }
}