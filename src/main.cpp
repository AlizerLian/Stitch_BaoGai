#include "Camera_input.hpp"
#include "KeyPoint.hpp"
#include "Est_exposure.hpp"
#include "Est_seam.hpp"
#include "Stitch_frame.hpp"
#include "local_stich/control_params.hpp"
#include <queue>
#include <thread>
#include <filesystem>
#include <atomic>
#include <chrono>
#include <opencv2/core/ocl.hpp>
#include <opencv2/highgui.hpp>

// yaml文件 ---
// define : 相机个数，成像大小
// warp类型
// 曝光器类型
// 融合器类型
// 相机参数
// 拼接缝搜索算法
// 相机参数计算更新策略
// 拼接缝更新策略
// ---

// 读视频流：队列不稳定，图像应用不全 --- 输入对齐

// 矫正 & 曝光 & 融合 & 主程序函数接口 --- 结构优化 --- done

// 拼接缝不稳定 --- 拼接缝更新方法
// ---> 可能方法：1，检测用户按键输入（cin>>）---额外线程检测输入，实时性  2，计时（不稳定---pass） 3，固定帧数更新（计时器---手动调）
// ---> 修改接口：1，曝光器调用：等待提取（无额外耗时）   2，拼接缝搜索：确定搜索时机（额外耗时）
// 
// 结构调整 --- 适应yaml参数 --- 多路视频流拼接 --- done : 类组 --- 函数接口配合（重载函数）

using namespace cv;
using namespace std;

#define CAMERA_NUM 2    // 相机数量
#define LOGLN(msg) std::cout << msg << std::endl    // 打印日志
#define TIME_LOG 0      // 打印时间
#define ORG_EXPOSE 0    // 使用原始图像
#define IsThread 0      // 是否使用多线程
#define videoHave 1     // 合成视频
#define use_new 1       // 使用新结构

bool exit_flag = false; //全局退出标志（当视频流读取为空）

Video_Capture img_queue1;
Video_Capture img_queue2;
Find_Feature feature1;
Find_Feature feature2;
Match_Features feature_match;
Est_exposure est_expose;
Est_seamm est_seam;
graph_blender blend_item;

///////////////////////////////////////thread1////////////////////////////////////////////////////
void video_input1(string path) {
    VideoCapture cap(path);
    if (!cap.isOpened()) {
        LOGLN("无法打开视频文件!");
        return;
    }
    //params
    Mat frame;
    int  wrap_flag;
    float wrap_scale;
    cv::detail::CameraParams cam_param;
    double t;
    int i = 0;
    while (cap.read(frame)) {
        if (exit_flag) { break; }
        i++;
        std::this_thread::sleep_for(std::chrono::milliseconds(15));
        t = getTickCount();
        //get org_img
        if (frame.empty()) continue;
        img_queue1.org_push(frame);
        //warp
        unique_lock<mutex> lk_cameras(feature_match.mtx_cameras);
        wrap_flag = feature_match.boot_flag;
        if (wrap_flag > 0) {
            wrap_scale = feature_match.warped_image_scale;
            feature_match.copy_camera(cam_param, feature_match.cameras[0]);
        }
        lk_cameras.unlock();
        if (wrap_flag > 0) {
            img_queue1.warp(frame, cam_param, wrap_scale);
#if 0
            Mat warp_img;
            img_queue1.rwc_pop(warp_img);
            warp_img.convertTo(warp_img, CV_8U);
            img_queue1.save_img(warp_img);
#endif
        }

#if TIME_LOG
        LOGLN("video_input1 : " << (getTickCount() - t) / getTickFrequency() << " s");
#endif
    }
    while (!img_queue1.org_rwc_empty()) { 

    }
    //exit_flag = true;
    exit_flag = true;
    img_queue1.signal_exit();
    LOGLN("----thread video 1 ---- 退出" << "total:" << i);
}
// 读取视频流结构优化 ------------------------------------
void video_input(Video_Capture* vid, Match_Features* match ,string path) {
    VideoCapture cap(path);
    if (!cap.isOpened()) {
        LOGLN("无法打开视频文件!");
        return;
    }
    //params
    Mat frame;
    int  wrap_flag;
    float wrap_scale;
    cv::detail::CameraParams cam_param;
    double t;
    int i = 0;
    while (cap.read(frame)) {
        if (exit_flag) { break; }
        i++;
        std::this_thread::sleep_for(std::chrono::milliseconds(15));
        t = getTickCount();
        //get org_img
        if (frame.empty()) continue;
        vid->org_push(frame);
        //warp
        unique_lock<mutex> lk_cameras(match->mtx_cameras);
        wrap_flag = match->boot_flag;
        if (wrap_flag > 0) {
            wrap_scale = match->warped_image_scale;
            match->copy_camera(cam_param, match->cameras[0]);
        }
        lk_cameras.unlock();
        if (wrap_flag > 0) {
            vid->warp(frame, cam_param, wrap_scale);
#if 0
            Mat warp_img;
            vid->rwc_pop(warp_img);
            warp_img.convertTo(warp_img, CV_8U);
            vid->save_img(warp_img);
#endif
        }

#if TIME_LOG
        LOGLN("video_input1 : " << (getTickCount() - t) / getTickFrequency() << " s");
#endif
    }
    while (!vid->org_rwc_empty()) {

    }
    //exit_flag = true;
    exit_flag = true;
    vid->signal_exit();
    LOGLN("----thread video 1 ---- 退出" << "total:" << i);

}
/////////////////////////////////////// thread2 ////////////////////////////////////////////////////
void video_input2(string path) {
    VideoCapture cap(path);
    if (!cap.isOpened()) {
        LOGLN("无法打开视频文件!");
        return;
    }
    double t;
    Mat frame;
    bool wrap_flag;
    float wrap_scale;
    cv::detail::CameraParams cam_param;

    while (cap.read(frame)) {
        if (exit_flag) { break; }
        std::this_thread::sleep_for(std::chrono::milliseconds(15));
        t = getTickCount();
        if (frame.empty()) continue;
        img_queue2.org_push(frame);
        //warp
        unique_lock<mutex> lk_cameras(feature_match.mtx_cameras);
        wrap_flag = feature_match.boot_flag;
        if (feature_match.boot_flag) {
            wrap_scale = feature_match.warped_image_scale;
            feature_match.copy_camera(cam_param, feature_match.cameras[1]);
        }
        lk_cameras.unlock();
        if (wrap_flag) {
            img_queue2.warp(frame, cam_param, wrap_scale);
#if 0
            Mat warp_img;
            img_queue2.rwc_pop(warp_img);
            warp_img.convertTo(warp_img, CV_8U);
            img_queue2.save_img(warp_img);
#endif
        }

#if TIME_LOG
        LOGLN("video_input2 : " << (getTickCount() - t) / getTickFrequency() << " s");
#endif
    }
    while (!img_queue2.org_rwc_empty()) {

    }
    exit_flag = true;
    img_queue2.signal_exit();
    LOGLN("----thread video 2 ---- 退出");
}
/////////////////////////////////////// thread3 ////////////////////////////////////////////////////
void find_feature_1() {
    Mat dst;
    double t;
    while (true) {
        if (exit_flag) { break; }
        if (img_queue1.org_pop(dst)) {
            t = getTickCount();
            feature1.find(dst);
#if TIME_LOG
            LOGLN("find_feature1 : " << (getTickCount() - t) / getTickFrequency() << " s");
#endif
        }
        else {
            LOGLN("----thread feature 1 ---- 退出");
            break;
        }
    }
}
// 特征点查找结构优化 ----------------------------------
void find_feature(Video_Capture* vid,Find_Feature* fea) {
    Mat dst;
    double t;
    while (true) {
        if (exit_flag) { break; }
        if (vid->org_pop(dst)) {
            t = getTickCount();
            fea->find(dst);
#if TIME_LOG
            LOGLN("find_feature1 : " << (getTickCount() - t) / getTickFrequency() << " s");
#endif
        }
        else {
            LOGLN("----thread feature 1 ---- 退出");
            break;
        }
    }
}
/////////////////////////////////////// thread4 ////////////////////////////////////////////////////
void find_feature_2() {
    Mat dst;
    double t;
    while (true) {
        if (exit_flag) { break; }
        if (img_queue2.org_pop(dst)) {
            t = getTickCount();
            feature2.find(dst);

#if TIME_LOG          
            LOGLN("find_feature2 : " << (getTickCount() - t) / getTickFrequency() << " s");
#endif
        }
        else {
            LOGLN("----thread feature 2 ---- 退出");
            break;
        }
    }
}
/////////////////////////////////////// thread5 ////////////////////////////////////////////////////
void match_feature() {
    vector<detail::ImageFeatures> features;
    vector<Mat> imgs;
    imgs.resize(CAMERA_NUM);
    features.resize(CAMERA_NUM);
    double t;
    LOGLN("等待特征点提取");
    unique_lock<mutex> lk_f1(feature1.mtx_feature);
    while (!feature1.boot_flag) {
        feature1.cond.wait(lk_f1);
    }
    lk_f1.unlock();
    unique_lock<mutex> lk_f2(feature2.mtx_feature);
    while (!feature2.boot_flag) {
        feature2.cond.wait(lk_f1);
    }
    lk_f2.unlock();
    LOGLN("开始匹配");
    while (1) {
        if (exit_flag) { break; }
        t = getTickCount();
        //copy
        feature1.mtx_feature.lock();
        feature2.mtx_feature.lock();
        //LOGLN("读取");
        imgs[0] = feature1.get_dst();
        imgs[1] = feature2.get_dst();
        features[0] = feature1.get_feature();
        features[1] = feature2.get_feature();
        feature1.mtx_feature.unlock();
        feature2.mtx_feature.unlock();
        //process
        feature_match.update_features_imgs(features, imgs);
        //LOGLN("匹配");
        feature_match.match();
        //LOGLN("匹配完成，开始估计相机参数");
        feature_match.est_params();
#if TIME_LOG   
        LOGLN("match_feature : " << (getTickCount() - t) / getTickFrequency() << " s");
#endif 

    }
    LOGLN("----thread match---- 退出");
}
// 特征点匹配结构优化 ------------------------------------------
void match_new(Find_Feature* fea1, Find_Feature* fea2, Match_Features* match) {
    vector<detail::ImageFeatures> features;
    vector<Mat> imgs;
    imgs.resize(CAMERA_NUM);
    features.resize(CAMERA_NUM);
    double t;
    LOGLN("等待特征点提取");
    unique_lock<mutex> lk_f1(fea1->mtx_feature);
    while (!fea1->boot_flag) {
        fea1->cond.wait(lk_f1);
    }
    lk_f1.unlock();
    unique_lock<mutex> lk_f2(fea2->mtx_feature);
    while (!fea2->boot_flag) {
        fea2->cond.wait(lk_f1);
    }
    lk_f2.unlock();
    LOGLN("开始匹配");
    while (1) {
        if (exit_flag) { break; }
        t = getTickCount();
        //copy
        fea1->mtx_feature.lock();
        fea2->mtx_feature.lock();
        //LOGLN("读取");
        imgs[0] = fea1->get_dst();
        imgs[1] = fea2->get_dst();
        features[0] = fea1->get_feature();
        features[1] = fea2->get_feature();
        fea1->mtx_feature.unlock();
        fea2->mtx_feature.unlock();
        //process
        match->update_features_imgs(features, imgs);
        //LOGLN("匹配");
        match->match();
        //LOGLN("匹配完成，开始估计相机参数");
        match->est_params();
#if TIME_LOG   
        LOGLN("match_feature : " << (getTickCount() - t) / getTickFrequency() << " s");
#endif 

    }
    LOGLN("----thread match---- 退出");
}
/////////////////////////////////////// thread6 ////////////////////////////////////////////////////
void stitch_frame() {
    // update -----------------------------------
    blend_union bu;
    // end --------------------------------------
    vector<Mat>images_blend_s{ Mat(),Mat() };
    vector<Mat>masks_blend_s{ Mat(),Mat() };
    LOGLN("等待特征点匹配");
    unique_lock<mutex> lk_cameras(feature_match.mtx_cameras);
    while (!feature_match.boot_flag) {
        feature_match.cond.wait(lk_cameras);
    }
    lk_cameras.unlock();
    LOGLN("等待曝光");
    unique_lock<mutex> lk_exposure(est_expose.mtx_esE);
    while (!est_expose.bootflag_exposure_done) {
        est_expose.cond.wait(lk_exposure);
    }
    lk_exposure.unlock();
    LOGLN("开始拼接");
    //std::this_thread::sleep_for(std::chrono::milliseconds(1000));   
    auto totol_start = chrono::high_resolution_clock::now();
    while (1) {
        if (exit_flag) { break; }
        if (!est_expose.corners_done_flag || !est_expose.graph_done_flag) { continue; }
        auto start = chrono::high_resolution_clock::now();
        /*
        // update ------------------------------------------
        images_blend_s[0] = bu.blend_img1.clone();
        images_blend_s[1] = bu.blend_img2.clone();
        masks_blend_s[0] = bu.blend_mask1.clone();
        masks_blend_s[1] = bu.blend_mask2.clone();
        // end ----------------------------------------------
        */
        est_expose.mtx_esE.lock();
        for (uint8_t i = 0;i < CAMERA_NUM;i++) {
            images_blend_s[i] = est_expose.use_imgs[i].clone();
            masks_blend_s[i] = est_expose.use_masks[i].clone();
        }
        blend_item.update_source(images_blend_s, masks_blend_s, est_expose.corners, est_expose.sizes);
        est_expose.mtx_esE.unlock();
        blend_item.graph_blend();

        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
        LOGLN("拼接处理时间: " << duration.count() << " ms");
    }
    auto total_end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(total_end - totol_start);
    double average_time = static_cast<double>(duration.count()) / blend_item.id;
    LOGLN("拼接平均时间: " << average_time << " ms");
    LOGLN("----thread blend---- 退出");
}
// 拼接融合结构优化 --------------------------------------------------
void stitch_main_new(Est_exposure* ese, Match_Features* match, graph_blender* blender) {
    // update -----------------------------------
    blend_union bu;
    // end --------------------------------------
    vector<Mat>images_blend_s{ Mat(),Mat() };
    vector<Mat>masks_blend_s{ Mat(),Mat() };
    vector<Point>blend_corners{ Point(), Point() };
    vector<Size>blend_sizes{ Size(), Size() };
    LOGLN("等待特征点匹配");
    unique_lock<mutex> lk_cameras(match->mtx_cameras);
    while (!match->boot_flag) {
        match->cond.wait(lk_cameras);
    }
    lk_cameras.unlock();
    LOGLN("等待曝光");
    unique_lock<mutex> lk_exposure(ese->mtx_esE);
    while (!ese->bootflag_exposure_done) {
        ese->cond.wait(lk_exposure);
    }
    lk_exposure.unlock();
    LOGLN("开始拼接");
    //std::this_thread::sleep_for(std::chrono::milliseconds(1000));   
    auto totol_start = chrono::high_resolution_clock::now();
    while (1) {
        if (exit_flag) { break; }
        if (!ese->corners_done_flag || !ese->graph_done_flag) { continue; }
        auto start = chrono::high_resolution_clock::now();
        /*
        // update ------------------------------------------
        images_blend_s[0] = bu.blend_img1.clone();
        images_blend_s[1] = bu.blend_img2.clone();
        masks_blend_s[0] = bu.blend_mask1.clone();
        masks_blend_s[1] = bu.blend_mask2.clone();
        // end ----------------------------------------------
        */
        ese->mtx_esE.lock();
        for (uint8_t i = 0;i < CAMERA_NUM;i++) {
            images_blend_s[i] = ese->use_imgs[i].clone();
            masks_blend_s[i] = ese->use_masks[i].clone();
            blend_corners[i] = ese->corners[i];
            blend_sizes[i] = ese->sizes[i];
        }
        blender->update_source(images_blend_s, masks_blend_s, blend_corners, blend_sizes);
        ese->mtx_esE.unlock();
        blender->graph_blend();

        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
        LOGLN("拼接处理时间: " << duration.count() << " ms");
    }
    auto total_end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(total_end - totol_start);
    double average_time = static_cast<double>(duration.count()) / blend_item.id;
    LOGLN("拼接平均时间: " << average_time << " ms");
    LOGLN("----thread blend---- 退出");
}
/////////////////////////////////////// thread7 ////////////////////////////////////////////////////
void Est_exp() {
    // update ------------------------------------
    rwc_org ro1, ro2;
    // end ---------------------------------------
    Mat dst1, dst2;
    vector<Point>corners;
    vector<Mat>warped_masks;
    vector<Mat>images_warped;
    vector<detail::CameraParams>cams = {
        detail::CameraParams(),
        detail::CameraParams()
    };
    float workscale;
    float warped_img_scale;
    est_expose.init_all();
    double t;
    LOGLN("等待相机参数");
    unique_lock<mutex> lk_cameras(feature_match.mtx_cameras);
    while (!feature_match.boot_flag) {
        feature_match.cond.wait(lk_cameras);
    }
    lk_cameras.unlock();
    LOGLN("相机参数获取，初始化曝光器");
#if ORG_EXPOSE
    img_queue1.org_pop(dst1);
    img_queue2.org_pop(dst2);
    feature1.mtx_feature.lock();
    workscale = feature1.work_scale;
    feature1.mtx_feature.unlock();
    img_queue1.mtx_corner.lock();
    warped_masks.push_back(img_queue1.mask_warped.clone());
    corners.push_back(img_queue1.corners);
    img_queue1.mtx_corner.unlock();
    img_queue2.mtx_corner.lock();
    warped_masks.push_back(img_queue2.mask_warped.clone());
    corners.push_back(img_queue2.corners);
    img_queue2.mtx_corner.unlock();
    feature_match.mtx_cameras.lock();
    feature_match.copy_cameras(cams, feature_match.cameras);
    warped_img_scale = feature_match.warped_image_scale;
    feature_match.mtx_cameras.unlock();
    est_expose.get_cams(cams, warped_img_scale);
    est_expose.get_images(dst1, dst2, workscale);
    est_expose.exposure_compensator_update(); // 更新曝光器
#else
    //img_queue1.rwc_pop(dst1);
    //img_queue2.rwc_pop(dst2);
    // update ------------------------
    img_queue1.rwc_org_pop(ro1);
    img_queue2.rwc_org_pop(ro2);
    // end ---------------------------
    feature1.mtx_feature.lock();
    workscale = feature1.work_scale;
    feature1.mtx_feature.unlock();
    img_queue1.mtx_corner.lock();
    warped_masks.push_back(img_queue1.mask_warped.clone());
    corners.push_back(img_queue1.corners);
    images_warped.push_back(img_queue1.img_warped.clone());
    img_queue1.mtx_corner.unlock();
    img_queue2.mtx_corner.lock();
    warped_masks.push_back(img_queue2.mask_warped.clone());
    corners.push_back(img_queue2.corners);
    images_warped.push_back(img_queue2.img_warped.clone());
    img_queue2.mtx_corner.unlock();
    feature_match.mtx_cameras.lock();
    feature_match.copy_cameras(cams, feature_match.cameras);
    warped_img_scale = feature_match.warped_image_scale;
    feature_match.mtx_cameras.unlock();
    est_expose.get_cams(cams, warped_img_scale);
    est_expose.get_images(ro1.org, ro2.org, workscale);
    est_expose.get_rwc_images(vector<Mat>{ro1.rwc, ro2.rwc}, warped_masks, corners, images_warped);
    warped_masks.pop_back();warped_masks.pop_back();
    corners.pop_back();corners.pop_back();
    images_warped.pop_back();images_warped.pop_back();
    est_expose.exposure_compensator_update_withrwc(); // 更新曝光器
#endif

    LOGLN("等待拼接缝搜索，更新掩码");
    unique_lock<mutex>lk_seam(est_seam.mtx_esS);
    while (!est_seam.bootflag_seam) {
        est_seam.cond.wait(lk_seam);
    }
    lk_seam.unlock();

    LOGLN("掩码更新完成，开始补偿");
    est_seam.mtx_esS.lock();
    est_expose.update_seamed_warpedmask(est_seam.find_mask);
    est_seam.mtx_esS.unlock();

    while (1) {
        if (exit_flag) { break; }

        t = getTickCount();
        // update --------------------------------
        //img_queue1.org_pop(dst1);
        //img_queue2.org_pop(dst2);
        img_queue1.rwc_org_pop(ro1);
        img_queue2.rwc_org_pop(ro2);
        // end -----------------------------------
        feature1.mtx_feature.lock();
        workscale = feature1.work_scale;
        feature1.mtx_feature.unlock();
        est_expose.get_images(ro1.org, ro2.org, workscale);
        est_expose.get_feed();
        est_expose.warp_compensate_img();
        if (est_seam.maskflag_seam) {
            est_seam.mtx_esS.lock();
            est_expose.update_seamed_warpedmask(est_seam.find_mask);
            est_seam.mtx_esS.unlock();
            // update------------------------------------
            //img_queue1.rwc_pop(dst1);
            //img_queue2.rwc_pop(dst2);
            // end --------------------------------------
            feature1.mtx_feature.lock();
            workscale = feature1.work_scale;
            feature1.mtx_feature.unlock();
            img_queue1.mtx_corner.lock();
            warped_masks.push_back(img_queue1.mask_warped.clone());
            corners.push_back(img_queue1.corners);
            images_warped.push_back(img_queue1.img_warped.clone());
            img_queue1.mtx_corner.unlock();
            img_queue2.mtx_corner.lock();
            warped_masks.push_back(img_queue2.mask_warped.clone());
            corners.push_back(img_queue2.corners);
            images_warped.push_back(img_queue2.img_warped.clone());
            img_queue2.mtx_corner.unlock();
            feature_match.mtx_cameras.lock();
            feature_match.copy_cameras(cams, feature_match.cameras);
            warped_img_scale = feature_match.warped_image_scale;
            feature_match.mtx_cameras.unlock();
            est_expose.get_cams(cams, warped_img_scale);
            est_expose.get_images(ro1.org, ro2.org, workscale);
            est_expose.get_rwc_images(vector<Mat>{ro1.rwc, ro2.rwc}, warped_masks, corners, images_warped);
            warped_masks.pop_back();warped_masks.pop_back();
            corners.pop_back();corners.pop_back();
            images_warped.pop_back();images_warped.pop_back();
            est_expose.exposure_compensator_update_withrwc(); // 更新曝光器
        }
        //std::cout << "完成曝光补偿" << endl;
#if TIME_LOG   
        LOGLN("曝光处理时间 : " << (getTickCount() - t) / getTickFrequency() << " s");
#endif 
    }
    LOGLN("----thread est_exp---- 退出");

}
// 曝光结构优化 -----------------------------------------
void expose_new(Video_Capture* vid1, Video_Capture* vid2, Find_Feature* fea1 ,Match_Features* match, Est_exposure* ese, Est_seamm* esm) {
    // update ------------------------------------
    rwc_org ro1, ro2;
    // end ---------------------------------------
    Mat dst1, dst2;
    vector<Point>corners;
    vector<Mat>warped_masks;
    vector<Mat>images_warped;
    vector<detail::CameraParams>cams = {
        detail::CameraParams(),
        detail::CameraParams()
    };
    float workscale;
    float warped_img_scale;
    ese->init_all();
    double t;
    LOGLN("等待相机参数");
    unique_lock<mutex> lk_cameras(match->mtx_cameras);
    while (!match->boot_flag) {
        match->cond.wait(lk_cameras);
    }
    lk_cameras.unlock();
    LOGLN("相机参数获取，初始化曝光器");
#if ORG_EXPOSE
    img_queue1.org_pop(dst1);
    img_queue2.org_pop(dst2);
    feature1.mtx_feature.lock();
    workscale = feature1.work_scale;
    feature1.mtx_feature.unlock();
    img_queue1.mtx_corner.lock();
    warped_masks.push_back(img_queue1.mask_warped.clone());
    corners.push_back(img_queue1.corners);
    img_queue1.mtx_corner.unlock();
    img_queue2.mtx_corner.lock();
    warped_masks.push_back(img_queue2.mask_warped.clone());
    corners.push_back(img_queue2.corners);
    img_queue2.mtx_corner.unlock();
    feature_match.mtx_cameras.lock();
    feature_match.copy_cameras(cams, feature_match.cameras);
    warped_img_scale = feature_match.warped_image_scale;
    feature_match.mtx_cameras.unlock();
    est_expose.get_cams(cams, warped_img_scale);
    est_expose.get_images(dst1, dst2, workscale);
    est_expose.exposure_compensator_update(); // 更新曝光器
#else
    //img_queue1.rwc_pop(dst1);
    //img_queue2.rwc_pop(dst2);
    // update ------------------------
    vid1->rwc_org_pop(ro1);
    vid2->rwc_org_pop(ro2);
    // end ---------------------------
    fea1->mtx_feature.lock();
    workscale = fea1->work_scale;
    fea1->mtx_feature.unlock();
    vid1->mtx_corner.lock();
    warped_masks.push_back(vid1->mask_warped.clone());
    corners.push_back(vid1->corners);
    images_warped.push_back(vid1->img_warped.clone());
    vid1->mtx_corner.unlock();
    vid2->mtx_corner.lock();
    warped_masks.push_back(vid2->mask_warped.clone());
    corners.push_back(vid2->corners);
    images_warped.push_back(vid2->img_warped.clone());
    vid2->mtx_corner.unlock();
    match->mtx_cameras.lock();
    match->copy_cameras(cams, match->cameras);
    warped_img_scale = match->warped_image_scale;
    match->mtx_cameras.unlock();
    ese->get_cams(cams, warped_img_scale);
    ese->get_images(ro1.org, ro2.org, workscale);
    ese->get_rwc_images(vector<Mat>{ro1.rwc, ro2.rwc}, warped_masks, corners, images_warped);
    warped_masks.pop_back();warped_masks.pop_back();
    corners.pop_back();corners.pop_back();
    images_warped.pop_back();images_warped.pop_back();
    ese->exposure_compensator_update_withrwc(); // 更新曝光器
#endif
    LOGLN("等待拼接缝搜索，更新掩码");
    unique_lock<mutex>lk_seam(esm->mtx_esS);
    while (!esm->bootflag_seam) {
        esm->cond.wait(lk_seam);
    }
    lk_seam.unlock();

    LOGLN("掩码更新完成，开始补偿");
    esm->mtx_esS.lock();
    ese->update_seamed_warpedmask(esm->find_mask);
    esm->mtx_esS.unlock();

    while (1) {
        if (exit_flag) { break; }
        t = getTickCount();
        // update --------------------------------
        //img_queue1.org_pop(dst1);
        //img_queue2.org_pop(dst2);
        vid1->rwc_org_pop(ro1);
        vid2->rwc_org_pop(ro2);
        // end -----------------------------------
        fea1->mtx_feature.lock();
        workscale = fea1->work_scale;
        fea1->mtx_feature.unlock();
        ese->get_images(ro1.org, ro2.org, workscale);
        ese->get_feed();
        ese->warp_compensate_img();
        if (esm->maskflag_seam) {
            esm->mtx_esS.lock();
            ese->update_seamed_warpedmask(esm->find_mask);
            esm->mtx_esS.unlock();
            // update------------------------------------
            //img_queue1.rwc_pop(dst1);
            //img_queue2.rwc_pop(dst2);
            // end --------------------------------------
            fea1->mtx_feature.lock();
            workscale = fea1->work_scale;
            fea1->mtx_feature.unlock();
            vid1->mtx_corner.lock();
            warped_masks.push_back(vid1->mask_warped.clone());
            corners.push_back(vid1->corners);
            images_warped.push_back(vid1->img_warped.clone());
            vid1->mtx_corner.unlock();
            vid2->mtx_corner.lock();
            warped_masks.push_back(vid2->mask_warped.clone());
            corners.push_back(vid2->corners);
            images_warped.push_back(vid2->img_warped.clone());
            vid2->mtx_corner.unlock();
            match->mtx_cameras.lock();
            match->copy_cameras(cams, match->cameras);
            warped_img_scale = match->warped_image_scale;
            match->mtx_cameras.unlock();
            ese->get_cams(cams, warped_img_scale);
            ese->get_images(ro1.org, ro2.org, workscale);
            ese->get_rwc_images(vector<Mat>{ro1.rwc, ro2.rwc}, warped_masks, corners, images_warped);
            warped_masks.pop_back();warped_masks.pop_back();
            corners.pop_back();corners.pop_back();
            images_warped.pop_back();images_warped.pop_back();
            ese->exposure_compensator_update_withrwc(); // 更新曝光器
        }
        //std::cout << "完成曝光补偿" << endl;
#if TIME_LOG   
        LOGLN("曝光处理时间 : " << (getTickCount() - t) / getTickFrequency() << " s");
#endif 
    }
    LOGLN("----thread est_exp---- 退出");
}
/////////////////////////////////////// thread8 ////////////////////////////////////////////////////
void Est_seam() {
    vector<Point>corners;
    vector<UMat> masks_warped_seam{ UMat(),UMat() };
    vector<UMat> images_warped_seam{ UMat(),UMat() };
    Mat dst1, dst2;
    double t;
    LOGLN("等待相机参数");
    unique_lock<mutex> lk_cameras(feature_match.mtx_cameras);
    while (!feature_match.boot_flag) {
        feature_match.cond.wait(lk_cameras);
    }
    lk_cameras.unlock();
    //////////////////////////////////////////////////////////////////
    unique_lock<mutex> lk_exposure(est_expose.mtx_esE);
    while (!est_expose.bootflag_exposure) {
        est_expose.cond.wait(lk_exposure);
    }
    lk_exposure.unlock();
    //////////////////////////////////////////////////////////////////
    LOGLN("开始搜索拼接缝");
    est_expose.mtx_esE.lock();
    for (uint8_t i = 0;i < CAMERA_NUM;++i) {
        masks_warped_seam[i] = est_expose.masks_warped_f[i].clone();
        images_warped_seam[i] = est_expose.images_warped_f[i].clone();
    }
    corners = est_expose.mycorner;
    est_expose.mtx_esE.unlock();
    est_seam.update_img(images_warped_seam, corners, masks_warped_seam);
    est_seam.init_seamfinder();
    while (1) {
        if (exit_flag) { break; }
        est_expose.mtx_esE.lock();
        for (uint8_t i = 0; i < CAMERA_NUM; ++i) {
            masks_warped_seam[i] = est_expose.masks_warped_f[i].clone();
            images_warped_seam[i] = est_expose.images_warped_f[i].clone();
        }
        corners = est_expose.mycorner;
        est_expose.mtx_esE.unlock();
        est_seam.update_img(images_warped_seam, corners, masks_warped_seam);
        est_seam.init_seamfinder();
        //std::cout << "拼接缝更新" << endl;

    }
    LOGLN("----thread est_seam---- 退出");
}
// 拼接缝搜索结构优化 --------------------------------------------
void seam_finder_new(Match_Features* match, Est_exposure* ese, Est_seamm* esm) {
    vector<Point>corners;
    vector<UMat> masks_warped_seam{ UMat(),UMat() };
    vector<UMat> images_warped_seam{ UMat(),UMat() };
    Mat dst1, dst2;
    double t;
    LOGLN("等待相机参数");
    unique_lock<mutex> lk_cameras(match->mtx_cameras);
    while (!match->boot_flag) {
        match->cond.wait(lk_cameras);
    }
    lk_cameras.unlock();
    //////////////////////////////////////////////////////////////////
    unique_lock<mutex> lk_exposure(ese->mtx_esE);
    while (!ese->bootflag_exposure) {
        ese->cond.wait(lk_exposure);
    }
    lk_exposure.unlock();
    //////////////////////////////////////////////////////////////////
    LOGLN("开始搜索拼接缝");
    ese->mtx_esE.lock();
    for (uint8_t i = 0;i < CAMERA_NUM;++i) {
        masks_warped_seam[i] = ese->masks_warped_f[i].clone();
        images_warped_seam[i] = ese->images_warped_f[i].clone();
    }
    corners = ese->mycorner;
    ese->mtx_esE.unlock();
    esm->update_img(images_warped_seam, corners, masks_warped_seam);
    esm->init_seamfinder();
    while (1) {
        if (exit_flag) { break; }
        ese->mtx_esE.lock();
        for (uint8_t i = 0; i < CAMERA_NUM; ++i) {
            masks_warped_seam[i] = ese->masks_warped_f[i].clone();
            images_warped_seam[i] = ese->images_warped_f[i].clone();
        }
        corners = ese->mycorner;
        ese->mtx_esE.unlock();
        esm->update_img(images_warped_seam, corners, masks_warped_seam);
        esm->init_seamfinder();
        //std::cout << "拼接缝更新" << endl;

    }
    LOGLN("----thread est_seam---- 退出");
}
/////////////////////////////////////// 合成视频函数 /////////////////////////////////////////////////
void generate_video() {
    VideoWriter writer;
    int frame_fps = 30;
    int frame_width = 720;
    int frame_height = 480;

    writer = VideoWriter("sample.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
        frame_fps, Size(frame_width, frame_height), true);
    std::cout << "frame_width is " << frame_width << endl;
    std::cout << "frame_height is " << frame_height << endl;
    std::cout << "frame_fps is " << frame_fps << endl;
    Mat img;
    for (int i = 1; i < 2000; i++) // 需要合成的图像编号
    {
        string image_name = "../out_lab/" + to_string(i) + ".jpg";

        img = imread(image_name);
        
        if (img.empty()) {
            cerr << "Failed to read image: " << image_name << endl;
            continue;
        }
        
        if (img.cols != frame_width || img.rows != frame_height) {
            resize(img, img, Size(frame_width, frame_height));
        }
        //cout << i << endl;
        
        if (!img.empty())
        {
            writer << img;
        }
    }
}
/////////////////////////////////////// 检测按键 /////////////////////////////////////////////////////
void control_info(Control_Params* pcontol) {
    // 循环检测用户输入
    while (1) {
        pcontol->watching_input_seam();
        if (exit_flag) {
            break;
        }
    }
    LOGLN("程序退出，停止检测");
}
/////////////////////////////////////// main ////////////////////////////////////////////////////////
int main() {
# if use_new
    std::cout << "测试---手动更新拼接缝，若需要更新，按下1" << endl; // 再开一个线程实时检测
    Video_Capture vid[CAMERA_NUM];  Video_Capture* pvid[CAMERA_NUM];
    Find_Feature fea[CAMERA_NUM];   Find_Feature* pfea[CAMERA_NUM];
    Match_Features match[CAMERA_NUM - 1];   Match_Features* pmatch[CAMERA_NUM - 1];
    Est_exposure ese;   Est_exposure* pese;
    Est_seamm esm;  Est_seamm* pesm;
    graph_blender blender;  graph_blender* pblender;
    // 视频路径
    string paths[CAMERA_NUM] = {"../left1.mp4","../right1.mp4"};
    // 指针 --- 地址赋值
    for (int i = 0;i < CAMERA_NUM;i++) {
        pvid[i] = &vid[i];
        pfea[i] = &fea[i];
    }
    for (int i = 0;i < CAMERA_NUM - 1; i++) {
        pmatch[i] = &match[i];
    }
    pese = &ese;
    pesm = &esm;
    pblender = &blender;
    // 线程操作
    LOGLN("System supports " << thread::hardware_concurrency() << " concurrent threads.");
    thread video_num1(video_input, pvid[0], pmatch[0], paths[0]);
    thread video_num2(video_input, pvid[1], pmatch[0], paths[1]);
    thread feature_num1(find_feature, pvid[0], pfea[0]);
    thread feature_num2(find_feature, pvid[1], pfea[1]);
    thread match_num1(match_new, pfea[0], pfea[1], pmatch[0]);
    thread expose_num1(expose_new, pvid[0], pvid[1], pfea[0], pmatch[0], pese, pesm);
    thread seam_num1(seam_finder_new, pmatch[0], pese, pesm);
    thread stitch_num1(stitch_main_new, pese, pmatch[0], pblender);

    // 销毁线程
    video_num1.join();
    video_num2.join();
    feature_num1.join();
    feature_num2.join();
    match_num1.join();
    expose_num1.join();
    seam_num1.join();
    stitch_num1.join();
# endif
#if IsThread
    LOGLN("System supports " << thread::hardware_concurrency() << " concurrent threads.");
    thread video1_input_thread(video_input1, "../left1.mp4");//left video1 load_1
    thread video2_input_thread(video_input2, "../right1.mp4");//right video2 load_2
    thread feature1_thread(find_feature_1);
    thread feature2_thread(find_feature_2);
    thread match_thread(match_feature);
    thread est_exposure_thread(Est_exp);
    thread est_seam_thread(Est_seam);
    thread stitch_thread(stitch_frame);

    video1_input_thread.join();
    video2_input_thread.join();
    feature1_thread.join();
    feature2_thread.join();
    match_thread.join();
    est_exposure_thread.join();
    est_seam_thread.join();
    stitch_thread.join();

    LOGLN("程序运行完成");
#elif videoHave
    generate_video();
#else
    VideoCapture cap1("../left1.mp4");
    if (!cap1.isOpened()) {
        LOGLN("无法打开视频文件!");
        return -1;
    }
    VideoCapture cap2("../right1.mp4");
    if (!cap2.isOpened()) {
        LOGLN("无法打开视频文件!");
        return -1;
    }
    Mat frame1, frame2;
    Mat mframe1, mframe2;
    vector<detail::ImageFeatures> features{ detail::ImageFeatures(),detail::ImageFeatures() };
    vector<Mat> imgs{ Mat(),Mat() };
    vector<detail::CameraParams>cams = {
        detail::CameraParams(),
        detail::CameraParams()
    };
    est_expose.init_all();
    while (1)
    {
        cap1.read(frame1);
        cap2.read(frame2);

        /*string img1_path = "1_img1.jpg";
        string img2_path = "1_img2.jpg";

        frame1 = cv::imread(img1_path,cv::IMREAD_COLOR);
        frame2 = cv::imread(img2_path, cv::IMREAD_COLOR);*/


        img_queue1.org_push(frame1);
        img_queue2.org_push(frame2);

        feature1.find(frame1);
        feature2.find(frame2);

        imgs[0] = feature1.get_dst();
        imgs[1] = feature2.get_dst();
        features[0] = feature1.get_feature();
        features[1] = feature2.get_feature();

        feature_match.update_features_imgs(features, imgs);
        feature_match.match();
        feature_match.est_params();

        img_queue1.warp(frame1, feature_match.cameras[0], feature_match.warped_image_scale);
        img_queue2.warp(frame2, feature_match.cameras[1], feature_match.warped_image_scale);

        // init_exposure
        img_queue1.rwc_pop(mframe1);
        img_queue2.rwc_pop(mframe2);
        feature_match.copy_cameras(cams, feature_match.cameras);
        est_expose.get_cams(cams, feature_match.warped_image_scale);
        est_expose.get_rwc_images(vector<Mat>{mframe1, mframe2}, vector<Mat>{img_queue1.mask_warped.clone(), img_queue2.mask_warped.clone()},
            vector<Point>{img_queue1.corners, img_queue2.corners}, vector<Mat>{img_queue1.img_warped.clone(), img_queue2.img_warped.clone()});
        est_expose.exposure_compensator_update_withrwc(); // 更新曝光器
        //seam
        //double t = getTickCount();
        est_seam.update_img(vector<UMat>{est_expose.images_warped_f[0].clone(), est_expose.images_warped_f[1].clone()},
            est_expose.mycorner, vector<UMat>{est_expose.masks_warped_f[0].clone(), est_expose.masks_warped_f[1].clone()});
        est_seam.init_seamfinder();
        //std::cout << (getTickCount() - t) / getTickFrequency() * 1000 << endl;
        //expose
        est_expose.update_seamed_warpedmask(est_seam.find_mask);
        est_expose.get_images(frame1, frame2, feature1.work_scale);
        est_expose.warp_compensate_img();
        //blend
        blend_item.update_source(vector<Mat>{est_expose.img_warped_s[0].clone(), est_expose.img_warped_s[1].clone()},
            vector<Mat>{est_expose.mask_warped_blend[0].clone(), est_expose.mask_warped_blend[1].clone()},
            est_expose.corners, est_expose.sizes);
        blend_item.graph_blend();
        cout << "拼接完成" << endl;
        if (img_queue1.org_is_empty() && img_queue2.org_is_empty()) {
            cout << "视频拼接完成" << endl;
            break;
        }
    }

#endif

    return 0;
}