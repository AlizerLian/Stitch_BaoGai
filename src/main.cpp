#include "Camera_input.hpp"
#include "KeyPoint.hpp"
#include "Est_exposure.hpp"
#include "Est_seam.hpp"
#include "Stitch_frame.hpp"
#include <queue>
#include <thread>
#include <filesystem>
#include <atomic>
#include <chrono>
#include <opencv2/core/ocl.hpp>
#include "Motion.h"

using namespace cv;
using namespace std;

#define CAMERA_NUM 2
#define LOGLN(msg) std::cout << msg << std::endl
#define TIME_LOG 0
#define ORG_EXPOSE 0
#define IsThread 1

bool exit_flag = false; //全局退出标志（当视频流读取为空）
Motion motion;
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
    int i=0;
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

    exit_flag = true;
    img_queue1.signal_exit();
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
    /*LOGLN("等待特征点提取");
    unique_lock<mutex> lk_f1(feature1.mtx_feature);
    while (!feature1.boot_flag) {
        feature1.cond.wait(lk_f1);
    }
    lk_f1.unlock();
    unique_lock<mutex> lk_f2(feature2.mtx_feature);
    while (!feature2.boot_flag) {
        feature2.cond.wait(lk_f1);
    }
    lk_f2.unlock();*/
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
/////////////////////////////////////// thread6 ////////////////////////////////////////////////////
void stitch_frame() {
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

        auto start = chrono::high_resolution_clock::now();
        est_expose.mtx_esE.lock();
        for (uint8_t i = 0;i < 2;++i) {
            images_blend_s[i] = est_expose.img_warped_s[i].clone();
            masks_blend_s[i] = est_expose.mask_warped_blend[i].clone();
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
/////////////////////////////////////// thread7 ////////////////////////////////////////////////////
void Est_exp() {
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
    img_queue1.rwc_pop(dst1);
    img_queue2.rwc_pop(dst2);
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
    est_expose.get_images(dst1, dst2, workscale);
    est_expose.get_rwc_images(vector<Mat>{dst1, dst2}, warped_masks, corners, images_warped);
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
    est_expose.update_seamed_warpedmask(est_seam.find_masks);
    est_seam.mtx_esS.unlock();

    while (1) {
        if (exit_flag) { break; }

        t = getTickCount();
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
        est_expose.get_images(dst1, dst2, workscale);
        est_expose.warp_compensate_img();
        est_seam.mtx_esS.lock();
        est_expose.update_seamed_warpedmask(est_seam.find_masks);
        est_seam.mtx_esS.unlock();
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

    }
    LOGLN("----thread est_seam---- 退出");
}
// 结构调整------------------------------------------------------------------------------------------
///////////////////////////////////////input thread////////////////////////////////////////////////////
void video_input_thread(const string& path, Video_Capture* vid, const string& camName, Match_Features* fea_match) {
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
        unique_lock<mutex> lk_cameras(fea_match->mtx_cameras);
        wrap_flag = fea_match->boot_flag;
        if (wrap_flag > 0) {
            wrap_scale = fea_match->warped_image_scale;
            fea_match->copy_camera(cam_param, fea_match->cameras[0]);
        }
        lk_cameras.unlock();
        if (wrap_flag > 0) {
            vid->warp(frame, cam_param, wrap_scale);
#if 0
            Mat warp_img;
            vid.rwc_pop(warp_img);
            warp_img.convertTo(warp_img, CV_8U);
            vid.save_img(warp_img);
#endif
        }

#if TIME_LOG
        LOGLN(camName << (getTickCount() - t) / getTickFrequency() << " s");
#endif
    }

    exit_flag = true;
    img_queue1.signal_exit();
    LOGLN(camName <<"退出" << "total:" << i);
}
/////////////////////////////////////// feature thread ////////////////////////////////////////////////////
void feature_find_thread(Video_Capture* vid, Find_Feature* fea) {
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
/////////////////////////////////////// match thread ////////////////////////////////////////////////////
void fea_match_thread(Find_Feature* fea[], Match_Features* fea_matches) {
    vector<detail::ImageFeatures> features;
    vector<Mat> imgs;
    imgs.resize(CAMERA_NUM);
    features.resize(CAMERA_NUM);
    double t;
    LOGLN("等待特征点提取");
    unique_lock<mutex> lk_f1(fea[0]->mtx_feature);
    while (!fea[0]->boot_flag) {
        fea[0]->cond.wait(lk_f1);
    }
    lk_f1.unlock();
    unique_lock<mutex> lk_f2(fea[1]->mtx_feature);
    while (!fea[1]->boot_flag) {
        fea[1]->cond.wait(lk_f1);
    }
    lk_f2.unlock();
    LOGLN("开始匹配");
    while (1) {
        if (exit_flag) { break; }
        t = getTickCount();
        //copy
        fea[0]->mtx_feature.lock();
        fea[1]->mtx_feature.lock();
        //LOGLN("读取");
        imgs[0] = fea[0]->get_dst();
        imgs[1] = fea[1]->get_dst();
        features[0] = fea[0]->get_feature();
        features[1] = fea[1]->get_feature();
        fea[0]->mtx_feature.unlock();
        fea[1]->mtx_feature.unlock();
        //process
        fea_matches->update_features_imgs(features, imgs);
        //LOGLN("匹配");
        fea_matches->match();
        //LOGLN("匹配完成，开始估计相机参数");
        fea_matches->est_params();
#if TIME_LOG   
        LOGLN("match_feature : " << (getTickCount() - t) / getTickFrequency() << " s");
#endif 

    }
    LOGLN("----thread match---- 退出");
}
/////////////////////////////////////// stitch thread////////////////////////////////////////////////////
void stitch_thread(Est_exposure* exposed, graph_blender* gbl) {
    vector<Mat>images_blend_s{ Mat(),Mat() };
    vector<Mat>masks_blend_s{ Mat(),Mat() };
    LOGLN("等待曝光");
    unique_lock<mutex> lk_exposure(exposed->mtx_esE);
    while (!exposed->bootflag_exposure_done) {
        exposed->cond.wait(lk_exposure);
    }
    lk_exposure.unlock();
    LOGLN("开始拼接");
    //std::this_thread::sleep_for(std::chrono::milliseconds(1000));   
    auto totol_start = chrono::high_resolution_clock::now();
    while (1) {
        if (exit_flag) { break; }

        auto start = chrono::high_resolution_clock::now();
        exposed->mtx_esE.lock();
        for (uint8_t i = 0;i < 2;++i) {
            images_blend_s[i] = exposed->img_warped_s[i].clone();
            masks_blend_s[i] = exposed->mask_warped_blend[i].clone();
        }
        gbl->update_source(images_blend_s, masks_blend_s, exposed->corners, exposed->sizes);
        exposed->mtx_esE.unlock();
        gbl->graph_blend();

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
/////////////////////////////////////// expose thread ////////////////////////////////////////////////////
void expose_thread(Video_Capture* vid[], Find_Feature* fea[], Est_exposure* exposed, Est_seamm* seammed, Match_Features* matched) {
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
    exposed->init_all();
    double t;
    LOGLN("等待相机参数");
    unique_lock<mutex> lk_cameras(matched->mtx_cameras);
    while (!matched->boot_flag) {
        matched->cond.wait(lk_cameras);
    }
    lk_cameras.unlock();
    LOGLN("相机参数获取，初始化曝光器");
    // 曝光器第一次更新------------------------------------------------------------------------
    while (1) {
        if (vid[0]->rwc_pop(dst1) && vid[1]->rwc_pop(dst2)) {
            fea[0]->mtx_feature.lock();
            workscale = fea[0]->work_scale;
            fea[0]->mtx_feature.unlock();
            vid[0]->mtx_corner.lock();
            warped_masks.push_back(vid[0]->mask_warped.clone());
            corners.push_back(vid[0]->corners);
            images_warped.push_back(vid[0]->img_warped.clone());
            vid[0]->mtx_corner.unlock();
            vid[1]->mtx_corner.lock();
            warped_masks.push_back(vid[1]->mask_warped.clone());
            corners.push_back(vid[1]->corners);
            images_warped.push_back(vid[1]->img_warped.clone());
            vid[1]->mtx_corner.unlock();
            matched->mtx_cameras.lock();
            matched->copy_cameras(cams, matched->cameras);
            warped_img_scale = matched->warped_image_scale;
            matched->mtx_cameras.unlock();
            exposed->get_cams(cams, warped_img_scale);
            exposed->get_images(dst1, dst2, workscale);
            exposed->get_rwc_images(vector<Mat>{dst1, dst2}, warped_masks, corners, images_warped);
            exposed->exposure_compensator_update_withrwc(); // 更新曝光器
            break;
        }
    }
    // end--------------------------------------------------------------------------------------
    LOGLN("等待拼接缝搜索，更新掩码");
    unique_lock<mutex>lk_seam(seammed->mtx_esS);
    while (!seammed->bootflag_seam) {
        seammed->cond.wait(lk_seam);
    }
    lk_seam.unlock();
    LOGLN("掩码更新完成，开始补偿");
    seammed->mtx_esS.lock();
    exposed->update_seamed_warpedmask(seammed->find_masks);
    seammed->mtx_esS.unlock();

    while (1) {
        if (exit_flag) { break; }

        t = getTickCount();
        if (vid[0]->org_pop(dst1) && vid[1]->org_pop(dst2)) {
            fea[0]->mtx_feature.lock();
            workscale = fea[0]->work_scale;
            fea[0]->mtx_feature.unlock();
            exposed->get_images(dst1, dst2, workscale);
            exposed->warp_compensate_img();
            seammed->mtx_esS.lock();
            exposed->update_seamed_warpedmask(seammed->find_masks);
            seammed->mtx_esS.unlock();
        }
#if TIME_LOG   
        LOGLN("曝光处理时间 : " << (getTickCount() - t) / getTickFrequency() << " s");
#endif 
    }
    LOGLN("----thread est_exp---- 退出");
}
/////////////////////////////////////// seam thread ////////////////////////////////////////////////////
void seam_thread(Est_exposure* exposed, Est_seamm* seammed) {
    vector<Point>corners;
    vector<UMat> masks_warped_seam{ UMat(),UMat() };
    vector<UMat> images_warped_seam{ UMat(),UMat() };
    Mat dst1, dst2;
    double t;
    //////////////////////////////////////////////////////////////////
    unique_lock<mutex> lk_exposure(exposed->mtx_esE);
    while (!exposed->bootflag_exposure) {
        exposed->cond.wait(lk_exposure);
    }
    lk_exposure.unlock();
    //////////////////////////////////////////////////////////////////
    LOGLN("开始搜索拼接缝");
    exposed->mtx_esE.lock();
    for (uint8_t i = 0;i < CAMERA_NUM;++i) {
        masks_warped_seam[i] = exposed->masks_warped_f[i].clone();
        images_warped_seam[i] = exposed->images_warped_f[i].clone();
    }
    corners = exposed->mycorner;
    exposed->mtx_esE.unlock();
    seammed->update_img(images_warped_seam, corners, masks_warped_seam);
    seammed->init_seamfinder();
    while (1) {
        if (exit_flag) { break; }
        exposed->mtx_esE.lock();
        for (uint8_t i = 0; i < CAMERA_NUM; ++i) {
            masks_warped_seam[i] = exposed->masks_warped_f[i].clone();
            images_warped_seam[i] = exposed->images_warped_f[i].clone();
        }
        corners = exposed->mycorner;
        exposed->mtx_esE.unlock();
        seammed->update_img(images_warped_seam, corners, masks_warped_seam);
        seammed->init_seamfinder();

    }
    LOGLN("----thread est_seam---- 退出");
}
/////////////////////////////////////// main ////////////////////////////////////////////////////////
int main() {

    LOGLN("左内参" << motion.camera_matrix_left);
    LOGLN("左畸变" << motion.dist_left);
    LOGLN("右内参" << motion.camera_matrix_right);
    LOGLN("右畸变" << motion.dist_right);
    LOGLN("旋转" << motion.R);
    LOGLN("平移" << motion.T);

    LOGLN("线程？" << motion.Is_Thread);
    LOGLN("调试？" << motion.Is_Log);
    LOGLN("相机数" << motion.camera_num);
    LOGLN("视频1"  << motion.video_address1);
    LOGLN("视频2" << motion.video_address2);

    LOGLN("投影器:" << motion.Warper_Type);

    LOGLN("曝光器:" << motion.Exposure_Type);
    LOGLN("曝光器更新？:" << motion.Expourse_Upgrade);
    LOGLN("曝光器时间间隔:" << motion.Exposure_time);
    LOGLN("曝光器帧数间隔:" << motion.Exposure_Num);

    LOGLN("拼接缝查找器:" << motion.SeamFinder_Type);
    LOGLN("拼接缝更新？:" << motion.Seam_Upgrade);
    LOGLN("拼接缝查找时间间隔:" << motion.Seam_time);
    LOGLN("拼接缝查找帧数间隔:" << motion.Seam_Num);
#if IsThread
    // 调整结构后----------------------------------------------------------------------------
   /* Video_Capture vid[CAMERA_NUM];
    Find_Feature fea[CAMERA_NUM];
    Match_Features fea_match;
    Est_exposure ese;
    Est_seamm ess;
    graph_blender gb;

    Video_Capture* pvid[CAMERA_NUM];
    Find_Feature* pfea[CAMERA_NUM];
    Match_Features* pfea_match = &fea_match;
    Est_exposure* pese = &ese;
    Est_seamm* pess = &ess;
    graph_blender* pgb = &gb;
    for (uint8_t i = 0;i < CAMERA_NUM;i++) {
        pvid[i] = &vid[i];
        pfea[i] = &fea[i];
    }

    string path[CAMERA_NUM];
    string camname[CAMERA_NUM];
    path[0] = "../video1.mp4";
    path[1] = "../video2.mp4";
    camname[0] = "cam1";
    camname[1] = "cam2";

    LOGLN("System supports " << thread::hardware_concurrency() << " concurrent threads.");
    thread input1Thread(video_input_thread, ref(path[0]), pvid[0], ref(camname[0]), pfea_match);
    thread input2Thread(video_input_thread, ref(path[1]), pvid[1], ref(camname[1]), pfea_match);
    thread fea1Thread(feature_find_thread, pvid[0], pfea[0]);
    thread fea2Thread(feature_find_thread, pvid[1], pfea[1]);
    thread matchThread(fea_match_thread, pfea, pfea_match);
    thread exposeThread(expose_thread, pvid, pfea, pese, pess, pfea_match);
    thread seamThread(seam_thread, pese, pess);
    thread stitchThread(stitch_thread, pese, pgb);

    input1Thread.join();
    input2Thread.join();
    fea1Thread.join();
    fea2Thread.join();
    matchThread.join();
    exposeThread.join();
    seamThread.join();
    stitchThread.join(); */ 
    // end------------------------------------------------------------------------------------

    //thread video1_input_thread(video_input1, "../video1.mp4");//left video1 load_1
    //thread video2_input_thread(video_input2, "../video2.mp4");//right video2 load_2
    thread video1_input_thread(video_input1, motion.video_address1);//left video1 load_1
    thread video2_input_thread(video_input2, motion.video_address2);//right video2 load_2
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

#else
    VideoCapture cap1("left.mp4");
    if (!cap1.isOpened()) {
        LOGLN("无法打开视频文件!");
        return -1;
    }
    VideoCapture cap2("right.mp4");
    if (!cap2.isOpened()) {
        LOGLN("无法打开视频文件!");
        return -1;
    }
    Mat frame1, frame2;
    Mat mframe1, mframe2;
    vector<detail::ImageFeatures> features{ detail::ImageFeatures(),detail::ImageFeatures()};
    vector<Mat> imgs{Mat(),Mat()};
    vector<detail::CameraParams>cams = {
        detail::CameraParams(),
        detail::CameraParams()
    };
    est_expose.init_all();
    while (1)
    {
        cap1.read(frame1);
        cap2.read(frame2);

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
        feature_match.copy_cameras(cams,feature_match.cameras);
        est_expose.get_cams(cams, feature_match.warped_image_scale);
        est_expose.get_rwc_images(vector<Mat>{mframe1, mframe2}, vector<Mat>{img_queue1.mask_warped.clone(), img_queue2.mask_warped.clone()},
            vector<Point>{img_queue1.corners, img_queue2.corners}, vector<Mat>{img_queue1.img_warped.clone(), img_queue2.img_warped.clone()});
        est_expose.exposure_compensator_update_withrwc(); // 更新曝光器
        //seam
        est_seam.update_img(vector<UMat>{est_expose.images_warped_f[0].clone(),est_expose.images_warped_f[1].clone()},
            est_expose.mycorner, vector<UMat>{est_expose.masks_warped[0].clone(),est_expose.masks_warped[1].clone()});
        est_seam.init_seamfinder();
        //expose
        est_expose.update_seamed_warpedmask(est_seam.mymask);
        est_expose.get_images(frame1, frame2, feature1.work_scale);
        est_expose.warp_compensate_img();
        //blend
        blend_item.update_source(vector<Mat>{est_expose.img_warped_s[0].clone(), est_expose.img_warped_s[1].clone()},
            vector<Mat>{est_expose.mask_warped_blend[0].clone(), est_expose.mask_warped_blend[1].clone()},
            est_expose.corners, est_expose.sizes);
        blend_item.graph_blend();
    }

#endif

    return 0;
}