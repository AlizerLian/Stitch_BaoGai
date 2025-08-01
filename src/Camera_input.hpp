#ifndef camera_queue_h
#define camera_queue_h
#include <filesystem>
namespace fs = std::filesystem;
#define LOGLN(msg) std::cout << msg << std::endl

#include "opencv2/opencv.hpp"
#include <iostream>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
using namespace std;
using namespace cv;

typedef struct rwc_org
{
    Mat rwc;
    Mat org;
};

class Video_Capture {
private:
    condition_variable cond;
    const size_t org_max_size = 1000;  // 防止内存溢出
    float seam_work_aspect = 0.5f;
    const size_t rwc_max_size = 1000;
    Ptr<WarperCreator> warper_creator = makePtr<cv::SphericalWarper>();

public:
    mutex mtx_org;
    mutex mtx_rwc;
    mutex mtx_corner;
    queue<Mat> org_queue;
    queue<Mat> rwc_queue;
    queue<rwc_org> rwc_org_queue;
    atomic<bool> exit_flag{ false };
    Point corners;
    Mat mask_warped;
    Mat img_warped;
    void warp(const Mat& frame, detail::CameraParams camera, float warped_image_scale);

    void org_push(const Mat& img);
    bool org_pop(Mat& img);
    void rwc_push(const Mat& img);
    bool rwc_pop(Mat& img);
    void rwc_org_push(const rwc_org& ro);
    bool rwc_org_pop(rwc_org& ro);
    rwc_org ro_clone(const rwc_org& ro) {
        rwc_org thisone;
        thisone.org = ro.org.clone();  // 深拷贝
        thisone.rwc = ro.rwc.clone();  // 深拷贝
        return thisone;
    }
    bool rwc_is_empty() {
        return rwc_queue.empty();
    }
    bool org_is_empty() {
        return org_queue.empty();
    }
    bool org_rwc_empty() {
        return rwc_org_queue.empty();
    }

    Mat warpedmask_get() {
        if (mask_warped.empty()) {
            cout << "error---warped mask haven't done";
        }
        return mask_warped.clone();
    }

    void signal_exit() {
        exit_flag = true;
        cond.notify_all();
    }
    void save_img(const Mat& frame) {
        fs::path dir("out/rwc");
        if (!fs::exists(dir)) {
            fs::create_directories(dir);
        }
        // 构造文件名
        fs::path file_path = dir / (to_string(id) + ".jpg");
        id++;
        // 保存图片
        bool success = imwrite(file_path.string(), frame);
    }
private:
    int id = 0;

};




#endif