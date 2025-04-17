#include "Est_seam.hpp"

void Est_seamm::update_img(vector<UMat>images_warped_f_in, vector<Point>corners_in, vector<UMat>masks_warped_in) {
    for (uint8_t i = 0;i < 2;++i) {
        myimg[i] = images_warped_f_in[i].clone();
        mymask[i] = masks_warped_in[i].clone();
    }
    corner = corners_in;
}

void Est_seamm::init_seamfinder() {
    mtx_esS.lock();
    Ptr<SeamFinder> seam_finder; //搜索器，函数内初始化(放到此处，减少搜索器生命周期)
    seam_finder = makePtr<detail::GraphCutSeamFinder>(detail::GraphCutSeamFinder::COST_COLOR);
    seam_finder->find(myimg, corner, mymask);
    for (uint8_t i = 0;i < 2;i++) {
        find_masks[i] = mymask[i].clone();
    }
    bootflag_seam = true;
    mtx_esS.unlock();
    cond.notify_all();
}