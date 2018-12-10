//
// Created by mike on 12/9/18.
//

#ifndef VLO_FRAME_H
#define VLO_FRAME_H

using namespace std;

class Frame {
public:
    cv::Mat left_image, right_image;
    cv::Mat descriptors_l, descriptors_r;
    vector<cv::Point2d> feature_obser_l_vec, feature_obser_r_vec;
    vector<cv::Point3d> feature_3d_vec;

    Frame(const cv::Mat & left_image_in, const cv::Mat & right_image_in, const cv::Mat & desc_l, const cv::Mat & desc_r,
          const vector<cv::Point2d> & obser_l_vec, const vector<cv::Point2d> & obser_r_vec, const vector<cv::Point3d> & feature_3d) :
            left_image(left_image_in), right_image(right_image_in),
            descriptors_l(desc_l), descriptors_r(desc_r),
            feature_obser_l_vec(obser_l_vec), feature_obser_r_vec(obser_r_vec),
            feature_3d_vec(feature_3d){}
};


#endif //VLO_FRAME_H
