//
// Created by mike on 12/8/18.
//

// intrinsics
double K_l_arr[3][3] = {{458.654, 0, 367.215}, {0, 457.296, 248.375}, {0, 0, 1}};
const cv::Mat K_l = cv::Mat(3, 3, CV_64F, K_l_arr);
double dist_coeff_l_arr[4] = {-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05};
const cv::Mat dist_coeff_l = cv::Mat(1, 4, CV_64F, dist_coeff_l_arr);

double K_r_arr[3][3] = {{457.587, 0, 379.999}, {0, 456.134, 255.238}, {0, 0, 1}};
const cv::Mat K_r = cv::Mat(3, 3, CV_64F, K_r_arr);
double dist_coeff_r_arr[4] = {-0.28368365,  0.07451284, -0.00010473, -3.55590700e-05};
const cv::Mat dist_coeff_r = cv::Mat(1, 4, CV_64F, dist_coeff_r_arr);

// extrinsics
double T_S2B_l_arr[4][4] ={{0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975},
                            {0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768},
                            {-0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949},
                            {0.0, 0.0, 0.0, 1.0}};
const cv::Mat T_S2B_l = cv::Mat(4, 4, CV_64F, T_S2B_l_arr);

double T_S2B_r_arr[4][4] = {{0.0125552670891, -0.999755099723, 0.0182237714554, -0.0198435579556},
                             {0.999598781151, 0.0130119051815, 0.0251588363115, 0.0453689425024},
                             {-0.0253898008918, 0.0179005838253, 0.999517347078, 0.00786212447038},
                             {0.0, 0.0, 0.0, 1.0}};
const cv::Mat T_S2B_r = cv::Mat(4, 4, CV_64F, T_S2B_r_arr);

const cv::Mat T_l2r = T_S2B_r.inv() * T_S2B_l;

const cv::Mat R_stereo = T_l2r.colRange(0,3).rowRange(0,3), t_stereo = T_l2r.col(3).rowRange(0,3);

double t_stereo_skew_arr[9] = {0, -t_stereo.at<double>(2,0), t_stereo.at<double>(1,0),
                                    t_stereo.at<double>(2,0), 0, -t_stereo.at<double>(0,0),
                                    -t_stereo.at<double>(1,0), t_stereo.at<double>(0,0), 0};
const cv::Mat t_stereo_skew = cv::Mat(3, 3, CV_64F, t_stereo_skew_arr);
const cv::Mat F_stereo = K_r.t().inv() * (t_stereo_skew*R_stereo) * K_l.inv();