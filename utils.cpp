//
// Created by mike on 12/7/18.
//

#define PATCH_SIZE 5

using namespace std;
using namespace cv::line_descriptor;


template <typename _Tp1, typename _Tp2>
vector<cv::Point_<_Tp2>> Point2_d_f_conversion(vector<cv::Point_<_Tp1>> input_vec)
{
    vector<cv::Point_<_Tp2>> output;
    for(size_t i=0; i<input_vec.size(); i++)
    {
        output.push_back(cv::Point_<_Tp2>(input_vec[i].x, input_vec[i].y));
    }
    return output;
};


void init_detectors_matchers(cv::Ptr<cv::DescriptorMatcher> & matcher, cv::Ptr<cv::ORB> & orb_detector,
                             cv::Ptr<cv::LineSegmentDetector> & line_detector, cv::Ptr<BinaryDescriptor> & line_descriptor,
                             cv::Ptr<BinaryDescriptorMatcher> & line_matcher)
{
    // ORB
    bool useProvidedKeypoints = true;
    float scaleFactor = 1.2f;
    int nlevels = 8;
    int patchSize = PATCH_SIZE;
    int edgeThreshold = patchSize;
    int firstLevel = 0;
    int WTA_K = 2;
    int scoreType = cv::ORB::HARRIS_SCORE;
    orb_detector = cv::ORB::create(4000, scaleFactor, nlevels,
                                                    edgeThreshold, firstLevel, WTA_K, scoreType,
                                                    patchSize);
    // descriptor matcher
    matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

    // line detection
    line_detector = cv::createLineSegmentDetector(cv::LSD_REFINE_STD);
    line_descriptor = BinaryDescriptor::createBinaryDescriptor();
    line_matcher = BinaryDescriptorMatcher::createBinaryDescriptorMatcher();
}

template <typename _Tp>
vector<cv::KeyPoint> Point2T_vec_2_Keypoint_vec(vector<cv::Point_<_Tp>> input_vec)
{
    vector<cv::KeyPoint> keypoints_vec;
    for( size_t i = 0; i < input_vec.size(); i++ ) {
        keypoints_vec.push_back(cv::KeyPoint(input_vec[i], 1.f));
    }
    return keypoints_vec;
}

template <typename _Tp>
vector<cv::Point_<_Tp>> Keypoint_vec_2_Point2T_vec(vector<cv::KeyPoint> input_vec)
{
    vector<cv::Point_<_Tp>> pt2T_vec;
    for( size_t i = 0; i < input_vec.size(); i++ ) {
        pt2T_vec.push_back(input_vec[i].pt);
    }
    return pt2T_vec;
}


template <typename _Tp>
vector<cv::Point3_<_Tp>> Mat_2_Keypoint_vec(cv::Mat tri_pt3d_mat)
{
    vector<cv::Point3_<_Tp>> tri_pt3d_vec;
    for(int pt_i=0; pt_i<tri_pt3d_mat.cols; pt_i++)
    {
        cv::Point3_<_Tp> cur_pt(tri_pt3d_mat.at<double>(0,pt_i)/tri_pt3d_mat.at<double>(3,pt_i),
                           tri_pt3d_mat.at<double>(1,pt_i)/tri_pt3d_mat.at<double>(3,pt_i),
                           tri_pt3d_mat.at<double>(2,pt_i)/tri_pt3d_mat.at<double>(3,pt_i));
        tri_pt3d_vec.push_back(cur_pt);
    }
    return tri_pt3d_vec;
}


template <typename T, typename matcher_T>
void knn_match(const vector<T> & left_feature, const vector<T> & right_feature,
               vector<T> & matched_l, vector<T> & matched_r,
               cv::Mat & left_desc, cv::Mat & right_desc, vector<cv::DMatch> & ORB_matches,
               vector< vector<cv::DMatch> > & matches, const cv::Ptr<matcher_T> & matcher,
               const float nn_match_ratio = 0.8f)
{
    matcher->knnMatch(left_desc, right_desc, matches, 2);

    cv::Mat left_desc_out, right_desc_out;
    for(unsigned i = 0; i < matches.size(); i++) {
        if(matches[i][0].distance < nn_match_ratio * matches[i][1].distance) {
            int idx_l = matches[i][0].queryIdx;
            int idx_r = matches[i][0].trainIdx;
            matched_l.push_back(left_feature[idx_l]);
            matched_r.push_back(right_feature[idx_r]);

            ORB_matches.push_back(matches[i][0]);
            left_desc_out.push_back(left_desc.row(idx_l));
            right_desc_out.push_back(right_desc.row(idx_r));
        }
    }

    left_desc_out.copyTo(left_desc);
    right_desc_out.copyTo(right_desc);
}


vector<cv::DMatch> detect_match_point_features(cv::Mat left_image, cv::Mat right_image,
                                               vector<cv::KeyPoint> & matched_l, vector<cv::KeyPoint> & matched_r,
                                               cv::Mat & left_desc, cv::Mat & right_desc,
                                               cv::Ptr<cv::ORB> & orb_detector, const cv::Ptr<cv::DescriptorMatcher> & matcher)
{
    vector<cv::DMatch> ORB_matches;
    vector< cv::Point2d > left_pt2d_vec, right_pt2d_vec;

    int maxCorners = 4000;
    double qualityLevel = 0.01;
    int patchSize = PATCH_SIZE;
    double minDistance = 10;
    cv::goodFeaturesToTrack(left_image, left_pt2d_vec, maxCorners, qualityLevel, minDistance,
                            cv::Mat(), patchSize);
    cv::goodFeaturesToTrack(right_image, right_pt2d_vec, maxCorners, qualityLevel, minDistance,
                            cv::Mat(), patchSize);

    vector<cv::KeyPoint> left_keypts = Point2T_vec_2_Keypoint_vec<double>(left_pt2d_vec);
    vector<cv::KeyPoint> right_keypts = Point2T_vec_2_Keypoint_vec<double>(right_pt2d_vec);

    orb_detector->compute(left_image, left_keypts, left_desc);
    orb_detector->compute(right_image, right_keypts, right_desc);

    // match points
    float nn_match_ratio = 0.8f;
    vector< vector<cv::DMatch> > matches;
    knn_match<cv::KeyPoint, cv::DescriptorMatcher>(left_keypts, right_keypts, matched_l, matched_r,
                                                   left_desc, right_desc, ORB_matches, matches, matcher, nn_match_ratio);

    // visualization
    cv::namedWindow("point matches");
    cv::Mat matches_image;
    cv::drawMatches(left_image, left_keypts, right_image, right_keypts,
                    ORB_matches, matches_image);
    imshow("point matches", matches_image );
    cv::waitKey(500);

    return ORB_matches;
}


void filter_short_lines(vector<KeyLine> & left_keyl, cv::Mat & left_keyl_desc,
                        vector<KeyLine> & right_keyl, cv::Mat & right_keyl_desc, double len_thresh = 50)
{
    vector<KeyLine> left_keyl_out, right_keyl_out;
    cv::Mat left_desc_out, right_desc_out;
    for(int i=0; i<left_keyl.size(); i++)
    {
        cv::Point2f line_vec = left_keyl[i].getEndPoint() - left_keyl[i].getStartPoint();
        double line_len = sqrt(line_vec.x*line_vec.x + line_vec.y*line_vec.y);
        if(line_len < len_thresh){
            continue;
        }else{
            left_keyl_out.push_back(left_keyl[i]);
            left_desc_out.push_back(left_keyl_desc.row(i));
        }
    }
    left_keyl = left_keyl_out;
    left_keyl_desc = left_desc_out;

    for(int i=0; i<right_keyl.size(); i++)
    {
        cv::Point2f line_vec = right_keyl[i].getEndPoint() - right_keyl[i].getStartPoint();
        double line_len = sqrt(line_vec.x*line_vec.x + line_vec.y*line_vec.y);
        if(line_len < len_thresh){
            continue;
        }else{
            right_keyl_out.push_back(right_keyl[i]);
            right_desc_out.push_back(right_keyl_desc.row(i));
        }
    }
    right_keyl = right_keyl_out;
    right_keyl_desc = right_desc_out;
}


vector<cv::DMatch>  detect_match_line_features(const cv::Mat & left_image, const cv::Mat & right_image,
                                               vector<KeyLine> & matched_l, vector<KeyLine> & matched_r,
                                               cv::Mat & left_keyl_desc, cv::Mat & right_keyl_desc,
                                               const cv::Ptr<BinaryDescriptor> & line_descriptor,
                                               const cv::Ptr<BinaryDescriptorMatcher> line_matcher)
{
    vector<cv::DMatch> line_matches;
    vector<KeyLine> left_keyl, right_keyl;

    // detect lines
    cv::Mat mask_l = cv::Mat::ones( left_image.size(), CV_8UC1 );
    cv::Mat mask_r = cv::Mat::ones( right_image.size(), CV_8UC1 );
    line_descriptor->detect(left_image, left_keyl, mask_l);
    line_descriptor->detect(right_image, right_keyl, mask_r);
    line_descriptor->compute( left_image, left_keyl, left_keyl_desc );
    line_descriptor->compute( right_image, right_keyl, right_keyl_desc );

    // match lines
    double len_thresh = 50;
    filter_short_lines(left_keyl, left_keyl_desc, right_keyl, right_keyl_desc, len_thresh);
    float nn_match_ratio = 0.6f;
    vector<vector<cv::DMatch>> matches;
    knn_match<KeyLine, BinaryDescriptorMatcher>(left_keyl, right_keyl, matched_l, matched_r,
                                                left_keyl_desc, right_keyl_desc, line_matches,
                                                matches, line_matcher, nn_match_ratio);

    // visualization
    show_line_matches(left_image, right_image, matched_l, matched_r);

    return line_matches;
}


vector<int> filter_by_F(const cv::Mat F, const cv::Mat& img1, const cv::Mat & img2,
                 vector<cv::Point2d> & points1, vector<cv::Point2d> & points2,
                 cv::Mat & desc1, cv::Mat & desc2, const float inlierDistance = -1, bool visualize=false)
{
    bool change_desc = desc1.rows > 0;
    vector<cv::Point2d> out_points1, out_points2;
    cv::Mat out_desc1, out_desc2;
    vector<int> inlier_idx;

    string title = "epipolar lines";
    CV_Assert(img1.size() == img2.size() && img1.type() == img2.type());

    cv::Mat outImg(img1.rows, img1.cols*2, CV_8UC3);
    cv::Rect rect1(0,0, img1.cols, img1.rows);
    cv::Rect rect2(img1.cols, 0, img1.cols, img1.rows);
    if(visualize)
    {
        if (img1.type() == CV_8U)
        {
            cv::cvtColor(img1, outImg(rect1), CV_GRAY2BGR);
            cv::cvtColor(img2, outImg(rect2), CV_GRAY2BGR);
        }
        else
        {
            img1.copyTo(outImg(rect1));
            img2.copyTo(outImg(rect2));
        }
    }
    vector<cv::Vec3d> epilines1, epilines2;
    cv::computeCorrespondEpilines(points1, 1, F, epilines1); //Index starts with 1
    cv::computeCorrespondEpilines(points2, 2, F, epilines2);

    CV_Assert(points1.size() == points2.size() &&
              points2.size() == epilines1.size() &&
              epilines1.size() == epilines2.size());

    cv::RNG rng(0);
    for(size_t i=1; i<points1.size(); i++)
//    for(size_t i=0; i<10; i++)
    {

        if(inlierDistance > 0 && (distancePointLine<double, double>(points1[i], epilines2[i]) > inlierDistance ||
           distancePointLine<double, double>(points2[i], epilines1[i]) > inlierDistance))
        {
            continue;
        }

        if(visualize)
        {
            cv::Scalar color(rng(256),rng(256),rng(256));

            cv::line(outImg(rect2),
                     cv::Point(0,-epilines1[i][2]/epilines1[i][1]),
                     cv::Point(img1.cols,-(epilines1[i][2]+epilines1[i][0]*img1.cols)/epilines1[i][1]),
                     color);
            cv::circle(outImg(rect1), points1[i], 3, color, -1, CV_AA);

            cv::line(outImg(rect1),
                     cv::Point(0,-epilines2[i][2]/epilines2[i][1]),
                     cv::Point(img2.cols,-(epilines2[i][2]+epilines2[i][0]*img2.cols)/epilines2[i][1]),
                     color);
            cv::circle(outImg(rect2), points2[i], 3, color, -1, CV_AA);
        }

        inlier_idx.push_back(i);
        out_points1.push_back(points1[i]);
        out_points2.push_back(points2[i]);
        if(change_desc)
        {
            out_desc1.push_back(desc1.row(i));
            out_desc2.push_back(desc2.row(i));
        }
    }

    points1 = out_points1;
    points2 = out_points2;
    if(change_desc)
    {
        out_desc1.copyTo(desc1);
        out_desc2.copyTo(desc2);
    }

    if(visualize)
    {
        cv::namedWindow(title);
        cv::imshow(title, outImg);
        cv::waitKey(500);
    }

    return inlier_idx;
}


void filter_matches(vector<cv::Point2d> & left_pt2d_vec, vector<cv::Point2d> & right_pt2d_vec,
                    cv::Mat & left_pt_desc, cv::Mat & right_pt_desc,
                    vector<KeyLine> & left_matched_keyl_vec, vector<KeyLine> & right_matched_keyl_vec,
                    cv::Mat & left_keyl_desc, cv::Mat & right_keyl_desc,
                    const cv::Mat & img1, const cv::Mat & img2, bool visualize=false)
{
    float dist_2_epipolar_line = 3;
    filter_by_F(F_stereo, img1, img2, left_pt2d_vec, right_pt2d_vec, left_pt_desc, right_pt_desc, dist_2_epipolar_line, visualize);
}


template <typename _Tp>
cv::Mat Point2T_vec_2_Mat(const vector<cv::Point_<_Tp>> & pt_vec)
{
    cv::Mat out_mat;
    for(int i=0; i<pt_vec.size(); i++)
    {
        double pt_arr[2] = {pt_vec[i].x, pt_vec[i].y};
        cv::Mat pt_mat = cv::Mat(2, 1, CV_64F, pt_arr);
        if(out_mat.cols == 0)
        {
            out_mat = pt_mat.clone();
        }else{
            cv::hconcat(out_mat, pt_mat, out_mat);
        }

    }

    return out_mat;
}


//void project()


void triangulate(vector<cv::KeyPoint> & left_pts_vec, vector<cv::KeyPoint> & right_pts_vec,
                 vector<cv::Point3d> & tri_pt3d_vec, cv::Mat & K_l, cv::Mat & Rt_l, cv::Mat & K_r, cv::Mat & Rt_r)
{
    cv::Mat P_l = K_l * Rt_l, P_r = K_r * Rt_r;
    cv::Mat P_l1 = P_l.row(0), P_l2 = P_l.row(1), P_l3 = P_l.row(2);
    cv::Mat P_r1 = P_r.row(0), P_r2 = P_r.row(1), P_r3 = P_r.row(2);

    for(int i=0; i<left_pts_vec.size(); i++)
    {
        double x_l = left_pts_vec[i].pt.x;
        double y_l = left_pts_vec[i].pt.y;
        double x_r = right_pts_vec[i].pt.x;
        double y_r = right_pts_vec[i].pt.y;
        cv::Mat D = x_l * P_l3 - P_l1;
        D.push_back(y_l * P_l3 - P_l2);
        D.push_back(x_r * P_r3 - P_r1);
        D.push_back(y_r * P_r3 - P_r2);
        cv::Mat sigma, u, vt, v;
        cv::SVD::compute(D.t()*D, sigma, u, vt, 4);
        v = vt.t();
        cv::Mat sol = v.col(v.cols-1);
        cv::Point3d cur_pt = cv::Point3d(sol.at<double>(0,0)/sol.at<double>(3,0),
                                         sol.at<double>(1,0)/sol.at<double>(3,0),
                                         sol.at<double>(2,0)/sol.at<double>(3,0));
        tri_pt3d_vec.push_back(cur_pt);
    }

}


vector<int> track_prev_frame(const cv::Mat & prev_image, const cv::Mat & image,
                                  vector<cv::Point2d> & prev_pt2d_vec, vector<cv::Point2d> & tracked_pt2d_vec)
{
    vector<int> tracked_idx;

    // init
    cv::Mat tracked_image = image.clone();
    cv::Size subPixWinSize(10,10), winSize(31,31);
    cv::TermCriteria termcrit(cv::TermCriteria::COUNT|cv::TermCriteria::EPS,20,0.03);
    vector<uchar> status;
    vector<float> err;

    // track
    vector<cv::Point2f> tracked_pt2f_vec;
    vector<cv::Point2f> prev_pt2f_vec = Point2_d_f_conversion<double, float>(prev_pt2d_vec);
    cornerSubPix(prev_image, prev_pt2f_vec, subPixWinSize, cv::Size(-1,-1), termcrit);
    cv::calcOpticalFlowPyrLK(prev_image, image, prev_pt2f_vec, tracked_pt2f_vec,
                             status, err, winSize, 3, termcrit, 0, 0.001);

    size_t i, k;
    for( i = k = 0; i < tracked_pt2f_vec.size(); i++ )
    {
        if( !status[i] )
            continue;
        tracked_idx.push_back(i);
        tracked_pt2f_vec[k++] = tracked_pt2f_vec[i];
        prev_pt2f_vec[k-1] = prev_pt2f_vec[i];
        circle( tracked_image, tracked_pt2f_vec[i], 3, cv::Scalar(0,255,0), -1, 8);
    }
    tracked_pt2f_vec.resize(k);
    prev_pt2f_vec.resize(k);

    prev_pt2d_vec = Point2_d_f_conversion<float, double>(prev_pt2f_vec);
    tracked_pt2d_vec = Point2_d_f_conversion<float, double>(tracked_pt2f_vec);

    // visualization
    vector<cv::KeyPoint> prev_keypt_vec = Point2T_vec_2_Keypoint_vec(prev_pt2d_vec);
    vector<cv::KeyPoint> tracked_keypt_vec = Point2T_vec_2_Keypoint_vec(tracked_pt2d_vec);
    draw_matches(prev_keypt_vec, tracked_keypt_vec, prev_image, image, "tracking");

    return  tracked_idx;
}


vector<int> track_stereo(const Frame & prev_frame, const cv::Mat & left_image,
                         const cv::Mat & right_image, vector<cv::Point2d> & tracked_pt2d_l_vec,
                         vector<cv::Point2d> & tracked_pt2d_r_vec, bool visualize)
{
    // init
    cv::Mat prev_left_img = prev_frame.left_image.clone(), prev_right_img = prev_frame.right_image.clone();
    vector<cv::Point2d> prev_pt2d_l_vec = prev_frame.feature_obser_l_vec;
    vector<cv::Point2d> prev_pt2d_r_vec = prev_frame.feature_obser_r_vec;

    // track
    vector<int> tracked_idx_l_vec = track_prev_frame(prev_left_img, left_image, prev_pt2d_l_vec, tracked_pt2d_l_vec);
    int n_l_tracked = tracked_idx_l_vec.size();
    vector<int> tracked_idx_r_vec = track_prev_frame(prev_right_img, right_image, prev_pt2d_r_vec, tracked_pt2d_r_vec);
    int n_r_tracked = tracked_idx_r_vec.size();

//    vector<int> tracked_idx_both_vec = n_l_tracked > n_r_tracked? tracked_idx_l_vec : tracked_idx_r_vec;
//    vector<int>::iterator it=std::set_intersection (tracked_idx_l_vec.begin(), tracked_idx_l_vec.begin()+n_l_tracked,
//            tracked_idx_r_vec.begin(), tracked_idx_r_vec.begin()+n_r_tracked, tracked_idx_both_vec.begin());
//    tracked_idx_both_vec.resize(it-tracked_idx_both_vec.begin());
    int k=0;
    for(int i=0, j=0; i<tracked_idx_l_vec.size() && j<tracked_idx_r_vec.size();)
    {
        int i_l = tracked_idx_l_vec[i];
        int i_r = tracked_idx_r_vec[j];

        if(i_l == i_r)
        {
            tracked_idx_l_vec[k] = i_l;
            tracked_pt2d_l_vec[k] = tracked_pt2d_l_vec[i];
            tracked_idx_r_vec[k] = i_r;
            tracked_pt2d_r_vec[k] = tracked_pt2d_r_vec[j];
            k++; i++; j++;
        }else if(i_l < i_r){
            i++;
        }else{
            j++;
        }
    }
    tracked_idx_l_vec.resize(k);
    tracked_idx_r_vec.resize(k);
    tracked_pt2d_l_vec.resize(k);
    tracked_pt2d_r_vec.resize(k);

    // filter
    float inlierDistance = 3;
    cv::Mat tmp1, tmp2;
    vector<int> inlier_idx = filter_by_F(F_stereo, prev_left_img, prev_right_img, tracked_pt2d_l_vec, tracked_pt2d_r_vec,
            tmp1, tmp2, inlierDistance, visualize);

    for(int i=0; i<inlier_idx.size(); i++)
    {
        int idx = inlier_idx[i];
        tracked_idx_l_vec[i] = tracked_idx_l_vec[idx];
        tracked_idx_r_vec[i] = tracked_idx_r_vec[idx];

        assert(tracked_idx_r_vec[i] == tracked_idx_l_vec[i]);
    }
    tracked_idx_l_vec.resize(inlier_idx.size());
    tracked_idx_r_vec.resize(inlier_idx.size());

    return tracked_idx_l_vec;
}
