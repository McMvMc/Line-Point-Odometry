#include <cstring>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>


#include "opencv2/core.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/line_descriptor.hpp"
#include <opencv2/calib3d/calib3d.hpp>
#include "opencv2/sfm.hpp"
#include <pangolin/pangolin.h>
#include "opencv2/video/tracking.hpp"

//#include "opencv2/viz.hpp"
//#include "opencv2/core/private.hpp"

#include "Frame.h"

#include "constant.cpp"
#include "math_helper.cpp"
#include "visualization.cpp"
#include "utils.cpp"
#include "data_loader.cpp"
#include "GREPPnX.cpp"

using namespace std;
using namespace cv::xfeatures2d;
using namespace cv::line_descriptor;

const std::string keys =
        "{help h usage ?    |                  | print this message   }"
        "{data_path         |/evo970/mav0/      | image1 for compare   }";

vector<Frame> frame_vec;


int main( int argc, char* argv[] )
{
    cout << "OpenCV version : " << CV_VERSION << endl;

    cv::CommandLineParser parser( argc, argv, keys );
    string data_path = parser.get<string>( "data_path" );

    string left_image_dir = data_path + "cam0/data/";
    string left_csv_path = data_path + "cam0/data.csv";
    string right_image_dir = data_path + "cam1/data/";
    string right_csv_path = data_path + "cam1/data.csv";

    bool visualize = true;

    vector<string> left_image_fn_vec, right_image_fn_vec;
    read_image_fn_vec(left_csv_path, right_csv_path, left_image_fn_vec, right_image_fn_vec);

    // init
    cv::Ptr<cv::DescriptorMatcher> pt_matcher;
    cv::Ptr<cv::ORB> orb_detector;
    cv::Ptr<cv::LineSegmentDetector> line_detector;
    cv::Ptr<BinaryDescriptor> line_descriptor;
    cv::Ptr<BinaryDescriptorMatcher> line_matcher;
    init_detectors_matchers(pt_matcher, orb_detector, line_detector, line_descriptor, line_matcher);

    // init 3D visualization
//    init_3d_visualization(s_cam);
    pangolin::CreateWindowAndBind("Map", 640, 480);
    glEnable(GL_DEPTH_TEST);

    // Define Projection and initial ModelView matrix
    pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 500),
            pangolin::ModelViewLookAt(0, 0, 0, 0, 10, 0, pangolin::AxisX)
    );

    // Create Interactive View in window
    pangolin::Handler3D handler(s_cam);
    pangolin::View& d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f / 480.0f)
            .SetHandler(&handler);
    pangolin::RegisterKeyPressCallback(' ', continue_track);

    // process each frame
    int i = 100;

    vector<cv::Point3d> tri_pt3d_vec;
    vector<cv::Vec6d> tri_l3d_vec;
    while (!pangolin::ShouldQuit()) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);

        if (i < left_image_fn_vec.size() && continue_track_flag) {
            // read image
            cv::Mat left_image, right_image;
            read_image_pair(left_image_dir + left_image_fn_vec[i], right_image_dir + right_image_fn_vec[i],
                            left_image, right_image);

            // ------ detect & match features ------ same frame

            // --- point features ---
            vector<cv::Point2f> left_pt2f_vec, right_pt2f_vec;
            vector<cv::KeyPoint> left_ORB_keypts_vec, right_ORB_keypts_vec, left_matched_keypts_vec, right_matched_keypts_vec;
            cv::Mat left_pt_desc, right_pt_desc, mask;

            // ORB_matches vector is relative to left/right ORB keypoints, not matched_keypts
            detect_match_point_features(left_image, right_image, left_matched_keypts_vec, right_matched_keypts_vec,
                                        left_pt_desc, right_pt_desc, orb_detector, pt_matcher);

            // --- line features ---
            vector<KeyLine> left_keyl_vec, right_keyl_vec, left_matched_keyl_vec, right_matched_keyl_vec;
            cv::Mat left_keyl_desc, right_keyl_desc;
            detect_match_line_features(left_image, right_image, left_matched_keyl_vec, right_matched_keyl_vec,
                                       left_keyl_desc, right_keyl_desc, line_descriptor, line_matcher);

            // --- filter bad matches ---
            vector<cv::Point2d> left_pt2d_vec = Keypoint_vec_2_Point2T_vec<double>(left_matched_keypts_vec);
            vector<cv::Point2d> right_pt2d_vec = Keypoint_vec_2_Point2T_vec<double>(right_matched_keypts_vec);
            filter_matches(left_pt2d_vec, right_pt2d_vec, left_pt_desc, right_pt_desc,
                           left_matched_keyl_vec, right_matched_keyl_vec, left_keyl_desc, right_keyl_desc,
                           left_image, right_image, visualize);

            // --- triangulation (points) ---
            cv::Mat P_l = K_l*cv::Mat::eye(3,4,CV_64F), P_r = K_r * T_l2r.clone().rowRange(0,3);
            cv::Mat left_pt_mat = Point2T_vec_2_Mat<double>(left_pt2d_vec);
            cv::Mat right_pt_mat = Point2T_vec_2_Mat<double>(right_pt2d_vec);
            cv::Mat tri_pt3d_mat;
            cv::triangulatePoints(P_l, P_r, left_pt_mat, right_pt_mat, tri_pt3d_mat);
            tri_pt3d_vec = Mat_2_Keypoint_vec<double>(tri_pt3d_mat);

            // --- triangulation (line) ---
            tri_l3d_vec = triangulate_lines(left_matched_keyl_vec, right_matched_keyl_vec);

            // --- match with previous frame (point) ---
            if(frame_vec.size() > 0){
                int last_frame_i = frame_vec.size()-1;
                vector<cv::Point2d> tracked_pt2d_l_vec, tracked_pt2d_r_vec;
                vector<int> tracked_idx = track_stereo(frame_vec[last_frame_i], left_image, right_image,
                        tracked_pt2d_l_vec, tracked_pt2d_r_vec, visualize);
            }

            // update database
            Frame new_frame = Frame(left_image, right_image, left_pt_desc, right_pt_desc, left_pt2d_vec, right_pt2d_vec, tri_pt3d_vec);
            frame_vec.push_back(new_frame);
            continue_track_flag = false;
            i++;
        }
        draw_3d_frame(tri_pt3d_vec, tri_l3d_vec);
        pangolin::FinishFrame();
    }
    return 0;
}
