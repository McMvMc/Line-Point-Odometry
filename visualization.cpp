//
// Created by mike on 12/8/18.
//

using namespace std;
using namespace cv::line_descriptor;

bool continue_track_flag;


void draw_matches(const vector<cv::KeyPoint> & left_matched_keypts_vec, const vector<cv::KeyPoint> & right_matched_keypts_vec,
                  const cv::Mat & left_image, const cv::Mat & right_image, string title="filtered matches")
{
    vector<cv::DMatch> filtered_matches;
    for(int i=0; i<left_matched_keypts_vec.size(); i++)
    {
        filtered_matches.push_back(cv::DMatch(i, i, 0));
    }
    cv::Mat filtered_image;
    cv::drawMatches(left_image, left_matched_keypts_vec, right_image, right_matched_keypts_vec,
                    filtered_matches, filtered_image);
    cv::imshow(title, filtered_image);
    cv::waitKey(500);
}


void continue_track()
{
    continue_track_flag = true;
    cout<<"track current frame"<<endl;
}


void init_3d_visualization()
{
//    pangolin::CreateWindowAndBind("Map", 640, 480);
//    glEnable(GL_DEPTH_TEST);
//
//    // Define Projection and initial ModelView matrix
//    s_cam = pangolin::OpenGlRenderState(
//            pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 500),
//            pangolin::ModelViewLookAt(0, 2, 2, 0, 0, 0, pangolin::AxisY));
//
//    // Create Interactive View in window
//    pangolin::Handler3D handler(s_cam);
//    pangolin::RegisterKeyPressCallback(' ', continue_track);
}


void draw_3d_frame(const vector<cv::Point3d> & tri_pt3d_vec, vector<cv::Vec6d> & tri_l3d_vec)
{
    // draw point
    glPointSize(3.0);
    glBegin(GL_POINTS);
    glColor3f(1.0, 0, 0);
    for (size_t i = 0; i < tri_pt3d_vec.size(); i++)
    {
        glVertex3f(-tri_pt3d_vec[i].y, tri_pt3d_vec[i].z, -tri_pt3d_vec[i].x);
    }
    glEnd();
    glPointSize(1.0);

    glEnable(GL_LINE_SMOOTH);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);


    // draw line
    glEnable(GL_LINE_SMOOTH);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);

    glLineWidth(2.0);
    glBegin(GL_LINES);
    glColor3f(0.0, 1.0, 0);
    for (size_t i = 0; i < tri_l3d_vec.size(); i++)
//    for (size_t i = 0; i < min(6, int(tri_l3d_vec.size())); i++)
    {
        cv::Point3d pt1(tri_l3d_vec[i][0], tri_l3d_vec[i][1], tri_l3d_vec[i][2]);
        cv::Point3d pt2(tri_l3d_vec[i][3], tri_l3d_vec[i][4], tri_l3d_vec[i][5]);

        glVertex3f(-pt1.y, pt1.z, -pt1.x);
        glVertex3f(-pt2.y, pt2.z, -pt2.x);
    }
    glLineWidth(1.0);
    glEnd();
}


void show_line_matches(const cv::Mat & left_image, const cv::Mat & right_image,
                       const vector<KeyLine> & matched_l, const vector<KeyLine> & matched_r)
{
    cv::namedWindow("line matches");
    cv::Mat left_image_rgb, right_image_rgb;
    cv::cvtColor(left_image, left_image_rgb, cv::COLOR_GRAY2BGR);
    cv::cvtColor(right_image, right_image_rgb, cv::COLOR_GRAY2BGR);

    cv::Mat matches_image(left_image.rows, left_image.cols * 2, CV_8UC3);
    drawKeylines(left_image_rgb, matched_l, left_image_rgb);
    drawKeylines(right_image_rgb, matched_r, right_image_rgb);
    left_image_rgb.copyTo(matches_image.colRange(0, left_image.cols).rowRange(0, left_image.rows));
    right_image_rgb.copyTo(matches_image.colRange(right_image.cols, right_image.cols * 2).rowRange(0, right_image.rows));

    for (size_t i = 0; i < matched_l.size(); i++)
    {
        int l_mid_x = (matched_l[i].startPointX + matched_l[i].endPointX)/2;
        int l_mid_y = (matched_l[i].startPointY + matched_l[i].endPointY)/2;
        int r_mid_x = (matched_r[i].startPointX + matched_r[i].endPointX)/2;
        int r_mid_y = (matched_r[i].startPointY + matched_r[i].endPointY)/2;
        cv::Point2d left_mid_pt2d = cv::Point2d(l_mid_x, l_mid_y);
        cv::Point2d right_mid_pt2d = cv::Point2d(r_mid_x+left_image.cols, r_mid_y);

        cv::line(matches_image, left_mid_pt2d, right_mid_pt2d, CV_RGB(0, 0, 255), 1, CV_AA);
    }

    imshow("line matches", matches_image );
    cv::waitKey(500);
}



