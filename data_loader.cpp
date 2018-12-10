//
// Created by mike on 12/8/18.
//

using namespace std;

void read_image_fn_vec(string left_csv_path, string right_csv_path,
                       vector<string> & left_image_fn_vec, vector<string> & right_image_fn_vec)
{
    string tmp_csv_line = "";

    // read image file names
    ifstream left_csv_file(left_csv_path);
    getline(left_csv_file, tmp_csv_line);
    while (getline(left_csv_file, tmp_csv_line)) {
        size_t start_pos = tmp_csv_line.find(",")+1;
        size_t end_pos = tmp_csv_line.find("\r");
        string tmp_fn = tmp_csv_line.substr (start_pos, end_pos-start_pos);
        left_image_fn_vec.push_back(tmp_fn);
    }

    ifstream right_csv_file(right_csv_path);
    getline(right_csv_file, tmp_csv_line);
    while (getline(right_csv_file, tmp_csv_line)) {
        size_t start_pos = tmp_csv_line.find(",")+1;
        size_t end_pos = tmp_csv_line.find("\r");
        string tmp_fn = tmp_csv_line.substr (start_pos, end_pos-start_pos);
        right_image_fn_vec.push_back(tmp_fn);
    }

    return;
}


void read_image_pair(string left_image_path, string right_image_path, cv::Mat & left_image, cv::Mat & right_image)
{
    cv::Mat left_image_raw = cv::imread(left_image_path, CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat right_image_raw = cv::imread(right_image_path, CV_LOAD_IMAGE_GRAYSCALE);

//    cv::imshow("before undistort left", left_image_raw);
//    cv::waitKey(100);
//    cv::imshow("before undistort right", right_image_raw);
//    cv::waitKey(0);
//    cout<<"dist_coeff_l = "<<dist_coeff_l<<endl;
//    cout<<"dist_coeff_r = "<<dist_coeff_r<<endl;
    undistort(left_image_raw, left_image, K_l, dist_coeff_l);
    undistort(right_image_raw, right_image, K_r, dist_coeff_r);
//    cv::imshow("after undistort left", left_image);
//    cv::waitKey(0);
//    cv::imshow("after undistort right", right_image);
//    cv::waitKey(0);
    return;
}
