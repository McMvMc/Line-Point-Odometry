//
// Created by mike on 12/9/18.
//

using namespace cv::line_descriptor;

template <typename _Tp1, typename _Tp2>
float distancePointLine(const cv::Point_<_Tp1> point, const cv::Vec<_Tp2, 3> & line)
{
    //Line is given as a*x + b*y + c = 0
    return fabsf(line(0)*point.x + line(1)*point.y + line(2))
           / sqrt(line(0)*line(0)+line(1)*line(1));
}


cv::Point3d unproject_2d_point(const cv::Point2d pt, const cv::Mat & K, const double depth=1.0)
{
    cv::Mat pt_mat(cv::Point3d(pt.x, pt.y, 1));
    cv::Mat out = depth*K.inv()*pt_mat;
//    cout<<"K.inv() = "<<K.inv()<<endl;
//    cout<<"out = "<<out<<endl;
    return cv::Point3d(out);
}


cv::Vec4d unproject_2d_line(const KeyLine & keyl, const cv::Mat & K)
{
    cv::Point3d pt1(0,0,0);

    cv::Point3d pt2 = unproject_2d_point(keyl.getStartPoint(), K, 10);

    cv::Point3d pt3 = unproject_2d_point(keyl.getEndPoint(), K, 10);

    cv::Vec3d dir1 = pt2 - pt1;
    cv::Vec3d dir2 = pt3 - pt1;

    cv::Vec3d normal = dir1.cross(dir2);
    cv::Vec4d plane(normal[0], normal[1], normal[2], 0);

    return plane;
}


cv::Point3d line_plane_intersection(const cv::Mat & dir_pt, const cv::Mat & plane)
{
    cv::Mat dir = dir_pt.rowRange(0, 3);
    cv::Mat pt = dir_pt.rowRange(3, 6);
    cv::Mat norm = plane.rowRange(0, 3);
    double d = plane.at<double>(3);

    double x = -(d+pt.dot(norm)) / dir.dot(norm);

    cv::Mat inter_pt = pt + x*dir;
    return cv::Point3d(inter_pt);
}


cv::Point3d change_frame(cv::Mat T, cv::Point3d a)
{
    cv::Vec4d a_homo(a.x, a.y, a.z, 1);
    cv::Mat b_mat = T*cv::Mat(a_homo);
    cv::Vec4d b(b_mat);

    return cv::Point3d(b[0], b[1], b[2]);
}


// left is [I 0], right is [R t]
vector<cv::Vec6d> triangulate_lines(vector<KeyLine> keyl_vec_l, vector<KeyLine> keyl_vec_r)
{
    vector<cv::Vec6d> out_3d_line; // dir, pt
    for(int i=0; i<keyl_vec_l.size(); i++)
    {
        cv::Vec4d plane = unproject_2d_line(keyl_vec_l[i], K_l);

        cv::Point3d origin_r(0, 0, 0);
        cv::Point3d origin_r_frame_l = change_frame(T_l2r.inv(), origin_r);

        // ray 1
        cv::Point3d pt3d_1_frame_r = unproject_2d_point(keyl_vec_r[i].getStartPoint(), K_r, 10);
        cv::Point3d pt3d_1 = change_frame(T_l2r.inv(), pt3d_1_frame_r);
        cv::Point3d dir_1 = pt3d_1 - origin_r_frame_l;
        cv::Vec6d ray_1(dir_1.x, dir_1.y, dir_1.z, origin_r_frame_l.x, origin_r_frame_l.y, origin_r_frame_l.z); // pt = [0 0 0], dir = pt3d - pt

        // ray 2
        cv::Point3d pt3d_2_frame_r = unproject_2d_point(keyl_vec_r[i].getEndPoint(), K_r, 10);
        cv::Point3d pt3d_2 = change_frame(T_l2r.inv(), pt3d_2_frame_r);
        cv::Point3d dir_2 = pt3d_2 - origin_r_frame_l;
        cv::Vec6d ray_2(dir_2.x, dir_2.y, dir_2.z, origin_r_frame_l.x, origin_r_frame_l.y, origin_r_frame_l.z); // pt = [0 0 0], dir = pt3d - pt

        // endpoints
        cv::Point3d l_endpt_1 = line_plane_intersection(cv::Mat(ray_1), cv::Mat(plane));
        cv::Point3d l_endpt_2 = line_plane_intersection(cv::Mat(ray_2), cv::Mat(plane));

        out_3d_line.push_back(cv::Vec6d(l_endpt_1.x, l_endpt_1.y, l_endpt_1.z,
                                        l_endpt_2.x, l_endpt_2.y, l_endpt_2.z));
//        out_3d_line.push_back(cv::Vec6d(origin_r_frame_l.x, origin_r_frame_l.y, origin_r_frame_l.z,
//                                        pt3d_1.x, pt3d_1.y, pt3d_1.z));
//        out_3d_line.push_back(cv::Vec6d(origin_r_frame_l.x, origin_r_frame_l.y, origin_r_frame_l.z,
//                                        pt3d_2.x, pt3d_2.y, pt3d_2.z));
//
//        cv::Point3d pt2 = unproject_2d_point(keyl_vec_l[i].getStartPoint(), K_l, 10);
//        cv::Point3d pt3 = unproject_2d_point(keyl_vec_l[i].getEndPoint(), K_l, 10);
//        out_3d_line.push_back(cv::Vec6d(0, 0, 0,
//                                        pt2.x, pt2.y, pt2.z));
//        out_3d_line.push_back(cv::Vec6d(0, 0, 0,
//                                        pt3.x, pt3.y, pt3.z));
    }

    return out_3d_line;
}