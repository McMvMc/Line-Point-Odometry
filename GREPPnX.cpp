#include <opencv2/opencv.hpp>
#include <iostream>


void printMat(cv::Mat& m) {
    std::cout<<"R: "<<m.rows<<", C: "<<m.cols<<", T: "<<m.type()<<std::endl;
}


void mat_sub_vec(cv::Mat& X, cv::Mat& mX, cv::Mat& cX) {
    cX = X - cv::repeat(mX, 1, X.cols);
}


void normalize(cv::Mat& P, cv::Mat& Ls, cv::Mat& Le,
               cv::Mat& cP, cv::Mat& cLs, cv::Mat& cLe, cv::Mat& m) {
    m = cv::Mat::zeros(3, 1, CV_64F);
    cv::Mat sP;
    if (P.cols > 0) {
        cv::reduce(P, sP, 1, cv::REDUCE_SUM);
        m += sP*2;
    }
    if (Ls.cols > 0) {
        cv::reduce(Ls, sP, 1, cv::REDUCE_SUM);
        m += sP;
        cv::reduce(Le, sP, 1, cv::REDUCE_SUM);
        m += sP;
    }
    m /= (P.cols*2 + Ls.cols + Le.cols);
    if (P.cols > 0) {
        mat_sub_vec(P, m, cP);
    } else {
        cP = P;
    }

    if (Ls.cols > 0) {
        mat_sub_vec(Ls, m, cLs);
        mat_sub_vec(Le, m, cLe);
    } else {
        cLs = Ls;
        cLe = Le;
    }
}

void convert_endpoint_to_line(cv::Mat& s, cv::Mat& e, cv::Mat& l) {
    // a2b3-a3b2, a3b1-a1b3, a1b2-a2b1
    if (s.cols == 0) {
        l = cv::Mat::zeros(3, 0, CV_64F);
        return;
    }

    cv::Mat tmp;
    cv::vconcat(s.row(1) - e.row(1), e.row(0) - s.row(0), tmp);
    cv::vconcat(tmp, s.row(0).mul(e.row(1))-s.row(1).mul(e.row(0)), l);
    tmp = l.row(0).mul(l.row(0)) + l.row(1).mul(l.row(1));
    cv::sqrt(tmp, tmp);
    l /= cv::repeat(tmp, 3, 1);
}


double get_threshold(cv::Mat& err, double ratio=0.25) {
    int n = err.cols*ratio;
    cv::Mat tmp;
    cv::sort(err, tmp, CV_SORT_EVERY_ROW + CV_SORT_ASCENDING);
    return tmp.at<double>(0, n)*2;
}


double get_inliers(cv::Mat& M, cv::Mat& b, cv::Mat& Cw, cv::Mat& mP,
        cv::Mat& R, cv::Mat& T, cv::Mat& D) {
    cv::Mat Cc, res, err;

    Cc = R*Cw + cv::repeat(T + R*mP, 1, Cw.cols);
    Cc = Cc.reshape(1, 12);
    res = M*Cc;
    if (b.cols == 1) res -= b;
    res = res.mul(res);
    res = res.reshape(1, 2);
    cv::reduce(res, err, 0, cv::REDUCE_SUM);
    double t = get_threshold(err);
    D = err < fmax(t, 1e-5);
    return t;
}


void get_alpha(cv::Mat& P, cv::Mat& A) {
    cv::Mat sP, tmp;
    cv::reduce(P, sP, 0, cv::REDUCE_SUM);
    cv::vconcat(P, 1-sP, tmp);
    cv::transpose(tmp, A);
}


void centralize(cv::Mat& X, cv::Mat& cX, cv::Mat& mX) {
    cv::reduce(X, mX, 1, cv::REDUCE_AVG);
    mat_sub_vec(X, mX, cX);
}


void
orthogonal_polar_factor(cv::Mat& A, cv::Mat& R, double& s) {
    cv::Mat W, U, Vt;
    cv::SVDecomp(A, W, U, Vt);
    s = cv::determinant(U*Vt);
    U(cv::Range::all(), cv::Range(2,3)) *= s;
    R = U*Vt;
    s = cv::sum(W)[0];
}


void
isotropic_procrutes(cv::Mat& X, cv::Mat& Y, cv::Mat& R, double& s) {
    cv::Mat A, Xt;
    cv::transpose(X, Xt);
    A = Y*Xt;
    orthogonal_polar_factor(A, R, s);
    s /= cv::sum(X.mul(X))[0];
}


void
build_point_target(cv::Mat& P3, cv::Mat& P2, cv::Mat& M1, cv::Mat& M2) {
    int n = P3.cols;
    if(n == 0) {
        M1 = cv::Mat::zeros(0, 12, CV_64F);
        M2 = cv::Mat::zeros(0, 12, CV_64F);
        return;
    }
    cv::Mat A;
    get_alpha(P3, A);

    cv::Mat tmp[3];
    tmp[0] = A;
    tmp[1] = cv::Mat::zeros(A.rows, A.cols, CV_64F);
    tmp[2] = -A.mul(cv::repeat(P2.row(0).t(), 1, 4));
    cv::hconcat(tmp, 3, M1);

    tmp[0] = tmp[1];
    tmp[1] = A;
    tmp[2] = -A.mul(cv::repeat(P2.row(1).t(), 1, 4));
    cv::hconcat(tmp, 3, M2);
}


void
build_point_offset(cv::Mat& P2, cv::Mat& Po, cv::Mat& T1, cv::Mat& T2) {
    int n = P2.cols;
    if(n == 0) {
        T1 = cv::Mat::zeros(0, 1, CV_64F);
        T2 = cv::Mat::zeros(0, 1, CV_64F);
        return;
    }
    cv::transpose(P2.row(0).mul(Po.row(2)) - Po.row(0), T1);
    cv::transpose(P2.row(1).mul(Po.row(2)) - Po.row(1), T2);

}


void
build_line_target(cv::Mat& Ls3, cv::Mat& Le3, cv::Mat& l, cv::Mat& M1, cv::Mat& M2) {
    int n = Ls3.cols;
    if(n == 0) {
        M1 = cv::Mat::zeros(0, 12, CV_64F);
        M2 = cv::Mat::zeros(0, 12, CV_64F);
        return;
    }
    cv::Mat A1, A2, lp;
    get_alpha(Ls3, A1);
    get_alpha(Le3, A2);

    cv::transpose(l, lp);
    lp = cv::repeat(lp.reshape(1, n*3), 1, 4);
    lp = lp.reshape(1, n);
    M1 = lp.mul(cv::repeat(A1, 1, 3));
    M2 = lp.mul(cv::repeat(A2, 1, 3));
}


void
build_line_offset(cv::Mat& l, cv::Mat& Lo, cv::Mat& T1, cv::Mat& T2) {
    int n = l.cols;
    if(n == 0) {
        T1 = cv::Mat::zeros(0, 1, CV_64F);
        T2 = cv::Mat::zeros(0, 1, CV_64F);
        return;
    }

    cv::Mat A;
    cv::reduce(l.mul(Lo), A, 0, cv::REDUCE_SUM);
    cv::transpose(-A, T1);
    T2 = T1;
}


void
prepare_data(cv::Mat& P3, cv::Mat& P2,
        cv::Mat& Ls3, cv::Mat& Le3, cv::Mat& l,
        cv::Mat& M, cv::Mat& b, cv::Mat& Cw) {
    Cw = cv::Mat::eye(3, 4, CV_64F);
    cv::Mat Ms[4];

    build_point_target(P3, P2, Ms[0], Ms[2]);
    build_line_target(Ls3, Le3, l, Ms[1], Ms[3]);
    cv::vconcat(Ms, 4, M);

    b = cv::Mat::zeros(M.rows, 0, CV_64F);
}

void
prepare_data(cv::Mat& P3, cv::Mat& P2, cv::Mat& Po,
        cv::Mat& Ls3, cv::Mat& Le3, cv::Mat& l, cv::Mat& Lo,
        cv::Mat& M, cv::Mat& b, cv::Mat& Cw) {
    Cw = cv::Mat::eye(3, 4, CV_64F);
    cv::Mat Ms[4];
    cv::Mat Ts[4];

    build_point_target(P3, P2, Ms[0], Ms[2]);
    build_line_target(Ls3, Le3, l, Ms[1], Ms[3]);
    cv::vconcat(Ms, 4, M);

    build_point_offset(P2, Po, Ts[0], Ts[2]);
    build_line_offset(l, Lo, Ts[1], Ts[3]);
    cv::vconcat(Ts, 4, b);
}


void
kernel_noise(cv::Mat& M, cv::Mat& b, int dims, cv::Mat& Km, cv::Mat& Cc) {
    cv::Mat W, U, Vt;
    cv::SVDecomp(M, W, U, Vt);
    cv::transpose(Vt(cv::Range(8,12),cv::Range::all()), Km);

    if (b.cols == 1) {
        Cc = ((b.t() * U)/W.t()) * Vt;
        Cc = Cc.reshape(1, 3);
    } else {
        Cc = cv::Mat::zeros(3, 0, CV_64F);
    }
}

double
kernel_PnP(cv::Mat& Cw, cv::Mat& Cc, cv::Mat& Km, cv::Mat& R, cv::Mat& T) {

    cv::Mat X, cX, mX, Y, cY, mY, ncY, nmY, Kmt, res;
    double s, err, nerr;

    cv::transpose(Km, Kmt);
    X = Cw;
    centralize(X, cX, mX);

    if (Cc.cols != 0) {
        Y = Cc.clone();
    } else {
        Y = Kmt.row(3).clone().reshape(1, 3);
    }
    // observations are assumed to be in front of the camera
    if (cv::sum(Y.row(2))[0] < 0) {
        Y = -Y;
    }
    centralize(Y, cY, mY);
    isotropic_procrutes(cX, cY, R, s);

    for (int i = 0; i < 10; i++) {
        Y = R * cX + cv::repeat(mY/s, 1, cX.cols);

        // project onto the effective null space
        if (Cc.cols != 0) Y -= Cc;
        Y = Y.reshape(1, 12);
        Y = Km*(Kmt*Y);
        Y = Y.reshape(1, 3);
        if (Cc.cols != 0) Y += Cc;

        centralize(Y, ncY, nmY);
        res = R*cX - ncY;
        nerr = sqrt(cv::sum(res.mul(res))[0]);

        if (i > 1 && nerr > err*0.95) break;
        err = nerr;

        centralize(Y, cY, mY);
        isotropic_procrutes(cX, cY, R, s);
    }

    T = mY/s - R*mX;
    return err;
}


void
test_func() {

    cv::Mat Km, Kmt, Y, Cc;

    Km = cv::Mat::ones(12, 4, CV_64F);
    Kmt = Km.row(2).mul(Km.row(3));
    printMat(Kmt);
}

double
solve_kernel_PnP(cv::Mat&M, cv::Mat& b, cv::Mat& Cw, cv::Mat& mP,
        cv::Mat& R, cv::Mat& T) {
    cv::Mat Km, Cc, Cc2;
    kernel_noise(M, b, 4, Km, Cc);
    double err = kernel_PnP(Cw, Cc, Km, R, T);
    T -= R * mP;
    return err;
}

double
solve_robust_kernel_PnP(cv::Mat&M, cv::Mat& b, cv::Mat& Cw, cv::Mat& mP,
        cv::Mat& R, cv::Mat& T) {
    int n = M.rows/2, inlier_count;
    float err, t;
    cv::Mat D, Df, D2, inliers, sM, sb;
    D = cv::Mat::ones(1, n, CV_8U);

    for (int i = 0; i < 30; i++) {
        D.convertTo(Df, CV_64F);
        cv::hconcat(Df, Df, inliers);
        inliers = inliers.reshape(1, n*2);

        sM = M.mul(cv::repeat(inliers, 1, 12));
        if (b.cols == 1) {
            sb = b.mul(inliers);
        }
        err = solve_kernel_PnP(sM, sb, Cw, mP, R, T);
        t = get_inliers(M, b, Cw, mP, R, T, D2);

        if (cv::sum(D!=D2)[0] == 0) break;
        inlier_count = cv::sum(D2)[0];
        if (inlier_count < fmax(n*0.1, 6)) break;
        D = D2.clone();
    }

    return err;
}


double
REPPnX(cv::Mat& R, cv::Mat& T,
       cv::Mat& P3, cv::Mat& P2,
       cv::Mat& Ls3, cv::Mat& Le3, cv::Mat& Ls2, cv::Mat& Le2,
       bool robust_kernel=false) {
    // R: 3x3 matrix
    // T: 3x1 matrix
    // P3: 3xn, points in world coordinate
    // P2: 2xn, points on image
    // Ls3: 3xn, starting point of line segments in world coordinate
    // Le3: 3xn, ending point of line segments in world coordinate
    // Ls2: 2xn, starting point of line segments on image
    // Le2: 2xn, ending point of line segments on image
    int n = P3.cols + Ls3.cols;
    if (n < 6) {
        R = cv::Mat::eye(3, 3, CV_64F);
        T = cv::Mat::zeros(3, 1, CV_64F);
        return 1e5;
    }

    cv::Mat cP, cLs, cLe, mP, l;
    normalize(P3, Ls3, Le3, cP, cLs, cLe, mP);
    convert_endpoint_to_line(Ls2, Le2, l);

    cv::Mat M, b, Cw;
    prepare_data(cP, P2, cLs, cLe, l, M, b, Cw);
    double err;
    if (robust_kernel) {
        err = solve_robust_kernel_PnP(M, b, Cw, mP, R, T);
    } else {
        err = solve_kernel_PnP(M, b, Cw, mP, R, T);
    }
    return err;

}


double
GREPPnX(cv::Mat& R, cv::Mat& T,
        cv::Mat& P3, cv::Mat& P2, cv::Mat& Po,
        cv::Mat& Ls3, cv::Mat& Le3, cv::Mat& Ls2, cv::Mat& Le2, cv::Mat& Lo,
        bool robust_kernel=false) {
    // R: 3x3 matrix
    // T: 3x1 matrix
    // P3: 3xn, points in world coordinate
    // P2: 2xn, points on image
    // Ls3: 3xn, starting point of line segments in world coordinate
    // Le3: 3xn, ending point of line segments in world coordinate
    // Ls2: 2xn, starting point of line segments on image
    // Le2: 2xn, ending point of line segments on image
    int n = P3.cols + Ls3.cols;
    if (n < 6) {
        R = cv::Mat::eye(3, 3, CV_64F);
        T = cv::Mat::zeros(3, 1, CV_64F);
        return 1e5;
    }

    cv::Mat cP, cLs, cLe, mP, l;
    normalize(P3, Ls3, Le3, cP, cLs, cLe, mP);
    convert_endpoint_to_line(Ls2, Le2, l);

    cv::Mat M, b, Cw;
    prepare_data(cP, P2, Po, cLs, cLe, l, Lo, M, b, Cw);

    double err;
    if (robust_kernel) {
        err = solve_robust_kernel_PnP(M, b, Cw, mP, R, T);
    } else {
        err = solve_kernel_PnP(M, b, Cw, mP, R, T);
    }
    return err;

}



