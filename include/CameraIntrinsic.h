#ifndef CAMERAINTRINSIC_H_
#define CAMERAINTRINSIC_H_

#include <Eigen/Core>

namespace DSLAM
{

typedef Eigen::Matrix<float, 5, 1> Vector5f;

struct CameraIntrinsic
{
EIGEN_MAKE_ALIGNED_OPERATOR_NEW
public:
    CameraIntrinsic();
    CameraIntrinsic(const Eigen::Vector4f intrin, const Vector5f distort);
    CameraIntrinsic(const CameraIntrinsic& other);

    Eigen::Matrix3f ConvertToMatrix();
    void scale(float factor);

    inline float get_fx() const
    {
        return intrinsic(0);
    }

    inline float get_fy() const
    {
        return intrinsic(1);
    }

    inline float get_cx() const
    {
        return intrinsic(2);
    }

    inline float get_cy() const
    {
        return intrinsic(3);
    }

public:
    Eigen::Vector4f intrinsic;
    Vector5f distortion;

};

}

#endif

