#include <CameraIntrinsic.h>

namespace DSLAM
{
    CameraIntrinsic::CameraIntrinsic()
    {
        intrinsic = Eigen::Vector4f::Zero();
        distortion = Vector5f::Zero();
    }

    CameraIntrinsic::CameraIntrinsic(const Eigen::Vector4f intrin, const Vector5f distort)
    {
        intrinsic = intrin;
        distortion = distort;
    }

    CameraIntrinsic::CameraIntrinsic(const CameraIntrinsic &other)
    {
        intrinsic = other.intrinsic;
        distortion = other.distortion;
    }

    Eigen::Matrix3f CameraIntrinsic::ConvertToMatrix()
    {
        Eigen::Matrix3f K = Eigen::Matrix3f::Zero();
        K(0,0) = intrinsic(0);
        K(0,2) = intrinsic(2);
        K(1,1) = intrinsic(1);
        K(1,2) = intrinsic(3);
        K(2,2) = 1.0f;

        return K;
    }

    void CameraIntrinsic::scale(float factor)
    {
        intrinsic *= factor;
    }
}
