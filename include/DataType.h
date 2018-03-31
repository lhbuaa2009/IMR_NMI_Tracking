#ifndef DATATYPE_H_
#define DATATYPE_H_

#include <limits>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <sophus/sim3.hpp>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

namespace DSLAM
{

    #define MAX_RES_PER_POINT 8
    #define NUM_THREADS 6

    static const float Invalid = std::numeric_limits<float>::quiet_NaN();

    typedef float NumType;

    typedef Eigen::Matrix<NumType, 13, 13> Mat13x13;
    typedef Eigen::Matrix<NumType, 12, 12> Mat12x12;
    typedef Eigen::Matrix<NumType, 7, 7> Mat7x7;
    typedef Eigen::Matrix<NumType, 6, 6> Mat6x6;
    typedef Eigen::Matrix<NumType, 3, 3> Mat3x3;
    typedef Eigen::Matrix<NumType, 2, 2> Mat2x2;
    typedef Eigen::Matrix<NumType, 1, 2> Mat1x2;
    typedef Eigen::Matrix<NumType, 2, 6> Mat2x6;
    typedef Eigen::Matrix<NumType, 2, 7> Mat2x7;
    typedef Eigen::Matrix<NumType, 2, 12> Mat2x12;
    typedef Eigen::Matrix<NumType, 2, 13> Mat2x13;
    typedef Eigen::Matrix<NumType, Eigen::Dynamic, Eigen::Dynamic> MatXX;


    typedef Eigen::Matrix<NumType, 13, 1> Vector13;
    typedef Eigen::Matrix<NumType, 12, 1> Vector12;
    typedef Eigen::Matrix<NumType, 8, 1> Vector8;
    typedef Eigen::Matrix<NumType, 7, 1> Vector7;
    typedef Eigen::Matrix<NumType, 6, 1> Vector6;
    typedef Eigen::Matrix<NumType, 4, 1> Vector4;
    typedef Eigen::Matrix<NumType, 3, 1> Vector3;
    typedef Eigen::Matrix<NumType, 2, 1> Vector2;
    typedef Eigen::Matrix<NumType, Eigen::Dynamic, 1> VectorX;
    typedef Eigen::Matrix<NumType, MAX_RES_PER_POINT, 1> VecResPattern;
//    typedef Eigen::Matrix<NumType, MAX_RES_PER_POINT, 1> VecResIdepth;
    typedef Eigen::Matrix<NumType, 2*MAX_RES_PER_POINT, 1> VecRes;

    typedef Eigen::Matrix<double, 6, 1> Vector6d;
    typedef Eigen::Matrix<double, 7, 1> Vector7d;
    typedef Eigen::Matrix<double, 6, 6> Mat6x6d;


    typedef Sophus::SE3d SE3;
    typedef Sophus::Sim3d Sim3;
    typedef Sophus::SO3d SO3;

    typedef Eigen::Transform<NumType,3, Eigen::Affine> AffineTransform;

    typedef Eigen::Affine3d AffineTransformd;
    typedef Eigen::Matrix<double, 6, 1> Vector6d;
    typedef Eigen::Matrix<double, 6, 6> Matrix6d;

}

#endif
