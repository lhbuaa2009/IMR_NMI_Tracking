#ifndef FRAME_H_
#define FRAME_H_

#include <common.h>
#include <DataType.h>
#include <FrameShell.h>
#include <CameraIntrinsic.h>

#include <Point.h>
#include <ImmaturePoint.h>

namespace DSLAM
{

struct FFPrecalc
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    static int instanceCounter;
    Frame* host;
    Frame* target;

    Mat3x3 PRE_Rot;
    Vector3 PRE_Tra;

    Mat3x3 PRE_Rot0;
    Vector3 PRE_Tra0;

    Mat3x3 PRE_KRotKi;
    Mat3x3 PRE_RotKi;
    Vector3 PRE_KTra;

    float distance;

    inline FFPrecalc() { host = target = 0; }
    inline ~FFPrecalc() {}

    void set(Frame* hostF, Frame* targetF, CameraIntrinsic& camera);
};

class Frame
{

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Frame();
    Frame(const cv::Mat& imGray, const cv::Mat& imDepth, const double& Timestamp, CameraIntrinsic& camera);
    Frame(const Frame& frame);

    ~Frame();
    void ConvertRawDepthImg(const cv::Mat& imDepth);
    void creatFramPyr();
    inline void calculateNormalandAngle(float u, float v, int level, CameraIntrinsic& camera,
                                 Vector3& Normal, float& Angle)
    {
        float* dep = depth[level].ptr<float>();
        int widt = width[level];
        Eigen::Matrix3f cam = camera.ConvertToMatrix();

        int u0 = (int)std::floor(u);
        int v0 = (int)std::floor(v);
        int idx = u0 + widt * v0;
        Vector3 backProjectPoint[4];

        backProjectPoint[0] = dep[idx] * cam.inverse() * Vector3(u0, v0, 1);
        backProjectPoint[1] = dep[idx+1] * cam.inverse() * Vector3(u0+1, v0, 1);
        backProjectPoint[2] = dep[idx+widt] * cam.inverse() * Vector3(u0, v0+1, 1);
        backProjectPoint[3] = dep[idx+widt+1] * cam.inverse() * Vector3(u0+1, v0+1, 1);

        Normal = (backProjectPoint[3] - backProjectPoint[0]).cross(backProjectPoint[2] - backProjectPoint[1]);
        Normal.normalize();

        Angle = std::acos(std::fabs(Normal[2]));
    }


    inline void calcuDepthSigma(int type, float z, float angle, float& sigma)
    {
        if(type == 0)   // select kinect 1.0 error model: (m/fb)(cm) * deltad(pixel) * z(m)

            sigma = 2.85 * 1e-5 * 0.5 * (1e4*z*z);   //cm

        else if(type == 1) // select kinect 2.0 error model: 1.5 - 0.5*z + 0.3*z*z + 0.1*sqrt(z*z*z)*theta*theta/(0.5*pi-theta)2
        {
            sigma = 1.5 - 0.5*z + 0.3*z*z + 0.1*sqrt(z*z*z)*angle*angle/((0.5*M_PI-angle)*(0.5*M_PI-angle));  //mm
            sigma = 0.1 * sigma;  //cm
        }
        else
        {
            std::cout << "Error model dosen't exist!" << std::endl;
            return;
        }
        return;
    }

public:

    CameraIntrinsic* intrinsic;
    double timestamp;

    FrameShell* shell;
    cv::Mat intensity[PYR_LEVELS];
    cv::Mat depth[PYR_LEVELS];

    Eigen::Vector2f* Grad_Int[PYR_LEVELS];
    Eigen::Vector2f* Grad_Dep[PYR_LEVELS];
    float* GradNorm[PYR_LEVELS];

    int width[PYR_LEVELS], height[PYR_LEVELS];

    static int Num_TotalFrames;
    int frameID;                   // index in allKeyFrameshistory, incrementally
    int idx;                       // index in activeKeyframes, incrementally
    int abIDx;                     // absolute index in datasets

    float frameEnergyTH;
    bool flag_Marginalization;

    std::vector<Point*> GoodPoints;
    std::vector<Point*> MarginalizedPoints;
    std::vector<Point*> OutPoints;
    std::vector<ImmaturePoint*> ImmaturePoints;

    Mat6x6d NullSpace_Pose;

    SE3 worldToFram_evalPT;
    Vector6d state_zero;
    Vector6d state;
    Vector6d state_backup;
    Vector6d iter_step;
    Vector6d iter_step_backup;

    Mat6x6 Hessian_state;

    EIGEN_STRONG_INLINE const SE3 &get_worldToFram_evalPT() const {return worldToFram_evalPT;}
    EIGEN_STRONG_INLINE const Vector6d &get_state_zero() const {return state_zero;}
    EIGEN_STRONG_INLINE const Vector6d &get_state() const {return state;}
    EIGEN_STRONG_INLINE const Vector6d get_state_minus_stateZero() const {return get_state() - get_state_zero();}

    SE3 PRE_worldToFram;
    SE3 PRE_FramToworld;
    std::vector<FFPrecalc, Eigen::aligned_allocator<FFPrecalc>> TargetPrecalc;

    void setNullSpace();
    inline void setState_zero(const Vector6d& state_zero_)
    {
        state_zero = state_zero_;
    }

    inline void setState(const Vector6d& state_)
    {
        state = state_;
        PRE_worldToFram = SE3::exp(state_)*get_worldToFram_evalPT();
        PRE_FramToworld = PRE_worldToFram.inverse();
    }

    inline void setEvaPT(const SE3& worldToFram_evalPT_)
    {
        worldToFram_evalPT = worldToFram_evalPT_;
        setNullSpace();
    }


};

}

#endif
