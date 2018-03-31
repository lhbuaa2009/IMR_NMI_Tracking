#ifndef IMMATUREPOINT_H_
#define IMMATUREPOINT_H_

#include <common.h>
#include <DataType.h>

#include <Residual.h>

namespace DSLAM
{

class Frame;

struct ImmaturePointResidual
{

    ResState res_State;
    ResState res_NewState;

    double res_Energy;
    double res_NewEnergy;

    Frame* targetF;
};

enum ImmaturePointStatus
{
    IPS_GOOD = 0,
    IPS_OOB,
    IPS_OUTLIER,
    IPS_SKIPPED,
    IPS_BADCONDITION,
    IPS_UNINITIALIZED,
    IPS_CONVERGED
};



class ImmaturePoint
{

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ImmaturePoint(int u_, int v_, Frame* host_, CameraIntrinsic& camera);
    ~ImmaturePoint();

    ImmaturePointStatus traceOn(Frame* frame, const SE3& RefToFrame, const Mat3x3& RefToFrame_KRKi, const Vector3& RefToFrame_Kt, const CameraIntrinsic& camera);
    bool depthFromTriangulation(const SE3& RefToFrame, const Vector3& ref, const Vector3& cur, float& depth);
    bool updateDepthandSigma(const float& idepthNew, const float& sigmaNew);
    double linearizeResidual(const CameraIntrinsic& camera, const float outlierTH, ImmaturePointResidual* ImRes,
                             float& Hdd, float& bd, float idepth);

public:
    static int Num_TotalImmaturePoints;
    Frame* hostF;
    uchar levelFound;

    float color[patternNum];
    float weights[patternNum];

    Mat2x2 gradH;
    float grad;
    float energyTH;
    float u, v;

    float quality;
    float pointType;

    float sigma;
    float idepth;
    float idepth_min, idepth_max;

    float idepth_ref;       // debug variable with RGB-D datasets

    ImmaturePointStatus lastTraceStatus;
    Vector2 lastTraceUV;
    float lastTraceInterval;

    int GoodObservations;
    bool debugPrint;

};

}

#endif

