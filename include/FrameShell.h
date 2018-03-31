#ifndef FRAMESHELL_H_
#define FRAMESHELL_H_

#include <DataType.h>

namespace DSLAM
{

class FrameShell
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    int id;
    static int next_id;
    double timestamp;

    SE3 FrameToTrackingRef;
    FrameShell* trackingRef;

    SE3 FrameToWorld;
    bool poseValid;

    int Num_OutlierRes;
    int Num_GoodRes;
    int marginalizedAt;

    inline FrameShell()
    {
        id = -1;
        poseValid = true;
        FrameToTrackingRef = SE3();
        timestamp = 0;
        marginalizedAt = -1;
        Num_GoodRes = Num_OutlierRes = 0;
        trackingRef = 0;
        FrameToWorld = SE3();
    }
};
}
#endif
