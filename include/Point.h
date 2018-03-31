#ifndef POINT_H_
#define POINT_H_

#include <common.h>
#include <DataType.h>


namespace DSLAM
{

class Frame;
class ImmaturePoint;
class Residual;

class Point
{

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Point();
    Point(const ImmaturePoint* candidate);

    void release();
    inline ~Point()
    {
        release();
        Num_TotalPoints--;
    }

public:
    static int Num_TotalPoints;

    enum PointStatus { ACTIVE=0, INACTIVE, OUTLIER, OOB, MARGINALIZED };
    PointStatus status;
    Frame* hostF;

    float color[patternNum];
    float weights[patternNum];

    float u, v;
    Vector3 Normal;
    int idx;
    float energyTH;
    bool hasDepthPrior;

    uchar levelFound;

    float depth_zero;
    float depth;
    float depth_backup;

    float idepth_zero;
    float idepth;
    float idepth_backup;

    float step;
    float step_backup;

    float nullspaces;
    float idepth_hessian;
    float maxRelBaseline;

    int Num_GoodRes;

    inline void setPointStatus(PointStatus s)
    { status = s; }

    inline void setIdepth(float idepth)
    {
        this->idepth = idepth;
        this->depth = 1.0f/(idepth);
    }

    inline void setIdepthZero(float idepth)
    {
        idepth_zero = idepth;
        depth_zero = 1.0f/(idepth);
    }

    std::vector<Residual*> residuals;
    std::pair<Residual*, ResState> last2residuals[2];

    inline bool isValid()    // before margnalization, this is good/inlier point or not
    {
        if(residuals.size() >= setting_minGoodResiduals &&               // before margnalization, number of good residuals satisfy the demand for a good point
                Num_GoodRes >= setting_minGoodResiduals)
            return true;

        return false;
    }


    bool isOOB(const std::vector<Frame*>& toKeep, const std::vector<Frame*>& toMarg);     // decide if this point should be margnalized or dropped

};
}

#endif
