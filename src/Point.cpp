#include <Point.h>
#include <Frame.h>
#include <ImmaturePoint.h>
#include <Residual.h>


namespace DSLAM
{

int Point::Num_TotalPoints = 0;

Point::Point(const ImmaturePoint* candidate)
{
    Num_TotalPoints++;
    hostF = candidate->hostF;
    hasDepthPrior = false;

    idepth_hessian = 0;
    maxRelBaseline = 0;
    Num_GoodRes = 0;

    u = candidate->u;
    v = candidate->v;
    assert((candidate->idepth_min > 0)&&std::isfinite(candidate->idepth_max));

    levelFound = candidate->levelFound;
    setIdepth(candidate->idepth);
    setPointStatus(PointStatus::INACTIVE);

    int n = patternNum;
    memcpy(color, candidate->color, sizeof(float)*n);
    memcpy(weights, candidate->weights, sizeof(float)*n);
    energyTH = candidate->energyTH;
}


void Point::release()
{
    for(size_t i = 0; i < residuals.size(); i++)
        delete residuals[i];
    residuals.clear();
}

bool Point::isOOB(const std::vector<Frame*>& toKeep, const std::vector<Frame*>& toMarg)     // decide if this point should be margnalized or dropped
{
    int ResnumInToMarg = 0;
    for(Residual* res : residuals)
    {
        if(res->res_State != ResState::IN)
            continue;

        for(Frame* f : toMarg)
        {
            if(res->targetF == f)
                ResnumInToMarg++;        // good residuals in margnalized frames

            if(res->hostF == f)         // this point is in the margnalized frames
                return true;
        }
    }

    if(residuals.size() >= setting_minGoodResiduals &&                   // before margnalization, number of good residuals satisfy the demand for a good point
         Num_GoodRes >= setting_minGoodResiduals &&
         residuals.size() - ResnumInToMarg < setting_minGoodResiduals)   // after margnalization, number of good residuals cannot satisfy
    {
        return true;
    }

    if(last2residuals[0].second == ResState::OOB)  return true;      // cannot project into the latest keyframe
    if(residuals.size() < 2)  return false;                          // new point ?
    if(last2residuals[0].second == ResState::OUTLIER && last2residuals[1].second == ResState::OUTLIER) // both outliers in the latest two frames
        return true;

    return false;       // keep active in optimization structures, not margnalize or drop
}



}
