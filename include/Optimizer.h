#ifndef OPTIMIZER_H_
#define OPTIMIZER_H_

#include <Frame.h>
#include <PixelSelector.h>
#include <PixelSelectorInPyr.h>

namespace DSLAM
{

class Optimizer
{
public:
    void static DirectOptimization(Frame* pKF1, Frame* pKF2, int nIterations = 5, const bool bRobust = true);  //binary edge
    void static DirectUnaryOptimization(const Frame* pKF1, const Frame* pKF2, const bool bRobust = true);
};

}

#endif
