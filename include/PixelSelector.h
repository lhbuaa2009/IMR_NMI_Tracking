#ifndef PIXELSELECTOR_H_
#define PIXELSELECTOR_H_

#include <DataType.h>
#include <Frame.h>
#include <globalFuncs.h>

namespace DSLAM
{

enum PixelSelectorStatus {PIXSEL_VOID=0, PIXSEL_1, PIXSEL_2, PIXSEL_3};

class PixelSelector
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    PixelSelector(int w, int h);
    ~PixelSelector();

    int makeMaps(
            const Frame* const fh, float* map_out, float density,
            int recursionsLeft=1, bool plot=false, float thFactor=1);
    int currentPotential;

    bool allowFast;
    void makeHists(const Frame* const fh);
private:

    Eigen::Vector3i select(const Frame* const fh,
            float* map_out, int pot, float thFactor=1);

    unsigned char* randomPattern;

    int* gradHist;
    float* ths;
    float* thsSmoothed;
    int thsStep;
    const Frame* gradHistFrame;

    inline int computHistQuantity(int* hist, float below)
    {
        int th = hist[0] * below + 0.5f;
        for(int i = 0; i < 90; i++)
        {
            th -= hist[i+1];
            if(th < 0)
                return i;
        }
        return 90;
    }
};

}

#endif
