#ifndef TESTPIXELSELECTOR_H_
#define TESTPIXELSELECTOR_H_

#include <DataType.h>
#include <CameraIntrinsic.h>
#include <common.h>

enum PixelSelectorStatus {PIXSEL_VOID=0, PIXSEL_1, PIXSEL_2, PIXSEL_3};

class testPixelSelector
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    testPixelSelector(const cv::Mat& imGray, const cv::Mat& imDepth, const DSLAM::CameraIntrinsic& camera);
    ~testPixelSelector();

    void ConvertRawDepthImg(const cv::Mat& imDepth);
    void creatFramPyr();
    int makeMaps(float* map_out, float density,
                  int recursionsLeft=1, bool plot=false, float thFactor=1);
    int currentPotential;

    bool allowFast;
    void makeHists();

public:
    const static uchar PYR_LEVELS = 5;
    cv::Mat intensity[PYR_LEVELS];
    cv::Mat depth[PYR_LEVELS];

    Eigen::Vector2f* Grad_Int[PYR_LEVELS];
    Eigen::Vector2f* Grad_Dep[PYR_LEVELS];
    float* GradNorm[PYR_LEVELS];

    int width[PYR_LEVELS], height[PYR_LEVELS];
private:

    Eigen::Vector3i select(float* map_out, int pot, float thFactor=1);

    unsigned char* randomPattern;

    int* gradHist;
    float* ths;
    float* thsSmoothed;
    int thsStep;

    bool SelectedFlag;
//    const Frame* gradHistFrame;

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

#endif
