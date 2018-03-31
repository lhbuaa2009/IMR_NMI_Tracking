#ifndef INTIALIZER_H_
#define INTIALIZER_H_

#include <common.h>
#include <DataType.h>
#include <Frame.h>
#include <PixelSelector.h>
#include <PixelSelectorInPyr.h>
#include <MatrixAccumulators.h>
#include <nanoflann.h>

namespace DSLAM
{

struct Pnt
{

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    float u, v;
    float idepth, depth;
    float idepth_new;
    float energy, energy_new;

    float idepth_0, depth_0;  // debug parameters, store raw depth values

    bool isGood;
    bool isGood_new;

    float lastHessian;
    float lastHessian_new;

    float alpha;              // debug parameter
    float idepth_prior;       // prior information of idepth
    float Hessian_prior;      // regularization term: (1/sigma)*(idepth - idepth_prior)2   ???
    float energyReg, energyReg_new;    // energy in regularization term

    float maxstep;   // max stepsize for idepth according to max movement in pixel space

    // index of closest point which is chosen on one pyramid level above
    int parentIdx;
    float parentDist;

    int childrenIdx[4];
    float childrenDist[4];

    float levelFound;
    float outlierTH;

};

class Initializer
{

public:    
   EIGEN_MAKE_ALIGNED_OPERATOR_NEW

   Initializer(int w, int h);
   ~Initializer();

   void setFirst(Frame* frame);
   bool trackFrame(Frame* frame);
   void calcGrads(Frame* frame);

   int frameID;
   bool printDebug;   
   bool RegFlag;

   Pnt* points[PYR_LEVELS];
   int numPoints[PYR_LEVELS];
   SE3 thisToNext;

   Frame* firstFrame;
   Frame* newFrame;

   Vector6* dataFirst[PYR_LEVELS];
   Vector6* dataNew[PYR_LEVELS];

private:
   Mat3x3 K[PYR_LEVELS];
   Mat3x3 Ki[PYR_LEVELS];
   double fx[PYR_LEVELS];
   double fy[PYR_LEVELS];
   double fxi[PYR_LEVELS];
   double fyi[PYR_LEVELS];
   double cx[PYR_LEVELS];
   double cy[PYR_LEVELS];
   double cxi[PYR_LEVELS];
   double cyi[PYR_LEVELS];
   int w[PYR_LEVELS];
   int h[PYR_LEVELS];

   void makePyramid(const CameraIntrinsic* camera);

   Vector8* JbBuffer;      // Jd * [Jxi, res, Jd]
   Vector8* JbBuffer_new;

   Accumulator7 acc7;
   Accumulator7 acc7SC;

   Vector2 calcResAndHessian(int level, Mat6x6& H, Vector6& b, Mat6x6& Hsc, Vector6& bsc,
                             const SE3& refToNew, bool plot);
   Vector3 calcEnergy(int level);
   void optReg(int level);

   void propagateUp(int srcLvl);
   void propagateDown(int srcLvl);

   void resetPoints(int level);
   void doStep(int level, float lamba, Vector6 inc);
   void applyStep(int level);
   void debugPlot(int level);
   void makeNN();

};

struct FLANNPointcloud
{
    inline FLANNPointcloud() {num=0; points=0;}
    inline FLANNPointcloud(int n, Pnt* p) :  num(n), points(p) {}
    int num;
    Pnt* points;
    inline size_t kdtree_get_point_count() const { return num; }
    inline float kdtree_distance(const float *p1, const size_t idx_p2,size_t /*size*/) const
    {
        const float d0=p1[0]-points[idx_p2].u;
        const float d1=p1[1]-points[idx_p2].v;
        return d0*d0+d1*d1;
    }

    inline float kdtree_get_pt(const size_t idx, int dim) const
    {
        if (dim==0) return points[idx].u;
        else return points[idx].v;
    }
    template <class BBOX>
        bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }
};

}

#endif
