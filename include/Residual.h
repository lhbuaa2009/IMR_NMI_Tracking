#ifndef RESIDUAL_H_
#define RESIDUAL_H_

#include <common.h>
#include <DataType.h>


#include <CameraIntrinsic.h>
#include <Frame.h>
#include <Point.h>

namespace DSLAM
{

/*
enum ResState {IN=0, OOB, OUTLIER};
enum DistributionMode {TDistribution=0, NormalDistribution};
*/
struct ResidualJacobian
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    VecResPattern resInt;
    VecResPattern resIdp;

    Vector6 Jpdxi[2];
    Vector2 Jpdd;

    VecResPattern JIdp[2];
    VecResPattern Jddp[2];

    Vector6 J_transDepth_dxi;
    float J_transDepth_dd;

    Mat2x2 JIdp2;
    Mat2x2 Jddp2;

    float resInt_sigma;   // assume that the sigma(variance) is the same for this pattern
    float resIdp_sigma;

    float ResWeight;    // also assume that the weight(TDistribution) is the same for this pattern
};


class Residual
{

public:
   EIGEN_MAKE_ALIGNED_OPERATOR_NEW

   Residual();
   Residual(Point* point_, Frame* host_, Frame* target_);

   ~Residual();

   void calcuResSigma(const Mat6x6& Hessian_h, const Mat6x6& Hessian_t, const float& Hessian_idepth,
                      float& resIntSig, float& resIdpSig);
   void calcuResWeight(DistributionMode mode, int Dof, float& resWeight, float resInt, float resDep);
   double linearizeRes(CameraIntrinsic& camera, int level);
   void updateRes();

public:


   static int Num_TotalResiduals;
   bool isNew;
   DistributionMode ResDistribution;

   ResState res_State;
   ResState res_NewState;
   double res_Energy;
   double res_NewEnergy;
   double res_NewEnergyWithOutlier;

   Point* point;
   Frame* hostF;
   Frame* targetF;
   ResidualJacobian* Jac;

   Vector2 projectedPT[patternNum];
   Vector3 centerprojectedPT;

   inline void setState(ResState s) {res_State = s;}
   inline void resetOOB()
   {
       res_Energy = res_NewEnergy = 0;
       res_NewState = ResState::OUTLIER;

       setState(ResState::IN);
   }
};
}

#endif
