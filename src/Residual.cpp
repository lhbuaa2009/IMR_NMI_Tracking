#include <globalFuncs.h>
#include <Eigen/Dense>

#include <Residual.h>



namespace DSLAM
{
    int Residual::Num_TotalResiduals = 0;

    Residual::Residual() { Num_TotalResiduals++; }
    Residual::Residual(Point *point_, Frame *host_, Frame *target_):
        point(point_), hostF(host_), targetF(target_)
    {
        Num_TotalResiduals++;
        resetOOB();
        Jac = new ResidualJacobian();
        assert( ((long)Jac)%16 == 0);

        isNew = true;
    }

    Residual::~Residual()
    {
        Num_TotalResiduals--;
        delete Jac;
    }

    double Residual::linearizeRes(CameraIntrinsic& camera, int level)
    {
        res_NewEnergyWithOutlier = -1;

        if( res_State == ResState::OOB)
        {
            res_NewState = ResState::OOB;
            return res_Energy;
        }

        const float* const intensity = point->color;
        const float* const weights = point->weights;

        FFPrecalc* precalc = &(hostF->TargetPrecalc[targetF->idx]);
        float energyLeft = 0;

        cv::Mat imgData = targetF->intensity[level];
        cv::Mat depData = targetF->depth[level];
        Eigen::Vector2f* imgGrad = targetF->Grad_Int[level];
        Eigen::Vector2f* depGrad = targetF->Grad_Dep[level];
        Eigen::Vector2i bound(targetF->width[level], targetF->height[level]);

        const Mat3x3 PRE_KRotKi = precalc->PRE_KRotKi;
        const Vector3 PRE_KTra = precalc->PRE_KTra;
        const Mat3x3 PRE_Rot0 = precalc->PRE_Rot0;
        const Vector3 PRE_Tra0 = precalc->PRE_Tra0;

        float fx = camera.get_fx(); float fy = camera.get_fy();
        float new_idepth;
        Vector3 normal; float angle;
        {
            Vector6 Jpdxi_x, Jpdxi_y, Jzdxi;
            float Jpdd_x, Jpdd_y, Jzdd;
            float idepth_ratio, u, v;
            float Ku, Kv;
            Vector3 TransformPT;

            if(!projectPoint(point->u, point->v, point->idepth_zero, 0, 0, camera,
                        PRE_Rot0, PRE_Tra0, idepth_ratio, u, v, Ku, Kv, new_idepth, bound))
            {
                res_NewState = ResState::OOB;
                return res_Energy;
            }

            centerprojectedPT = Vector3(Ku, Kv, new_idepth);
            targetF->calculateNormalandAngle(Ku, Kv, 0, camera, normal, angle);

            TransformPT = (1.0f/new_idepth) * Vector3(u, v, 1.0f);

            Jpdd_x = idepth_ratio * (PRE_Tra0[0] - PRE_Tra0[2]*u) * fx;
            Jpdd_y = idepth_ratio * (PRE_Tra0[1] - PRE_Tra0[2]*v) * fy;

            Jpdxi_x[0] = new_idepth * fx;
            Jpdxi_x[1] = 0;
            Jpdxi_x[2] = -new_idepth * u * fx;
            Jpdxi_x[3] = -u * v * fx;
            Jpdxi_x[4] = (1 + u*u) * fx;
            Jpdxi_x[5] = -v * fx;

            Jpdxi_y[0] = 0;
            Jpdxi_y[1] = new_idepth * fy;
            Jpdxi_y[2] = -new_idepth * v * fy;
            Jpdxi_y[3] = -(1 + v*v) * fy;
            Jpdxi_y[4] = u * v * fy;
            Jpdxi_y[5] = u * fy;

            Jac->Jpdxi[0] = Jpdxi_x; Jac->Jpdxi[1] = Jpdxi_y;
            Jac->Jpdd[0] = Jpdd_x; Jac->Jpdd[1] = Jpdd_y;

            Jzdxi[0] = Jzdxi[1] = 0;
            Jzdxi[2] = 1;
            Jzdxi[3] = TransformPT[1];
            Jzdxi[4] = -TransformPT[0];
            Jzdxi[5] = 0;

            Jzdd = -(TransformPT[2] - PRE_Tra0[2]) * (1.0f/point->idepth_zero);

            Jac->J_transDepth_dxi = Jzdxi; Jac->J_transDepth_dd = Jzdd;
        }

        float JIdp2_00, JIdp2_10, JIdp2_11;
        float Jddp2_00, Jddp2_10, Jddp2_11;
        float SumGrad = 0;
        for(int idx = 0; idx < patternNum; idx++)
        {
            float Ku, Kv;
            float sigma;
            float transformDepth;
            if(!projectPoint(point->u+pattern[idx][0], point->v+pattern[idx][1], point->idepth, camera, transformDepth, PRE_KRotKi, PRE_KTra, Ku, Kv, bound))
                {
                    res_NewState = ResState::OOB;
                    return res_Energy;
                }
            projectedPT[idx] = Vector2(Ku, Kv);

            Vector6 hitData = bilinearInterpolationWithDepthBuffer(imgData, depData, imgGrad, depGrad, Ku, Kv, transformDepth);
            if( std::isnan(hitData[0]) )
            {
                res_NewState = ResState::OOB;
                return res_Energy;
            }

            float res_int = 1.0f/255 * (hitData[0] - intensity[idx]);  // scale intensity error in order to match depth error
            float res_dep = hitData[1] - transformDepth;               // unit: meter

            targetF->calcuDepthSigma(0, hitData[1], angle, sigma);  // uncetainty of depth observations
            if(fabs(res_dep) > 5 * sigma * 0.01)
            {
                res_NewState = ResState::OOB;
                return res_Energy;
            }

            {

               Jac->JIdp[0][idx] = hitData[2];
               Jac->JIdp[1][idx] = hitData[3];
               Jac->Jddp[0][idx] = hitData[4];
               Jac->Jddp[1][idx] = hitData[5];

               if(idx == 0)
               {
                   calcuResSigma(hostF->Hessian_state, targetF->Hessian_state, point->idepth_hessian, Jac->resInt_sigma, Jac->resIdp_sigma);
                   calcuResWeight(DistributionMode::TDistribution, 5, Jac->ResWeight, res_int, res_dep);
               }


               float residual = sqrt( Jac->ResWeight * (Jac->resInt_sigma*res_int*res_int + Jac->resIdp_sigma*res_dep*res_dep) );  // Approxiamtion
               float hw = fabsf(residual) < setting_huberTH2 ? 1 : setting_huberTH2 / fabsf(residual);

               Jac->resInt[idx] = hw * res_int;
               Jac->resIdp[idx] = hw * res_dep;
               energyLeft += hw * residual * residual * (2-hw);

               SumGrad += hw * hw *(hitData[2] * hitData[2] + hitData[3] * hitData[3]);

               JIdp2_00 += hitData[2] * hitData[2];
               JIdp2_11 += hitData[3] * hitData[3];
               JIdp2_10 += hitData[2] * hitData[3];

               Jddp2_00 += hitData[4] * hitData[4];
               Jddp2_11 += hitData[5] * hitData[5];
               Jddp2_10 += hitData[4] * hitData[5];
            }

        }

        Jac->JIdp2(0,0) = JIdp2_00; Jac->JIdp2(1,1) = JIdp2_11;
        Jac->JIdp2(0,1) = Jac->JIdp2(1,0) = JIdp2_10;

        Jac->Jddp2(0,0) = Jddp2_00; Jac->Jddp2(1,1) = Jddp2_11;
        Jac->Jddp2(0,1) = Jac->Jddp2(1,0) = Jddp2_10;

        res_NewEnergyWithOutlier = energyLeft;

        if(energyLeft > std::max<float>(hostF->frameEnergyTH, targetF->frameEnergyTH) || SumGrad < 2)
        {
            energyLeft = std::max<float>(hostF->frameEnergyTH, targetF->frameEnergyTH);
            res_NewState = ResState::OUTLIER;  // set up new state of residual
        }
        else
        {
            res_NewState = ResState::IN;     // set up new state of residual
        }

        res_NewEnergy = energyLeft;
        return energyLeft;
    }

    void Residual::updateRes()
    {
        setState(res_NewState);
        res_NewEnergy = res_Energy;
    }

    void Residual::calcuResSigma(const Mat6x6& Hessian_h, const Mat6x6& Hessian_t, const float& Hessian_idepth,
                                 float& resIntSig, float& resIdpSig)
    {
        // Calculate variance of residuals based on chain rule, approximate
        // Only considering diagonal matrix, without covariance

       Mat6x6 FrameH_Sigma = Hessian_h.inverse();
       Mat6x6 FrameT_Sigma = Hessian_t.inverse();
       float idepth_Sigma = 1.0f/Hessian_idepth;

       SE3 HostToTarget = targetF->get_worldToFram_evalPT() * hostF->get_worldToFram_evalPT().inverse();
       Mat6x6 AdjointTH = (HostToTarget.Adj()).cast<float>();

       Vector6 JIdxi = Jac->JIdp[0][0] * Jac->Jpdxi[0] + Jac->JIdp[1][0] * Jac->Jpdxi[1];
       float JIdd = Jac->JIdp[0][0] * Jac->Jpdd[0] +  Jac->JIdp[1][0] * Jac->Jpdd[1];

       resIntSig = JIdxi.transpose() * (AdjointTH*FrameH_Sigma*AdjointTH.transpose() + FrameT_Sigma) * JIdxi + JIdd*JIdd*idepth_Sigma;

       Vector6 Jddxi = Jac->Jddp[0][0] * Jac->Jpdxi[0] + Jac->Jddp[1][0] * Jac->Jpdxi[1] - Jac->J_transDepth_dxi;
       float Jddd = Jac->Jddp[0][0] * Jac->Jpdd[0] +  Jac->Jddp[1][0] * Jac->Jpdd[1] - Jac->J_transDepth_dd;

       resIdpSig = Jddxi.transpose() * (AdjointTH*FrameH_Sigma*AdjointTH.transpose() + FrameT_Sigma) * Jddxi + Jddd*Jddd*idepth_Sigma;

       if(std::isnan(1.0f/resIntSig))
           resIntSig = 0.5;              // assume weight = 0.5
       else
           resIntSig = 1.0f/resIntSig;

       if(std::isnan(1.0f/resIdpSig))
           resIdpSig = 0.5;
       else
           resIdpSig = 1.0f/resIdpSig;

       return;
    }

    void Residual::calcuResWeight(DistributionMode mode, int Dof, float& resWeight, float resInt, float resDep)
    {
        float sigmaI = Jac->resInt_sigma;
        float sigmaD = Jac->resIdp_sigma;

        if(mode == DistributionMode::TDistribution)     // residuals follow T-dsitribution
        {
            resWeight = (5.0 + 2.0)/(5.0 + resInt*sigmaI*resInt + resDep*sigmaD*resDep);
        }
        else if(mode == DistributionMode::NormalDistribution)  // follow Normal-distribution
        {
            resWeight = 1;
        }
    }
}
