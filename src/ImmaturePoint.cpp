#include <globalFuncs.h>
#include <Eigen/Dense>

#include <ImmaturePoint.h>
#include <Frame.h>

namespace DSLAM
{

int ImmaturePoint::Num_TotalImmaturePoints = 0;

ImmaturePoint::ImmaturePoint(int u_, int v_, Frame *host_, CameraIntrinsic &camera):
    u(u_), v(v_), hostF(host_), idepth(0), idepth_min(0), idepth_max(NAN), lastTraceStatus(IPS_UNINITIALIZED), GoodObservations(0)
{
    Num_TotalImmaturePoints++;

    grad = 0;
    for(int idx = 0; idx < patternNum; idx++)
    {
        int dx = pattern[idx][0];
        int dy = pattern[idx][1];

        Vector6 data = bilinearInterpolation(hostF->intensity[0], hostF->depth[0], hostF->Grad_Int[0],
                                          hostF->Grad_Dep[0], u+dx, v+dy);

        color[idx] = data[0];
        if(idx == 0)
        {
            Vector3 Normal = Vector3::Zero();
            float angle = 0;

            hostF->calculateNormalandAngle(u+dx, v+dy, 0, camera, Normal, angle);
            hostF->calcuDepthSigma(0, data[1], angle, sigma);     // set initial value of uncertainty in depth observation

/*          float max_depth = data[1] + sigma * 0.01;
            float min_depth = data[1] - sigma * 0.01;   */
            idepth = 1.0f/data[1];                      // set initial value of inverse depth
            sigma = (0.01*sigma)/(idepth*idepth);       // set initial value of uncertainty in inverse depth

            idepth_min = idepth - sigma;
            idepth_max = idepth + sigma;                       
        }

        if(!std::isfinite(color[idx]) || std::isnan(idepth))
        {
            energyTH = NAN;
            return;
        }

        gradH += data.segment<2>(2) * data.segment<2>(2).transpose();

        weights[idx] = sqrtf(setting_outlierTHSumComponent/(setting_outlierTHSumComponent + data.segment<2>(2).squaredNorm()));
    }

    debugPrint = false;
    energyTH = patternNum * setting_outlierTH;
    quality = 10000;

}

ImmaturePoint::~ImmaturePoint() {}

ImmaturePointStatus ImmaturePoint::traceOn(Frame* frame, const SE3& RefToFrame, const Mat3x3& RefToFrame_KRKi, const Vector3& RefToFrame_Kt, const CameraIntrinsic& camera)
{
    if(lastTraceStatus == IPS_OOB) return lastTraceStatus;
//    if(lastTraceStatus == IPS_CONVERGED) return lastTraceStatus;

    int width = frame->width[0]; int height = frame->height[0];
    float maxPixelSearch = (width + height) * setting_maxPixelSearch;

    Vector3 pr = RefToFrame_KRKi * Vector3(u,v,1);
    Vector3 ptpMin = pr + RefToFrame_Kt * idepth_min;
    float uMin = ptpMin[0]/ptpMin[2];
    float vMin = ptpMin[1]/ptpMin[2];

    if(!(uMin > 4 && vMin > 4 && uMin < width-5 && vMin < height-5))
    {
         lastTraceUV = Vector2(-1,-1);
         lastTraceInterval = 0;
         return lastTraceStatus = IPS_OOB;
    }

    float dist, uMax, vMax;
    Vector3 ptpMax = Vector3::Zero();
    if(std::isfinite(idepth_max))
    {
        ptpMax = pr + RefToFrame_Kt * idepth_max;
        uMax = ptpMax[0] / ptpMax[2];
        vMax = ptpMax[1] / ptpMax[2];


        if(!(uMax > 4 && vMax > 4 && uMax < width-5 && vMax < height-5))
        {
            lastTraceUV = Vector2(-1,-1);
            lastTraceInterval = 0;
            return lastTraceStatus = IPS_OOB;
        }

 // ============== check their distance. everything below 2px is OK (-> skip). ===================
        dist = (uMin-uMax)*(uMin-uMax) + (vMin-vMax)*(vMin-vMax);
        dist = sqrtf(dist);
        if(dist < setting_trace_slackInterval)
        {
            if(debugPrint)
                 printf("TOO CERTAIN ALREADY (dist %f)!\n", dist);

            lastTraceUV = Vector2(uMax+uMin, vMax+vMin)*0.5;
            lastTraceInterval = dist;
            return lastTraceStatus = ImmaturePointStatus::IPS_SKIPPED;
        }

        assert(dist>0);
    }
    else
    {
        dist = maxPixelSearch;

        // project to arbitrary depth to get direction.
        ptpMax = pr + RefToFrame_Kt*0.01;
        uMax = ptpMax[0] / ptpMax[2];
        vMax = ptpMax[1] / ptpMax[2];

       // direction.
       float dx = uMax-uMin;
       float dy = vMax-vMin;
       float d = 1.0f / sqrtf(dx*dx+dy*dy);

       // set to [setting_maxPixSearch].
       uMax = uMin + dist*dx*d;
       vMax = vMin + dist*dy*d;

       // may still be out!
       if(!(uMax > 4 && vMax > 4 && uMax < width-5 && vMax < height-5))
       {
 //          if(debugPrint) printf("OOB uMax-coarse %f %f %f!\n", uMax, vMax,  ptpMax[2]);
             lastTraceUV = Vector2(-1,-1);
             lastTraceInterval=0;
             return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
       }
       assert(dist>0);
    }

    // set OOB if scale change too big.
    if(idepth_min < 0 || ptpMin[2] < 0.75 || ptpMin[2] > 1.5)
    {
        lastTraceUV = Vector2(-1,-1);
        lastTraceInterval = dist;
        return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
    }

    float dx = uMax - uMin;
    float dy = vMax - vMin;

    // compute error-bounds on result in pixel. why why why? cannot understand at all.....
    float errorInPixel = 0;
    {
        float a = (Vector2(dx,dy).transpose() * gradH * Vector2(dx,dy));
        float b = (Vector2(dy,-dx).transpose() * gradH * Vector2(dy,-dx));
        errorInPixel = 0.2f + 0.2f * (a+b) / a;

        if(errorInPixel > 10) errorInPixel = 10;
    }

    dx /= dist; dy /= dist;     // search step size

    if(dist > maxPixelSearch)
    {
        uMax = uMin + maxPixelSearch*dx;
        vMax = vMin + maxPixelSearch*dy;
        dist = maxPixelSearch;
    }

    int numSteps = 1.9999f + dist;
    Mat2x2 Rplane = RefToFrame_KRKi.topLeftCorner<2,2>();

    float randShift = uMin*1000 - std::floor(uMin*1000);
    float ptx = uMin - randShift*dx;
    float pty = vMin - randShift*dy;

    Vector2 rotatePattern[patternNum];
    for(int idx=0; idx < patternNum; idx++)
    {
        rotatePattern[idx] = Rplane * Vector2(pattern[idx][0], pattern[idx][1]);
    }

    float errors[100];
    float bestU = 0, bestV = 0, bestEnergy = 1e10;
    int bestIdx = -1;
    if(numSteps >= 100) numSteps = 99;

    for(int i = 0; i < numSteps; i++)
    {
        float energy = 0;
        for(int idx = 0; idx < patternNum; idx++)
        {
            Vector2 hitData = bilinearInterpolation(frame->intensity[0], frame->depth[0], ptx+rotatePattern[idx][0], pty+rotatePattern[idx][1]);

            if(!std::isfinite(hitData[0]) || !std::isfinite(hitData[1]))
            {
                energy += 1e5;
                continue;
            }

            float resInt = hitData[0] - color[idx];                  // only consider intensity residuals when tracing
            float hw = fabs(resInt) < setting_huberTH ? 1 : setting_huberTH / fabs(resInt);
            energy += hw * resInt * resInt * (2-hw);
        }

        errors[i] = energy;
        if(energy < bestEnergy)
        {
            bestU = ptx; bestV = pty; bestEnergy = energy; bestIdx = i;
        }

        ptx += dx; pty += dy;
    }

    // find best score outside a +-2px radius.
    float secondBest = 1e10;
    for(int i = 0; i< numSteps; i++)
    {
        if((i < bestIdx - setting_minTraceTestRadius || i > bestIdx + setting_minTraceTestRadius) && errors[i] < secondBest)
                    secondBest = errors[i];
    }
    float newQuality = secondBest / bestEnergy;
    if(newQuality < quality || numSteps > 10) quality = newQuality;

    // ============== do GN optimization ===================
    float uBack=bestU, vBack=bestV, gnstepsize=1, stepBack=0;
    if(setting_trace_GNIterations>0) bestEnergy = 1e5;
    int gnStepsGood=0, gnStepsBad=0;
    float sigmaPT = 0;
    for(int it = 0; it < setting_trace_GNIterations; it++)
    {
        float H = 1, b = 0, energy = 0;
        for(int idx = 0; idx < patternNum; idx++)
        {
            Vector6 hitData = bilinearInterpolation(frame->intensity[0], frame->depth[0], frame->Grad_Int[0], frame->Grad_Dep[0],
                                                    bestU+rotatePattern[idx][0], bestV+rotatePattern[idx][1]);
            if(!std::isfinite(hitData[0]) || !std::isfinite(hitData[1]))
            {
                energy += 1e5;
                continue;
            }

            float res = hitData[0] - color[idx];
            float J = dx * hitData[2] + dy * hitData[3];
            float hw = fabs(res) < setting_huberTH ? 1 : setting_huberTH / fabs(res);

            H += hw * J * J;
            b += hw * J * res;
            energy += weights[idx]*weights[idx]*hw *res*res*(2-hw);
        }

        if(energy > bestEnergy)
        {
            gnStepsBad++;

            stepBack *= 0.5;
            bestU = uBack + stepBack * dx;
            bestV = vBack + stepBack * dy;
        }
        else
        {
            gnStepsGood++;

            float step = -gnstepsize * b/H;
            if(step < - 0.5)  step = -0.5;
            if(step > 0.5) step = 0.5;

            if(!std::isfinite(step)) step = 0;

            uBack = bestU; vBack = bestV; stepBack = step;
            bestU += step * dx; bestV += step * dy; bestEnergy = energy;
            sigmaPT = 1/(H - 1);                          // approximated uncertainty of estimated pixel position along the epipolar line
            if(sigmaPT > 2) sigmaPT = 1;
        }

        if(fabsf(stepBack) < setting_trace_GNThreshold) break;
    }


    if(bestEnergy > energyTH * setting_trace_extraSlackOnTH)
    {
        lastTraceInterval = 0;
        lastTraceUV = Vector2(-1,-1);
        if(lastTraceStatus == IPS_OUTLIER)
            return lastTraceStatus = IPS_OOB;  // set OOB, when tracing outliying in two sequent frames
        else
            return lastTraceStatus = IPS_OUTLIER;
    }

    // calculate corresponding depth and uncertainty by triangulation
    float depth, idepth_, sigma_;
    float depth_min, idepth_min_;
    float depth_max, idepth_max_;

    Mat3x3 K, Ki;
    K << camera.get_fx(), 0, camera.get_cx(),
          0, camera.get_fy(), camera.get_cy(),
            0, 0, 1;
    Ki = K.inverse();

    Vector3 NormalPT_ref = Vector3(Ki*Vector3(u, v, 1)).normalized();
    Vector3 NormalPT_cur = Vector3(Ki*Vector3(bestU, bestV, 1)).normalized();
    if(!depthFromTriangulation(RefToFrame, NormalPT_ref, NormalPT_cur, depth))
    {
        lastTraceUV = Vector2(-1,-1);
        lastTraceInterval=0;
        return lastTraceStatus = ImmaturePointStatus::IPS_OUTLIER;
    }
    else
    {
        idepth_ = 1.0f/depth;
    }


    sigmaPT = errorInPixel;
    sigmaPT = 2;
    Vector3 NormalPT_cur_min = Vector3(Ki*Vector3(bestU - sigmaPT*dx, bestV - sigmaPT*dy, 1)).normalized();
    Vector3 NormalPT_cur_max = Vector3(Ki*Vector3(bestU + sigmaPT*dx, bestV + sigmaPT*dy, 1)).normalized();
    if(!depthFromTriangulation(RefToFrame, NormalPT_ref, NormalPT_cur_min, depth_min) ||
       !depthFromTriangulation(RefToFrame, NormalPT_ref, NormalPT_cur_max, depth_max))
    {
        lastTraceUV = Vector2(-1,-1);
        lastTraceInterval=0;
        return lastTraceStatus = ImmaturePointStatus::IPS_OUTLIER;
    }
    else
    {
        idepth_min_ = 1.0f/depth_max;
        idepth_max_ = 1.0f/depth_min;
        if(idepth_min_ > idepth_max_) std::swap<float>(idepth_min_, idepth_max_);
 //       idepth_min = idepth_min_; idepth_max = idepth_max_;

        sigma_ = std::max(std::abs(idepth_ - idepth_min_), std::abs(idepth_ - idepth_max_));
        if(((idepth_ - sigma_) < 0) || !std::isfinite(idepth_ + sigma_))
        {
            lastTraceUV = Vector2(-1,-1);
            lastTraceInterval = 0;
            return lastTraceStatus = ImmaturePointStatus::IPS_OUTLIER;
        }
    }


/*
    {
        int errorInPixel = 2;
        if(dx*dx>dy*dy)
        {
            idepth_min = (pr[2]*(bestU-errorInPixel*dx) - pr[0]) / (RefToFrame_Kt[0] - RefToFrame_Kt[2]*(bestU-errorInPixel*dx));
            idepth_max = (pr[2]*(bestU+errorInPixel*dx) - pr[0]) / (RefToFrame_Kt[0] - RefToFrame_Kt[2]*(bestU+errorInPixel*dx));
        }
        else
        {
            idepth_min = (pr[2]*(bestV-errorInPixel*dy) - pr[1]) / (RefToFrame_Kt[1] - RefToFrame_Kt[2]*(bestV-errorInPixel*dy));
            idepth_max = (pr[2]*(bestV+errorInPixel*dy) - pr[1]) / (RefToFrame_Kt[1] - RefToFrame_Kt[2]*(bestV+errorInPixel*dy));
        }
        if(idepth_min > idepth_max) std::swap<float>(idepth_min, idepth_max);

        if(!std::isfinite(idepth_min) || !std::isfinite(idepth_max) || (idepth_max<0))
        {

            lastTraceUV = Vector2(-1,-1);
            return lastTraceStatus = ImmaturePointStatus::IPS_OUTLIER;
        }
    }
*/
    // update idepth and sigma using new observations based on Gaussian model
    if(lastTraceStatus == ImmaturePointStatus::IPS_UNINITIALIZED ||
        (lastTraceStatus!=ImmaturePointStatus::IPS_UNINITIALIZED && GoodObservations==0))
    {
        idepth = idepth_;
        sigma = sigma_;
        idepth_min = idepth - sigma; idepth_max = idepth + sigma;
    }
    else
    {
/*
        bool updateFlag = updateDepthandSigma(idepth_, sigma_);
        if(!updateFlag)
        {
            lastTraceUV = Vector2(-1,-1);
            lastTraceInterval=0;
            return lastTraceStatus = ImmaturePointStatus::IPS_OUTLIER;
        }
*/

        idepth_min = idepth_min_; idepth_max = idepth_max_;
        idepth = 0.5 * (idepth_min + idepth_max);

    }

    lastTraceInterval = dist;
    lastTraceUV = Vector2(bestU, bestV);
    GoodObservations++;
    if((sigma*idepth*idepth) <= 0.02)
        return lastTraceStatus = IPS_CONVERGED;
    else return lastTraceStatus = ImmaturePointStatus::IPS_GOOD;
}

bool ImmaturePoint::depthFromTriangulation(const SE3 &RefToFrame, const Vector3 &ref, const Vector3 &cur, float &depth)
{
    Eigen::Matrix<double, 3, 2> A;
    A.col(0) = -RefToFrame.rotationMatrix()*ref.cast<double>();
    A.col(1) = cur.cast<double>();
    const Eigen::Matrix2d AtA = A.transpose() * A;
    if(AtA.determinant() < 1e-6)
        return false;

    const Eigen::Vector2d depth2 = AtA.inverse() * A.transpose() * RefToFrame.translation();
    depth = (float)(fabs(depth2[0]));
    depth *= ref(2);

/*
    {
        Eigen::Matrix3d ISyMatrix;
        ISyMatrix << 0, -cur(2), cur(1),
                  cur(2), 0, -cur(0),
                 -cur(1), cur(0), 0;
        Eigen::Vector3d J = ISyMatrix * A.col(0);
        if(J.norm() < 1e-6)
            return false;
        Eigen::Vector3d b = ISyMatrix * RefToFrame.translation();
        depth = (float)(fabs((J.transpose()*b) / (J.transpose()*J)));
    }
*/
    return true;

}

bool ImmaturePoint::updateDepthandSigma(const NumType &idepthNew, const NumType &sigmaNew)
{
    float idepth_back = idepth, sigma_back = sigma;

    float sigma2 = sigma*sigma;
    float sigmaNew2 = sigmaNew*sigmaNew;
    idepth = (sigmaNew2 * idepth + sigma2 * idepthNew)/(sigmaNew2 + sigma2);
    sigma = (sigma2 * sigmaNew2)/(sigma2 + sigmaNew2);
    idepth_min = idepth - sigma;
    idepth_max = idepth + sigma;

    if((idepth_min < 0) || !std::isfinite(idepth_max))
    {

        idepth = idepth_back;
        sigma = sigma_back;
        idepth_min = idepth - sigma; idepth_max = idepth + sigma;

      return false;
    }

    return true;
}

double ImmaturePoint::linearizeResidual(const CameraIntrinsic &camera, const float outlierTH, ImmaturePointResidual *ImRes,
                                        NumType &Hdd, NumType &bd, float idepth)
{
    if(ImRes->res_State = ResState::OOB)
    {
        ImRes->res_NewState = ResState::OOB;
        return ImRes->res_Energy;
    }

    FFPrecalc* precalc = &(hostF->TargetPrecalc[ImRes->targetF->idx]);
    const Mat3x3 PRE_Rot = precalc->PRE_Rot;
    const Vector3 PRE_Tra = precalc->PRE_Tra;

    cv::Mat imgData = ImRes->targetF->intensity[0];
    cv::Mat depData = ImRes->targetF->depth[0];
    Eigen::Vector2f* imgGrad = ImRes->targetF->Grad_Int[0];
    Eigen::Vector2f* depGrad = ImRes->targetF->Grad_Dep[0];
    Eigen::Vector2i bound(ImRes->targetF->width[0], ImRes->targetF->height[0]);

    float energyleft = 0;
    float fx = camera.get_fx(); float fy = camera.get_fy();
    float new_idepth;
    float sigmaDep;

    {
        float Jpdd_x, Jpdd_y, Jzdd;
        float idepth_ratio, u, v;
        float Ku, Kv;
        Vector3 TransformPT;

        for(int idx = 0; idx < patternNum; idx++)
        {
            int dx = pattern[idx][0];
            int dy = pattern[idx][1];

            if(!projectPoint(this->u, this->v, this->idepth, dx, dy, camera,
                        PRE_Rot, PRE_Tra, idepth_ratio, u, v, Ku, Kv, new_idepth, bound))
            {
                ImRes->res_NewState = ResState::OOB;
                return ImRes->res_Energy;
            }

            TransformPT = (1.0f/this->idepth) * Vector3(u, v, 1.0f/idepth_ratio);
            Jpdd_x = idepth_ratio * (PRE_Tra[0] - PRE_Tra[2]*u) * fx;
            Jpdd_y = idepth_ratio * (PRE_Tra[1] - PRE_Tra[2]*v) * fy;

            Jzdd = -(TransformPT[2] - PRE_Tra[2]) * (1.0f/this->idepth);

            Vector6 hitData = bilinearInterpolationWithDepthBuffer(imgData, depData, imgGrad, depGrad, Ku, Kv, TransformPT[2]);
            if( std::isnan(hitData[0]) )
            {
                ImRes->res_NewState = ResState::OOB;
                return ImRes->res_Energy;
            }

            float res_int = 1.0f/255 * (hitData[0] - color[idx]);  // scale intensity error in order to match depth error
            float res_dep = hitData[1] - TransformPT[2];               // unit: meter
            float residual = 0.5 * res_int + 0.5 * res_dep;      // approximation

            ImRes->targetF->calcuDepthSigma(0, hitData[1], 0, sigmaDep);  // uncetainty of depth observations
            if(fabs(res_dep) > 5 * sigmaDep * 0.01)
            {
                ImRes->res_NewState = ResState::OOB;
                return ImRes->res_Energy;
            }

            float hw = fabsf(residual) < setting_huberTH ? 1 : setting_huberTH / fabsf(residual);
            energyleft += weights[idx] * weights[idx] * hw * residual * residual * (2-hw);

            Vector2 JIdp = Vector2(hitData[2], hitData[3]);
            Vector2 Jddp = Vector2(hitData[4], hitData[5]);

            float JIdd = weights[idx] * (JIdp[0] * Jpdd_x + JIdp[1] * Jpdd_y);
            float Jddd = weights[idx] * (Jddp[0] * Jpdd_x + Jddp[1] * Jpdd_y - Jzdd);

            Hdd += hw * (JIdd * JIdd + Jddd * Jddd);
            bd += hw * (JIdd * res_int + Jddd * res_dep);

        }

        if(energyleft > energyTH * outlierTH)
        {
            energyleft = energyTH * outlierTH;
            ImRes->res_NewState = ResState::OUTLIER;
        }
        else
        {
            ImRes->res_NewState = ResState::IN;
        }

        ImRes->res_NewEnergy = energyleft;
        return energyleft;

    }



}

}
