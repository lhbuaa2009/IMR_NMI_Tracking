#include <Tracker.h>
#include <iomanip>

namespace DSLAM
{

template<int b, typename T>
T* allocAligned(int size, std::vector<T*> &rawPtrVec)
{
    const int padT = 1 + ((1 << b)/sizeof(T));
    T* ptr = new T[size + padT];
    rawPtrVec.push_back(ptr);
    T* alignedPtr = (T*)(( ((uintptr_t)(ptr+padT)) >> b) << b);
    return alignedPtr;
}

Tracker::Tracker(int w, int h, const string Config) : paraReader(Config)
{
    for(int lvl = 0; lvl < PYR_LEVELS; lvl++)
    {
        int wl = w >> lvl;
        int hl = h >> lvl;

        idepth[lvl] = allocAligned<4, float>(wl*hl, ptrToDelete);
        weightSums[lvl] = allocAligned<4,float>(wl*hl, ptrToDelete);

        SeltPnts[lvl] = 0;
        NumPnts[lvl]  = 0;
    }

    // warped buffers for SSD-based optimization
    buf_warped_idepth = allocAligned<4,float>(w*h, ptrToDelete);
    buf_warped_u = allocAligned<4,float>(w*h, ptrToDelete);
    buf_warped_v = allocAligned<4,float>(w*h, ptrToDelete);
    buf_warped_idx = allocAligned<4,float>(w*h, ptrToDelete);
    buf_warped_idy = allocAligned<4,float>(w*h, ptrToDelete);
    buf_warped_refColor = allocAligned<4,float>(w*h, ptrToDelete);
    buf_warped_color_residual = allocAligned<4,float>(w*h, ptrToDelete);

    buf_warped_ddx = allocAligned<4,float>(w*h, ptrToDelete);
    buf_warped_ddy = allocAligned<4,float>(w*h, ptrToDelete);
    buf_warped_depth_residual = allocAligned<4,float>(w*h, ptrToDelete);

    buf_warped_ref_idx = allocAligned<4,float>(w*h, ptrToDelete);
    buf_warped_ref_idy = allocAligned<4,float>(w*h, ptrToDelete);
    buf_warped_ref_ddx = allocAligned<4,float>(w*h, ptrToDelete);
    buf_warped_ref_ddy = allocAligned<4,float>(w*h, ptrToDelete);

    buf_warped_weight = allocAligned<4,float>(w*h, ptrToDelete);

    newFrame = 0;
    lastRef = 0;
    refFrameID = -1;
    debugPlot = debugPrint = false;
    UseDoubleEM = false;

    string useEM = paraReader.getData<string>("UseEM");
    if(useEM == "false")  UseEM = false;
        else UseEM = true;

    IterationStatus.IsFirstIteration = false;
    IterationStatus.IteInfo.resize(PYR_LEVELS);

    for(int lvl = 0; lvl < PYR_LEVELS; lvl++)     // set Scale Matrix to a Unit
    {
        Scale[lvl] << 1.0, 0, 0, 1.0;
    }

    //***************************************************************************************
    //warp buffers for MI-based optimization
    string debug = paraReader.getData<string>("debugInfo");
    string converg = paraReader.getData<string>("UseConvergEstimate");
    string weight = paraReader.getData<string>("UseWeight");
    string norma = paraReader.getData<string>("UseNormMInfo");
    string SecondDerv = paraReader.getData<string>("Use2ndDervs");
    if(debug == "true")  debugMInfo = true;
        else debugMInfo = false;
    if(converg == "true") UseConvergEstimate = true;
        else UseConvergEstimate = false;
    if(weight == "false") UseWeight = false;
        else UseWeight = true;
    if(norma == "true") UseNormMInfo = true;
        else UseNormMInfo = false;
    if(SecondDerv == "true") Use2ndDerivates = true;
        else Use2ndDerivates = false;

    testParaReader();
    buf_MI_ref_color = allocAligned<4,float>(NumPntsInMI, ptrToDelete);
    buf_MI_ref_idepth = allocAligned<4,float>(NumPntsInMI, ptrToDelete);
    buf_MI_ref_u = allocAligned<4,float>(NumPntsInMI, ptrToDelete);
    buf_MI_ref_v = allocAligned<4,float>(NumPntsInMI, ptrToDelete);
    buf_MI_ref_idx = allocAligned<4,float>(NumPntsInMI, ptrToDelete);
    buf_MI_ref_idy = allocAligned<4,float>(NumPntsInMI, ptrToDelete);
    buf_MI_ref_scaledIntensity = allocAligned<4,float>(NumPntsInMI, ptrToDelete);
    buf_MI_ref_weight = allocAligned<4,float>(NumPntsInMI, ptrToDelete);
    buf_MI_visibility = allocAligned<4,float>(NumPntsInMI, ptrToDelete);
    buf_MI_energy = allocAligned<4,float>(NumPntsInMI, ptrToDelete);

    for(int bin = 0; bin < BinsOfHis; bin++)
    {
        buf_MI_ref_BSpline[bin] = allocAligned<4,float>(NumPntsInMI, ptrToDelete);
        buf_MI_ref_BSpline_derivas[bin] = allocAligned<4,float>(NumPntsInMI, ptrToDelete);
        buf_MI_cur_BSpline[bin] = allocAligned<4,float>(NumPntsInMI, ptrToDelete);
        buf_MI_ref_BSpline_derivas2[bin] = allocAligned<4,float>(NumPntsInMI, ptrToDelete);
    }

    for(int dim = 0; dim < 6; dim++)
    {
        buf_MI_PrecalGauss[dim] = allocAligned<4,float>(NumPntsInMI, ptrToDelete);
    }

}

Tracker::~Tracker()
{
    for(float* ptr : ptrToDelete)
    {
        delete[] ptr;
    }
    ptrToDelete.clear();

    for(int lvl = 0; lvl < PYR_LEVELS; lvl++)
    {
        delete[] SeltPnts[lvl];
    }
}
void Tracker::resetTracker()
{
    // reset Scale and IterationStatus
    for(int lvl = 0; lvl < PYR_LEVELS; lvl++)     // set Scale Matrix to a Unit
    {
        Scale[lvl] << 1.0, 0, 0, 1.0;
        IterationStatus.IterationNumOfGEM[lvl] = 0;

        int num = NumPnts[lvl];
        SelectPnt* pntsl = SeltPnts[lvl];
        for(int i = 0; i < num; i++)         // reset initial weights to 1.0f
        {
            pntsl[i].Tweight = 1.0f;
            pntsl[i].energy = pntsl[i].energy_new = 0;
            pntsl[i].isGood = pntsl[i].isGood_new = true;
        }

        IterationStatus.IteInfo[lvl].clear();
    }

    IterationStatus.TDistributionLogLikelihood = 0;
    IterationStatus.IsFirstIteration = false;

}

void Tracker::makePyramid(const CameraIntrinsic* camera)
{

    w[0] = lastRef->width[0];
    h[0] = lastRef->height[0];

    fx[0] = camera->get_fx();
    fy[0] = camera->get_fy();
    cx[0] = camera->get_cx();
    cy[0] = camera->get_cy();

    for(int lvl = 1; lvl < PYR_LEVELS; lvl++)
    {
        w[lvl] = w[0] >> lvl;
        h[lvl] = h[0] >> lvl;
        fx[lvl] = fx[lvl-1] * 0.5f;
        fy[lvl] = fy[lvl-1] * 0.5f;
        cx[lvl] = (cx[0] + 0.5) / ((int)1 << lvl) - 0.5;
        cy[lvl] = (cy[0] + 0.5) / ((int)1 << lvl) - 0.5;

    }

    for(int lvl = 0; lvl < PYR_LEVELS; lvl++)
    {
        K[lvl] << fx[lvl], 0.0f, cx[lvl], 0.0f, fy[lvl], cy[lvl], 0.0f, 0.0f, 1.0f;
        Ki[lvl] = K[lvl].inverse();
        fxi[lvl] = K[lvl](0,0);
        fyi[lvl] = K[lvl](1,1);
        cxi[lvl] = K[lvl](0,2);
        cyi[lvl] = K[lvl](1,2);
    }
}



void Tracker::makeTrackDepth(std::vector<Frame*> keyframes)
{
    memset(idepth[0], 0, sizeof(float)*w[0]*h[0]);
    memset(weightSums[0], 0, sizeof(float)*w[0]*h[0]);

    for(Frame* fh : keyframes)
    {
        for(Point* ph : fh->GoodPoints)
        {
            if(ph->last2residuals[0].first != 0 && ph->last2residuals[0].second == ResState::IN)
            {
                Residual* res = ph->last2residuals[0].first;
 //             assert(r->efResidual->isActive() && r->target == lastRef);    // not add efResidual yet
                assert(res->targetF == lastRef);
                int u = res->centerprojectedPT[0] + 0.5f;
                int v = res->centerprojectedPT[1] + 0.5f;
                float new_idepth = res->centerprojectedPT[2];
 //             float weight = sqrtf(1e-3 / (ph->efPoint->HdiF+1e-12));   // not add efPoint yet
                float weight = 1.0f;                                      // set weight = 1 temporarily...

                idepth[0][u + v*w[0]] += new_idepth * weight;
                weightSums[0][u + v*w[0]] += weight;
            }
        }
    }

    float* depth0 = lastRef->depth[0].ptr<float>();
    for(int v = 0; v < h[0]; v++)
        for(int u = 0; u < w[0]; u++)
        {
            int idx = u + v*w[0];

            if(weightSums[0][idx] <= 0)
            {
                if(depth0[idx] == 0)
                    idepth[0][idx] = 0;
                else
//                    idepth[0][idx] = 1.0f/depth0[idx];
                    idepth[0][idx] = 0;
            }
            else
            {
                NumPnts[0]++;

                if(depth0[idx] == 0)
                    idepth[0][idx] = idepth[0][idx] / weightSums[0][idx];
                else
                    idepth[0][idx] = (idepth[0][idx] / weightSums[0][idx] + 2*(1.0f/depth0[idx])) / 3.0f;  // still to be considered carefully...
            }
        }

    for(int lvl = 1; lvl < PYR_LEVELS; lvl++)
    {
        int wl = w[lvl];
        int hl = h[lvl];
        int wll = w[lvl-1];

        float* idepthLvl = idepth[lvl];
        float* idepthLvll = idepth[lvl-1];
        for(int v = 0; v < hl; v++)
            for(int u = 0; u < wl; u++)
            {
                int idx = u + v*wl;
                int idx2 = 2*u + 2*v*wll;

                idepthLvl[idx] = 0.25 * (idepthLvll[idx2] + idepthLvll[idx2+1] +
                                          idepthLvll[idx2+wll] + idepthLvll[idx2+wll+1]);

                if(idepthLvl[idx] > 0)   NumPnts[lvl]++;
            }
    }

    // initialize Point structure
    for(int lvl = 0; lvl < PYR_LEVELS; lvl++)
    {
        if(SeltPnts[lvl] != 0) delete[] SeltPnts[lvl];
        SeltPnts[lvl] = new SelectPnt[NumPnts[lvl]];

        int wl = w[lvl], hl = h[lvl];
        SelectPnt* pntsl = SeltPnts[lvl];
        float* idepthL = idepth[lvl];
        float* color = lastRef->intensity[lvl].ptr<float>();
        int count = 0;

        for(int y = 3; y < hl - 3; y++)
            for(int x = 3; x < wl - 3; x++)
            {
                int idx = x + y*wl;
                if(idepthL[idx] > 0)
                {
                    pntsl[count].u = x;
                    pntsl[count].v = y;

                    if(!std::isfinite(color[idx]) || idepthL[idx] < 0.1 || idepthL[idx] > 2.0)
                    {
                       idepthL[idx] = 0;
                       continue;
                    }

                    pntsl[count].color = color[idx];
                    pntsl[count].idepth = idepthL[idx];
                    pntsl[count].Tweight = 1.0f;          // initial all weights to 1.0

                    pntsl[count].energy = pntsl[count].energy_new = 0;
                    pntsl[count].isGood = pntsl[count].isGood_new = true;
                    pntsl[count].outlierTH = patternNum * setting_outlierTH;  // to be adjusted...

                    count++;

                    assert(count < NumPnts[lvl]);
                }
            }

        NumPnts[lvl] = count;
    }
}

void Tracker::makeTrackPointsbyFrame(Frame *frame)      // only for testing track algorithem
{
    lastRef = frame;
    makePyramid(lastRef->intrinsic);

    int w0 = w[0], h0 = h[0];
    PixelSelector selector(w0, h0);

    float* statusMap = new float[w0*h0];
    bool*  statusMapB = new bool[w0*h0];
    float densities[] = {0.03, 0.05, 0.15, 0.5, 1.0};
    int potInPyr = 5;

    for(int level = 0; level < PYR_LEVELS; level++)
    {
        selector.currentPotential = 3;   // not necessary
        int npts;

        if(level == 0)
        {
            npts = selector.makeMaps(lastRef, statusMap, w0*h0*densities[0], 1, true, 2);

            // debug depth recovery algorithm using depth filter, nothing to do with tracking
            // should be removed after debugging
            {
                Vector4 intrinsic = Vector4(517.3, 516.5, 318.6, 255.3);
                Vector5f distortion = Vector5f::Zero();
                CameraIntrinsic camera(intrinsic, distortion);

                lastRef->ImmaturePoints.reserve(npts * 1.2f);
                for(int y = 3; y < h[0] - 3; y++)
                    for(int x = 3; x < w[0] - 3; x++)
                    {
                        int idx = x + y*w[0];
                        if(statusMap[idx] == 0)
                            continue;
                        float depth = lastRef->depth[0].at<float>(y,x);
                        if(depth < 0.2 || depth> 10.0)
                        {
                           continue;
                        }

                        ImmaturePoint* impt = new ImmaturePoint(x, y, lastRef, camera);                        
                        impt->idepth_ref = 1.0f / depth;
                        impt->idepth = 0,0;
                        impt->sigma = 0.0;
                        impt->idepth_min = 0.0f;
                        impt->idepth_max = NAN;

                        if(!std::isfinite(impt->energyTH))  delete impt;
                        else lastRef->ImmaturePoints.push_back(impt);
                    }
            }

        }
        else
        {
            npts = PixelSelectorInPyr(lastRef, level, statusMapB, potInPyr, w0*h0*densities[0]);
        }

        if(SeltPnts[level] != 0)   delete[] SeltPnts[level];
        SeltPnts[level] = new SelectPnt[npts];

        // initialize selected points
        int wl = w[level], hl = h[level];
        SelectPnt* ptl = SeltPnts[level];
        float* depth = lastRef->depth[level].ptr<float>();
        float* color = lastRef->intensity[level].ptr<float>();
        Vector2* color_Grad = lastRef->Grad_Int[level];
        Vector2* depth_Grad = lastRef->Grad_Dep[level];
        float* norm = lastRef->GradNorm[level];
        int numl = 0;
        for(int y = 3; y < hl - 3; y++)
            for(int x = 3; x < wl - 3; x++)
            {
                int idx = x + y*wl;
                if( (level == 0 && statusMap[idx] != 0) ||
                       (level != 0 && statusMapB[idx]) )
                {

                    if(!std::isfinite(color[idx]) || depth[idx] < 0.4 || depth[idx] > 10.0)    // 0.4~10
                    {
                       depth[idx] = 0;
                       continue;
                    }

                    ptl[numl].u = x + 0.1;
                    ptl[numl].v = y + 0.1;

                    ptl[numl].idepth = 1.0f / depth[idx];
                    ptl[numl].color  = color[idx];

                    ptl[numl].colorGrad = color_Grad[idx];
                    if(!isnan(depth_Grad[idx](0)) && !isnan(depth_Grad[idx](1)))
                    {
                        ptl[numl].depthGrad = depth_Grad[idx];
                    }
                    else
                    {
                        ptl[numl].depthGrad = Vector2(0,0);
                    }
                    ptl[numl].norm = norm[idx];

                    ptl[numl].Tweight = 1.0;
                    ptl[numl].isGood  = true;
                    ptl[numl].isGood_new = false;
                    ptl[numl].energy = ptl[numl].energy_new = 0;


                    ptl[numl].outlierTH = patternNum * setting_outlierTH;

                    numl++;
                    assert(numl <= npts);
                }
            }

        NumPnts[level] = numl;
    }

    delete[] statusMap;
    delete[] statusMapB;
}

void Tracker::setTrackingRef(std::vector<Frame*> keyframes)
{
    assert(keyframes.size() > 0);
    lastRef = keyframes.back();
    makeTrackDepth(keyframes);

    refFrameID = lastRef->shell->id;
    firstTrackRMSE = -1;
}

Vector6 Tracker::calcRes(int lvl, const SE3 &refToNew, float cutoffTH)
{
    Mat2x2 ScaleL = Scale[lvl];   // scale/precision Matrix of 2-d residual vector

    float E = 0;
    int numTermsInE = 0;
    int numTermsInWarped = 0;
    int numSaturated = 0;

    int wl = w[lvl], hl = h[lvl];
    float fxl = fx[lvl], fyl = fy[lvl];
    float cxl = cx[lvl], cyl = cy[lvl];
    cv::Mat gray = newFrame->intensity[lvl];
    cv::Mat depth = newFrame->depth[lvl];
    Eigen::Vector2f* igradNew = newFrame->Grad_Int[lvl];
    Eigen::Vector2f* dgradNew = newFrame->Grad_Dep[lvl];


    Mat3x3 RKi = (refToNew.rotationMatrix().cast<float>()) * Ki[lvl];
    Vector3 t = refToNew.translation().cast<float>();

    float sumSquaredShiftT = 0;
    float sumSquaredShiftRT = 0;
    float sumSquaredShiftNum = 0;

    float maxEnergy = 2 * setting_huberTH2 * cutoffTH - setting_huberTH2 * setting_huberTH2;   // why why why...

    if(debugPlot)
    {
        // to do...
    }

    int num = NumPnts[lvl];
    SelectPnt* pntsl = SeltPnts[lvl];
    for(int i = 0; i < num; i++)
    {

        if(UseEM)
        {
            if(!IterationAccepted)
            {
                if(!pntsl[i].isGood)
                {
                    E += pntsl[i].energy;
                    numTermsInE++;
                    pntsl[i].energy_new = pntsl[i].energy;
                    pntsl[i].isGood_new = false;

                    continue;
                }
            }
        }

        float T_weight = pntsl[i].Tweight;

        float x = pntsl[i].u;
        float y = pntsl[i].v;
        float idpt = pntsl[i].idepth;
        float refcolor = pntsl[i].color;
        Vector2 colorGrad = pntsl[i].colorGrad;
        Vector2 depthGrad = pntsl[i].depthGrad;

        Vector3 pt = RKi * Vector3(x, y, 1) + t*idpt;
        float u = pt[0] / pt[2];
        float v = pt[1] / pt[2];
        float Ku = fxl*u + cxl;
        float Kv = fyl*v + cyl;
        float new_idepth = idpt/pt[2];

        if(lvl==0 && i%32==0)
        {
            // translation only (positive)
            Vector3 ptT = Ki[lvl] * Vector3(x, y, 1) + t*idpt;
            float uT = ptT[0] / ptT[2];
            float vT = ptT[1] / ptT[2];
            float KuT = fxl * uT + cxl;
            float KvT = fyl * vT + cyl;
            //printf("COUGHT INF / NAN minmax depth (%f %f)!\n", idepth_min, idepth_max);

            // translation only (negative)
            Vector3 ptT2 = Ki[lvl] * Vector3(x, y, 1) - t*idpt;
            float uT2 = ptT2[0] / ptT2[2];
            float vT2 = ptT2[1] / ptT2[2];
            float KuT2 = fxl * uT2 + cxl;
            float KvT2 = fyl * vT2 + cyl;

            //translation and rotation (negative)
            Vector3 pt3 = RKi * Vector3(x, y, 1) - t*idpt;
            float u3 = pt3[0] / pt3[2];
            float v3 = pt3[1] / pt3[2];
            float Ku3 = fxl * u3 + cxl;
            float Kv3 = fyl * v3 + cyl;

            //translation and rotation (positive)
            //already have it.

            sumSquaredShiftT += (KuT-x)*(KuT-x) + (KvT-y)*(KvT-y);
            sumSquaredShiftT += (KuT2-x)*(KuT2-x) + (KvT2-y)*(KvT2-y);
            sumSquaredShiftRT += (Ku-x)*(Ku-x) + (Kv-y)*(Kv-y);
            sumSquaredShiftRT += (Ku3-x)*(Ku3-x) + (Kv3-y)*(Kv3-y);
            sumSquaredShiftNum+=2;
        }

        if(!(Ku > 2 && Kv > 2 && Ku < wl-3 && Kv < hl-3 && new_idepth > 0.1 && new_idepth < 2.5))
        {

            if(UseEM)
            {
                E += pntsl[i].energy;
                numTermsInE++;
                pntsl[i].energy_new = pntsl[i].energy;
                pntsl[i].isGood_new = false;
            }

            continue;
        }
        Vector6 hitData = bilinearInterpolationWithDepthBuffer(gray, depth, igradNew, dgradNew, Ku, Kv, 1.0f/new_idepth);
        if(isnan(hitData[1]) || !std::isfinite(hitData[0]) || hitData[1] < 0.2 || hitData[1] > 10)
        {

            if(UseEM)
            {
                E += pntsl[i].energy;
                numTermsInE++;
                pntsl[i].energy_new = pntsl[i].energy;
                pntsl[i].isGood_new = false;
            }

            continue;
        }

        float color_residual = 1.0f/255 * (hitData[0] - refcolor);  // adjust scale of color residual
        float depth_residual = hitData[1] - 1.0f/new_idepth;       //  to match the scle of depth res..

        float depSigma;
        newFrame->calcuDepthSigma(0, hitData[1], 0, depSigma);    // not sure about this formular...
//        depSigma = 0.0012 + 0.0019 * (hitData[1] - 0.4f) * (hitData[1] - 0.4f);
/*
        if(i == 0)                                           // debug output lines
            cout << "Sigma_depth : ";
        else if(i < 15)
        {
            cout << depSigma << ", ";
        }
        else if(i == 15)
        {
            cout << depSigma << endl;
        }
*/

        if(fabs(depth_residual) > 10*depSigma*0.01)
        {

            if(UseEM)
            {
                E += pntsl[i].energy;
                numTermsInE++;
                pntsl[i].energy_new = pntsl[i].energy;
                pntsl[i].isGood_new = false;
            }
            continue;
        }


        Vector2 residual = Vector2(color_residual, depth_residual);
        float tmpRes = residual.transpose() * ScaleL.inverse() * residual;
        float hw = sqrtf(fabs(tmpRes)) < setting_huberTH2 ? 1 : setting_huberTH2 / sqrtf(fabs(tmpRes));   // setting_huberTH2 = 1.32, may need to be adjusted

        if(sqrt(fabs(tmpRes)) > 1e6)    // how to choose the value of cutoffTH : 1e6, 10, 5
        {
            if(debugPlot) {};  // to do
  //          E += maxEnergy;
  //          numSaturated++;
            E += pntsl[i].energy;
            numTermsInE++;
            pntsl[i].energy_new = pntsl[i].energy;
            pntsl[i].isGood_new = false;
        }
        else
        {
            if(debugPlot) {};
            if(UseEM)
            {
                if(IterationAccepted)
                {
                    pntsl[i].Tweight = calcSigResWeight(residual[0], residual[1], Vector2(0,0), Scale[lvl].inverse());
                    T_weight = pntsl[i].Tweight;
                }
            }
            else
            {
                if(IterationStatus.IsFirstIteration)   // only calc weights on the first iteration in the M-step of each level
                {
                    pntsl[i].Tweight = calcSigResWeight(residual[0], residual[1], Vector2(0,0), Scale[lvl].inverse());
                    T_weight = pntsl[i].Tweight;
                }
                else if(pntsl[i].Tweight == 1.0f)    // re-cacl weights fow new added points
                {
                    pntsl[i].Tweight = calcSigResWeight(residual[0], residual[1], Vector2(0,0), Scale[lvl].inverse());
                    T_weight = pntsl[i].Tweight;
                }
            }

//          hw = 1.0f;
            T_weight = 1.0f;                 // forbiden T-weights
            pntsl[i].isGood_new = true;
            pntsl[i].energy_new = T_weight * hw * tmpRes * tmpRes * (2-hw);

            E += pntsl[i].energy_new;
            numTermsInE++;

            buf_warped_idepth[numTermsInWarped] = new_idepth;
            buf_warped_u[numTermsInWarped] = u;
            buf_warped_v[numTermsInWarped] = v;
            buf_warped_idx[numTermsInWarped] = hitData[2];
            buf_warped_idy[numTermsInWarped] = hitData[3];
            buf_warped_refColor[numTermsInWarped] = refcolor;
            buf_warped_color_residual[numTermsInWarped] = residual[0];       // was scaled by 1.0/255

            buf_warped_ddx[numTermsInWarped] = hitData[4];
            buf_warped_ddy[numTermsInWarped] = hitData[5];
            buf_warped_depth_residual[numTermsInWarped] = residual[1];

            buf_warped_ref_idx[numTermsInWarped] = colorGrad(0);
            buf_warped_ref_idy[numTermsInWarped] = colorGrad(1);
            buf_warped_ref_ddx[numTermsInWarped] = depthGrad(0);
            buf_warped_ref_ddy[numTermsInWarped] = depthGrad(1);

            buf_warped_weight[numTermsInWarped] = hw * T_weight;
            numTermsInWarped++;
        }
    }

    while(numTermsInWarped%4 != 0)  // in order to perform SSE
    {
        buf_warped_idepth[numTermsInWarped] = 1.0f;
        buf_warped_u[numTermsInWarped] = 0;
        buf_warped_v[numTermsInWarped] = 0;
        buf_warped_idx[numTermsInWarped] = 0;
        buf_warped_idy[numTermsInWarped] = 0;
        buf_warped_refColor[numTermsInWarped] = 0;
        buf_warped_color_residual[numTermsInWarped] = 0;                       // was scaled by 1.0/255

        buf_warped_ddx[numTermsInWarped] = 0;
        buf_warped_ddy[numTermsInWarped] = 0;
        buf_warped_depth_residual[numTermsInWarped] = 0;

        buf_warped_ref_idx[numTermsInWarped] = 0;
        buf_warped_ref_idy[numTermsInWarped] = 0;
        buf_warped_ref_ddx[numTermsInWarped] = 0;
        buf_warped_ref_ddy[numTermsInWarped] = 0;

        buf_warped_weight[numTermsInWarped] = 0;
        numTermsInWarped++;
    }

    buf_warped_num = numTermsInWarped;

    if(IterationAccepted) IterationAccepted = false;

    if(IterationStatus.IsFirstIteration)
    {
        if(debugPrint)
        {
            cout << "Update TDsitribution weights : " << endl;     //debug lines
            for(int i = 0; i < 20; i++)
            {
                cout << buf_warped_weight[i] << ", ";
            }
            cout << endl;
        }

        IterationStatus.IsFirstIteration = false;
    }

    // debug lines
    if(debugPrint)//printf("COUGHT INF / NAN minmax depth (%f %f)!\n", idepth_min, idepth_max);

    {
         cout << "E, numInE, numInBuf, avRes : " << E << ", " << numTermsInE << ", " << buf_warped_num << ", "
                                                 << sqrtf((float)(E/numTermsInE)) << endl;
    }


    if(debugPlot)  {};   // to do...


    Vector6 result;
    result[0] = E;
    result[1] = numTermsInE;
    result[2] = sumSquaredShiftT/(sumSquaredShiftNum+0.1);
    result[3] = 0;
    result[4] = sumSquaredShiftRT/(sumSquaredShiftNum+0.1);
    result[5] = numSaturated / (float)numTermsInE;

    return result;
}

void Tracker::calcGSSSE(int lvl, Mat6x6 &H_out, Vector6 &b_out, const SE3 &refToNew)
{
    acc7.initialize();

    Mat2x2 precision = Scale[lvl].inverse();

    __m128 fxl = _mm_set1_ps(fx[lvl]);
    __m128 fyl = _mm_set1_ps(fy[lvl]);

    __m128 one = _mm_set1_ps(1.0f);
    __m128 zero = _mm_set1_ps(0);
    __m128 half = _mm_set1_ps(0.5);
    __m128 factor = _mm_set1_ps(1.0f/255);


    int n = buf_warped_num;

    assert(n%4==0);
    for(int i = 0; i < n; i+=4)
    {       
/*
        __m128 idx = _mm_mul_ps(_mm_mul_ps(half, factor), _mm_mul_ps(_mm_add_ps(_mm_load_ps(buf_warped_idx+i), _mm_load_ps(buf_warped_ref_idx+i)), fxl));
        __m128 idy = _mm_mul_ps(_mm_mul_ps(half, factor), _mm_mul_ps(_mm_add_ps(_mm_load_ps(buf_warped_idy+i), _mm_load_ps(buf_warped_ref_idy+i)), fyl));
        __m128 ddx = _mm_mul_ps(_mm_add_ps(_mm_load_ps(buf_warped_ddx+i), _mm_load_ps(buf_warped_ref_ddx)), _mm_mul_ps(half, fxl));
        __m128 ddy = _mm_mul_ps(_mm_add_ps(_mm_load_ps(buf_warped_ddy+i), _mm_load_ps(buf_warped_ref_ddy)), _mm_mul_ps(half, fyl));
*/

        __m128 idx = _mm_mul_ps(factor, _mm_mul_ps(_mm_load_ps(buf_warped_idx+i), fxl));
        __m128 idy = _mm_mul_ps(factor, _mm_mul_ps(_mm_load_ps(buf_warped_idy+i), fyl));

        __m128 ddx = _mm_mul_ps(_mm_load_ps(buf_warped_ddx+i), fxl);
        __m128 ddy = _mm_mul_ps(_mm_load_ps(buf_warped_ddy+i), fyl);

        __m128 u = _mm_load_ps(buf_warped_u+i);
        __m128 v = _mm_load_ps(buf_warped_v+i);
        __m128 idepth = _mm_load_ps(buf_warped_idepth+i);
        __m128 depth = _mm_div_ps(one, idepth);

        __m128 J00 = _mm_mul_ps(idepth, idx);

        __m128 J01 = _mm_mul_ps(idepth, idy);

        __m128 J02 = _mm_sub_ps(zero, _mm_mul_ps(idepth, _mm_add_ps(_mm_mul_ps(u, idx), _mm_mul_ps(v, idy))));

        __m128 J03 =  _mm_sub_ps(zero, _mm_add_ps(
                                     _mm_mul_ps(_mm_mul_ps(u,v),idx),
                                     _mm_mul_ps(_mm_add_ps(one,_mm_mul_ps(v,v)), idy)));
        __m128 J04 =_mm_add_ps(
                    _mm_mul_ps(_mm_add_ps(one, _mm_mul_ps(u,u)), idx),
                    _mm_mul_ps(_mm_mul_ps(u,v), idy));

        __m128 J05 = _mm_sub_ps(_mm_mul_ps(u,idy), _mm_mul_ps(v,idx));

        __m128 J06 = _mm_load_ps(buf_warped_color_residual+i);


        __m128 J10 = _mm_mul_ps(idepth, ddx);
        __m128 J11 = _mm_mul_ps(idepth, ddy);
        __m128 J12 = _mm_add_ps(one, _mm_sub_ps(zero, _mm_mul_ps(idepth, _mm_add_ps(_mm_mul_ps(u, ddx), _mm_mul_ps(v, ddy)))));
        __m128 J13 = _mm_sub_ps(_mm_sub_ps(zero, _mm_add_ps(
                                                            _mm_mul_ps(_mm_mul_ps(u,v),ddx),
                                                            _mm_mul_ps(_mm_add_ps(one,_mm_mul_ps(v,v)), ddy))),
                                                         _mm_mul_ps(depth, v));
        __m128 J14 = _mm_add_ps(_mm_mul_ps(depth, u),
                                _mm_add_ps(_mm_mul_ps(_mm_add_ps(one, _mm_mul_ps(u,u)), ddx),
                                           _mm_mul_ps(_mm_mul_ps(u,v), ddy)));
        __m128 J15 = _mm_sub_ps(_mm_mul_ps(u,ddy), _mm_mul_ps(v,ddx));
        __m128 J16 = _mm_load_ps(buf_warped_depth_residual+i);

        __m128 weight = _mm_load_ps(buf_warped_weight+i);

        const float a = precision(0,0);
        const float b = 0.1 * precision(0,1);      //0.1
        const float c = 0.001 * precision(1,1);    //0.001
//        const float c = 0;
//        const float b = 0;


        acc7.updateSSE_weighted(J00, J01, J02, J03, J04, J05, J06,
                                J10, J11, J12, J13, J14, J15, J16,
                                weight, a, b, c);

/*
        acc7.updateSSE_weighted(
                    _mm_mul_ps(idepth, idx),        // Jacobian entries about color residual
                    _mm_mul_ps(idepth, idy),
                    _mm_sub_ps(zero, _mm_mul_ps(idepth, _mm_add_ps(_mm_mul_ps(u, idx), _mm_mul_ps(v, idy)))),
                    _mm_sub_ps(zero, _mm_add_ps(
                                         _mm_mul_ps(_mm_mul_ps(u,v),idx),
                                         _mm_mul_ps(_mm_add_ps(one,_mm_mul_ps(v,v)), idy))),
                    _mm_add_ps(
                              _mm_mul_ps(_mm_add_ps(one, _mm_mul_ps(u,u)), idx),
                              _mm_mul_ps(_mm_mul_ps(u,v), idy)),
                    _mm_sub_ps(_mm_mul_ps(u,idy), _mm_mul_ps(v,idx)),
                    _mm_load_ps(buf_warped_color_residual+i),
                    _mm_mul_ps(idepth, ddx),        // Jacobian entries about depth residual
                    _mm_mul_ps(idepth, ddy),
                    _mm_add_ps(one, _mm_sub_ps(zero, _mm_mul_ps(idepth, _mm_add_ps(_mm_mul_ps(u, ddx), _mm_mul_ps(v, ddy))))),
                    _mm_sub_ps(_mm_sub_ps(zero, _mm_add_ps(
                                         _mm_mul_ps(_mm_mul_ps(u,v),ddx),
                                         _mm_mul_ps(_mm_add_ps(one,_mm_mul_ps(v,v)), ddy))),
                               _mm_mul_ps(depth, v)),
                    _mm_add_ps(_mm_mul_ps(depth, u),
                               _mm_add_ps(_mm_mul_ps(_mm_add_ps(one, _mm_mul_ps(u,u)), ddx),
                                          _mm_mul_ps(_mm_mul_ps(u,v), ddy))),
                    _mm_sub_ps(_mm_mul_ps(u,ddy), _mm_mul_ps(v,ddx)),
                    _mm_load_ps(buf_warped_depth_residual+i),
                    _mm_load_ps(buf_warped_weight+i),   // Weight of T-distribution
                    precision(0,0),             // Scale entries
                    precision(0,1),
                    precision(1,1));
*/
    }

    acc7.finish();
//    H_out = acc7.H.topLeftCorner<6,6>() * (1.0f/n);   // what if without factor 1.0f/n ...
//    b_out = acc7.H.topRightCorner<6,1>() * (1.0f/n);

    H_out = acc7.H.topLeftCorner<6,6>() * (0.1/(n));   // what if without factor 1.0f/n ...
    b_out = acc7.H.topRightCorner<6,1>() * (0.1/(n));
}

// add "__declspec(align(16))" to pointer argumetns
Vector2 Tracker::calcResMean(const float *res_color, const float *res_depth, const float *weight)
{
    int n = buf_warped_num;
    assert(n%4==0);

    EIGEN_ALIGN16 float sum_color[4] = {0, 0, 0, 0};
    EIGEN_ALIGN16 float sum_depth[4] = {0, 0, 0, 0};
    EIGEN_ALIGN16 float sum_weight[4] = {0, 0, 0, 0};
    for(int i = 0; i < n; i+=4)
    {
        __m128 resC = _mm_load_ps(res_color+i);
        __m128 resD = _mm_load_ps(res_depth+i);
        __m128 wT = _mm_load_ps(weight+i);

        _mm_store_ps(sum_color, _mm_add_ps(_mm_load_ps(sum_color), _mm_mul_ps(resC, wT)));
        _mm_store_ps(sum_depth, _mm_add_ps(_mm_load_ps(sum_depth), _mm_mul_ps(resD, wT)));
        _mm_store_ps(sum_weight, _mm_add_ps(_mm_load_ps(sum_weight), wT));
    }

    float tmp1=0, tmp2=0, tmp3=0;
    for(int i = 0; i < 4; i++)
    {
        tmp1 += sum_color[i];
        tmp2 += sum_depth[i];
        tmp3 += sum_weight[i];
    }

    return Vector2(tmp1/tmp3, tmp2/tmp3);
}

Mat2x2 Tracker::calcResScaleSSE(const float *res_color, const float *res_depth, const Vector2 mean, const float *weight)
{
    int n = buf_warped_num;
    assert(n%4==0);

    EIGEN_ALIGN16 float K1[4] = {0, 0, 0, 0};
    EIGEN_ALIGN16 float K2[4] = {0, 0, 0, 0};
    EIGEN_ALIGN16 float K3[4] = {0, 0, 0, 0};

    __m128 mean_resC = _mm_set1_ps(mean[0]);
    __m128 mean_resD = _mm_set1_ps(mean[1]);
    __m128 err_color, err_depth, weights;
    for(int i = 0; i < n; i+=4)
    {
        err_color = _mm_sub_ps(_mm_load_ps(res_color+i), mean_resC);
        err_depth = _mm_sub_ps(_mm_load_ps(res_depth+i), mean_resD);
        weights = _mm_load_ps(weight+i);

        _mm_store_ps(K1, _mm_add_ps(_mm_load_ps(K1), _mm_mul_ps(weights, _mm_mul_ps(err_color, err_color))));
        _mm_store_ps(K3, _mm_add_ps(_mm_load_ps(K3), _mm_mul_ps(weights, _mm_mul_ps(err_depth, err_depth))));
        _mm_store_ps(K2, _mm_add_ps(_mm_load_ps(K2), _mm_mul_ps(weights, _mm_mul_ps(err_color, err_depth))));
    }

    Mat2x2 covariance = Mat2x2::Zero();
    for(int i = 0; i < 4; i++)
    {
        covariance(0,0) += K1[i];
        covariance(0,1) += K2[i];
        covariance(1,1) += K3[i];
    }
    covariance(1,0) = covariance(0,1);

    return  (1.0f/(n-2))*covariance;
}

Mat2x2 Tracker::calcResScale(const float *res_color, const float *res_depth, const Vector2 mean, const float *weight)
{
    int n = buf_warped_num;
    float factor = 1.0f / (n-2);
    Mat2x2 covariance = Mat2x2::Zero();

    for(int i = 0; i < n; i++)
    {
        Vector2 res = Vector2(res_color[i], res_depth[i]);
        float w = weight[i];

        covariance += w * res * res.transpose();
    }

    return covariance*factor;
}


float Tracker::calcSigResWeight(const float res_color, const float res_depth, const Vector2 &mean, const Mat2x2 &precision)
{
    Vector2 error = Vector2(res_color, res_depth) - mean;
    return (2.0 + 5.0f) / (5.0f + error.transpose() * precision * error);
}

void Tracker::calcResWeights(const float *res_color, const float *res_depth, const Vector2 &mean, const Mat2x2 &precision, float *weights)
{
    int n = buf_warped_num;
    assert(n%4==0);

    __m128 two = _mm_set1_ps(2.0);
    __m128 five = _mm_set1_ps(5.0);
    __m128 sevn = _mm_set1_ps(7.0);

    __m128 mean_resC = _mm_set1_ps(mean[0]);
    __m128 mean_resD = _mm_set1_ps(mean[1]);
    __m128 a0 = _mm_set1_ps(precision(0,0));
    __m128 a1 = _mm_set1_ps(precision(0,1));
    __m128 a2 = _mm_set1_ps(precision(1,1));
    for(int i = 0; i < n; i+=4)
    {
        __m128 err_color = _mm_sub_ps(_mm_load_ps(res_color+i), mean_resC);
        __m128 err_depth = _mm_sub_ps(_mm_load_ps(res_depth+i), mean_resD);

        __m128 elem00 = _mm_mul_ps(a0, _mm_mul_ps(err_color, err_color));
        __m128 elem01 = _mm_mul_ps(two, _mm_mul_ps(a1, _mm_mul_ps(err_color, err_depth)));
        __m128 elem11 = _mm_mul_ps(a2, _mm_mul_ps(err_depth, err_depth));

        __m128 elemSum = _mm_add_ps(elem00, _mm_add_ps(elem01, elem11));

        _mm_store_ps(weights+i, _mm_div_ps(sevn, _mm_add_ps(five, elemSum)));
    }

}

bool Tracker::trackNewFrame(Frame *newFrame, SE3 &lastToNew_out, int maxlvl, Vector5f minResForAbort)
{
    resetTracker();          // reset Scale and IterationStatus information

    debugPrint = false;
    assert(maxlvl < PYR_LEVELS);

    lastResiduals.setConstant(NAN);
    lastFlowIndicators.setConstant(1000);

    this->newFrame = newFrame;
    int maxIterations[] = {10,20,20,20,20};
    float lambaExtrapolationLimit = 0.001;

    SE3 refToNew_current = lastToNew_out;
    bool haveRepeated = false;
    Vector2 mean = Vector2::Zero();
    for(int lvl = maxlvl; lvl >= 0; lvl--)
    {
        if(debugPrint)
        {
            cout << "Track in " << lvl << "th level :" << endl;
        }

        int IterationNum = 0;                                          // debug varies
        int AcceptedIterationNum = 0;

        if(!IterationStatus.IsFirstIteration)
            IterationStatus.IsFirstIteration = true;

        Mat6x6 H; Vector6 b;
        float levelCutoffRepeat = 1;
        Vector6 resOld = calcRes(lvl, refToNew_current, setting_coarseCutoffTH*levelCutoffRepeat);  // update weights and calc new weighted-residual
        applyStep(lvl);

        while(resOld[5] > 0.6 && levelCutoffRepeat < 50)
        {
            levelCutoffRepeat *= 2;
            resOld = calcRes(lvl, refToNew_current, setting_coarseCutoffTH*levelCutoffRepeat);
        }

        if((IterationStatus.IterationNumOfGEM[lvl] != 0) &&
                IterationStatus.IterationNumOfGEM[lvl] < IterationStatus.MaxIterationNumOfGEM[lvl])
        {
            float UpdatedRes = sqrtf((float)(resOld[0] / resOld[1]));   // residual with new weights
            if(UpdatedRes > IterationStatus.TDistributionLogLikelihood)
            {
                cout << "LogLikelihood decrease, end EM optimization in current lvl!" << endl;

                if(lvl != 0)
                    Scale[lvl-1] = Scale[lvl];     // pass current scale params to next level

                continue;
            }
        }

        calcGSSSE(lvl, H, b, refToNew_current);     // calc Jacobian and hessian params
        if(debugPrint)
        {
            cout << "H = " << H << endl;            // output debug lines
            cout << "b = " << b << endl;
        }

        float lamba = 0.1;
        if(debugPrint)
        {
            // to do.....
        }

        IterationPrintInfo IteInfoLvl;
        for(int iteration = 0; iteration < maxIterations[lvl]; iteration++)
        {
            IterationNum++;

            Mat6x6 Hl = H;
            for(int i = 0; i < 6; i++)  Hl(i,i) *= (1+lamba);
            Vector6 inc = Hl.ldlt().solve(-b);

            float extrapFac = 1;
            if(lamba < lambaExtrapolationLimit) extrapFac = sqrt(sqrt(lambaExtrapolationLimit / lamba));
            inc *= extrapFac;             // why why why...

            SE3 refToNew_new = SE3::exp(inc.cast<double>()) * refToNew_current;
            Vector6 resNew = calcRes(lvl, refToNew_new, setting_coarseCutoffTH*levelCutoffRepeat);

            bool accept = (resNew[0] / resNew[1]) < (resOld[0] / resOld[1]);   // average energy...
//            bool accept = resNew[0] < resOld[0];
            if(debugPrint) { };  // to do...

            if(accept)
            {
                IterationAccepted = true;
                AcceptedIterationNum++;
                if(debugPrint)
                    cout << "Accepted Inc : " << inc.transpose() << endl;

/*                if(lvl == maxlvl && AcceptedIterationNum == 1)
                {
                    cout << "[color_res, depth_res, weight] : " << endl;
                    cout << std::fixed;
                    for(int j = 175; j < 188; j++)
                    {
                        cout << "[" << setprecision(4) << buf_warped_color_residual[j] << "," << setprecision(4) << buf_warped_depth_residual[j] <<
                                "," << buf_warped_weight[j] <<"]" << " ";
                    }
                    cout << endl;
                    cout.unsetf(ios_base::fixed);
                }
*/

                // after inc accpeted,// update scale matrix in current level/*
//                mean = calcResMean(buf_warped_color_residual, buf_warped_depth_residual, buf_warped_weight);
                Scale[lvl] = calcResScaleSSE(buf_warped_color_residual, buf_warped_depth_residual, mean, buf_warped_weight);
                if(debugPrint)
                {
                    cout << "Scale = " << endl << Scale[lvl] << endl;                // output debug lines
                    cout << "Precision = " << endl << Scale[lvl].inverse() << endl;
                }

                resOld = calcRes(lvl, refToNew_new, setting_coarseCutoffTH*levelCutoffRepeat);  // calc res using updated scale
                applyStep(lvl);
                calcGSSSE(lvl, H, b, refToNew_new);
                refToNew_current = refToNew_new;
                lamba *= 0.5;

            }
            else
            {
                lamba *= 4;
                if(lamba < lambaExtrapolationLimit)  lamba = lambaExtrapolationLimit;
                if(lamba > 1e4) lamba = 1e4;
            }

            if(!(inc.norm() > 1e-4))
            {
                if(debugPrint)
                {
                    cout << "#### inc too small, end optimization in current level!" << endl;
                }

                break;
            }
        }

        lastResiduals[lvl] = sqrtf((float)(resOld[0] / resOld[1]));
        lastFlowIndicators = resOld.segment<3>(2);

        IteInfoLvl.currentlvl = lvl;
        IteInfoLvl.currentNumInEM = IterationStatus.IterationNumOfGEM[lvl] + 1;
        IteInfoLvl.iterations = IterationNum;
        IteInfoLvl.acceptedIterations = AcceptedIterationNum;
        IteInfoLvl.lamba = lamba; IteInfoLvl.leftEnergy = lastResiduals[lvl];
        IteInfoLvl.num_GoodRes = buf_warped_num;

        IterationStatus.IteInfo[lvl].push_back(IteInfoLvl);

        float ratio = (float)(buf_warped_num) / (float)(NumPnts[lvl]);
        if(ratio <= 0.002) return false;                // when ratio < 0.2, tacking failed

        // debug lines
        if(debugPrint)
        {
            cout << "GN information in level " << lvl << " :" << endl
                  <<"   " << "iterations : " << IterationNum << endl
                  <<"   " << "acceptedIterations : " << AcceptedIterationNum << endl
                  <<"   " << "lamba : " << lamba << endl
                  <<"   " << "leftEnergy : " << lastResiduals[lvl] << endl
                  <<"   " << "num_GoodRes : " << buf_warped_num << endl;
            cout << "********************************************************************************" << endl;
        }

        IterationStatus.IterationNumOfGEM[lvl]++;
        IterationStatus.TDistributionLogLikelihood = lastResiduals[0];
        if(IterationStatus.IterationNumOfGEM[lvl] < IterationStatus.MaxIterationNumOfGEM[lvl])
        {                       
            if(debugPrint)
                cout << "Continue a new EM pipeline in level " << lvl << "..." << endl;
            lvl++;
        }
        else
        {
            if(lvl != 0)
                Scale[lvl-1] = Scale[lvl];     // pass current scale params to next level
        }

 //       if(lastResiduals[lvl] > 1.5*minResForAbort[lvl])  return false;   // how to set minResForAbort ?
        if(levelCutoffRepeat > 1 && !haveRepeated)
        {
            lvl++;                            // doubt it ?
            haveRepeated = true;
            cout << "Repeat current level" << endl;
        }
    }

    lastToNew_out = refToNew_current;
    return true;
}

void Tracker::applyStep(int lvl)
{
    SelectPnt* pts = SeltPnts[lvl];
    int npt = NumPnts[lvl];

    for(int i = 0; i < npt; i++)
    {
        pts[i].isGood = pts[i].isGood_new;
        pts[i].energy = pts[i].energy_new;
    }
}

//*************************************************************************************************
//Implementaion of MI-based optimizaion functions
void Tracker::SampleRefPnts(int lvl, float threshold, float ratio)
{
    float Scale = (float)((BinsOfHis)/ 255.0f);     // used to scale pixel intensity
    int NumInLvl = (int)(ratio*NumPntsInMI);

    int num = NumPnts[lvl];
    SelectPnt* pnts = SeltPnts[lvl];
    vector<SelectPnt> Candidates;
    Candidates.clear();

    for(int i = 0; i < num; i++)
    {
        SelectPnt pnt = pnts[i];
        if(!pnt.isGood || (pnt.norm < threshold))
            continue;
        Candidates.push_back(pnt);
    }

    if(Candidates.size() < NumInLvl)
    {
        NumSampPnts = Candidates.size();
        std::sort(Candidates.begin(), Candidates.end(), greater<SelectPnt>());   // from large to small
    }
    else
    {
        std::sort(Candidates.begin(), Candidates.end(), greater<SelectPnt>());
        Candidates.resize(NumInLvl);
        NumSampPnts = NumInLvl;
    }

    for(int i = 0; i < NumSampPnts; i++)
    {
        SelectPnt pnt = Candidates[i];
        Vector3 pt = Ki[lvl] * Vector3(pnt.u, pnt.v, 1);
        float pu = pt[0] / pt[2];
        float pv = pt[1] / pt[2];

        buf_MI_ref_color[i] = pnt.color;
        buf_MI_ref_idepth[i] = pnt.idepth;
        buf_MI_ref_u[i] = pu;
        buf_MI_ref_v[i] = pv;
        buf_MI_ref_idx[i] = pnt.colorGrad(0);
        buf_MI_ref_idy[i] = pnt.colorGrad(1);
        buf_MI_ref_scaledIntensity[i] = pnt.color * Scale;        
        buf_MI_energy[i] = 0;
    }

    while(NumSampPnts%4 != 0)
    {
        buf_MI_ref_color[NumSampPnts] = 0;
        buf_MI_ref_idepth[NumSampPnts] = -1.0f;
        buf_MI_ref_u[NumSampPnts] = 0;
        buf_MI_ref_v[NumSampPnts] = 0;
        buf_MI_ref_idx[NumSampPnts] = 0;
        buf_MI_ref_idy[NumSampPnts] = 0;
        buf_MI_ref_scaledIntensity[NumSampPnts] = 0;
        buf_MI_energy[NumSampPnts] = 0;
        NumSampPnts++;
    }
}

void Tracker::calcRefSplineBuf(const float *scaleIntensity)
{
    int n = NumSampPnts;
    assert(n%4==0);

    for(int bin = 0; bin < BinsOfHis; bin++)
    {
        for(int i = 0; i < n; i++)
        {
            float err = (bin+1) - scaleIntensity[i];
            BSpline_interpolation(err, buf_MI_ref_BSpline[bin][i], true);
            BSpline_interpolation_deriv(err, buf_MI_ref_BSpline_derivas[bin][i], true);
            BSpline_interpolation_deriv2(err, buf_MI_ref_BSpline_derivas2[bin][i], true);
        }
    }
}

void Tracker::calcRefHisto(Histo &refHisto)
{
    int n = NumSampPnts;
    assert(n%4==0);

    EIGEN_ALIGN16 float sum[4] = {0, 0, 0, 0};
    for(int bin = 0; bin < BinsOfHis; bin++)
    {
        memset(sum, 0, sizeof(float)*4);
        for(int k = 0; k < n; k+=4)
        {
            __m128 weight = _mm_load_ps(buf_MI_ref_weight+k);
            _mm_store_ps(sum, _mm_add_ps(_mm_load_ps(sum), _mm_mul_ps(weight, _mm_load_ps(buf_MI_ref_BSpline[bin]+k))));
        }

        refHisto(bin) = sum[0] + sum[1] + sum[2] + sum[3];
    }
}

void Tracker::calcCurSplineBuf(int lvl, const SE3 &refTonew)
{
    float Scale = (float)(BinsOfHis / 255.0f);     // used to scale pixel intensity
    int n = NumSampPnts;
    assert(n%4==0);

    Mat3x3 Kl = K[lvl];
    Mat3x3 Kil = Ki[lvl];
    float fxl = fx[lvl], fyl = fy[lvl];
    float cxl = cx[lvl], cyl = cy[lvl];
    float wl = w[lvl], hl = h[lvl];

    cv::Mat gray = newFrame->intensity[lvl];
    cv::Mat depth = newFrame->depth[lvl];
    Mat3x3 R = refTonew.rotationMatrix().cast<float>();
    Vector3 t = refTonew.translation().cast<float>();
    Mat3x3 KRKi = Kl * R * Kil;
    Mat2x2 RPlane = KRKi.topLeftCorner<2,2>();

    NormalizedConst = 0;
    NumVisibility = 0;
    for(int i = 0; i < n; i++)
    {
        float ref_color = buf_MI_ref_color[i];
        float x = buf_MI_ref_u[i];
        float y = buf_MI_ref_v[i];
        float idpt = buf_MI_ref_idepth[i];
        if(idpt == -1)
        {
            buf_MI_visibility[i] = 0;
            buf_MI_ref_weight[i] = 0;
            for(int bin = 0; bin < BinsOfHis; bin++)
            {
                buf_MI_cur_BSpline[bin][i] = 0;
            }
            continue;
        }

        Vector3 pt = R * Vector3(x, y, 1) + t*idpt;
        float Ku = fxl*(pt[0]/pt[2]) + cxl;
        float Kv = fyl*(pt[1]/pt[2]) + cyl;

        if(!(Ku > 3 && Kv > 3 && Ku < wl-3 && Kv < hl-3))
        {
            buf_MI_visibility[i] = 0;
            buf_MI_ref_weight[i] = 0;
            for(int bin = 0; bin < BinsOfHis; bin++)
            {
                buf_MI_cur_BSpline[bin][i] = 0;

            }
            continue;
        }

        Vector2 hitData = bilinearInterpolation(gray, depth, Ku, Kv);
        if(!std::isfinite(hitData[0]))
        {
            buf_MI_visibility[i] = 0;
            buf_MI_ref_weight[i] = 0;
            for(int bin = 0; bin < BinsOfHis; bin++)
            {
                buf_MI_cur_BSpline[bin][i] = 0;
            }
        }
        else
        {
//            buf_MI_energy[i] = (ref_color - hitData[0])*(ref_color - hitData[0]);
            buf_MI_energy[i] = fabs(ref_color - hitData[0]);          //update energy term for good points
            buf_MI_visibility[i] = 1.0f;
            if(UseWeight)
            {
                float u = fxl*x + cxl; float v = fyl*y + cyl;
                buf_MI_ref_weight[i] = calcWeights(Vector2(u,v), Vector2(Ku,Kv),
                                                   lastRef->intensity[lvl], gray, RPlane);
            }
            else
            {
                buf_MI_ref_weight[i] = 1.0f;
            }
            NormalizedConst += buf_MI_ref_weight[i];
            NumVisibility += buf_MI_visibility[i];

            for(int bin = 0; bin < BinsOfHis; bin++)
            {
                float err = (bin+1) - hitData[0] * Scale;
                BSpline_interpolation(err, buf_MI_cur_BSpline[bin][i], true);
            }
        }

    }
}

void Tracker::calcCurHisto(Histo &curHisto)
{
    int n = NumSampPnts;
    assert(n%4==0);

    EIGEN_ALIGN16 float sum[4] = {0, 0, 0, 0};
    for(int bin = 0; bin < BinsOfHis; bin++)
    {
        memset(sum, 0, sizeof(float)*4);
        for(int k = 0; k < n; k+=4)
        {
            __m128 weight = _mm_load_ps(buf_MI_ref_weight+k);
            _mm_store_ps(sum, _mm_add_ps(_mm_load_ps(sum), _mm_mul_ps(weight, _mm_load_ps(buf_MI_cur_BSpline[bin]+k))));
        }

        curHisto(bin) = sum[0] + sum[1] + sum[2] + sum[3];
    }
}

float Tracker::calcWeights(const Vector2 &p, const Vector2 &Kp,
                          const cv::Mat &ref, const cv::Mat &cur, const Mat2x2 &Rplane)
{
    Vector2 refCentroid = Vector2::Zero();
    Vector2 curCentroid = Vector2::Zero();
    Vector2 traCentroid = Vector2::Zero();

    refCentroid = ComputeGrayCentroid(ref, p(0), p(1));
    curCentroid = ComputeGrayCentroid(cur, Kp(0), Kp(1));
    traCentroid = Rplane * refCentroid;

    float cosine = (traCentroid.dot(curCentroid)) / (traCentroid.norm()*curCentroid.norm());

    return fabs(cosine);
}

void Tracker::calcJointHisto(float **refbuff, float **curbuff, const float* weight, JointHisto &jhisto)
{
    int n = NumSampPnts;
    assert(n%4==0);

    EIGEN_ALIGN16 float SSEData[4*BinsOfHis*BinsOfHis];
    memset(SSEData, 0, sizeof(float)*4*BinsOfHis*BinsOfHis);
    float* pt = SSEData;

    for(int r = 0; r < BinsOfHis; r++)
       for(int t = 0; t < BinsOfHis; t++)
       {
            const float* refbuffBin = refbuff[r];
            const float* curbuffBin = curbuff[t];
            for(int i = 0; i < n; i+=4)
            {
                _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_mul_ps(_mm_load_ps(weight+i), _mm_mul_ps(_mm_load_ps(refbuffBin+i), _mm_load_ps(curbuffBin+i)))));
            }

            jhisto(r,t) = *pt + *(pt+1) + *(pt+2) + *(pt+3);
            pt+=4;
       }
}

void Tracker::calcHistoFromJointHisto(Histo &refHisto, Histo &curHisto, const JointHisto &Joint)
{
    for(int i = 0; i < BinsOfHis; i++)
    {
        refHisto = Joint.rowwise().sum().transpose();
        curHisto = Joint.colwise().sum();
    }
}

void Tracker::PrecalcGaussian(int lvl)
{
    int n = NumSampPnts;
    assert(n%4==0);

    float Scale = (float)(BinsOfHis / 255.0f);  // Scale factor
//    Scale = 1.0f;

    __m128 fxl = _mm_set1_ps(fx[lvl] * Scale);
    __m128 fyl = _mm_set1_ps(fy[lvl] * Scale);
    __m128 one = _mm_set1_ps(1.0f);
    __m128 zero = _mm_set1_ps(0);

    for(int i = 0; i < n; i+=4)
    {
        __m128 idx = _mm_mul_ps(_mm_load_ps(buf_MI_ref_idx+i), fxl);
        __m128 idy = _mm_mul_ps(_mm_load_ps(buf_MI_ref_idy+i), fyl);
        __m128 u = _mm_load_ps(buf_MI_ref_u+i);
        __m128 v = _mm_load_ps(buf_MI_ref_v+i);
        __m128 idepth = _mm_load_ps(buf_MI_ref_idepth+i);

        _mm_store_ps(buf_MI_PrecalGauss[0]+i, _mm_mul_ps(idepth, idx));  // J00

        _mm_store_ps(buf_MI_PrecalGauss[1]+i, _mm_mul_ps(idepth, idy));  // J01

        _mm_store_ps(buf_MI_PrecalGauss[2]+i, _mm_sub_ps(zero, _mm_mul_ps(idepth, _mm_add_ps(_mm_mul_ps(u, idx), _mm_mul_ps(v, idy))))); // J02

        _mm_store_ps(buf_MI_PrecalGauss[3]+i, _mm_sub_ps(zero, _mm_add_ps(
                                     _mm_mul_ps(_mm_mul_ps(u,v),idx),
                                     _mm_mul_ps(_mm_add_ps(one,_mm_mul_ps(v,v)), idy)))); // J03
        _mm_store_ps(buf_MI_PrecalGauss[4]+i, _mm_add_ps(
                                                _mm_mul_ps(_mm_add_ps(one, _mm_mul_ps(u,u)), idx),
                                                _mm_mul_ps(_mm_mul_ps(u,v), idy)));  // J04

        _mm_store_ps(buf_MI_PrecalGauss[5]+i, _mm_sub_ps(_mm_mul_ps(u,idy), _mm_mul_ps(v,idx))); //J05
    }
}

void Tracker::calcHistDerivates(Vector6 *joint_Derv, Vector6 *joint_converg_Derv, Vector6 *ref_Derv)
{
    int n = NumSampPnts;
    assert(n%4==0);

    __m128 zero = _mm_set1_ps(0);

    EIGEN_ALIGN16 float tmp1[4*6];
    EIGEN_ALIGN16 float tmp2[4*6];
    EIGEN_ALIGN16 float tmp3[4*6];

    EIGEN_ALIGN16 float J0_bak[n];
    EIGEN_ALIGN16 float J1_bak[n];
    EIGEN_ALIGN16 float J2_bak[n];
    EIGEN_ALIGN16 float J3_bak[n];
    EIGEN_ALIGN16 float J4_bak[n];
    EIGEN_ALIGN16 float J5_bak[n];

    bool Precalc = true;
    for(int r = 0; r < BinsOfHis; r++)
    {
        if(histo_ref(r) == 0)
            continue;

        memset(J0_bak, 0, sizeof(float)*n);
        memset(J1_bak, 0, sizeof(float)*n);
        memset(J2_bak, 0, sizeof(float)*n);
        memset(J3_bak, 0, sizeof(float)*n);
        memset(J4_bak, 0, sizeof(float)*n);
        memset(J5_bak, 0, sizeof(float)*n);

        memset(tmp1, 0, sizeof(float)*4*6);
        Precalc = true;
        for(int t = 0; t < BinsOfHis; t++)
        {
            if(histo_cur(t)==0 || JHisto(r,t)==0)
                continue;

            memset(tmp2, 0, sizeof(float)*4*6);
            memset(tmp3, 0, sizeof(float)*4*6);
            for(int i = 0; i < n; i+=4)
            {
               __m128 weight = _mm_load_ps(buf_MI_ref_weight+i);

               if(Precalc)
               {
                   __m128 BSpline_deriv = _mm_mul_ps(weight, _mm_sub_ps(zero, _mm_load_ps(buf_MI_ref_BSpline_derivas[r]+i)));
                   __m128 J00 = _mm_load_ps(buf_MI_PrecalGauss[0]+i);
                   __m128 J01 = _mm_load_ps(buf_MI_PrecalGauss[1]+i);
                   __m128 J02 = _mm_load_ps(buf_MI_PrecalGauss[2]+i);
                   __m128 J03 = _mm_load_ps(buf_MI_PrecalGauss[3]+i);
                   __m128 J04 = _mm_load_ps(buf_MI_PrecalGauss[4]+i);
                   __m128 J05 = _mm_load_ps(buf_MI_PrecalGauss[5]+i);

                   _mm_store_ps(J0_bak+i, _mm_mul_ps(J00, BSpline_deriv));
                   _mm_store_ps(J1_bak+i, _mm_mul_ps(J01, BSpline_deriv));
                   _mm_store_ps(J2_bak+i, _mm_mul_ps(J02, BSpline_deriv));
                   _mm_store_ps(J3_bak+i, _mm_mul_ps(J03, BSpline_deriv));
                   _mm_store_ps(J4_bak+i, _mm_mul_ps(J04, BSpline_deriv));
                   _mm_store_ps(J5_bak+i, _mm_mul_ps(J05, BSpline_deriv));

                   float* pt = tmp1;
                   _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_load_ps(J0_bak+i))); pt+=4;
                   _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_load_ps(J1_bak+i))); pt+=4;
                   _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_load_ps(J2_bak+i))); pt+=4;
                   _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_load_ps(J3_bak+i))); pt+=4;
                   _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_load_ps(J4_bak+i))); pt+=4;
                   _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt), _mm_load_ps(J5_bak+i))); pt+=4;

               }

               __m128 BSpline_value = _mm_load_ps(buf_MI_cur_BSpline[t]+i);     // Actual, for Jocabian matrix

               __m128 WJ00 = _mm_mul_ps(_mm_load_ps(J0_bak+i), BSpline_value);
               __m128 WJ01 = _mm_mul_ps(_mm_load_ps(J1_bak+i), BSpline_value);
               __m128 WJ02 = _mm_mul_ps(_mm_load_ps(J2_bak+i), BSpline_value);
               __m128 WJ03 = _mm_mul_ps(_mm_load_ps(J3_bak+i), BSpline_value);
               __m128 WJ04 = _mm_mul_ps(_mm_load_ps(J4_bak+i), BSpline_value);
               __m128 WJ05 = _mm_mul_ps(_mm_load_ps(J5_bak+i), BSpline_value);

               float* pt2 = tmp2;
               _mm_store_ps(pt2, _mm_add_ps(_mm_load_ps(pt2), WJ00)); pt2+=4;
               _mm_store_ps(pt2, _mm_add_ps(_mm_load_ps(pt2), WJ01)); pt2+=4;
               _mm_store_ps(pt2, _mm_add_ps(_mm_load_ps(pt2), WJ02)); pt2+=4;
               _mm_store_ps(pt2, _mm_add_ps(_mm_load_ps(pt2), WJ03)); pt2+=4;
               _mm_store_ps(pt2, _mm_add_ps(_mm_load_ps(pt2), WJ04)); pt2+=4;
               _mm_store_ps(pt2, _mm_add_ps(_mm_load_ps(pt2), WJ05)); pt2+=4;

               if(UseConvergEstimate)
               {
                   BSpline_value = _mm_load_ps(buf_MI_ref_BSpline[t]+i);       // Approximation, for Hessian matrix at converged point

                   WJ00 = _mm_mul_ps(_mm_load_ps(J0_bak+i), BSpline_value);
                   WJ01 = _mm_mul_ps(_mm_load_ps(J1_bak+i), BSpline_value);
                   WJ02 = _mm_mul_ps(_mm_load_ps(J2_bak+i), BSpline_value);
                   WJ03 = _mm_mul_ps(_mm_load_ps(J3_bak+i), BSpline_value);
                   WJ04 = _mm_mul_ps(_mm_load_ps(J4_bak+i), BSpline_value);
                   WJ05 = _mm_mul_ps(_mm_load_ps(J5_bak+i), BSpline_value);

                   float* pt3 = tmp3;
                   _mm_store_ps(pt3, _mm_add_ps(_mm_load_ps(pt3), WJ00)); pt3+=4;
                   _mm_store_ps(pt3, _mm_add_ps(_mm_load_ps(pt3), WJ01)); pt3+=4;
                   _mm_store_ps(pt3, _mm_add_ps(_mm_load_ps(pt3), WJ02)); pt3+=4;
                   _mm_store_ps(pt3, _mm_add_ps(_mm_load_ps(pt3), WJ03)); pt3+=4;
                   _mm_store_ps(pt3, _mm_add_ps(_mm_load_ps(pt3), WJ04)); pt3+=4;
                   _mm_store_ps(pt3, _mm_add_ps(_mm_load_ps(pt3), WJ05)); pt3+=4;
                }
            }
            Precalc = false;

            Vector6 J2 = Vector6::Zero();
            for(int k = 0; k <= 5; k++)
            {
                int idx = 4*k;
                J2[k] = tmp2[idx] + tmp2[idx+1] + tmp2[idx+2] + tmp2[idx+3];
            }
            int index = r*BinsOfHis + t;
            joint_Derv[index] = J2;

            if(UseConvergEstimate)
            {
                Vector6 J3 = Vector6::Zero();
                for(int k = 0; k <= 5; k++)
                {
                    int idx = 4*k;
                    J3[k] = tmp3[idx] + tmp3[idx+1] + tmp3[idx+2] + tmp3[idx+3];
                }
                int index = r*BinsOfHis + t;
                joint_converg_Derv[index] = J3;
            }
        }

        Vector6 J1;
        for(int k = 0; k <= 5; k++)
        {
            int idx = 4*k;
            J1[k] = tmp1[idx] + tmp1[idx+1] + tmp1[idx+2] + tmp1[idx+3];
        }
        ref_Derv[r] = J1;
    }
}

void Tracker::calcHist2ndDerivates(Mat6x6 *joint_2nd_Derv)
{
    int n = NumSampPnts;
    assert(n%4==0);

    Accumulator7 accRT;
    accRT.initialize();

    __m128 zero = _mm_set1_ps(0);

    EIGEN_ALIGN16 float J0_bak[n];
    EIGEN_ALIGN16 float J1_bak[n];
    EIGEN_ALIGN16 float J2_bak[n];
    EIGEN_ALIGN16 float J3_bak[n];
    EIGEN_ALIGN16 float J4_bak[n];
    EIGEN_ALIGN16 float J5_bak[n];

    bool Precalc = true;
    for(int r = 0; r < BinsOfHis; r++)
    {
        if(histo_ref(r) == 0)
            continue;

        memset(J0_bak, 0, sizeof(float)*n);
        memset(J1_bak, 0, sizeof(float)*n);
        memset(J2_bak, 0, sizeof(float)*n);
        memset(J3_bak, 0, sizeof(float)*n);
        memset(J4_bak, 0, sizeof(float)*n);
        memset(J5_bak, 0, sizeof(float)*n);

        Precalc = true;
        for(int t = 0; t < BinsOfHis; t++)
        {
            if(histo_cur(t)==0 || JHisto(r,t)==0)
                continue;

            for(int i = 0; i < n; i+=4)
            {
                __m128 weight = _mm_load_ps(buf_MI_ref_weight+i);

                if(Precalc)
                {
                    __m128 BSpline_deriv_2 = _mm_mul_ps(weight, _mm_sub_ps(zero, _mm_load_ps(buf_MI_ref_BSpline_derivas2[r]+i)));
                    __m128 J00 = _mm_load_ps(buf_MI_PrecalGauss[0]+i);
                    __m128 J01 = _mm_load_ps(buf_MI_PrecalGauss[1]+i);
                    __m128 J02 = _mm_load_ps(buf_MI_PrecalGauss[2]+i);
                    __m128 J03 = _mm_load_ps(buf_MI_PrecalGauss[3]+i);
                    __m128 J04 = _mm_load_ps(buf_MI_PrecalGauss[4]+i);
                    __m128 J05 = _mm_load_ps(buf_MI_PrecalGauss[5]+i);

                    _mm_store_ps(J0_bak+i, _mm_mul_ps(J00, BSpline_deriv_2));
                    _mm_store_ps(J1_bak+i, _mm_mul_ps(J01, BSpline_deriv_2));
                    _mm_store_ps(J2_bak+i, _mm_mul_ps(J02, BSpline_deriv_2));
                    _mm_store_ps(J3_bak+i, _mm_mul_ps(J03, BSpline_deriv_2));
                    _mm_store_ps(J4_bak+i, _mm_mul_ps(J04, BSpline_deriv_2));
                    _mm_store_ps(J5_bak+i, _mm_mul_ps(J05, BSpline_deriv_2));
                }

                __m128 BSpline_value = _mm_load_ps(buf_MI_cur_BSpline[t]+i);     // Actual, for Jocabian matrix
                if(UseConvergEstimate)
                    BSpline_value = _mm_load_ps(buf_MI_ref_BSpline[t]+i);

                __m128 WJ00 = _mm_mul_ps(_mm_load_ps(J0_bak+i), BSpline_value);
                __m128 WJ01 = _mm_mul_ps(_mm_load_ps(J1_bak+i), BSpline_value);
                __m128 WJ02 = _mm_mul_ps(_mm_load_ps(J2_bak+i), BSpline_value);
                __m128 WJ03 = _mm_mul_ps(_mm_load_ps(J3_bak+i), BSpline_value);
                __m128 WJ04 = _mm_mul_ps(_mm_load_ps(J4_bak+i), BSpline_value);
                __m128 WJ05 = _mm_mul_ps(_mm_load_ps(J5_bak+i), BSpline_value);

                accRT.updateSSE_weighted(WJ00, WJ01, WJ02, WJ03, WJ04, WJ05, zero, weight);
            }

            accRT.finish();
            int index = r*BinsOfHis + t;
            joint_2nd_Derv[index] = -accRT.H.block<6,6>(0,0);
        }
        Precalc = false;
    }
}

void Tracker::calcHessian(Mat6x6 &H_out, const Vector6* joint_Derv, const Vector6* ref_Derv)
{
    AccumulatorXX<6,6> accRT; accRT.initialize();
    AccumulatorXX<6,6> accR;  accR.initialize();

    JointHisto ApproxiJhisto;
    Histo Approxihisto_ref;
    if(UseConvergEstimate)
    {
//        calcJointHisto(buf_MI_ref_BSpline, buf_MI_ref_BSpline, buf_MI_ref_weight, ApproxiJhisto);    // Approxiamation ??
//        Approxihisto_ref = ApproxiJhisto.rowwise().sum().transpose();
        ApproxiJhisto = this->JHisto;                             // actual
        Approxihisto_ref = this->histo_ref;
    }
    else
    {
        ApproxiJhisto = this->JHisto;                             // actual
        Approxihisto_ref = this->histo_ref;
    }

    if(debugMInfo)
    {
        cout << "ApproxiJhisto = " << endl << ApproxiJhisto << endl;
    }


    for(int r = 0; r < BinsOfHis; r++)
    {
        if(Approxihisto_ref(r) == 0) continue;

        for(int t = 0; t < BinsOfHis; t++)
        {
            if(histo_cur(t)==0 || ApproxiJhisto(r,t)==0)
                continue;

            int index = r * BinsOfHis + t;
            Vector6 J2 = joint_Derv[index];
            float weight2 = 1.0f / ApproxiJhisto(r,t);
            accRT.update(J2, J2, weight2);
        }

        Vector6 J1 = ref_Derv[r];
        float weight1 = 1.0f / Approxihisto_ref(r);
        accR.update(J1, J1, weight1);
    }

    accR.finish();
    accRT.finish();

    H_out = (0.01/(NormalizedConst))*(accRT.A1m - accR.A1m);             // 1.0f / NumSampPnts

/*
    //debug eigenvector
    Eigen::SelfAdjointEigenSolver<Mat6x6> eigensolver(H_out);
    cout << "The eigenvalues of H : " << eigensolver.eigenvalues().transpose() << endl;
*/
}

void Tracker::calcJocabian(Vector6 &b_out, const Vector6* joint_Derv)
{
    AccumulatorX<6> accJ;
    accJ.initialize();

    for(int r = 0; r < BinsOfHis; r++)
    {
        if(histo_ref(r) == 0) continue;

        for(int t = 0; t < BinsOfHis; t++)
        {
            if(JHisto(r,t) == 0) continue;

            int index = r*BinsOfHis + t;
            Vector6 J2 = joint_Derv[index];

            float value = JHisto(r,t) / histo_ref(r);
            float weight2 = 1.0f + log(value);                     // loge
 //         float weight2 = 1.0f + log10(value) / log10(2);        // log2

            accJ.update(J2, weight2);
        }

    }
    accJ.finish();
    b_out = (0.01/(NormalizedConst))*accJ.A1m;
}

void Tracker::calcJocaAndHessn(Mat6x6 &H_out, Vector6 &b_out, const float &refEnpy, const float &curEnpy, const float &jotEnpy,
                               const Vector6 *joint_Derv, const Vector6* joint_converg_dev, const Vector6 *ref_Derv, const Mat6x6* joint_2nd_Derv)
{
    float n = NormalizedConst;
    float constant = log(n);

    float A = jotEnpy;     // joint Entropy
    float B = refEnpy + curEnpy;             // sum of Entropy

    // compute derivate of A and B : Ad, Bd**************************************************
    Vector6 Ad, Bd, Ad_converg, Bd_converg;
    AccumulatorX<6> accAd, accBd, accAd_converg, accBd_converg;
    accAd.initialize(); accBd.initialize(); accAd_converg.initialize(); accBd_converg.initialize();
    for(int r = 0; r < BinsOfHis; r++)
    {
        if(histo_ref(r) == 0) continue;
        for(int t = 0; t < BinsOfHis; t++)
        {
            if(JHisto(r,t) == 0) continue;

            int index = r*BinsOfHis + t;
            Vector6 J2 = joint_Derv[index];

            float weight1 = log(histo_ref(r)) - constant;
            float weight2 = 1.0f + log(JHisto(r,t)) - constant;                     // loge

            accAd.update(J2, weight2);
            accBd.update(J2, weight1);

            if(UseConvergEstimate)
            {
                Vector6 J2_converg = joint_converg_dev[index];
                accAd_converg.update(J2_converg, weight2);
                accBd_converg.update(J2_converg, weight1);
            }
        }

    }
    accAd.finish(); accBd.finish();
    Ad = (1.0f/n)*accAd.A1m; Bd = (1.0f/n)*accBd.A1m;
    if(UseConvergEstimate)
    {
        accAd_converg.finish(); accBd_converg.finish();
        Ad_converg = (1.0f/n)*accAd_converg.A1m; Bd_converg = (1.0f/n)*accBd_converg.A1m;
    }

    // compute jocabian and hessian matrix of normalized mutual info *****************************
    b_out = (0.01/(A*A)) * (A*Bd - B*Ad);     // jocabian vector

    Mat6x6 term1, term2, term3, term4 = Mat6x6::Zero();
    AccumulatorXX<6,6> accRT; accRT.initialize();
    AccumulatorXX<6,6> accR;  accR.initialize();

    if(UseConvergEstimate)
        term1 = 2 * ( B/(A*A*A)*(Ad_converg*Ad_converg.transpose()) - 1.0f/(A*A)*(Ad_converg*Bd_converg.transpose()) );
    else
        term1 = 2 * ( B/(A*A*A)*(Ad*Ad.transpose()) - 1.0f/(A*A)*(Ad*Bd.transpose()) );

    for(int r = 0; r < BinsOfHis; r++)
    {
        if(histo_ref(r) == 0) continue;

        for(int t = 0; t < BinsOfHis; t++)
        {
            if(histo_cur(t)==0 || JHisto(r,t)==0)
                continue;

            int index = r * BinsOfHis + t;
            Vector6 J2 = joint_Derv[index];
            if(UseConvergEstimate)
                J2 = joint_converg_dev[index];
            float weight2 = 1.0f / JHisto(r,t);
            accRT.update(J2, J2, weight2);

            if(Use2ndDerivates)
            {
                float weight3 = (1.0f/A)*(log(histo_ref(r)) - constant) - (B/(A*A))*(1.0f + log(JHisto(r,t)) - constant);
                term4 += weight3 * joint_2nd_Derv[index];
            }
        }

        Vector6 J1 = ref_Derv[r];
        float weight1 = 1.0f / histo_ref(r);
        accR.update(J1, J1, weight1);
    }

    accR.finish(); accRT.finish();
    term2 = (1.0f/n)*B/(A*A)*accRT.A1m;
    term3 = (1.0f/n)*(1.0f/A)*accR.A1m;
    term4 = (1.0f/n)*term4;

    if(!Use2ndDerivates)
        H_out = 0.01*(term1 - term2 + term3);
    else
        H_out = 0.01*(term1 - term2 + term3 + term4);       // Hessian matrix, ignoring 2nd-order term

}

double Tracker::calcMutualInfo(const float &ref, const float &cur, const float &joint)
{
    return -(ref + cur - joint);
}

double Tracker::calcNormaMI(const float& ref, const float& cur, const float& joint)
{
    return (ref + cur) / joint;
}

float Tracker::checkEnergy()
{
    int num = NumSampPnts;
    assert(n%4==0);

    float E = 0;
    float numTermsInE = num;
    EIGEN_ALIGN16 float sumE[4] = {0,0,0,0};

    for(int i = 0; i < num; i+=4)
    {
        _mm_store_ps(sumE, _mm_add_ps(_mm_load_ps(sumE), _mm_mul_ps(_mm_load_ps(buf_MI_ref_weight+i), _mm_load_ps(buf_MI_energy+i))));
    }
    E = sumE[0] + sumE[1] + sumE[2] + sumE[3];

    return (E/numTermsInE);
}

bool Tracker::OptimizMI(SE3 &refTocur_out, Vector5f MInfoTH, Vector5f EnergyTH, int UsedLvl)
{
    int UsePyrLvls = UsedLvl;                    // default: PYR_LEVELS

    string weight = paraReader.getData<string>("UseWeight");
    if(weight == "true") UseWeight = true;
    else UseWeight = false;

    int maxIterations[] = {20,20,30,50,50};         //10,20,25,25,25; 20,20,30,50,50
//    float ratio[] = {0.6, 0.6, 0.5, 0.4, 0.2};      //0.6, 0.6, 0.5, 0.4, 0.2

    lastMInfo = Vector5f::Zero();
    lastEnergy = Vector5f::Constant(1e6);
    MInfOptiInfo.clear();                         // record optimization Info in current frame
//    cout << "##########################################################################################" << endl;
    SE3 refTocur_current = refTocur_out;
    float MIEstimate = 0; double MInfo_Old = 0; float Energy_Old = 0;
    float refEnpy = 0, curEnpy = 0, jotEnpy = 0;
    Vector6* joint_Derv = new Vector6[BinsOfHis*BinsOfHis];
    Vector6* joint_converg_Derv = new Vector6[BinsOfHis*BinsOfHis];
    Vector6* ref_Derv   = new Vector6[BinsOfHis];
    Mat6x6* joint_2nd_Derv = new Mat6x6[BinsOfHis*BinsOfHis];
    for(int lvl = UsePyrLvls - 1; lvl >= 0; lvl--)
    {
//        if(lvl == 0) UseWeight = true;

        IterationPrintInfo InfoLvl;              // record optimiz Info in current level
        SampleRefPnts(lvl, 8*8, ratio[lvl]);     // sample points in ref frame

        // initial respective Histo and Joint-Histo
        calcRefSplineBuf(buf_MI_ref_scaledIntensity);
        calcCurSplineBuf(lvl, refTocur_current);
 //       calcRefHisto(this->histo_ref); calcCurHisto(this->histo_cur);
        calcJointHisto(buf_MI_ref_BSpline, buf_MI_cur_BSpline, buf_MI_ref_weight, this->JHisto);
        calcHistoFromJointHisto(this->histo_ref, this->histo_cur, this->JHisto);
        calcEntropy(this->histo_ref, this->histo_cur, this->JHisto, refEnpy, curEnpy, jotEnpy);

        if(debugMInfo)            // debug lines
        {
            cout << "Histo_Ref = " << histo_ref << endl;
            cout << "Histo_Cur = " << histo_cur << endl;
            cout << "JointHisto = " << endl << JHisto << endl;
        }

        Energy_Old = checkEnergy();
        if(UseNormMInfo)
            MInfo_Old = calcNormaMI(refEnpy, curEnpy, jotEnpy);
        else
            MInfo_Old = calcMutualInfo(refEnpy, curEnpy, jotEnpy);
//        cout << "MIOld = " << MInfo_Old << endl;

        PrecalcGaussian(lvl);           // precalc gaussians, usually perform only once in each lvl
        calcHistDerivates(joint_Derv, joint_converg_Derv, ref_Derv);

        if(Use2ndDerivates)
            calcHist2ndDerivates(joint_2nd_Derv);

        Mat6x6 H; Vector6 J;
        if(UseNormMInfo)
            calcJocaAndHessn(H, J, refEnpy, curEnpy, jotEnpy, joint_Derv, joint_converg_Derv, ref_Derv, joint_2nd_Derv);
        else
        {
            calcJocabian(J, joint_Derv);
            if(UseConvergEstimate)
                calcHessian(H, joint_converg_Derv, ref_Derv);         // initialize Hessian matrix, not change afterwards
            else
                calcHessian(H, joint_Derv, ref_Derv);
        }


        if(debugMInfo)
        {
            cout << "Hessian = " << endl << H << endl;
            cout << "Jocabian = " << J.transpose() << endl;
        }

        int IterationNum = 0, AcceptedIterations = 0;
        double final_MI = MInfo_Old;
        double MInfo_New = 0; float Energy_New = 0;
        float lamba = 0.1, lamba2 = 0.1;                               // lamba2 = 0.05
        float factor1 = 0.5, factor2 = 4;
        int failsNum = 0, successNum = 0;
        for(int ite = 1; ite <= maxIterations[lvl]; ite++)
        {
            IterationNum++;

            Mat6x6 Hl = H;
            for(int i = 0; i < 6; i++)
            {
                Hl(i,i) *= (1+lamba);
            }
            Vector6 inc = Hl.ldlt().solve(J);
            inc *= lamba2;

            SE3 refTocur_update = refTocur_current * SE3::exp(-inc.cast<double>());

            calcCurSplineBuf(lvl, refTocur_update);
 //           calcCurHisto(this->histo_cur);
            calcJointHisto(buf_MI_ref_BSpline, buf_MI_cur_BSpline, buf_MI_ref_weight, this->JHisto);
            calcHistoFromJointHisto(this->histo_ref, this->histo_cur, this->JHisto);
            calcEntropy(this->histo_ref, this->histo_cur, this->JHisto, refEnpy, curEnpy, jotEnpy);
            Energy_New = checkEnergy();
            if(UseNormMInfo)
                MInfo_New = calcNormaMI(refEnpy, curEnpy, jotEnpy);
            else
                MInfo_New = calcMutualInfo(refEnpy, curEnpy, jotEnpy);
     //     cout << "Inc = " << inc.transpose() << "; " << "MINew = " << MInfo_New << endl;

//            bool accept = (MInfo_New > MInfo_Old) && (Energy_New < Energy_Old);
            bool accept = (MInfo_New > MInfo_Old);
            if(accept)
            {
                AcceptedIterations++;
                final_MI = MInfo_New;
                //debug lines
                if(debugMInfo)
                    cout << "Inc = " << inc.transpose() << "; " << "MINew = " << MInfo_New << endl;

                MInfo_Old = MInfo_New; Energy_Old = Energy_New;
                refTocur_current = refTocur_update;

                calcHistDerivates(joint_Derv, joint_converg_Derv, ref_Derv);

                if(Use2ndDerivates)
                    calcHist2ndDerivates(joint_2nd_Derv);

                if(UseNormMInfo)
                    calcJocaAndHessn(H, J, refEnpy, curEnpy, jotEnpy, joint_Derv, joint_converg_Derv, ref_Derv, joint_2nd_Derv);
                {
                    calcJocabian(J, joint_Derv);
                    if(UseConvergEstimate)
                        calcHessian(H, joint_converg_Derv, ref_Derv);    // if not use estimate in convergence point, then recalc Hessian matrix
                    else
                        calcHessian(H, joint_Derv, ref_Derv);
                }

                lamba *= factor1;
                if(lamba < 1e-4) lamba = 1e-4;

                failsNum = 0;
                successNum++;
                if(successNum >= 2) lamba2 *= 1.05;
            }
            else
            {
                successNum = 0;
                failsNum++;
                if(failsNum >= 2)  lamba2 *= 0.65;

                lamba *= factor2;
                if(lamba > 1e4) lamba = 1e4;
            }

            if(!(inc.norm() > 1e-4))
            {
//                cout << "inc too small!" << endl;
                break;
            }

        }        
        lastMInfo(lvl) = MInfo_Old;  lastEnergy(lvl) = Energy_Old;
        MIEstimate = final_MI;

        InfoLvl.MuInfo = final_MI;
        InfoLvl.currentlvl = lvl;
        InfoLvl.iterations = IterationNum;
        InfoLvl.acceptedIterations = AcceptedIterations;
        InfoLvl.lamba = lamba;
        InfoLvl.iteStep = lamba2;
        InfoLvl.num_GoodRes = NumSampPnts;
        InfoLvl.ratio = (float)(NumVisibility) / (float)(NumSampPnts);

        MInfOptiInfo.push_back(InfoLvl);

        if(!isfinite(MInfo_Old))  return false;
        if(MInfo_Old < MInfoTH(lvl) && Energy_Old > 1.0*EnergyTH(lvl))   return false;
  //      if(MInfo_Old < MInfoTH(lvl))   return false;
    }

    refTocur_out = refTocur_current;

    delete[] joint_Derv;        // delete temporal pointers
    delete[] ref_Derv;
    delete[] joint_converg_Derv;
    delete[] joint_2nd_Derv;

    return true;
    //debug lines
//    cout << "Num_of_samplpoints -------> " << NumSampPnts << endl;
//    cout << "Num_of_visibility -------> " << NumVisibility << endl;
}

double Tracker::CalcRefMInfo(int lvl, const SE3 &refTocur)
{
    float refEnpy = 0, curEnpy = 0, jotEnpy = 0;

    SampleRefPnts(lvl, 8*8, ratio[lvl]);            // fill the buffer in current level
    calcRefSplineBuf(buf_MI_ref_scaledIntensity);   // calc ref-spline values in current level

    calcCurSplineBuf(lvl, refTocur);
    calcJointHisto(buf_MI_ref_BSpline, buf_MI_cur_BSpline, buf_MI_ref_weight, this->JHisto);
    calcHistoFromJointHisto(this->histo_ref, this->histo_cur, this->JHisto);
    calcEntropy(this->histo_ref, this->histo_cur, this->JHisto, refEnpy, curEnpy, jotEnpy);
    if(UseNormMInfo)
        return calcNormaMI(refEnpy, curEnpy, jotEnpy);
    else
        return calcMutualInfo(refEnpy, curEnpy, jotEnpy);
}

}
