
#include <Intializer.h>
namespace DSLAM
{

    Initializer::Initializer(int w, int h) : thisToNext( SE3() )
    {
        for(int level = 0; level < PYR_LEVELS; level++)
        {
            points[level] = 0;
            numPoints[level] = 0;
        }

        JbBuffer = new Vector8[w*h];
        JbBuffer_new = new Vector8[w*h];

        frameID = -1;
        printDebug = false;
        RegFlag = true;      // whether to use regularization term or not
    }

    Initializer::~Initializer()
    {
        for(int level = 0; level < PYR_LEVELS; level++)
        {
            delete[] points[level];
        }

        delete[] JbBuffer;
        delete[] JbBuffer_new;
    }

    void Initializer::setFirst(Frame *frame)
    {              
        const CameraIntrinsic* camera = frame->intrinsic;

        firstFrame = frame;
        makePyramid(camera);

        int w0 = frame->width[0], h0 = frame->height[0];
        PixelSelector selector(w0, h0);

        // debug function
        if(JbBuffer != 0 || JbBuffer_new != 0)
        {
            delete[] JbBuffer;
            delete[] JbBuffer_new;
            JbBuffer = new Vector8[w0*h0];
            JbBuffer_new = new Vector8[w0*h0];
        }


        float* statusMap = new float[w0*h0];
        bool* statusMapB = new bool[w0*h0];

        float densities[] = {0.03, 0.05, 0.15, 0.5, 1.0};
        int potInPyr = 5;

        for(int level = 0; level < PYR_LEVELS; level++)
        {
            selector.currentPotential = 3;   // not necessary
            int npts;

            if(level == 0)
            {
                npts = selector.makeMaps(frame, statusMap, w0*h0*densities[0], 1, true, 2);
            }
            else
            {
                npts = PixelSelectorInPyr(frame, level, statusMapB, potInPyr, w0*h0*densities[0]);
            }

            if(points[level] != 0)   delete[] points[level];
            points[level] = new Pnt[npts];

            // initialize selected points
            int wl = frame->width[level], hl = frame->height[level];
            Pnt* ptl = points[level];
            int numl = 0;
            for(int y = 3; y < hl - 3; y++)
                for(int x = 3; x < wl -3; x++)
                {
                    int idx = x + y*wl;
                    if( (level == 0 && statusMap[idx] != 0) ||
                           (level != 0 && statusMapB[idx]) )
                    {
                        ptl[numl].u = x + 0.1;
                        ptl[numl].v = y + 0.1;
                        ptl[numl].depth = frame->depth[level].at<float>(y, x);
                        ptl[numl].idepth = 1.0f / ptl[numl].depth;
                        if(!std::isfinite(ptl[numl].idepth))
                            ptl[numl].idepth = 1;

                        ptl[numl].depth_0 = ptl[numl].depth;
                        ptl[numl].idepth_0 = ptl[numl].idepth;

                        ptl[numl].alpha = 1 * 20;            // debug parameter
                        ptl[numl].idepth_prior = ptl[numl].idepth;
                        ptl[numl].Hessian_prior = 0;
                        ptl[numl].energyReg = ptl[numl].energyReg_new = 0;

                        ptl[numl].isGood = true;
                        ptl[numl].energy = 0;

                        // calculate initial sigma/uncertainty in depth/idepth
                        {

                            ptl[numl].lastHessian = 0;
                            ptl[numl].lastHessian_new = 0;
                            ptl[numl].levelFound = (level!=0)? 1 : statusMap[idx];
                        }

                        ptl[numl].outlierTH = patternNum * setting_outlierTH;

                        numl++;
                        assert(numl <= npts);
                    }
                }

            numPoints[level] = numl;
        }

        delete[] statusMap;
        delete[] statusMapB;

        makeNN();
        frameID = 0;
    }

    void Initializer::makeNN()
    {
        const float NNDistFactor=0.05;

        typedef nanoflann::KDTreeSingleIndexAdaptor<
                        nanoflann::L2_Simple_Adaptor<float, FLANNPointcloud> ,
                        FLANNPointcloud,2> KDTree;

        FLANNPointcloud pcs[PYR_LEVELS];
        KDTree* indexes[PYR_LEVELS];
        for(int i = 0; i < PYR_LEVELS; i++)
        {
            pcs[i] = FLANNPointcloud(numPoints[i], points[i]);
            indexes[i] = new KDTree(2, pcs[i], nanoflann::KDTreeSingleIndexAdaptorParams(5) );
            indexes[i]->buildIndex();
        }

        const int nn = 4;
        for(int level = 0; level < PYR_LEVELS; level++)
        {
            Pnt* pts = points[level];
            int npts = numPoints[level];

            int ret_index[nn];
            float ret_dist[nn];
            nanoflann::KNNResultSet<float, int, int> resultSet(nn);
            nanoflann::KNNResultSet<float, int, int> resultSetChild(nn);

            for(int i = 0; i < npts; i++)
            {
                Vector2 pt = Vector2(pts[i].u, pts[i].v);

                if(level > 0)     // Set children index and dist in the level below
                {
                    resultSetChild.init(ret_index, ret_dist);
                    Vector2 pt1 = pt*2.0f + Vector2(0.5f, 0.5f);
                    indexes[level-1]->findNeighbors(resultSetChild, (float*)&pt1, nanoflann::SearchParams());

                    for(int j = 0; j < nn; j++)
                    {
                        pts[i].childrenIdx[j] = ret_index[j];
                        float dist = std::exp(-ret_dist[j] * NNDistFactor);
                        pts[j].childrenDist[j] = dist;

                        assert(ret_index[j] >= 0 && ret_index[j] < npts);
                    }
                }
                else
                {
                    for(int j = 0; j < nn; j++)
                    {
                        pts[i].childrenIdx[j] = -1;
                        pts[i].childrenDist[j] = -1.0f;
                    }
                }


                if(level < PYR_LEVELS - 1)     // Set parent index and dist in the level above
                {
                    resultSet.init(ret_index, ret_dist);
                    Vector2 pt2 = pt*0.5f - Vector2(0.25f, 0.25f);
                    indexes[level+1]->findNeighbors(resultSet, (float*)&pt2, nanoflann::SearchParams());

                    pts[i].parentIdx = ret_index[0];
                    pts[i].parentDist = std::exp(-ret_dist[0] * NNDistFactor);

                    assert(ret_index[0] >= 0 && ret_index[0] < numPoints[level+1]);
                }
                else
                {
                    pts[i].parentIdx = -1;
                    pts[i].parentDist = -1;
                }

            }
        }
    }

    // consider to add residuals in depth ....
    Vector2 Initializer::calcResAndHessian(int level, Mat6x6 &H, Vector6 &b, Mat6x6 &Hsc, Vector6 &bsc,
                                           const SE3 &refToNew, bool plot)
    {
        int wl = firstFrame->width[level], hl = firstFrame->height[level];
        cv::Mat imgRef = firstFrame->intensity[level];   //  data in ref frame
        cv::Mat depRef = firstFrame->depth[level];

        cv::Mat imgNew = newFrame->intensity[level];     //  data in new frame
        cv::Mat depNew = newFrame->depth[level];
        Eigen::Vector2f* igradNew = newFrame->Grad_Int[level];
        Eigen::Vector2f* dgradNew = newFrame->Grad_Dep[level];

        Mat3x3 RKi = refToNew.rotationMatrix().cast<float>() * Ki[level];
        Vector3 t = refToNew.translation().cast<float>();

        float fxl = fx[level];
        float fyl = fy[level];
        float cxl = cx[level];
        float cyl = cy[level];

        Accumulator11 E;
        acc7.initialize();
        acc7SC.initialize();
        E.initialize();

        int npts = numPoints[level];
        Pnt* ptl = points[level];
        for(int i = 0; i < npts; i++)
        {
            Pnt* point = ptl + i;

            float alpha = point->alpha;
            point->maxstep = 1e10;
/*            if(!point->isGood)                             // doubt it ????????
            {
                E.updateSingle(point->energy);
                point->energy_new = point->energy;
                point->isGood_new = false;
                continue;
            }
*/
            VecResPattern dp0, dp1, dp2, dp3, dp4, dp5;
            VecResPattern dd, res;
            JbBuffer_new[i] = Vector8::Zero();

            // sum residuals in the pattern
            bool isGood = true;
            float energy = 0;
            for(int idx = 0; idx < patternNum; idx++)
            {
                int dx = pattern[idx][0];
                int dy = pattern[idx][1];

                Vector3 pt = RKi * Vector3(point->u+dx, point->v+dy, 1) + t * point->idepth_new;
                float u = pt[0]/pt[2];
                float v = pt[1]/pt[2];
                float Ku = fxl * u + cxl;
                float Kv = fyl * v + cyl;
                float new_idepth = point->idepth_new/pt[2];

                if(!(Ku > 1 && Kv > 1 && Ku < wl-2 && Kv < hl-2 && new_idepth > 0))
                {
                    isGood = false;
                    break;
                }

                Vector6 hitData = bilinearInterpolation(imgNew, depNew, igradNew, dgradNew, Ku, Kv);
                Vector2 refData = bilinearInterpolation(imgRef, depRef, point->u+dx, point->v+dy);

                if(!std::isfinite(hitData[0]) || !std::isfinite(refData[0]))
                {
                    isGood = false;
                    break;
                }

                float residual = hitData[0] - refData[0];
                float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH/fabs(residual);
                energy += hw * residual * residual * (2-hw);

                float dxdd = (t[0]-t[2]*u)/pt[2];
                float dydd = (t[1]-t[2]*v)/pt[2];

                if(hw < 1) hw = sqrtf(hw);
                float dxInterp = hw*hitData[2]*fxl;
                float dyInterp = hw*hitData[3]*fyl;
                dp0[idx] = new_idepth * dxInterp;
                dp1[idx] = new_idepth * dyInterp;
                dp2[idx] = -new_idepth * (u*dxInterp + v*dyInterp);
                dp3[idx] = -u*v*dxInterp - (1+v*v)*dyInterp;
                dp4[idx] = (1+u*u)*dxInterp + u*v*dyInterp;
                dp5[idx] = -v*dxInterp + u*dyInterp;

                dd[idx] = dxInterp * dxdd  + dyInterp * dydd;
                res[idx] = hw*residual;
                float maxstep = 1.0f / Vector2(dxdd*fxl, dydd*fyl).norm();
                if(maxstep < point->maxstep) point->maxstep = maxstep;

                // immediately compute dp*dd' and dd*dd' in JbBuffer1.
                JbBuffer_new[i][0] += dp0[idx]*dd[idx];
                JbBuffer_new[i][1] += dp1[idx]*dd[idx];
                JbBuffer_new[i][2] += dp2[idx]*dd[idx];
                JbBuffer_new[i][3] += dp3[idx]*dd[idx];
                JbBuffer_new[i][4] += dp4[idx]*dd[idx];
                JbBuffer_new[i][5] += dp5[idx]*dd[idx];
                JbBuffer_new[i][6] += res[idx]*dd[idx];
                JbBuffer_new[i][7] += dd[idx]*dd[idx];

                // add regularization term: alpha*(d - d_prior)*(d - d_prior)
                if(RegFlag)
                {

                    float res2 = alpha * (point->idepth_new - point->idepth_prior);
                    energy += res2 * res2;

                    JbBuffer_new[i][6] += alpha * res2;
                    JbBuffer_new[i][7] += alpha * alpha;
                }
            }




            if(!isGood || energy > point->outlierTH * 20)
            {
                E.updateSingle(point->energy);
                point->isGood_new = false;
                point->energy_new = point->energy;
                continue;
            }

            // add into Energy
            E.updateSingle(energy);
            point->isGood_new = true;
            point->energy_new = energy;
            point->lastHessian_new = JbBuffer_new[i][7];

            //update Hessian matrix
            for(int i=0; i+3 < patternNum; i+=4)
                acc7.updateSSE(
                            _mm_load_ps(((float*)(&dp0))+i),
                            _mm_load_ps(((float*)(&dp1))+i),
                            _mm_load_ps(((float*)(&dp2))+i),
                            _mm_load_ps(((float*)(&dp3))+i),
                            _mm_load_ps(((float*)(&dp4))+i),
                            _mm_load_ps(((float*)(&dp5))+i),
                            _mm_load_ps(((float*)(&res))+i) );


            for(int i = ((patternNum>>2)<<2); i < patternNum; i++)
                acc7.updateSingle(
                            (float)dp0[i],(float)dp1[i],(float)dp2[i],(float)dp3[i],
                            (float)dp4[i],(float)dp5[i],(float)res[i]);

            JbBuffer_new[i][7] = 1.0 / (1.0 + JbBuffer_new[i][7]);
            acc7SC.updateSingleWeighted(
                        (float)JbBuffer_new[i][0],(float)JbBuffer_new[i][1],(float)JbBuffer_new[i][2],(float)JbBuffer_new[i][3],
                        (float)JbBuffer_new[i][4],(float)JbBuffer_new[i][5],(float)JbBuffer_new[i][6],(float)JbBuffer_new[i][7]);

        }

        E.finish();
        acc7.finish();
        acc7SC.finish();

        H = acc7.H.topLeftCorner<6,6>();
        b = acc7.H.topRightCorner<6,1>();
        Hsc = acc7SC.H.topLeftCorner<6,6>();
        bsc = acc7SC.H.topRightCorner<6,1>();

        return Vector2(E.A, E.num);
    }

    bool Initializer::trackFrame(Frame *frame)
    {
        newFrame = frame;

        int maxIterations[] = {5, 5, 10, 30, 50};

        SE3 refToNew_current = thisToNext;
        Vector2 latestRes = Vector2::Zero();
        for(int level = PYR_LEVELS - 1; level >= 0; level--)
        {
            if(frameID != 0)
            {
                if(level < PYR_LEVELS - 1)
                    propagateDown(level + 1);
            }

            int acceptIterations = 0;

            Mat6x6 H, Hsc; Vector6 b, bsc;
            resetPoints(level);
            Vector2 resOld = calcResAndHessian(level, H, b, Hsc, bsc, refToNew_current, false);
            applyStep(level);

            if(printDebug)
            {
                // to do ....
            }

            float lamba = 0.1, eps = 1e-4;
            int fails = 0;
            int iteration = 0;
            while(true)
            {
                Mat6x6 Hl = H;
                for(int i = 0; i < 6; i++)  Hl(i,i) *= (1 + lamba);
                Hl -= Hsc * (1.0 / (1 + lamba));

                Vector6 bl = b - bsc * (1.0 / (1 + lamba));

                Hl = Hl * (0.01 / (w[level] * h[level]));     // why? what if not scaling....
                bl = bl * (0.01 / (w[level] * h[level]));

                Vector6 inc = -Hl.ldlt().solve(bl);

                SE3 refToNew_new = SE3::exp(inc.cast<double>()) * refToNew_current;
                doStep(level, lamba, inc);                // update idepth value

                Mat6x6 H_new, Hsc_new; Vector6 b_new, bsc_new;
                Vector2 resNew = calcResAndHessian(level, H_new, b_new, Hsc_new, bsc_new, refToNew_new, false);

                float eTotalNew = resNew[0];
                float eTotalOld = resOld[0];
                bool accept = eTotalNew < eTotalOld;

                if(printDebug)
                {
                    // to do....
                }

                if(accept)
                {
                    acceptIterations++;

                    H = H_new;
                    b = b_new;
                    Hsc = Hsc_new;
                    bsc = bsc_new;
                    resOld = resNew;
                    refToNew_current = refToNew_new;
                    applyStep(level);

                    lamba *= 0.5;
                    fails = 0;
                    if(lamba < 1e-4) lamba = 1e-4;
                }
                else
                {
                    fails++;
                    lamba *= 4;
                    if(lamba > 1e4) lamba = 1e4;
                }

                bool quitOpt = false;

                if(!(inc.norm() > eps) || iteration >= maxIterations[level] || fails >= 2)
                {
                    quitOpt = true;
                }

                if(quitOpt)  break;
                iteration++;
            }

            latestRes = resOld;
/*
            // debug function
            int Num_GoodRes = 0;
            Pnt* ptp = points[level];
            for(int i = 0; i < numPoints[level]; i++)
            {
                Pnt pt = ptp[i];
                if(pt.isGood)
                    Num_GoodRes++;
            }

            cout << "GN information in level " << level << " :" << endl
                  <<"   " << "iterations : " << iteration << endl
                  <<"   " << "acceptediterations : " << acceptIterations << endl
                  <<"   " << "lamba : " << lamba << endl
                  <<"   " << "leftEnergy : " << latestRes[0] << endl
                  <<"   " << "num_GoodRes : " << Num_GoodRes << endl;
            cout << "********************************************************************************" << endl;
*/
        }

/*        // debug Hessian of idepth
        cout << "Point information : " << endl;
        for(int i = 1000; i < 1010; i++)
        {
            cout << "(" << points[0][i].lastHessian << ", " << points[0][i].idepth
                 << ", " << points[0][i].depth_0 << "); ";
            cout << points[0][i].u << ", " << points[0][i].v << endl;
        }
        cout << endl;
*/
        thisToNext = refToNew_current;
        if(RegFlag)
        {
            for(int i = 0; i < PYR_LEVELS; i++)         // update prior information of points
            {
                Pnt* point = points[i];
                int num = numPoints[i];
                for(int j = 0; j < num; j++)
                {
                    point[j].idepth_prior = point[j].idepth;
                    point[j].Hessian_prior = point[j].lastHessian;

                    if(point[j].isGood)                                // adjust weight
                    {
                        double weight = exp( -1.0f / (sqrt(point[j].lastHessian)) );
                        point[j].alpha *= (1 + weight);
                    }
                }
            }
        }

        for(int i = 0; i < PYR_LEVELS; i++)
            propagateUp(i);

        frameID++;

        return frameID > 3;
    }

    void Initializer::propagateUp(int srcLvl)
    {
        assert(srcLvl + 1 < PYR_LEVELS);

        int nptsS = numPoints[srcLvl];
        int nptsT = numPoints[srcLvl+1];

        Pnt* ptsS = points[srcLvl];
        Pnt* ptsT = points[srcLvl+1];

        int num1= 0, num2=0;
        for(int i = 0; i < nptsT; i++)
        {
            Pnt* point = ptsT + i;

            if(point->isGood)
            {
                float tempIdepth = point->idepth, tempHessian = point->lastHessian;
                int count = 0;
                for(int k = 0; k < 4; k++)
                {
                    Pnt* child = ptsS + point->childrenIdx[k];                    
                    if(!child->isGood) continue;

                    // chi2 test: (d1 - d 2)^2/sigma1 + (d1 - d2)^2/sigma2  < 5.99
                    float chi2 = (point->idepth - child->idepth)*(point->idepth - child->idepth)* point->lastHessian +
                            (point->idepth - child->idepth)*(point->idepth - child->idepth) * child->lastHessian;
                    if(chi2 < 5.99)
                    {
                        tempIdepth = (tempIdepth*tempHessian + child->idepth*child->lastHessian)/(tempHessian + child->lastHessian);
                        tempHessian += child->lastHessian;
                        count++;
                    }
                }

                if(std::isfinite(tempIdepth) && count != 0)
                {
                    point->idepth = tempIdepth;
                    point->lastHessian = tempHessian;
                    num1++;
                }

            }
            else
            {
                float tempIdepth = 0, tempHessian = 0;
                int count = 0;
                for(int k = 0; k < 4; k++)
                {
                    Pnt* child = ptsS + point->childrenIdx[k];                   
                    if(!child->isGood) continue;

                    // chi2 test
                    float chi2 = (point->idepth - child->idepth)*(point->idepth - child->idepth) * child->lastHessian;
                    if(chi2 < 3.84)
                    {
                        tempIdepth = (tempIdepth*tempHessian + child->idepth*child->lastHessian)/(tempHessian + child->lastHessian);
                        tempHessian += child->lastHessian;
                        count++;
                    }
                }

                if(std::isfinite(tempIdepth) && (count != 0))
                {
                    point->idepth = tempIdepth;
                    point->lastHessian = tempHessian;
                    point->isGood = true;
                    num2++;
                }
                else
                {
                    point->lastHessian = 0;
                }

            }

        }

//        cout << "num1 = " << num1 << "; " << "num2 = " << num2 << endl;


    }

    void Initializer::propagateDown(int srcLvl)
    {
        assert(srcLvl > 0);

        int nptsT = numPoints[srcLvl-1];
        Pnt* ptsT = points[srcLvl-1];
        Pnt* ptsS = points[srcLvl];

        for(int i = 0; i < nptsT; i++)
        {
            Pnt* point = ptsT + i;
            Pnt* parent = ptsS + point->parentIdx;

            if(!parent->isGood || parent->lastHessian < 0.1)  continue;

            if(point->isGood)
            {
                // chi2 test
                float chi2 = (point->idepth - parent->idepth)*(point->idepth - parent->idepth) * (parent->lastHessian + point->lastHessian);
                if(chi2 < 5.99)
                {

                    float newHessian = point->lastHessian + parent->lastHessian;
                    float newidepth = (point->idepth*point->lastHessian + parent->idepth*parent->lastHessian) / newHessian;
                    point->idepth = newidepth; point->lastHessian = newHessian;
                }
            }
            else
            {
                // chi2 test
                float chi2 = (point->idepth - parent->idepth)*(point->idepth - parent->idepth) * parent->lastHessian;
                if(chi2 < 3.84)
                {
                    point->idepth = parent->idepth;
                    point->lastHessian = parent->lastHessian;
                    point->isGood = true;
                }
                else
                {
                    point->lastHessian = 0;
                }
            }

        }
    }

    void Initializer::resetPoints(int level)
    {
        Pnt* pts = points[level];
        int npts = numPoints[level];
        for(int i = 0; i < npts; i++)
        {
            pts[i].energy = 0;
            pts[i].idepth_new = pts[i].idepth;
            pts[i].lastHessian_new = pts[i].lastHessian = 0;
        }

        {
            // need to get a average idepth ???
        }
    }

    void Initializer::doStep(int level, float lamba, Vector6 inc)
    {
        const float maxPixelStep = 0.25;
        const float idMaxStep = 1e10;

        Pnt* pts = points[level];
        int npts = numPoints[level];
        for(int i = 0; i < npts; i++)
        {
            if(!pts[i].isGood)   continue;

            float b = JbBuffer[i][6] + JbBuffer[i].head<6>().dot(inc);
            float step = -b * JbBuffer[i][7] * (1.0f/(1+lamba));         // minus ???

            float maxstep = maxPixelStep * pts[i].maxstep;
            if(maxstep > idMaxStep) maxstep = idMaxStep;

            if(step > maxstep)  step = maxstep;
            if(step < -maxstep) step = -maxstep;

            float newIdepth = pts[i].idepth + step;
            if(newIdepth < 1e-1) newIdepth = 1e-1;        // kinect range: 0.4 ~ 10 m
            if(newIdepth > 2.5)  newIdepth = 2.5;

            pts[i].idepth_new = newIdepth;
        }
    }

    void Initializer::applyStep(int level)
    {
        Pnt* pts = points[level];
        int npts = numPoints[level];
        for(int i = 0; i < npts; i++)
        {
            if(!pts[i].isGood)
            {
                pts[i].idepth = pts[i].idepth_new;
                pts[i].depth = 1.0f / pts[i].idepth;
                continue;
            }

            pts[i].energy = pts[i].energy_new;
            pts[i].isGood = pts[i].isGood_new;
            pts[i].idepth = pts[i].idepth_new;
            pts[i].lastHessian = pts[i].lastHessian_new;

            pts[i].depth = 1.0f / pts[i].idepth;
        }

        std::swap<Vector8*>(JbBuffer, JbBuffer_new);
    }

    void Initializer::makePyramid(const CameraIntrinsic* camera)
    {

        w[0] = firstFrame->width[0];
        h[0] = firstFrame->height[0];

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
}
