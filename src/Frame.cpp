
#include <Frame.h>

namespace DSLAM
{

    int Frame::Num_TotalFrames = 0;

    void FFPrecalc::set(Frame *hostF, Frame *targetF, CameraIntrinsic &camera)
    {
        host = hostF;
        target = targetF;

        SE3 Transform_0 = host->worldToFram_evalPT.inverse() * target->worldToFram_evalPT;
        PRE_Rot0 = (Transform_0.rotationMatrix()).cast<float>();
        PRE_Tra0 = (Transform_0.translation()).cast<float>();

        SE3 Transform = host->PRE_FramToworld * target->PRE_worldToFram;
        PRE_Rot = (Transform.rotationMatrix()).cast<float>();
        PRE_Tra = (Transform.translation()).cast<float>();
        distance = PRE_Tra.norm();

        Eigen::Matrix3f K = camera.ConvertToMatrix();
        PRE_KRotKi = K * PRE_Rot * K.inverse();
        PRE_KTra = K * PRE_Tra;
        PRE_RotKi = PRE_Rot * K.inverse();
    }

    Frame::Frame()
    {
        intrinsic =0;
        shell = 0;
        Num_TotalFrames++;
        frameID = idx = abIDx = -1;

        flag_Marginalization = false;
        frameEnergyTH = 8*8*patternNum;

        PRE_worldToFram = PRE_FramToworld = worldToFram_evalPT = SE3();
        state_zero = state = state_backup = iter_step = iter_step_backup = Vector6d::Zero();
        NullSpace_Pose = Mat6x6d::Zero();

    }

    Frame::Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &Timestamp, CameraIntrinsic& camera)
    {
        intrinsic = &camera;
        timestamp = Timestamp;

        for( int level = 0; level < PYR_LEVELS; level++)
        {
            width[level] = (imGray.cols) >> level;
            height[level] = (imGray.rows) >> level;

            intensity[level] = cv::Mat::zeros(height[level], width[level], CV_32FC1);
            depth[level] = cv::Mat::zeros(height[level], width[level], CV_32FC1);
        }

        imGray.convertTo(intensity[0], CV_32FC1);
        ConvertRawDepthImg(imDepth);
        creatFramPyr();

        shell = 0;
        Num_TotalFrames++;
        frameID = idx = abIDx = -1;

        flag_Marginalization = false;
        frameEnergyTH = 8*8*patternNum;

        PRE_worldToFram = PRE_FramToworld = worldToFram_evalPT = SE3();
        state_zero = state = state_backup = iter_step = iter_step_backup = Vector6d::Zero();
        NullSpace_Pose = Mat6x6d::Zero();
    }

    Frame::Frame(const Frame &frame) {}  // not implement right now

    void Frame::ConvertRawDepthImg(const cv::Mat& imDepth)
    {
        depth[0].create(imDepth.rows, imDepth.cols, CV_32FC1);

        const ushort* img_ptr = imDepth.ptr<ushort>();
        float* dep_ptr = depth[0].ptr<float>();

        for(int idx = 0; idx < (int)(imDepth.rows * imDepth.cols); idx++, img_ptr++, dep_ptr++)
        {
            if(*img_ptr == 0)
     //           *dep_ptr = Invalid;
                *dep_ptr = 0;
            else
                *dep_ptr = (float)(*img_ptr)*(1.0f/depthScale);
        }
    }

    void Frame::creatFramPyr()
    {
        for(int i = 0; i < PYR_LEVELS; i++)
        {
            Grad_Int[i] = new Eigen::Vector2f[width[i]*height[i]];
            Grad_Dep[i] = new Eigen::Vector2f[width[i]*height[i]];

            GradNorm[i] = new float[width[i]*height[i]];
        }

        for(int level = 0; level < PYR_LEVELS; level++)
        {
            int wl = width[level]; int hl = height[level];
            Eigen::Vector2f* Igrad = Grad_Int[level];
            Eigen::Vector2f* Dgrad = Grad_Dep[level];

            float* g_Norm = GradNorm[level];

            float* currentI = intensity[level].ptr<float>();
            float* currentD = depth[level].ptr<float>();
            if(level != 0)
            {
                int lastW = width[level-1];

                for(int y = 0; y < hl; y++)
                    for(int x = 0; x < wl; x++)
                    {
                        float* lastI = intensity[level-1].ptr<float>(2*y, 2*x);
                        float* lastD = depth[level-1].ptr<float>(2*y, 2*x);

                        currentI[x + y*wl] = 0.25*(*lastI + *(lastI+1) + *(lastI+lastW) +
                                                   *(lastI+lastW+1));
                        currentD[x + y*wl] = 0.25*(*lastD + *(lastD+1) + *(lastD+lastW) +
                                                   *(lastD+lastW+1));
                    }
            }

            for(int y = 0; y < hl; y++)
                for(int x = 0; x < wl; x++)
                {
                    int idx = x + y*wl;

                    if(x == 0)
                    {
                        Igrad[idx](0) = currentI[idx+1] - currentI[idx];
                        Dgrad[idx](0) = currentD[idx+1] - currentD[idx];
                    }
                    else if(x == wl-1)
                    {
                        Igrad[idx](0) = currentI[idx] - currentI[idx-1];
                        Dgrad[idx](0) = currentD[idx] - currentD[idx-1];
                    }
                    else
                    {
                        Igrad[idx](0) = 0.5 * (currentI[idx+1] - currentI[idx-1]);
                        Dgrad[idx](0) = 0.5 * (currentD[idx+1] - currentD[idx-1]);
                    }

                    if(y == 0)
                    {
                        Igrad[idx](1) = currentI[idx+wl] - currentI[idx];
                        Dgrad[idx](1) = currentD[idx+wl] - currentD[idx];
                    }
                    else if(y == hl-1)
                    {
                        Igrad[idx](1) = currentI[idx] - currentI[idx-wl];
                        Dgrad[idx](1) = currentD[idx] - currentD[idx-wl];
                    }
                    else
                    {
                        Igrad[idx](1) = 0.5 * (currentI[idx+wl] - currentI[idx-wl]);
                        Dgrad[idx](1) = 0.5 * (currentD[idx+wl] - currentD[idx-wl]);
                    }

                    if(!std::isfinite(Igrad[idx](0))) Igrad[idx](0) = Invalid;
                    if(!std::isfinite(Igrad[idx](1))) Igrad[idx](1) = Invalid;
                    if(!std::isfinite(Dgrad[idx](0)) || (currentD[idx] < 0.25)) Dgrad[idx](0) = Invalid;
                    if(!std::isfinite(Dgrad[idx](1)) || (currentD[idx] < 0.25)) Dgrad[idx](1) = Invalid;

                    if((Igrad[idx](0) == Invalid) || (Igrad[idx](1) == Invalid))
                        g_Norm[idx] = -1;
                    else
                        g_Norm[idx] = Igrad[idx].squaredNorm();
                }
        }
    }


    void Frame::setNullSpace()  // still don't understand why to solve nullspace like this
    {
        for(int i=0;i<6;i++)
        {
            Vector6d eps = Vector6d::Zero(); eps[i] = 1e-3;
            SE3 EepsP = Sophus::SE3::exp(eps);
            SE3 EepsM = Sophus::SE3::exp(-eps);
            SE3 w2c_leftEps_P_x0 = (get_worldToFram_evalPT() * EepsP) * get_worldToFram_evalPT().inverse();
            SE3 w2c_leftEps_M_x0 = (get_worldToFram_evalPT() * EepsM) * get_worldToFram_evalPT().inverse();

            NullSpace_Pose.col(i) = (w2c_leftEps_P_x0.log() - w2c_leftEps_M_x0.log())/(2e-3);
        }
    }

    Frame::~Frame()
    {
//        delete shell;

        for(Point* r : GoodPoints) delete r;
        for(Point* r : MarginalizedPoints) delete r;
        for(Point* r : OutPoints) delete r;
        for(ImmaturePoint* r : ImmaturePoints) delete r;

        GoodPoints.clear(); MarginalizedPoints.clear();
        OutPoints.clear(); ImmaturePoints.clear();

        Num_TotalFrames--;
        for(int i = 0; i < PYR_LEVELS; i++)
        {
            delete[] Grad_Int[i];
            delete[] Grad_Dep[i];
            delete[] GradNorm[i];

            intensity[i].release();
            depth[i].release();
        }

    }
}
