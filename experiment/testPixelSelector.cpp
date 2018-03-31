
#include <testPixelSelector.h>

testPixelSelector::testPixelSelector(const cv::Mat &imGray, const cv::Mat &imDepth, const DSLAM::CameraIntrinsic &camera)
{
    // image pyramid intialization

    std::cout << "Intialize color and depth matrix...." << endl;

    for( int level = 0; level < PYR_LEVELS; level++)
    {
        width[level] = (imGray.cols) >> level;
        height[level] = (imGray.rows) >> level;

        std::cout << "Size " << level << " = " << height[level] <<", " << width[level] << endl;

        intensity[level] = cv::Mat::zeros(height[level], width[level], CV_32FC1);
        depth[level] = cv::Mat::zeros(height[level], width[level], CV_32FC1);
    }
    std::cout << "Intialization completed !" << endl;

    imGray.convertTo(intensity[0], CV_32FC1);
    ConvertRawDepthImg(imDepth);

    // debug function
    std::cout << "Test of ucharTofloat conversion: " << endl;
    for(int i = 0; i < 10; i++)
    {
        std::cout << "Row " << i << " :" << endl;
        for(int j = 0; j < 10; j++)
        {
            cout << "(" << static_cast<int>(imGray.at<uchar>(i,j)) << ", " << intensity[0].at<float>(i,j) << ")" << " ";
        }
        std::cout << endl;
    }


    std::cout << "Start creat image pyramid...." << endl;

    creatFramPyr();

    std::cout << "Image Pyramid creation successfully !" << endl;

    // pixel selector intialization

    int w = width[0], h = height[0];
    randomPattern = new uchar[w*h];
    std::srand(3141592);	                             // want to be deterministic.
    for(int i=0;i<w*h;i++)
        randomPattern[i] = rand() & 0xFF;

    currentPotential = 3;

    gradHist = new int[100*(1+w/32)*(1+h/32)];
    ths = new float[(w/32)*(h/32)+100];
    thsSmoothed = new float[(w/32)*(h/32)+100];

    allowFast = false;
    SelectedFlag = false;
}

testPixelSelector::~testPixelSelector()
{
    for(int i = 0; i < PYR_LEVELS; i++)
    {
        delete[] Grad_Int[i];
        delete[] Grad_Dep[i];
        delete[] GradNorm[i];

        intensity[i].release();
        depth[i].release();
    }

    delete[] randomPattern;
    delete[] gradHist;
    delete[] ths;
    delete[] thsSmoothed;
}

void testPixelSelector::ConvertRawDepthImg(const cv::Mat& imDepth)
{
    const ushort* img_ptr = imDepth.ptr<ushort>();
    float* dep_ptr = depth[0].ptr<float>();

    for(int idx = 0; idx < (int)(imDepth.rows * imDepth.cols); idx++, img_ptr++, dep_ptr++)
    {
        if(*img_ptr == 0)
//            *dep_ptr = DSLAM::Invalid;
            *dep_ptr = 0.0f;
        else
            *dep_ptr = (float)(*img_ptr)*(1.0f/depthScale);
    }
}

void testPixelSelector::creatFramPyr()
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

            std::cout << "Create " << level << " " << "level pyramid...." << endl;
            std::cout << "(rows, cols) = " << hl << ", " << wl << endl;
            for(int y = 0; y < hl; y++)
            {
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

            std::cout << level << " level pyramid creation successfully !" << endl;
        }


        for(int y = 1; y < hl - 1; y++)
            for(int x = 1; x < wl - 1; x++)
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
                    Igrad[idx](0) = 0.5f * (currentI[idx+1] - currentI[idx-1]);
                    Dgrad[idx](0) = 0.5f * (currentD[idx+1] - currentD[idx-1]);
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
                    Igrad[idx](1) = 0.5f * (currentI[idx+wl] - currentI[idx-wl]);
                    Dgrad[idx](1) = 0.5f * (currentD[idx+wl] - currentD[idx-wl]);
                }

                if(!std::isfinite(Igrad[idx](0))) Igrad[idx](0) = DSLAM::Invalid;
                if(!std::isfinite(Igrad[idx](1))) Igrad[idx](1) = DSLAM::Invalid;
                if(!std::isfinite(Dgrad[idx](0)) || (currentD[idx] == 0)) Dgrad[idx](0) = DSLAM::Invalid;
                if(!std::isfinite(Dgrad[idx](1)) || (currentD[idx] == 0)) Dgrad[idx](1) = DSLAM::Invalid;

                if((Igrad[idx](0) == DSLAM::Invalid) || (Igrad[idx](1) == DSLAM::Invalid))
                    g_Norm[idx] = -1;
                else
                    g_Norm[idx] = Igrad[idx].squaredNorm();
            }
    }

    // debug function
/*  std::cout << endl;
    std::cout << "Gradients of image pixels: " << endl;
    for(int i = 200; i < 250; i++)
    {
        for(int j = 200; j < 240; j++)
        {
            int idx = j + i * width[0];
            std::cout << GradNorm[0][idx] << ", ";
        }

        std::cout << endl;
    }  */
}

void testPixelSelector::makeHists()
{
    SelectedFlag = true;
    float* mapmax0 = GradNorm[0];

    int w = width[0];
    int h = height[0];

    int w32 = w/32; int h32 = h/32;
    thsStep = w32;

    for(int y = 0; y < h32; y++)
        for(int x = 0; x < w32; x++)
        {
            float* map0 = mapmax0 + 32*y*w + 32*x;
            int* hist0 = gradHist;
            memset(hist0, 0, sizeof(int)*50);

            for(int j = 0; j < 32; j++)
                for(int i = 0; i < 32; i++)
                {
                    int u = x*32 + i;
                    int v = y*32 + j;

                    if(u > w-2 || v > h-2 || u < 1 || v < 1)   continue;
                    if(map0[j*w + i] == -1)   continue;

                    int g = (int)sqrt(map0[j*w + i]);
                    if(g > 48) g = 48;
                    hist0[g+1]++;
                    hist0[0]++;
                }
            ths[x+y*w32] = computHistQuantity(hist0, setting_minGradHistCut) + setting_minGradHistAdd;
        }

    for(int y = 0; y < h32; y++)
        for(int x = 0; x < w32; x++)
        {
            float sum = 0, num = 0;
            if(x > 0)
            {
                sum++;
                num += ths[x-1 + y*w32];
                if(y > 0)
                {
                    sum++;
                    num += ths[x-1 + (y-1)*w32];
                }
                if(y < h32 - 1)
                {
                    sum++;
                    num += ths[x-1 + (y+1)*w32];
                }

            }

            if(x < w32-1)
            {
                sum++;
                num += ths[x+1 + y*w32];
                if(y > 0)
                {
                    sum++;
                    num += ths[x+1 + (y-1)*w32];
                }
                if(y < h32 - 1)
                {
                    sum++;
                    num += ths[x+1 + (y+1)*w32];
                }
            }

            if(y > 0)
            {
                sum++;
                num += ths[x + (y-1)*w32];
            }
            if(y < h32 -1)
            {
                sum++;
                num += ths[x + (y+1)*w32];
            }

            thsSmoothed[x+y*w32] = (num/sum)*(num/sum);
        }


    // debug function
    std::cout << endl;
    std::cout << "Gradients of image blocks: " << endl;
    for(int y = 0; y < h32; y++)
    {
        for(int x = 0; x < w32; x++)
        {
            std::cout << thsSmoothed[x + y*w32] << " ";
        }

        std::cout << endl;
    }
}


int testPixelSelector::makeMaps(float *map_out, float density, int recursionsLeft, bool plot, float thFactor)
{
    float numHave = 0;
    float numWant = density;
    float quotia;
    int idealPotential = currentPotential;


    {
        if(!SelectedFlag) makeHists();

        Eigen::Vector3i n = this->select(map_out, currentPotential, thFactor);
        numHave = n[0] + n[1] + n[2];
        quotia = numWant/numHave;

        float K = numHave * (currentPotential+1) * (currentPotential+1);
        idealPotential = sqrtf(K/numWant) - 1;
        if(idealPotential < 1) idealPotential = 1;

        if(recursionsLeft>0 && quotia > 1.25 && currentPotential > 1)
        {
            if(idealPotential >= currentPotential)
                idealPotential = currentPotential - 1;

            currentPotential = idealPotential;
            return makeMaps(map_out, density, recursionsLeft-1, plot, thFactor);
        }
        else if(recursionsLeft > 0 && quotia < 0.25)
        {
            if(idealPotential <= currentPotential)
                idealPotential = currentPotential + 1;

            currentPotential = idealPotential;
            return makeMaps(map_out, density, recursionsLeft-1, plot, thFactor);
        }
    }

    int numHaveSub = numHave;
    if(quotia < 0.95)
    {
        int total = width[0] * height[0];
        int rn = 0;
        uchar charTH = 255 * quotia;
        for(int i = 0; i < total; i++)
        {
            if(map_out[i] != 0)
            {
                if(randomPattern[rn] > charTH)
                {
                    map_out[i] = 0;
                    numHaveSub--;
                }
                rn++;
            }

        }
    }

    currentPotential = idealPotential;
    if(plot)
    {
        // to do using openCV
        // display the image and selected points
        int w = width[0];
        int h = height[0];

        cv::Mat img;
        intensity[0].convertTo(img, CV_8UC1);
        cv::cvtColor(img, img, CV_GRAY2BGR);

        for(int y = 0; y < h; y++)
            for(int x = 0; x < w; x++)
            {
                int idx = x + y * w;
                if(map_out[idx] == 1)
                    cv::circle(img, cv::Point(x,y), 2, cv::Scalar(0, 0, 255), 1);
                if(map_out[idx] == 2)
                    cv::circle(img, cv::Point(x,y), 2, cv::Scalar(0, 255, 0), 1);
                if(map_out[idx] == 4)
                    cv::circle(img, cv::Point(x,y), 2, cv::Scalar(255, 0, 0), 1);
            }

        cv::namedWindow("Selected Points", cv::WINDOW_AUTOSIZE);
        cv::imshow("Selected Points", img);
        cv::waitKey(0);
    }

    return numHaveSub;
}

Eigen::Vector3i testPixelSelector::select(float* map_out, int pot, float thFactor)
{
    Eigen::Vector2f const* const colorGrad = Grad_Int[0];
    Eigen::Vector2f const* const depthGrad = Grad_Dep[0];

    float* mapMax0 = GradNorm[0];
    float* mapMax1 = GradNorm[1];
    float* mapMax2 = GradNorm[2];

    int w0 = width[0], w1 = width[1], w2 = width[2];
    int h0 = height[0];

    const Eigen::Vector2f directions[16] = {
             Eigen::Vector2f(0,    1.0000),
             Eigen::Vector2f(0.3827,    0.9239),
             Eigen::Vector2f(0.1951,    0.9808),
             Eigen::Vector2f(0.9239,    0.3827),
             Eigen::Vector2f(0.7071,    0.7071),
             Eigen::Vector2f(0.3827,   -0.9239),
             Eigen::Vector2f(0.8315,    0.5556),
             Eigen::Vector2f(0.8315,   -0.5556),
             Eigen::Vector2f(0.5556,   -0.8315),
             Eigen::Vector2f(0.9808,    0.1951),
             Eigen::Vector2f(0.9239,   -0.3827),
             Eigen::Vector2f(0.7071,   -0.7071),
             Eigen::Vector2f(0.5556,    0.8315),
             Eigen::Vector2f(0.9808,   -0.1951),
             Eigen::Vector2f(1.0000,    0.0000),
             Eigen::Vector2f(0.1951,   -0.9808)};
    memset(map_out, 0, w0*h0*sizeof(PixelSelectorStatus));

    float dw1 = setting_gradDownweightPerLevel;
    float dw2 = dw1 * dw1;

    int n2 = 0, n3 = 0, n4 = 0;
    for(int y4 = 0; y4 < h0; y4+=(4*pot))
        for(int x4 = 0; x4 < w0; x4+=(4*pot))
        {
            int my3 = std::min((4*pot), h0-y4);
            int mx3 = std::min((4*pot), w0-x4);
            int bestIdx4 = -1; float bestVal4 = 0;
            Eigen::Vector2f dir4 = directions[randomPattern[n2] & 0xf];
            for(int y3 = 0; y3 < my3; y3+=(2*pot))
                for(int x3 = 0; x3 < mx3; x3+=(2*pot))
                {
                    int y34 = y4 + y3;
                    int x34 = x4 + x3;
                    int my2 = std::min((2*pot), h0-y34);
                    int mx2 = std::min((2*pot), w0-x34);
                    int bestIdx3 = -1; float bestVal3 = 0;
                    Eigen::Vector2f dir3 = directions[randomPattern[n3] & 0xf];
                    for(int y2 = 0; y2 < my2; y2+=pot)
                        for(int x2 = 0; x2 < mx2; x2+=pot)
                        {
                            int y234 = y2 + y34;
                            int x234 = x2 + x34;
                            int my1 = std::min(pot, h0-y234);
                            int mx1 = std::min(pot, w0-x234);
                            int bestIdx2 = -1; float bestVal2 = 0;
                            Eigen::Vector2f dir2 = directions[randomPattern[n2] & 0xf];
                            for(int y1 = 0; y1 < my1; y1++)
                                for(int x1 = 0; x1 < mx1; x1++)
                                {
                                    assert(y234+y1 < h0);
                                    assert(x234+x1 < w0);
                                    int idx = x1 + x234 + w0*(y1+y234);
                                    int xf = x1 + x234;
                                    int yf = y1 + y234;

                                    if(xf<4 || xf>=w0-5 || yf<4 || yf>h0-4) continue;

                                    float pixelTH0 = thsSmoothed[(xf>>5) + (yf>>5) * thsStep];
                                    float pixelTH1 = pixelTH0 * dw1;
                                    float pixelTH2 = pixelTH1 * dw2;

                                    float ag0 = mapMax0[idx];
                                    if(ag0 > pixelTH0 * thFactor)
                                    {
                                        Eigen::Vector2f ag0dir = colorGrad[idx];
                                        float dirNorm = fabsf((float)(ag0dir.dot(dir2)));
                                        if(!setting_selectDirectionDistribution) dirNorm = ag0;

                                        if(dirNorm > bestVal2)
                                        {
                                            bestVal2 = dirNorm;
                                            bestIdx2 = idx;
                                            bestIdx3 = bestIdx4 = -2;
                                        }
                                    }
                                    if(bestIdx3 == -2) continue;

                                    float ag1 = mapMax1[(int)(xf*0.5f + 0.25f) + (int)(yf*0.5f + 0.25f) * w1];
                                    if(ag1 > pixelTH1 * thFactor)
                                    {
                                        Eigen::Vector2f ag0dir = colorGrad[idx];
                                        float dirNorm = fabsf((float)(ag0dir.dot(dir3)));
                                        if(!setting_selectDirectionDistribution) dirNorm = ag1;

                                        if(dirNorm > bestVal3)
                                        {
                                            bestVal3 = dirNorm;
                                            bestIdx3 = idx;
                                            bestIdx4 = -2;
                                        }
                                    }
                                    if(bestIdx4 == -2) continue;

                                    float ag2 = mapMax2[(int)(xf*0.25f + 0.125f) + (int)(yf*0.25f + 0.125f) * w2];
                                    if(ag2 > pixelTH2 * thFactor)
                                    {
                                        Eigen::Vector2f ag0dir = colorGrad[idx];
                                        float dirNorm = fabsf((float)(ag0dir.dot(dir4)));
                                        if(!setting_selectDirectionDistribution) dirNorm = ag2;

                                        if(dirNorm > bestVal4)
                                        {
                                            bestVal4 = dirNorm;
                                            bestIdx4 = idx;
                                        }
                                    }

                                }

                            if(bestIdx2 > 0)
                            {
                                map_out[bestIdx2] = 1;
                                bestVal3 = 1e10;
                                n2++;
                            }
                        }

                    if(bestIdx3 > 0)
                    {
                        map_out[bestIdx3] = 2;
                        bestVal4 = 1e10;
                        n3++;
                    }
                }

            if(bestIdx4 > 0)
            {
                map_out[bestIdx4] = 4;
                n4++;
            }
        }

    return Eigen::Vector3i(n2, n3, n4);
}
