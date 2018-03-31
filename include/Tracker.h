#ifndef TRACKER_H_
#define TRACKER_H_

#include <common.h>
#include <parameter_reader.h>
#include <DataType.h>
#include <Frame.h>
#include <Residual.h>
#include <MatrixAccumulators.h>
#include <nanoflann.h>
#include <PixelSelector.h>
#include <PixelSelectorInPyr.h>

namespace DSLAM
{

struct SelectPnt
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    int level;        // level of pyramid
    int u, v;
    float color;
    float idepth;
    Vector2 colorGrad;
    Vector2 depthGrad;
    float norm;

    float Tweight;      // weight value under T distribution
    float energy, energy_new;
    float outlierTH;

    bool isGood, isGood_new;

    bool operator >(const SelectPnt &Pnt) const
    {
        return this->norm > Pnt.norm;
    }
};

struct IterationPrintInfo
{
    int currentlvl;
    int currentNumInEM;
    int iterations;
    int acceptedIterations;
    float lamba;
    float leftEnergy;
    float ratio;
    int num_GoodRes;

    float iteStep;
    float MuInfo;   // mutual inforamtion

    IterationPrintInfo()
    {
        currentlvl = currentNumInEM = iterations = acceptedIterations = num_GoodRes = 0;
        ratio = lamba = leftEnergy = iteStep = MuInfo = 0.0f;
    }
};

struct IterationInfoInLvl0
{
    double TDistributionLogLikelihood;
    bool IsFirstIteration;

//    const int MaxIterationNumOfGEM[PYR_LEVELS] = {2,4,5,5,5};
    const int MaxIterationNumOfGEM[PYR_LEVELS] = {1,1,1,1,1};   // allowed maximal iteration num of Expectaion-Maximization Algorithm
    int IterationNumOfGEM[PYR_LEVELS];      // current iteration num of EM algorithm

    vector< vector<IterationPrintInfo> > IteInfo;
};

const float ratio[] = {0.6, 0.6, 0.5, 0.4, 0.3};    // ratio of smapling points in MInfo optimization

class Tracker
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Tracker(int w, int h, const string Config);
    ~Tracker();

    bool trackNewFrame(Frame* newFrame, SE3& lastToNew_out, int maxlvl, Vector5f minResForAbort);
    void setTrackingRef(std::vector<Frame*> keyframes);
    void makePyramid(const CameraIntrinsic* camera);

    bool OptimizMI(SE3& refTocur_out, Vector5f MInfoTH, Vector5f EnergyTH, int UsedLvl = PYR_LEVELS);          // debug MI optimization
    double CalcRefMInfo(int lvl, const SE3& refTocur);    // debug MI optimization

    void makeTrackPointsbyFrame(Frame* frame);       //only used for debugging track algorithm

    bool debugPrint, debugPlot;

    Mat3x3 K[PYR_LEVELS];
    Mat3x3 Ki[PYR_LEVELS];
    float  fx[PYR_LEVELS];
    float  fy[PYR_LEVELS];
    float  fxi[PYR_LEVELS];
    float  fyi[PYR_LEVELS];
    float  cx[PYR_LEVELS];
    float  cy[PYR_LEVELS];
    float  cxi[PYR_LEVELS];
    float  cyi[PYR_LEVELS];
    int w[PYR_LEVELS];
    int h[PYR_LEVELS];

    void debugPlotIdepthMap();     // remains to do...

    Frame* lastRef;
    Frame* newFrame;
    int refFrameID;

    SelectPnt* SeltPnts[PYR_LEVELS];
    int NumPnts[PYR_LEVELS];

    // SSD status recorder
    IterationInfoInLvl0 IterationStatus;
    Vector6 lastResiduals;
    Vector3 lastFlowIndicators;
    double firstTrackRMSE;

    // (N)MI status recorder
    vector<IterationPrintInfo> MInfOptiInfo;
    Vector5f lastMInfo;
    Vector5f lastEnergy;

    ParameterReader paraReader;
    inline void testParaReader()
    {
        cout << "Tracker params: ==========================================================================" << endl
             << "UseEM = " << UseEM << endl
             << "debugMInfo = " << debugMInfo << endl
             << "UseConvergEstimate = " << UseConvergEstimate << endl
             << "UseWeight = " << UseWeight << endl
             << "UseNormMInfo = " << UseNormMInfo << endl
             << "Use2ndDervs = " << Use2ndDerivates << endl;
    }

    inline bool IsWeighted() { return UseWeight; }
    inline bool IsNMInfo() {return UseNormMInfo; }
    inline bool IsConvergd() {return UseConvergEstimate; }
private:

    void resetTracker();

    void makeTrackDepth(std::vector<Frame*> keyframes);  //adjust depth map using observations
    float* idepth[PYR_LEVELS];
    float* weightSums[PYR_LEVELS];

    // precison or scale Matrix of the 2d residual vector : [res_color, res_depth]
    Mat2x2 Scale[PYR_LEVELS];
    Vector2 calcResMean(const float* res_color, const float* res_depth, const float* weight);
    Mat2x2 calcResScaleSSE(const float* res_color, const float* res_depth, const Vector2 mean, const float* weight);
    Mat2x2 calcResScale(const float* res_color, const float* res_depth, const Vector2 mean, const float* weight);
    float calcSigResWeight(const float res_color, const float res_depth, const Vector2 &mean, const Mat2x2 &precision);
    void calcResWeights(const float* res_color, const float* res_depth, const Vector2 &mean, const Mat2x2 &precision, float* weights);
    float calcLogLikelyhood(const float* res_color, const float* res_depth, const float* weight, const Mat2x2 &scale);

    Vector6 calcRes(int lvl, const SE3 &refToNew, float cutoffTH);
    void calcGS(int lvl, Mat6x6 &H_out, Vector6 &b_out, const SE3 &refToNew);
    void calcGSSSE(int lvl, Mat6x6 &H_out, Vector6 &b_out, const SE3 &refToNew);
    Vector6 calcResAndGS(int lvl, Mat6x6 &H_out, Vector6 &b_out, const SE3 &refToNew, float cutoffTH);

    void applyStep(int lvl);
    //warped buffers, used for calculating redisuals, Jacobian & Hessiam Matrix
    //parameters for calc hessian about color residual
    float* buf_warped_idepth;            // note: the color residaul was scaled by the factor 1/255 to
    float* buf_warped_u;                 //       match the scale of depth residual. Therefore, all hessian
    float* buf_warped_v;                 //       about color residual should be also scaled with the same factor 1/255.
    float* buf_warped_idx;
    float* buf_warped_idy;
    float* buf_warped_refColor;    
    float* buf_warped_color_residual;    

    //parameters for calc hessain about depth residual
    float* buf_warped_ddx;
    float* buf_warped_ddy;
    float* buf_warped_depth_residual;

    //parameters in reference image for Second-Order Minimization(ESM)
    float* buf_warped_ref_idx;
    float* buf_warped_ref_idy;
    float* buf_warped_ref_ddx;
    float* buf_warped_ref_ddy;

    //residual weights in T-Distribution
    float* buf_warped_weight;
    int buf_warped_num;
    std::vector<float*> ptrToDelete;

    bool IterationAccepted;
    bool UseEM;              // debug variable
    bool UseDoubleEM;        // debug variable

    Accumulator7_2 acc7;

    //*****************************************************************************************
    // parameters for Mutual Information-based optimization
    bool runMIOptimizaion;
    bool UseConvergEstimate;
    bool UseWeight;
    bool UseNormMInfo;
    bool Use2ndDerivates;
    bool debugMInfo;

    float* buf_MI_ref_color;
    float* buf_MI_ref_idepth;
    float* buf_MI_ref_u;
    float* buf_MI_ref_v;
    float* buf_MI_ref_idx;
    float* buf_MI_ref_idy;
    float* buf_MI_ref_scaledIntensity;
    float* buf_MI_ref_weight;
    float* buf_MI_visibility;   // used for Hessian compution at converged point
    float* buf_MI_energy;       // used for check the reliability of estimated transformation

    float* buf_MI_ref_BSpline[BinsOfHis];
    float* buf_MI_ref_BSpline_derivas[BinsOfHis];
    float* buf_MI_ref_BSpline_derivas2[BinsOfHis];
    float* buf_MI_cur_BSpline[BinsOfHis];

    float* buf_MI_PrecalGauss[6];
    int NumSampPnts;

    typedef Eigen::Matrix<float, 1, BinsOfHis> Histo;                 // histogram of image
    typedef Eigen::Matrix<float, BinsOfHis, BinsOfHis> JointHisto;    // jont histogram of reference and current images

    Histo histo_ref;
    Histo histo_cur;
    JointHisto JHisto;
    float NormalizedConst;
    float NumVisibility;

    void SampleRefPnts(int lvl, float threshold, float ratio);
    void calcRefSplineBuf(const float *scaleIntensity);
    void calcCurSplineBuf(int lvl, const SE3 &refTonew);
    void calcRefHisto(Histo& refHisto);
    void calcCurHisto(Histo& curHisto);
    void calcJointHisto(float** refbuff, float** curbuff, const float* weight, JointHisto& jhisto);
    void calcHistoFromJointHisto(Histo& refHisto, Histo& curHisto, const JointHisto& Joint);   // use joint probability to compute marginal probability
    double calcMutualInfo(const float& ref, const float& cur, const float& joint);       // calc mutual info:               MI = H(A) + H(B) - H(A,B)
    double calcNormaMI(const float& ref, const float& cur, const float& joint);          // calc normalized mutual info    NMI = (H(A) + H(B)) / H(A,B)

    float checkEnergy();

    void PrecalcGaussian(int lvl);
    void calcHistDerivates(Vector6* joint_Derv, Vector6* joint_converg_Derv, Vector6* ref_Derv);
    void calcHist2ndDerivates(Mat6x6* joint_2nd_Derv);
    void calcHessian(Mat6x6& H_out, const Vector6* joint_Derv, const Vector6* ref_Derv);
    void calcJocabian(Vector6& b_out, const Vector6* joint_Derv);
    void calcJocaAndHessn(Mat6x6& H_out, Vector6& b_out, const float& refEnpy, const float& curEnpy, const float& jotEnpy,
                          const Vector6* joint_Derv, const Vector6* joint_converg_dev ,const Vector6* ref_Derv, const Mat6x6* joint_2nd_Derv);          // for Normalized MI
    float calcWeights(const Vector2& p, const Vector2& Kp, const cv::Mat& ref,
                      const cv::Mat& cur, const Mat2x2& Rplane);
//    void OptimizMI(SE3& refTocur_out);

    // comput entropy from histogram/pdf
    inline float calcEntropy(const Histo& histo)     // minus Entropy: -H
    {
        double sum = 0;
        float n = NormalizedConst;
        float constant = log(n);
        for(int i = 0; i < BinsOfHis; i++)
        {
            if(histo(i) == 0) continue;
            sum += histo(i) * (log(histo(i)) - constant);
        }

        return (1.0f/n)*sum;
    }

    inline float calcJointEntropy(const JointHisto& joint)
    {
        double sum = 0;
        float n = NormalizedConst;
        float constant = log(n);
        for(int i = 0; i < BinsOfHis; i++)
            for(int j = 0; j < BinsOfHis; j++)
            {
                if(joint(i,j) == 0) continue;
                sum += joint(i,j) * (log(joint(i,j)) - constant);
            }
        return (1.0f/n)*sum;
    }

    inline void calcEntropy(const Histo& ref, const Histo& cur, const JointHisto& joint,
                            float& refEnpy, float& curEnpy, float& jotEnpy)
    {
        refEnpy = calcEntropy(ref);
        curEnpy = calcEntropy(cur);
        jotEnpy = calcJointEntropy(joint);
    }


    // BSpline-interpolaion
    inline void BSpline_interpolation(float x, float& y, bool thirdOrder = true)
    {
        float result;
        if(thirdOrder)
        {
            if(x <= -2 || x >= 2) result = 0;
            else if(x <= -1)      result = std::pow(2+x, 3);
            else if(x <= 0)       result = 1.0f + 3 * (1+x) + 3 * std::pow(1+x, 2) - 3 * std::pow(1+x, 3);
            else if(x < 1)        result = 1.0f + 3 * (1-x) + 3 * std::pow(1-x, 2) - 3 * std::pow(1-x, 3);
            else if(x < 2)        result = std::pow(2-x, 3);

            y = (1.0f/6.0f) * result;
        }
        else
        {
            if(x < -1 || x > 1)  y = 0;
            else if(x < 0)       y = x + 1;
            else                 y = -x + 1;
        }
    }

    // 1st derivatives of BSpline function
    inline void BSpline_interpolation_deriv(float x, float& y, bool thirdOrder = true)
    {
        if(thirdOrder)                  // perform 3rd-order B-Spline interpolation
        {
            if(x <= -2 || x >= 2) y = 0;
            else if(x <= -1)      y = 0.5 * std::pow(2+x, 2);
            else if(x <= 0)       y = 0.5 * (1.0f + 2 * (1+x) - 3 * std::pow(1+x, 2));
            else if(x < 1)        y = 0.5 * (-1.0f - 2 * (1-x) + 3 * std::pow(1-x, 2));
            else if(x < 2)        y = -0.5 * std::pow(2-x, 2);
        }
        else                         // perform 1st-order B-Spline interpolation
        {
            if(x < -1 || x > 1)  y = 0;
            else if(x < 0)       y = 1;
            else                 y = -1;
        }

    }

    // 2nd derivatives of BSpline function
    inline void BSpline_interpolation_deriv2(float x, float& y, bool thirdOrder = true)
    {
        if(thirdOrder)                  // perform 3rd-order B-Spline interpolation
        {
            if(x <= -2 || x >= 2) y = 0;
            else if(x <= -1)      y = (2+x);
            else if(x <= 0)       y = 1.0f - 3 * (1+x);
            else if(x < 1)        y = 1.0f - 3 * (1-x);
            else if(x < 2)        y = (2-x);
        }
        else                         // perform 1st-order B-Spline interpolation
        {
            if(x < -1 || x > 1)  y = 0;
            else if(x < 0)       y = 0;
            else                 y = 0;
        }
    }


};

}

template<typename CharT, typename Traits>
std::ostream& operator<< (std::basic_ostream<CharT, Traits> &out, const DSLAM::IterationPrintInfo &Info)
{
    out
    << "current lvl: " << Info.currentlvl << endl
    << "num In EM: "   << Info.currentNumInEM << endl
    << "total iterations: "   << Info.iterations << endl
    << "accpted iterations: " << Info.acceptedIterations << endl
    << "final lamba: " << Info.lamba << endl
    << "leftEnergy: "  << Info.leftEnergy << endl
    << "num of GoodRes: " << Info.num_GoodRes << endl;

    return out;
}

template<typename CharT, typename Traits>
std::ostream& operator<< (std::basic_ostream<CharT, Traits> &out, const DSLAM::IterationInfoInLvl0 &result)
{
    for(int lvl = result.IteInfo.size() - 1; lvl >= 0; lvl--)
    {
        int numEM = result.IteInfo[lvl].size();
        out << "Result in level " << lvl << ": " << "numOfEM = " << numEM << endl
            << "numInEM" << "   " << "iterations" << "  " << "acceptIterations" << "    " << "lamba" << "   "
            << "leftEnergy" << "  " << "num_GoodRes" << endl;
        for(vector<DSLAM::IterationPrintInfo>::const_iterator it = result.IteInfo[lvl].begin(); it!=result.IteInfo[lvl].end(); it++)
        {
            out << it->currentNumInEM << "  "
                << it->iterations << "    "
                << it->acceptedIterations << "  "
                << it->lamba << "   "
                << it->leftEnergy << "  "
                << it->num_GoodRes << " "
                << endl;
        }
        out << endl;
    }

    return out;
}

template<typename CharT, typename Traits>
std::ostream& operator<< (std::basic_ostream<CharT, Traits> &out, const vector<DSLAM::IterationPrintInfo> &result)
{
    int lvl = result.size();

    out << "Result in current frame " << ": " << endl
        << "level" << "   " << "iterations" << "  " << "acceptIterations" << "    " << "lamba" << "   " << "iteStep" << "   "
        << "finalMuInfo" << "  " << "numSamPnts" << endl;
    for(vector<DSLAM::IterationPrintInfo>::const_iterator it = result.begin(); it != result.end(); it++)
    {
        out << lvl-- << "  "
            << it->iterations << "    "
            << it->acceptedIterations << "  "
            << it->lamba << "   "
            << it->iteStep << "  "
            << it->MuInfo << "  "
            << it->num_GoodRes << " "
            << endl;
    }

    return out;
}
#endif
