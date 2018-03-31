#ifndef COMMON_H
#define COMMON_H


#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <map>
#include <string>
#include <cmath>
#include <algorithm>
#include <functional>
using namespace std;

// Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include <Eigen/StdVector>

// Sophus
#include <sophus/sim3.hpp>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// boost
#include <boost/format.hpp>
#include <boost/timer.hpp>
#include <boost/lexical_cast.hpp>

#define PYR_LEVELS 5
#define patternNum 8
#define patchSize 9            // 7*7 patch, used to compute the center of intensity
//#define depthScale 1000.0f
#define depthScale 5000.0f    // Tum RGB datasets

#define BinsOfHis 8           // bins of histogram, scale = 255 / 8
static float NumPntsInMI = 1000;       // number of points selected for Mutual Information optimization

static float setting_minGradHistCut = 0.5;
static float setting_minGradHistAdd = 7;
static float setting_gradDownweightPerLevel = 0.75;
static bool  setting_selectDirectionDistribution = true;

static float setting_maxShiftWeightT= 0.04f * (640+480);
static float setting_maxShiftWeightR= 0.0f * (640+480);
static float setting_maxShiftWeightRT= 0.02f * (640+480);
static float setting_kfGlobalWeight = 1;   // general weight on threshold, the larger the more KF's are taken (e.g., 2 = double the amount of KF's).

static float setting_minGoodResiduals = 3;
static float setting_huberTH = 9;
static float setting_huberTH2 = 1.345;
static float setting_outlierTHSumComponent = 50*50;
static float setting_outlierTH = 12*12;
static float setting_maxPixelSearch = 0.027;
static float setting_trace_slackInterval = 1.5;
static float setting_trace_extraSlackOnTH = 1.2;
static int setting_minTraceTestRadius = 2;
static int setting_trace_GNIterations = 3;				// max # GN iterations
static int setting_trace_GNThreshold = 0.1;

static float setting_coarseCutoffTH = 3.5;

const static int pattern[8][2] = {{0,0},	  {-1,-1},	   {1,-1},		{-2,0},
                           {0,-2},	  {2,0},	   {-1,1},		{0,2}};

enum ResState {IN=0, OOB, OUTLIER};
enum DistributionMode {TDistribution=0, NormalDistribution};

#endif
