#ifndef GLOBALFUNCS_H_
#define GLOBALFUNCS_H_

#include <common.h>
#include <DataType.h>
#include <CameraIntrinsic.h>

namespace DSLAM
{

    EIGEN_STRONG_INLINE float bilinearInterpolation(const cv::Mat& img, float x, float y)
    {
        int x0 = (int) std::floor(x);
        int y0 = (int) std::floor(y);

        const float x1_weight = x - x0;
        const float x0_weight = 1.0f - x1_weight;
        const float y1_weight = y - y0;
        const float y0_weight = 1.0f - y1_weight;

        int width = img.cols;
        int idx = x0 + y0*width;

        const float* dataI = img.ptr<float>();

        float interpolated = x0_weight * y0_weight * dataI[idx] +
                               x1_weight * y0_weight * dataI[idx+1] +
                               x0_weight * y1_weight * dataI[idx+width] +
                               x1_weight * y1_weight * dataI[idx+width+1];
        return interpolated;

    }

    EIGEN_STRONG_INLINE Vector2 bilinearInterpolation(const cv::Mat& img, const cv::Mat& depth, float x, float y)
    {
        int x0 = (int) std::floor(x);
        int y0 = (int) std::floor(y);
        int x1 = x0 + 1;
        int y1 = y0 + 1;

        const float x1_weight = x - x0;
        const float x0_weight = 1.0f - x1_weight;
        const float y1_weight = y - y0;
        const float y0_weight = 1.0f - y1_weight;

        int width = img.cols;
        int idx = x0 + y0*width;

        const float* dataI = img.ptr<float>();
        const float* dataD = depth.ptr<float>();

        Vector2 interpolated = x0_weight * y0_weight * Vector2(dataI[idx], dataD[idx]) +
                               x1_weight * y0_weight * Vector2(dataI[idx+1], dataD[idx+1]) +
                               x0_weight * y1_weight * Vector2(dataI[idx+width], dataD[idx+width]) +
                               x1_weight * y1_weight * Vector2(dataI[idx+width+1], dataD[idx+width+1]);
        return interpolated;

    }


    EIGEN_STRONG_INLINE Vector6 bilinearInterpolation(const cv::Mat& img, const cv::Mat& depth,
                                                     const Eigen::Vector2f* intGrad, const Eigen::Vector2f* depGrad,
                                                     float x, float y)
   {
     int x0 = (int) std::floor(x);
     int y0 = (int) std::floor(y);
     int x1 = x0 + 1;
     int y1 = y0 + 1;

     const float x1_weight = x - x0;
     const float x0_weight = 1.0f - x1_weight;
     const float y1_weight = y - y0;
     const float y0_weight = 1.0f - y1_weight;

     int width = img.cols;
     int idx = x0 + y0*width;

     const float* dataI = img.ptr<float>();
     const float* dataD = depth.ptr<float>();
     const Eigen::Vector2f* dataIG = intGrad;
     const Eigen::Vector2f* dataDG = depGrad;

     Vector6 temp1, temp2, temp3, temp4;
     temp1 << dataI[idx], dataD[idx], dataIG[idx](0), dataIG[idx](1), dataDG[idx](0), dataDG[idx](1);
     temp2 << dataI[idx+1],dataD[idx+1],dataIG[idx+1],dataDG[idx+1];
     temp3 << dataI[idx+width],dataD[idx+width],dataIG[idx+width],dataDG[idx+width];
     temp4 << dataI[idx+width+1],dataD[idx+width+1],dataIG[idx+width+1],dataDG[idx+width+1];

     Vector6 interpolated =
         temp1 * x0_weight * y0_weight +
         temp2 * x1_weight * y0_weight +
         temp3 * x0_weight * y1_weight+
         temp4 * x1_weight * y1_weight;

     return (interpolated);
   }

   EIGEN_STRONG_INLINE Vector6 bilinearInterpolationWithDepthBuffer(const cv::Mat& intensity, const cv::Mat& depth,
                                                                            const Eigen::Vector2f* intGrad, const Eigen::Vector2f* depGrad,
                                                                            float x, float y, float z)
   {
     const int x0 = (int) std::floor(x);
     const int y0 = (int) std::floor(y);
     const int x1 = x0 + 1;
     const int y1 = y0 + 1;

     int width = intensity.cols;
     int idx = x0 + y0*width;

     if(x1 >= intensity.cols || y1 >= intensity.rows)
     {
         Vector6 result;
         result << Invalid, Invalid, Invalid, Invalid, Invalid, Invalid;

         return result;
     }

     const float x1_weight = x - x0;
     const float x0_weight = 1.0f - x1_weight;
     const float y1_weight = y - y0;
     const float y0_weight = 1.0f - y1_weight;
     const float z_eps = 0.05f;

     Vector6 val = Vector6::Zero();
     int numInTerms = 0;

     const float* dataI = intensity.ptr<float>();
     const float* dataD = depth.ptr<float>();
     const Eigen::Vector2f* dataIG = intGrad;
     const Eigen::Vector2f* dataDG = depGrad;
     if((dataD[idx] > 0.25) && (fabs(dataD[idx] - z) < z_eps) && !isnan(dataIG[idx][0]) && !isnan(dataIG[idx][1]) && !isnan(dataDG[idx][0]) && !isnan(dataIG[idx][1]))
     {       
       Vector6 result;
       result << dataI[idx], dataD[idx], dataIG[idx](0), dataIG[idx](1), dataDG[idx](0), dataDG[idx](1);
       val += x0_weight * y0_weight * result;
       numInTerms++;
     }

     if((dataD[idx+1] > 0.25) && (fabs(dataD[idx+1] - z) < z_eps) && !isnan(dataIG[idx+1][0]) && !isnan(dataIG[idx+1][1]) && !isnan(dataDG[idx+1][0]) && !isnan(dataIG[idx+1][1]))
     {
       Vector6 result;
       result << dataI[idx+1], dataD[idx+1], dataIG[idx+1](0), dataIG[idx+1](1), dataDG[idx+1](0), dataDG[idx+1](1);
       val += x1_weight * y0_weight * result;
       numInTerms++;
     }

     if((dataD[idx+width] > 0.25) && (fabs(dataD[idx+width] - z) < z_eps) && !isnan(dataIG[idx+width][0]) && !isnan(dataIG[idx+width][1]) &&
                                                                         !isnan(dataDG[idx+width][0]) && !isnan(dataIG[idx+width][1]))
     {
       Vector6 result;
       result << dataI[idx+width], dataD[idx+width], dataIG[idx+width](0), dataIG[idx+width](1), dataDG[idx+width](0), dataDG[idx+width](1);
       val += x0_weight * y1_weight * result;
       numInTerms++;
     }

     if((dataI[idx+width+1] > 0.25) && (fabs(dataD[idx+width+1] - z) < z_eps) && !isnan(dataIG[idx+width+1][0]) && !isnan(dataIG[idx+width+1][1]) &&
                                                                         !isnan(dataDG[idx+width+1][0]) && !isnan(dataIG[idx+width+1][1]))
     {
       Vector6 result;
       result << dataI[idx+width+1], dataD[idx+width+1], dataIG[idx+width+1](0), dataIG[idx+width+1](1), dataDG[idx+width+1](0), dataDG[idx+width+1](1);
       val += x1_weight * y1_weight * result;
       numInTerms++;
     }

     if(numInTerms == 0)
     {
       val(1) = Invalid;
     }

     return val;
   }

   EIGEN_STRONG_INLINE Eigen::Vector2f derive_idepth( const Vector3 &t, const float &u, const float &v,
                                            const float &drescale)
   {
           return Eigen::Vector2f((t[0]-t[2]*u), (t[1]-t[2]*v)) * drescale;
   }



   EIGEN_STRONG_INLINE bool projectPoint(const float &u_pt, const float &v_pt, const float &idepth, const CameraIntrinsic& camera,
                                   float& transformdepth, const Mat3x3 &KRKi, const Vector3 &Kt, float &Ku, float &Kv, Eigen::Vector2i bound)
   {
           Mat3x3 cam;
           cam << camera.get_fx(), 0, camera.get_cx(),
                  0, camera.get_fy(), camera.get_cy(),
                   0, 0, 1;

           Vector3 ptp = KRKi * Vector3(u_pt,v_pt, 1) + Kt*idepth;
           Ku = ptp[0] / ptp[2];
           Kv = ptp[1] / ptp[2];

           Vector3 PT = (1.0f/idepth) * (cam.inverse() * ptp);
           transformdepth = PT[2];
           return Ku>1.1f && Kv>1.1f && Ku<(float)(bound(0)-3) && Kv<(float)(bound(1)-3);
   }



   EIGEN_STRONG_INLINE bool projectPoint(const float &u_pt, const float &v_pt, const float &idepth,
                                   const int &dx, const int &dy, const CameraIntrinsic& camera,
                                   const Mat3x3 &R, const Vector3 &t,float &drescale,
                                   float &u, float &v, float &Ku, float &Kv,
                                   float &new_idepth, Eigen::Vector2i bound)
   {
           Vector3 ptp = Vector3(
                           (u_pt + dx - camera.get_cx()) * (1.0f/camera.get_fx()),
                           (v_pt + dy - camera.get_cy()) * (1.0f/camera.get_fy()),
                           1);

           ptp = R * ptp + t*idepth;
           drescale = 1.0f/ptp[2];
           new_idepth = idepth*drescale;

           if(!(drescale>0)) return false;

           u = ptp[0] * drescale;
           v = ptp[1] * drescale;
           Ku = u * camera.get_fx() + camera.get_cx();
           Kv = v * camera.get_fy() + camera.get_cy();

           return Ku>1.1f && Kv>1.1f && Ku<(float)(bound(0)-3) && Kv<(float)(bound(1)-3);
   }

   EIGEN_STRONG_INLINE Vector2 ComputeGrayCentroid(const cv::Mat& img, float x, float y)
   {
       int x0 = (int) std::floor(x);
       int y0 = (int) std::floor(y);

       int size = 0.5 * (patchSize - 1);    // size = 3
       int width = img.cols;

       const float* data = img.ptr<float>(y0,x0);

       float sumGray = 0;
       float sumX = 0, sumY = 0;
       for(int i = 1; i <= size; ++i)
           for(int j = 1; j <= size; ++j)
           {
               int bais1 = width * i + j;
               int bais2 = width * i - j;

               sumX += -j * (*(data-bais1) + *(data+bais2)) + j * (*(data-bais2) + *(data+bais1));
               sumY += i * (*(data-bais1) + *(data-bais2)) - i * (*(data+bais2) + *(data+bais1));
               sumGray += *(data-bais1) + *(data-bais2) + *(data+bais1) + *(data+bais2);
           }

       for(int k = 1; k <= size; k++)
       {
           sumX += k * (*(data+k) - *(data-k));
           sumY += k* (*(data - k*width) - *(data + k*width));
           sumGray += *(data+k) + *(data-k) + *(data+k*width) + *(data-k*width);
       }

       return Vector2(sumX/sumGray, sumY/sumGray);
   }
}

#endif
