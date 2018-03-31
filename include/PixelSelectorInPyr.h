#ifndef PIXELSELECTORPYR_H_
#define PIXELSELECTORPYR_H_

#include <Frame.h>

namespace DSLAM
{

const float minUseGrad_pixel = 10;

template<int pot>
inline int gridMaxSelection(const Frame* const frame, int level, bool* map_out, float THFac)
{
    int wl = frame->width[level];
    int hl = frame->height[level];

    memset(map_out, 0, sizeof(bool)*wl*hl);

    int numGood = 0;

    for(int y = 1; y < hl - pot; y+=pot)
        for(int x = 1; x < wl - pot; x+=pot)
        {
            int bestXXID = -1;
            int bestYYID = -1;
            int bestXYID = -1;
            int bestYXID = -1;

            float bestXX = 0, bestYY = 0, bestXY = 0, bestYX = 0;

            float* gradNorm = frame->GradNorm[level] + x + y*wl;
            Eigen::Vector2f* grad = frame->Grad_Int[level] + x + y*wl;
            for(int dy = 0; dy < pot; dy++)
                for(int dx = 0; dx < pot; dx++)
                {
                    int idx = dx + dy*wl;
                    float gNorm = gradNorm[idx];
                    Eigen::Vector2f gXY = grad[idx];
                    float TH = THFac * minUseGrad_pixel * 0.75;

                    if(gNorm > TH*TH)
                    {
                        float agx = fabs(gXY[0]);
                        if(agx > bestXX)
                        {
                            bestXX = agx; bestXXID = idx;
                        }

                        float agy = fabs(gXY[1]);
                        if(agy > bestYY)
                        {
                            bestYY = agy; bestYYID = idx;
                        }

                        float agxy = fabs(gXY[0] - gXY[1]);
                        if(agxy > bestXY)
                        {
                            bestXY = agxy; bestXYID = idx;
                        }

                        float agyx = fabs(gXY[0] + gXY[1]);
                        if(agyx > bestYX)
                        {
                            bestYX = agyx; bestYXID = idx;
                        }
                    }
                }

            bool* map0 = map_out + x + y*wl;

            if(bestXXID >= 0)
            {
                if(!map0[bestXXID])
                    numGood++;
                map0[bestXXID] = true;
            }

            if(bestYYID >= 0)
            {
                if(!map0[bestYYID])
                    numGood++;
                map0[bestYYID] = true;
            }

            if(bestXYID >= 0)
            {
                if(!map0[bestXYID])
                    numGood++;
                map0[bestXYID] = true;
            }

            if(bestYXID >= 0)
            {
                if(!map0[bestYXID])
                    numGood++;
                map0[bestYXID] = true;
            }
        }

    return numGood;

}

inline int PixelSelectorInPyr(const Frame* const frame, int level, bool* map_out, int& potential,
                              float desiredDensity, int recsLeft = 5, float THFac = 1)
{
   if(potential < 1)  potential = 1;

   int numGoodPoints = 0;

   if(potential==1) numGoodPoints = gridMaxSelection<1>(frame, level, map_out, THFac);
   else if(potential==2) numGoodPoints = gridMaxSelection<2>(frame, level, map_out, THFac);
   else if(potential==3) numGoodPoints = gridMaxSelection<3>(frame, level, map_out, THFac);
   else if(potential==4) numGoodPoints = gridMaxSelection<4>(frame, level, map_out, THFac);
   else if(potential==5) numGoodPoints = gridMaxSelection<5>(frame, level, map_out, THFac);
   else if(potential==6) numGoodPoints = gridMaxSelection<6>(frame, level, map_out, THFac);
   else if(potential==7) numGoodPoints = gridMaxSelection<7>(frame, level, map_out, THFac);
   else if(potential==8) numGoodPoints = gridMaxSelection<8>(frame, level, map_out, THFac);
   else if(potential==9) numGoodPoints = gridMaxSelection<9>(frame, level, map_out, THFac);
   else if(potential==10) numGoodPoints = gridMaxSelection<10>(frame, level, map_out, THFac);
   else if(potential==11) numGoodPoints = gridMaxSelection<11>(frame, level, map_out, THFac);
   else
       std::cout << "Please choose potential again...." << endl;

   float quotia = (float)(numGoodPoints / desiredDensity);

   int newPotential = (potential * sqrtf(quotia)) + 0.7f;

   if(newPotential < 1) newPotential = 1;

   float oldTHFac = THFac;
   if(newPotential == 1 && potential == 1)  THFac *= 0.5;

   if((abs(newPotential - potential) < 1 && THFac == oldTHFac) || (quotia > 0.8 && quotia < 1.25)
           || recsLeft == 0)
   {
       potential = newPotential;
       return numGoodPoints;
   }
   else
   {
       potential = newPotential;
       return PixelSelectorInPyr(frame, level, map_out, potential, desiredDensity, recsLeft - 1, THFac);
   }

}

}

#endif
