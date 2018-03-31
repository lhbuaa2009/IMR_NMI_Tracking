/*
 * compute respective gray centroid of selected pixels,
 * using a 7*7 pathch
*/

#include <common.h>
#include <Frame.h>
#include <FrameReader.h>
#include <globalFuncs.h>
#include <CameraIntrinsic.h>
#include <PixelSelector.h>
#include <sstream>

using namespace DSLAM;

int main(int argc, char** argv)
{
    ParameterReader para;
    FrameReader fr(para);

    //-----------------------Read reference image------------------------------------
    const int Index = 0;
    const Vector7d ref_pose_first = fr.getCurrentPose();  // groundtruth of the first frame pose
    Frame* frame = fr.getSpecificFrame(Index);
    if(!frame)
    {
        cout << "The specific frame dosen't exist in datasets!" << endl;
        return 0;
    }
    cout << "Found the template image !" << endl;

    //-----------------------Read current image---------------------------------------
    int curIndx = 2;
    Frame* newFrame = fr.getSpecificFrame(curIndx);
    const Vector7d ref_pose_new = fr.getSpecificPose(curIndx);
    if(!newFrame)
    {
        cout << "The next frame dosen't exist in datasets!" << endl;
        return 0;
    }

    //----------------------calc relative tranformation--------------------------------
    Eigen::Quaterniond q0(ref_pose_first(6), ref_pose_first(3), ref_pose_first(4), ref_pose_first(5));
    Eigen::Quaterniond q1(ref_pose_new(6), ref_pose_new(3), ref_pose_new(4), ref_pose_new(5));

    SE3 PoseFirst(q0, ref_pose_first.head(3));
    SE3 PoseNew(q1, ref_pose_new.head(3));
    SE3 firstToNew = PoseNew.inverse() *  PoseFirst;   // relative transformation in grouthtruth

    //-----------------------Select pixels in ref image-------------------------------------------
    int w = frame->width[0], h = frame->height[0];
    float* statusMap = new float[w*h];
    float density = 0.03*w*h;
    PixelSelector selector(w,h);
    int npts = selector.makeMaps(frame, statusMap, density, 1, true, 2);
    if(npts < 50)
    {
        cout << "too few points" << endl;
        return -1;
    }

    //-----------------------Compute gray centroids------------------------------------
    Vector2* centroids = new Vector2[(int)(npts*1.1)];
    Vector2* corCentroids = new Vector2[(int)(npts*1.1)];
    Vector2* transCentroids = new Vector2[(int)(npts*1.1)];
    vector<float> variations;

    CameraIntrinsic* camera = frame->intrinsic;
    Mat3x3 K = camera->ConvertToMatrix();
    Mat3x3 Ki = K.inverse();

    Mat3x3 KRKi = K * (firstToNew.rotationMatrix().cast<float>()) * Ki;
    Vector3 Kt = K * firstToNew.translation().cast<float>();
    Mat2x2 Rplane = KRKi.topLeftCorner<2,2>();

    float* depth = frame->depth[0].ptr<float>();

    int num = 0;
    for(int y = 4; y < h-4; ++y)
        for(int x = 4; x < w-4; ++x)
        {
            int idx = x + w*y;
            if(statusMap[idx] == 0 ||
                 depth[idx] < 0.01 || depth[idx] > 10)
                continue;

            centroids[num] = ComputeGrayCentroid(frame->intensity[0], x, y);

            Vector3 Pnt = KRKi*Vector3(x,y,1) + Kt/depth[idx];
            float Ku = Pnt[0], Kv = Pnt[1];
            if(!(Ku > 4 && Kv > 4 && Ku < w-4 && Kv < h-4))
            {
                corCentroids[num] = Vector2(0.0f, 0.0f);
                transCentroids[num] = Vector2(0.0f, 0.0f);
                continue;
            }
            transCentroids[num] = Rplane * centroids[num];
            corCentroids[num] = ComputeGrayCentroid(newFrame->intensity[0], Ku, Kv);

            float cosine = (transCentroids[num].dot(corCentroids[num])) / (transCentroids[num].norm()*corCentroids[num].norm());
            variations.push_back(cosine);

            num++;
        }

    //-----------------------Display function--------------------------------------
    int count = 50;
    for(int i = 0; i < 100; i++)
    {
        cout << "<" << variations[count] << ">" << " ";
        count += 30;
    }
    cout << endl;

}
