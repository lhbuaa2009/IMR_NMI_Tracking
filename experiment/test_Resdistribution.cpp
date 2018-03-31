#include <common.h>
#include <Frame.h>
#include <FrameReader.h>
#include <globalFuncs.h>
#include <CameraIntrinsic.h>
#include <PixelSelector.h>
#include <sstream>

using namespace DSLAM;

const int Baies = 200;
int calcIntensityRes(const Frame* ref, const Frame* cur, const SE3& refTocur,
                      cv::Mat& Output, bool* Mask, int* histo);

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

    //----------------------calc intensity residuals------------------------------------
    int w = frame->width[0], h = frame->height[0];
    cv::Mat output(h, w, CV_32FC1);
    cv::Mat display;
    bool* mask = new bool[w*h];
    int* histo = new int[400];

    int Num = calcIntensityRes(frame, newFrame, firstToNew, output, mask, histo);
    cout << "Num of Projected Points = " << Num << endl;

    output.convertTo(display, CV_8UC1);
    cv::namedWindow("Intensity Residuals", cv::WINDOW_AUTOSIZE);
    cv::imshow("Intensity Residuals", display);
    cv::waitKey(0);

    cv::imwrite("/home/luo/Pictures/Residuals.png", display);
/*
    cout << "Histogram of intensity residuals: " << endl;
    for(int i = 0; i < 400; i++)
        cout << histo[i] << " ";
    cout << endl;
*/
}

int calcIntensityRes(const Frame *ref, const Frame *cur, const SE3& refTocur,
                      cv::Mat& Output, bool* Mask, int* histo)
{

    CameraIntrinsic* camera = ref->intrinsic;
    Mat3x3 K = camera->ConvertToMatrix();
    Mat3x3 Ki = K.inverse();
    int w = ref->width[0], h = ref->height[0];

    cv::Mat refColor = ref->intensity[0];
    cv::Mat curColor = cur->intensity[0];
    cv::Mat refDepth = ref->depth[0];
    cv::Mat curDepth = cur->depth[0];

    memset(Mask, false, sizeof(bool)*w*h);
    memset(histo, 0, sizeof(int)*400);

    Mat3x3 KRKi = K * (refTocur.rotationMatrix().cast<float>()) * Ki;
    Vector3 Kt = K * refTocur.translation().cast<float>();

    float* gray = refColor.ptr<float>();
    float* depth = refDepth.ptr<float>();
    float* out = Output.ptr<float>();

    float* statusMap = new float[w*h];
    float density = 0.03*w*h;
    PixelSelector selector(w,h);
    int npts = selector.makeMaps(ref, statusMap, density, 1, true, 2);
    if(npts < 50)
    {
        cout << "too few points" << endl;
        return -1;
    }

    int Num = 0;
    float minValue = 1e6;
    float maxValue = -1e6;
    ofstream fout("../resiudals.txt");
    ostringstream oss;
    for(int y = 1; y < h-2; y++)
        for(int x = 1; x < w-2; x++)
        {
            int idx = x + w*y;
            if(statusMap[idx] == 0 ||
                 depth[idx] < 0.01 || depth[idx] > 10)
            {
                out[idx] = 0.0f;
                continue;
            }

            Vector3 Pnt = KRKi*Vector3(x,y,1) + Kt/depth[idx];
            float Ku = Pnt[0], Kv = Pnt[1];
            if(!(Ku > 1 && Kv > 1 && Ku < w-2 && Kv < h-2))
            {
                out[idx] = 0.0f;
                continue;
            }

            Vector2 hitData = bilinearInterpolation(curColor, curDepth, Ku, Kv);
            if(isnan(hitData[1]) || !std::isfinite((hitData[0])))
            {
                out[idx] = 0.0f;
                continue;
            }

            out[idx] = (gray[idx] - hitData[0]) + Baies;   // set bias = 50
            if(out[idx] > maxValue)   maxValue = out[idx];
            if(out[idx] < minValue)   minValue = out[idx];
            Mask[idx] = true;
            Num++;

            oss.str("");
            oss << out[idx];
            fout << oss.str() << " ";

            if(out[idx] < 0) continue;
            if(out[idx] > 400) continue;
            int entry = (int)(out[idx]);
            histo[entry]++;
        }

    fout.close();
    cout << "minValue = " << minValue << ", " << "maxValue = " << maxValue << endl;
    return Num;
}
