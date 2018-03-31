#include <FrameReader.h>

using namespace DSLAM;
int main(int argc, char** argv)
{
    ParameterReader para;
    FrameReader fr(para);

    while(Frame* frame = fr.getNextFrame())
    {
        cv::Mat rgb;
        frame->intensity[0].convertTo(rgb, CV_8UC1);
        string name = "Room_Img";

        cv::namedWindow(name, cv::WINDOW_AUTOSIZE);
        cv::imshow(name, rgb);
        cv::waitKey(0);

     }

    return 0;
}
