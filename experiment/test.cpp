
#include <testPixelSelector.h>

const double cx = 325.5;
const double cy = 253.5;
const double fx = 518.0;
const double fy = 519.0;

int main(int argc, char** argv)
{
    string imgDir = "../data/";
    if(argc > 1)
    {
        imgDir = argv[1];
    }

    string colorName = imgDir + "rgb.png";
    string depthName = imgDir + "depth.png";

    cv::Mat color, depth;
//    color = cv::imread(colorName.c_str(), cv::IMREAD_GRAYSCALE);

    color = cv::imread(colorName.c_str(), CV_LOAD_IMAGE_COLOR);
    depth = cv::imread(depthName.c_str(), -1);

    cv::cvtColor(color, color, CV_BGR2GRAY);

    if(color.empty() || depth.empty())
    {
        std::cout << "Could not find or open the image !" << endl;
        return -1;
    }
    else
    {
        std::cout << "Image size :" << " " << color.cols << " * " << color.rows << endl;
    }

    DSLAM::CameraIntrinsic camera(Eigen::Vector4f(fx,cx,fy,cy), Eigen::Vector4f::Zero());

    std::cout << "Camera Intrinsic: " << fx <<" " << cx <<" " << fy <<" " << cy << endl;

    testPixelSelector selector(color, depth, camera);

    int w = selector.width[0];
    int h = selector.height[0];
    float* statusMap = new float[w*h];
    float density = (float)(0.03 * w * h);

    int NumPT = selector.makeMaps(statusMap, density, 1, true, 2);

    std::cout << "Total Num of selected pixels = " << " " << NumPT << endl;

    return 0;
}
