#ifndef FRAMEREADER_H_
#define FRAMEREADER_H_

#include <parameter_reader.h>
#include <Frame.h>

namespace DSLAM
{

class FrameReader
{

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    FrameReader(const ParameterReader& para) : paraReader(para)
    {
        initia_Datasets();
    }

    Frame* getNextFrame();
    Frame* getSpecificFrame(const int idx);

    inline void reset()
    {
        cout << "reset frame reader" << endl;
        current_idx = start_idx;

        rgbfiles.clear();
        depthfiles.clear();
    }

    Vector7d current_pose;

    inline const Vector7d& getCurrentPose() const
    {
        return truth.at(current_idx);
    }

    inline const Vector7d& getSpecificPose(const int idx) const
    {
        return truth.at(idx);
    }

    inline int getSizeOfData() const
    {
        return TimeStamps.size();
    }

    CameraIntrinsic camera;
protected:

    void initia_Datasets();

    int current_idx;
    int start_idx;

    const ParameterReader paraReader;

    vector<string> rgbfiles, depthfiles;
    vector<double> TimeStamps;
    vector<Vector7d, Eigen::aligned_allocator<Vector7d>> truth;

    string dataset_dir;


};

}

#endif
