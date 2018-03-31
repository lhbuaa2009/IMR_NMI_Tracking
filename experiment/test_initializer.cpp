#include <Intializer.h>
#include <FrameReader.h>
#include <iomanip>

using namespace DSLAM;
int main(int argc, char** argv)
{
    ParameterReader para;
    FrameReader fr(para);

    const int Index = 0;
    const Vector7d ref_pose_first = fr.getCurrentPose();  // groundtruth of the first frame pose
    Frame* frame = fr.getSpecificFrame(Index);
    if(!frame)
    {
        cout << "The specific frame dosen't exist in datasets!" << endl;
        return 0;
    }

    cout << "Found the specific image !" << endl;

    // if this Frame exist
    int w = frame->width[0], h = frame->height[0];
    Initializer* initializer = new Initializer(w, h);
    initializer->setFirst(frame);      // set the first frame

    for(int lvl = 0; lvl < PYR_LEVELS; lvl++)
    {
        cout << "Num of points in level " << lvl << ": " << initializer->numPoints[lvl];
        cout << endl;
    }

    // track the next frame*********************************************************************
/*
    for(int i = 1; i <= 4; i++)
    {
        const Vector7d ref_pose_new = fr.getCurrentPose();
        Frame* newFrame = fr.getNextFrame();

        if(!newFrame)
        {
            cout << "The next frame dosen't exist in datasets!" << endl;
        }

        initializer->trackFrame(newFrame);

        cout << "Track Result on Frame " << i << " :" << "*************************************************" << endl
             << "Estimate pose : " << initializer->thisToNext.log().transpose() << endl;

        {
            // calculate corresponding relative transformation using ground truth
            Eigen::Quaterniond q0(ref_pose_first(6), ref_pose_first(3), ref_pose_first(4), ref_pose_first(5));
            Eigen::Quaterniond q1(ref_pose_new(6), ref_pose_new(3), ref_pose_new(4), ref_pose_new(5));

            SE3 PoseFirst(q0, ref_pose_first.head(3));
            SE3 PoseNew(q1, ref_pose_new.head(3));
            SE3 firstToNew = PoseNew.inverse() *  PoseFirst;   // relative transformation in grouthtruth

            cout << "Reference pose : " << firstToNew.log().transpose() << endl;

            // approximate distance
            SE3 dis = firstToNew * initializer->thisToNext.inverse();
            double distance = dis.log().norm();

            cout << "distance = " << distance << endl;
        }

    }
*/


    cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << endl;
/*    int index = 1;
    Frame* newFrame = fr.getSpecificFrame(index);               // groundtruth of the new frame pose
    const Vector7d ref_pose_new = fr.getSpecificPose(index);
    initializer->trackFrame(newFrame);
    cout << "Track Result on Frame " << index << " :" << "*************************************************" << endl
         << "Estimate pose : " << initializer->thisToNext.log().transpose() << endl;

    {
        // calculate corresponding relative transformation using ground truth
        Eigen::Quaterniond q0(ref_pose_first(6), ref_pose_first(3), ref_pose_first(4), ref_pose_first(5));
        Eigen::Quaterniond q1(ref_pose_new(6), ref_pose_new(3), ref_pose_new(4), ref_pose_new(5));

        SE3 PoseFirst(q0, ref_pose_first.head(3));
        SE3 PoseNew(q1, ref_pose_new.head(3));
        SE3 firstToNew = PoseNew.inverse() *  PoseFirst;   // relative transformation in grouthtruth

        cout << "Reference pose : " << firstToNew.log().transpose() << endl;

        // approximate distance
        SE3 dis = firstToNew * initializer->thisToNext.inverse();
        double distance = dis.log().norm();

        cout << "distance = " << distance << endl;
    }

*/
    Eigen::Matrix<double, 3, 3> R;
    R << 1,0,0,
         0,1,0,
         0,0,1;
    Eigen::Vector3d t(0,0,0);
    SE3 refToNew(R,t);          // set initial value of SE3 transformation

    double totalError = 0;
    double meanError = 0;
    int FrameSize = 15;
    for(int j = 1; j <= FrameSize; j++)
    {
        initializer->setFirst(frame);
//        initializer->thisToNext = refToNew;

        Frame* newFrame = fr.getSpecificFrame(j);               // groundtruth of the new frame pose
        const Vector7d ref_pose_new = fr.getSpecificPose(j);

        if(!newFrame)
        {
            cout << "The next frame dosen't exist in datasets!" << endl;
            continue;
        }

        initializer->trackFrame(newFrame);
        SE3 PoseInWorld = initializer->thisToNext.inverse();
        Vector3 CameraCenter = PoseInWorld.translation().cast<float>();
        Eigen::Quaterniond rot(PoseInWorld.rotationMatrix());

        cout << "Track Result on Frame " << j << " :" << "*************************************************" << endl;
        cout << std::fixed;
        cout << "timeStamp : " << setprecision(6) << newFrame->timestamp << endl;
        cout.unsetf(ios_base::fixed);
        cout << "Estimate pose : " << CameraCenter.transpose() << " " << rot.x() << " "<< rot.y() << " " << rot.z() << " " << rot.w() << endl;

        {
            // calculate corresponding relative transformation using ground truth
            Eigen::Quaterniond q0(ref_pose_first(6), ref_pose_first(3), ref_pose_first(4), ref_pose_first(5));
            Eigen::Quaterniond q1(ref_pose_new(6), ref_pose_new(3), ref_pose_new(4), ref_pose_new(5));

            SE3 PoseFirst(q0, ref_pose_first.head(3));
            SE3 PoseNew(q1, ref_pose_new.head(3));
            SE3 firstToNew = PoseNew.inverse() *  PoseFirst;   // relative transformation in
            SE3 RefPose = firstToNew.inverse();                // ref camera pose in world coordinate (i.e., the coordinate of the first frame)
            Vector3 RefCameraCenter = RefPose.translation().cast<float>();
            Eigen::Quaterniond Refrot(RefPose.rotationMatrix());

            cout << "Reference pose : " << RefCameraCenter.transpose() << " " << Refrot.x() << " "<< Refrot.y() << " " << Refrot.z() << " " << Refrot.w() << endl;

            // approximate distance
//            SE3 dis = firstToNew * initializer->thisToNext.inverse();
//            double distance = dis.log().head(3).norm();
            Vector3 transError = CameraCenter - RefCameraCenter;
            double distance = transError.norm();
            cout << "distance = " << distance << endl;

            meanError += distance;
            totalError += transError.squaredNorm();
        }
    }

    cout << "Tracking done ############################################################" << endl;
    cout << "RMSE: " << sqrtf(totalError / FrameSize) << "; " << "MError: " << meanError / FrameSize << endl;

/*
    cout << "Track has been completed !" << endl;

    cout << "Estimate pose : *************************" << endl;
    cout << "Rotation Matrix : " << endl << initializer->thisToNext.rotationMatrix() << endl;
    cout << "Translation : " << endl << initializer->thisToNext.translation() << endl;
    cout << "se3 : " << initializer->thisToNext.log().transpose() << endl << endl;
*/


    return 0;
}
