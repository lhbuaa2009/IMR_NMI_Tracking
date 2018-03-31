#include <Tracker.h>
#include <FrameReader.h>
#include <Optimizer.h>
#include <iomanip>

using namespace DSLAM;

int DSLAM::FrameShell::next_id = 0;

const int w = 640;
const int h = 480;

int main()
{
/*
 {
    ParameterReader para;
    FrameReader fr(para);

    Optimizer optimizer;

    const int Index = 0;

//    Frame* frame = fr.getSpecificFrame(1);
    Tracker* tracker = new Tracker(w,h,"./Tracker.txt");

    for(int lvl = 0; lvl < PYR_LEVELS; lvl++)
    {
        cout << "Num of points in level " << lvl << ": " << tracker->NumPnts[lvl];
        cout << endl;
    }

    SE3 refToNew = SE3();          // set initial value of SE3 transformation
    SE3 refToNew_MI = SE3();
    SE3 refToNew_backup = SE3();

    Vector5f MInfoTH = Vector5f::Zero();
    Vector5f EnergyTH = Vector5f::Constant(1e6);
    Vector7d ref_pose_new = Vector7d::Zero();
    Vector7d ref_pose_first = fr.getSpecificPose(Index);

    std::vector<Frame*> activeFrames;
    std::vector< SE3 > refPoses;
    double totalError=0, totalError2=0, totalError3=0;
    double meanError=0, meanError2=0, meanError3=0;
    int FrameSize = 5;
    for(int j = Index, id = 0; j <= Index+FrameSize; j++, id++)
    {
        MInfoTH = Vector5f::Zero();
        EnergyTH = Vector5f::Constant(1e6);

        Frame* newFrame = fr.getSpecificFrame(j);               // groundtruth of the new frame pose
        if(!newFrame)
        {
            cout << "The next frame dosen't exist in datasets!" << endl;
            continue;
        }

        FrameShell* shell = new FrameShell();
        shell->timestamp = newFrame->timestamp;
        shell->id = shell->marginalizedAt = id;
        shell->next_id = id + 1;

        newFrame->shell = shell;
        activeFrames.push_back(newFrame);

        if(j==Index)
        {
            tracker->makeTrackPointsbyFrame(newFrame);    // set refFrame for tracking
            refPoses.push_back(SE3());
            continue;
        }
        else  ref_pose_new = fr.getSpecificPose(j);

        tracker->trackNewFrame(newFrame, refToNew, PYR_LEVELS-1, Vector5f::Zero());
        refToNew_backup = refToNew;
        bool flag1 = tracker->OptimizMI(refToNew_MI, MInfoTH, EnergyTH);

        double MInfo_Ref = tracker->CalcRefMInfo(0, refToNew_MI);
        double MInfo_SSD = tracker->CalcRefMInfo(0, refToNew);

        if(flag1)
        {
            MInfoTH = tracker->lastMInfo;
            EnergyTH = tracker->lastEnergy;
        }

        bool flag2 = tracker->OptimizMI(refToNew, MInfoTH, EnergyTH, 2);

        if(flag2)  refToNew_MI = refToNew;
        else if(flag1) refToNew = refToNew_MI;
             else { refToNew = refToNew_backup;
                    refToNew_MI = refToNew_backup;
        }

        newFrame->shell->FrameToWorld = refToNew.inverse();

        SE3 PoseInWorld = refToNew.inverse();
        Vector3 CameraCenter = PoseInWorld.translation().cast<float>();
        Eigen::Quaterniond rot(PoseInWorld.rotationMatrix());

        SE3 PoseInWorld2 = refToNew_MI.inverse();
        Vector3 CameraCenter2 = PoseInWorld2.translation().cast<float>();
        Eigen::Quaterniond rot2(PoseInWorld2.rotationMatrix());

        cout << "##########################################################################################" << endl;
        cout << "Track Result on Frame " << j << " :"  << endl;
        cout << std::fixed;
        cout << "timeStamp : " << setprecision(6) << newFrame->timestamp << endl;
        cout.unsetf(ios_base::fixed);
        cout << "Estimate pose : " << CameraCenter.transpose() << " " << rot.x() << " "<< rot.y() << " " << rot.z() << " " << rot.w() << endl;
        cout << "Estimate pose2 : " << CameraCenter2.transpose() << " " << rot2.x() << " "<< rot2.y() << " " << rot2.z() << " " << rot2.w() << endl;

        {
            // calculate corresponding relative transformation using ground truth
            Eigen::Quaterniond q0(ref_pose_first(6), ref_pose_first(3), ref_pose_first(4), ref_pose_first(5));
            Eigen::Quaterniond q1(ref_pose_new(6), ref_pose_new(3), ref_pose_new(4), ref_pose_new(5));

            SE3 PoseFirst(q0, ref_pose_first.head(3));
            SE3 PoseNew(q1, ref_pose_new.head(3));
            SE3 firstToNew = PoseNew.inverse() *  PoseFirst;   // relative transformation in grouthtruth
            double MInfo_Tru = tracker->CalcRefMInfo(0, firstToNew);

            SE3 RefPose = firstToNew.inverse();                // ref camera pose in world coordinate (i.e., the coordinate of the first frame)
            Vector3 RefCameraCenter = RefPose.translation().cast<float>();
            Eigen::Quaterniond Refrot(RefPose.rotationMatrix());

            refPoses.push_back(RefPose);

            cout << "Reference pose : " << RefCameraCenter.transpose() << " " << Refrot.x() << " "<< Refrot.y() << " " << Refrot.z() << " " << Refrot.w() << endl;

            // approximate distance
//            SE3 dis = firstToNew * refToNew.inverse();
//            double distance = dis.log().head(3).norm();
            Vector3 transError = CameraCenter - RefCameraCenter;
            Vector3 transError2 = CameraCenter2 - RefCameraCenter;
            double distance = transError.norm();
            double distance2 = transError2.norm();
            cout << "distance = " << distance << ", " << "distance2 = " << distance2
                  << ", " << "MIRef = " << MInfo_Ref << ", " << "MISsd = " << MInfo_SSD << ", MITru = " << MInfo_Tru << endl;

            meanError += distance; meanError2 += distance2;
            totalError += transError.squaredNorm(); totalError2 += transError2.squaredNorm();
        }

    }

    cout << "Tracking done ############################################################" << endl
         << "RMSE: " << sqrtf(totalError / FrameSize) << "; " << "MError: " << meanError / FrameSize << endl
         << "RMSE2: " << sqrtf(totalError2 / FrameSize) << "; " << "MError2: " << meanError2 / FrameSize << endl;

    //--------------------------------do optimization--------------------------------------------------------
    cout << "do g2o_optimization ===========================================================================" << endl;
    int index1 = 1, index2 = 3;
    const Frame* pKF1 = activeFrames[index1];
    const Frame* pKF2 = activeFrames[index2];
    pKF1->shell->FrameToWorld = refPoses[index1];

    optimizer.DirectOptimization(pKF1, pKF2, 30, true);

    cout << "Optimized Result on Frame " << index1 << " :" << endl
         << "Optimized pose : " << activeFrames[index1]->shell->FrameToWorld.translation().transpose() << endl;
    cout << "Optimized Result on Frame " << index2 << " :" << endl
         << "Optimized pose : " << activeFrames[index2]->shell->FrameToWorld.translation().transpose() << endl;
 }
*/
    ParameterReader para;
    FrameReader fr(para);

    Optimizer optimizer;

    const int Index = 0;

    SE3 EstimatedPose = SE3();          // set initial value of SE3 transformation

    Vector7d ref_pose_new = Vector7d::Zero();
    Vector7d ref_pose_first = fr.getSpecificPose(Index);

    std::vector<Frame*> activeFrames;
    std::vector< SE3 > refPoses;
    double totalError=0;
    double meanError=0;
    int FrameSize = 8;
    for(int j = Index, id = 0; j <= Index+FrameSize; j++, id++)
    {
        Frame* newFrame = fr.getSpecificFrame(j);               // extract new frame from datasets
        if(!newFrame)
        {
            cout << "The next frame dosen't exist in datasets!" << endl;
            continue;
        }

        FrameShell* shell = new FrameShell();
        shell->timestamp = newFrame->timestamp;
        shell->id = shell->marginalizedAt = id;
        shell->next_id = id + 1;

        shell->FrameToWorld = EstimatedPose;

        newFrame->shell = shell;
        activeFrames.push_back(newFrame);

        if(j==Index)
        {
            refPoses.push_back(SE3());
            continue;
        }
        else  ref_pose_new = fr.getSpecificPose(j);

        //--------------------------------do optimization--------------------------------------------------------
        cout << "=============================do g2o_optimization ==============================================" << endl;
        Frame* pKF1 = activeFrames[0];
 //       optimizer.DirectOptimization(pKF1, newFrame, 30, true);
        optimizer.DirectUnaryOptimization(pKF1, newFrame, true);

        if(activeFrames.size() >= 3)
        {
            Frame* pKF2 = activeFrames[activeFrames.size()-2];
            optimizer.DirectOptimization(pKF2, newFrame, 30, true);
        }

        // estimated pose from g2o optimization
        EstimatedPose = newFrame->shell->FrameToWorld;
        Vector3 CameraCenter = EstimatedPose.translation().cast<float>();
        Eigen::Quaterniond rot(EstimatedPose.rotationMatrix());

//        cout << "###################################################" << endl;
        cout << "Track Result on Frame " << j << " :"  << endl;
        cout << std::fixed;
        cout << "timeStamp : " << setprecision(6) << newFrame->timestamp << endl;
        cout.unsetf(ios_base::fixed);
        cout << "Estimate pose : " << CameraCenter.transpose() << " " << rot.x() << " "<< rot.y() << " " << rot.z() << " " << rot.w() << endl;

        // reference pose from ground truth
        Eigen::Quaterniond q0(ref_pose_first(6), ref_pose_first(3), ref_pose_first(4), ref_pose_first(5));
        Eigen::Quaterniond q1(ref_pose_new(6), ref_pose_new(3), ref_pose_new(4), ref_pose_new(5));

        SE3 PoseFirst(q0, ref_pose_first.head(3));
        SE3 PoseNew(q1, ref_pose_new.head(3));
        SE3 RefPose = PoseFirst.inverse() * PoseNew;
        Vector3 RefCameraCenter = RefPose.translation().cast<float>();
        Eigen::Quaterniond Refrot(RefPose.rotationMatrix());
        refPoses.push_back(RefPose);

        cout << "Reference pose : " << RefCameraCenter.transpose() << " " << Refrot.x() << " "<< Refrot.y() << " " << Refrot.z() << " " << Refrot.w() << endl;

        Vector3 transError = CameraCenter - RefCameraCenter;
        double distance = transError.norm();
        cout << "distance = " << distance << endl;

        meanError += distance;
        totalError += transError.squaredNorm();
    }

    cout << "Tracking done ############################################################" << endl
         << "RMSE: " << sqrtf(totalError / FrameSize) << "; " << "MError: " << meanError / FrameSize << endl;

    return 0;

}
