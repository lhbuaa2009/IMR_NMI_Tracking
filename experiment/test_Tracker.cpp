#include <test_Tracker.h>
#include <Tracker.h>
#include <FrameReader.h>
#include <iomanip>

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>


using namespace DSLAM;

int main(int argc, char** argv)
{
    ParameterReader para;
    FrameReader fr(para);

    const int Index = 0;

    Frame* frame = fr.getSpecificFrame(Index);
    const Vector7d ref_pose_first = fr.getSpecificPose(Index);  // groundtruth of the first frame pose
    if(!frame)
    {
        cout << "The specific frame dosen't exist in datasets!" << endl;
        return 0;
    }

    cout << "Found the specific image !" << endl;

    int w = frame->width[0], h = frame->height[0];
    cout << "width = " << w << ", " << "height = " << h << endl;

    Tracker* tracker = new Tracker(w,h,"./Tracker.txt");
    tracker->makeTrackPointsbyFrame(frame);    // set refFrame for tracking
    for(int lvl = 0; lvl < PYR_LEVELS; lvl++)        
    {
        cout << "Num of points in level " << lvl << ": " << tracker->NumPnts[lvl];
        cout << endl;
    }
/*
    pcl::visualization::CloudViewer viewer("viewer");
    PointCloud::Ptr tmp( new PointCloud() );
    {
      //-----------Viewer: display the reference frame, has nothing to do with tracking process
       //----------------- should be removed/commented after debugging
        string dir = "/windows/literature/TumDatasets/rgbd_dataset_freiburg1_room/rgb/1305031910.765238.png";
        cv::Mat rgb = cv::imread(dir, -1);
        int n = tracker->NumPnts[0];
        DSLAM::SelectPnt* pnts = tracker->SeltPnts[0];
        float factor = 2.0;
        for(int i = 0; i < n; i++)
        {
            float idepth = pnts[i].idepth;
            if(!std::isfinite(idepth) || idepth < 1e-4)
                continue;

            PointT p;
            float depth = 1.0f/idepth;
            Vector3 Pnt3 = depth * (tracker->Ki[0] * Vector3(pnts[i].u, pnts[i].v, 1));
            p.z = depth;           p.z *= factor;
            p.x = Pnt3(0);         p.x *= factor;
            p.y = Pnt3(1);         p.y *= factor;

            int u = pnts[i].u, v = pnts[i].v;
  //          p.b = rgb.ptr<uchar>(v)[u*3];
  //          p.g = rgb.ptr<uchar>(v)[u*3+1];
  //          p.r = rgb.ptr<uchar>(v)[u*3+2];
            p.b = 0; p.g = 255; p.r = 0;
            p.a = 1;

            tmp->points.push_back(p);
        }

        tmp->is_dense = true;
        viewer.showCloud(tmp);
        cv::waitKey(0);
    }
*/

    SE3 refToPre = SE3();          // store SE3 transformation of the previous frame
    SE3 refToNew = SE3();          // set initial value of SE3 transformation
    SE3 refToNew_MI = SE3();
    SE3 refToNew_backup = SE3();
    Vector5f MInfoTH = Vector5f::Zero();
    Vector5f EnergyTH = Vector5f::Constant(1e6);
    double totalError=0, totalError2=0;
    double meanError=0, meanError2=0;
    int FrameSize = 8;
 //   int FrameTrace = 0;  // debug depth recovery algorithm
    for(int j = Index+1; j <= Index+FrameSize; j++)
    {
        MInfoTH = Vector5f::Zero();
        EnergyTH = Vector5f::Constant(1e6);

        Frame* newFrame = fr.getSpecificFrame(j);               // groundtruth of the new frame pose
        const Vector7d ref_pose_new = fr.getSpecificPose(j);

        if(!newFrame)
        {
            cout << "The next frame dosen't exist in datasets!" << endl;
            continue;
        }

        tracker->trackNewFrame(newFrame, refToNew, PYR_LEVELS-1, Vector5f::Zero());
        refToNew_backup = refToNew;
        bool flag1 = tracker->OptimizMI(refToNew_MI, MInfoTH, EnergyTH);
/*
        if(flag1)
        {
            MInfoTH = tracker->lastMInfo;
            EnergyTH = tracker->lastEnergy;
        }
        bool flag2 = tracker->OptimizMI(refToNew, MInfoTH, EnergyTH);
        if(flag2)
        {
          refToNew_MI = refToNew;
          refToNew = refToNew_backup;
        }
*/
        double MInfo_Ref = tracker->CalcRefMInfo(0, refToNew_MI);

        SE3 PoseInWorld = refToNew.inverse();
        Vector3 CameraCenter = PoseInWorld.translation().cast<float>();
        Eigen::Quaterniond rot(PoseInWorld.rotationMatrix());

        SE3 PoseInWorld2 = refToNew_MI.inverse();
        Vector3 CameraCenter2 = PoseInWorld2.translation().cast<float>();
        Eigen::Quaterniond rot2(PoseInWorld2.rotationMatrix());

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

            cout << "Reference pose : " << RefCameraCenter.transpose() << " " << Refrot.x() << " "<< Refrot.y() << " " << Refrot.z() << " " << Refrot.w() << endl;

            // approximate distance
//            SE3 dis = firstToNew * refToNew.inverse();
//            double distance = dis.log().head(3).norm();
            Vector3 transError = CameraCenter - RefCameraCenter;
            Vector3 transError2 = CameraCenter2 - RefCameraCenter;
            double distance = transError.norm();
            double distance2 = transError2.norm();
            cout << "distance = " << distance << ", " << "distance2 = " << distance2
                  << ", " << "MIRef = " << MInfo_Ref << ", " << "MITru = " << MInfo_Tru << endl;

            meanError += distance; meanError2 += distance2;
            totalError += transError.squaredNorm(); totalError2 += transError2.squaredNorm();
/*
            // debug depth recovery algorithm using depth filter, nothing to do with tracking
            // should be removed after debugging
            {
                FrameTrace++;
                float factor = 2.0;
                if(FrameTrace <= 15)      // trace 3 frames
                {
                    int trace_total = 0, trace_good = 0, trace_oob = 0, trace_out = 0, trace_skip = 0, trace_badcondition = 0, trace_converged = 0;
 //                   Mat3x3 KRKi = tracker->K[0] * refToNew.rotationMatrix().cast<float>() * tracker->Ki[0];
 //                   Vector3 Kt = tracker->K[0] * refToNew.translation().cast<float>();

                    Mat3x3 KRKi = tracker->K[0] * firstToNew.rotationMatrix().cast<float>() * tracker->Ki[0];
                    Vector3 Kt = tracker->K[0] * firstToNew.translation().cast<float>();

                    PointCloud::Ptr update( new PointCloud() );
                    int count = 0; double err = 0, err2 = 0;
                    cout << "------------------------start tracking------------------------" << endl;
                    for(ImmaturePoint* impt : frame->ImmaturePoints)
                    {
                        impt->traceOn(newFrame, refToNew, KRKi, Kt, fr.camera);

                        if(impt->lastTraceStatus == ImmaturePointStatus::IPS_GOOD) trace_good++;
                        if(impt->lastTraceStatus == ImmaturePointStatus::IPS_BADCONDITION) trace_badcondition++;
                        if(impt->lastTraceStatus == ImmaturePointStatus::IPS_OUTLIER) trace_out++;
                        if(impt->lastTraceStatus == ImmaturePointStatus::IPS_SKIPPED) trace_skip++;
                        if(impt->lastTraceStatus == ImmaturePointStatus::IPS_OOB) trace_oob++;
                        if(impt->lastTraceStatus == ImmaturePointStatus::IPS_CONVERGED) trace_converged++;
                        trace_total++;

                        if(impt->lastTraceStatus == ImmaturePointStatus::IPS_GOOD || impt->lastTraceStatus == ImmaturePointStatus::IPS_SKIPPED
                                || impt->lastTraceStatus == ImmaturePointStatus::IPS_CONVERGED)
                        {
                            count++;
                            if(count <= 20)
                            {
                                cout << "[" << impt->idepth_ref << ", " << impt->idepth << ", "
                                            << impt->idepth_min << ", " << impt->idepth_max << "]" << ", ";
                            }

                            err += (impt->idepth_ref - impt->idepth)*(impt->idepth_ref - impt->idepth);
                            err2 += (1.0f/impt->idepth_ref - 1.0f/impt->idepth)*(1.0f/impt->idepth_ref - 1.0f/impt->idepth);

                        }

                        if(impt->GoodObservations > 0)
                        {
                         //----------------viewer: show updated point cloud
                            float depth = (1.0f) / impt->idepth;
                            if(!std::isfinite(depth) || depth == 0)
                                continue;

                            float u = impt->u, v = impt->v;
                            PointT p;
                            Vector3 Pnt3 = depth * (tracker->Ki[0] * Vector3(u,v,1));
                            p.z = factor * Pnt3(2);
                            p.x = factor * Pnt3(0);
                            p.y = factor * Pnt3(1);
                            p.b = 0; p.g = 255; p.r = 0; p.a = 1;

                            update->points.push_back(p);
                        }

                    }

                    update->is_dense = true;
                    viewer.showCloud(update);
                    cv::waitKey(0);

                    cout << endl;
                    cout << "NumPnts displayed = " << update->points.size() << endl;
                    cout << "Trace Result : "
                         << "total=" << trace_total << ", " << "good=" << trace_good << ", " << "skip=" << trace_skip << ", "
                         << "out=" << trace_out << ", " << "oob=" << trace_oob << ", " << "badcondition=" << trace_badcondition << ", "
                         <<  "converged=" << trace_converged << endl;
                    cout << "RMSE of idepth : " << sqrtf(err / count) << ", "
                         << "RMSE of depth : " << sqrtf(err2 / count) << endl;
                }
            } */

        }
    /*
        SE3 tmp = refToNew;
        refToNew = (refToNew * refToPre.inverse()) * refToNew;   // estimate initial pose of next frame
        refToPre = tmp;  */
     }

    cout << "Tracking done ############################################################" << endl;
    cout << "RMSE: " << sqrtf(totalError / FrameSize) << "; " << "MError: " << meanError / FrameSize << endl;
    cout << "RMSE2: " << sqrtf(totalError2 / FrameSize) << "; " << "MError2: " << meanError2 / FrameSize << endl;

//    cout << tracker->IterationStatus;
    return 0;
}
