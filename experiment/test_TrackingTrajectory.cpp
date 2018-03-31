#include <Tracker.h>
#include <FrameReader.h>
#include <iomanip>

using namespace DSLAM;

int DSLAM::FrameShell::next_id = 0;

const int width = 640;
const int height = 480;

struct FrameInfo
{
    bool GeometryCheck, BrightnessCheck;
    float energy;
    float ratio;
    float disparity;
    float abDevation;
    float distance;

    FrameInfo()
    {
        GeometryCheck = BrightnessCheck = false;
        energy = ratio = disparity = abDevation = distance = 0;
    }
};

template<typename CharT, typename Traits>
std::ostream& operator<< (std::basic_ostream<CharT, Traits> &out, const FrameInfo &Info)
{
    out
    << "GeometryCheck=" << Info.GeometryCheck << ", "
    << "BrightnessCheck="   << Info.BrightnessCheck << ", "
    << "energy="   << Info.energy << ", "
    << "ratio=" << Info.ratio << ", "
    << "disparity=" << Info.disparity << ", "
    << "error="  << Info.abDevation << endl;

    return out;
}

double computeDev(const SE3& Ref_Pose, const SE3& Est_Pose);
void saveTrajectory(const string dir, const vector<FrameShell*> &results);
int main(int argc, char** argv)
{
    if(argc != 2)
    {
        cout << "Usage--->: ./test_TrackingTrajectory < PathToConfig >" << endl;
        return -1;
    }
    string PathToConfig = argv[1];

    //------------------------------ConfigParams reader---------------------------------
    ParameterReader confpara(PathToConfig);
    const float normTH = confpara.getData<float>("normTH");
    const float ratioTH = confpara.getData<float>("ratioTH");
    const float disparityTH = confpara.getData<float>("disparityTH");
    const int startIndex = confpara.getData<int>("start");
    const int dataSize = confpara.getData<int>("dataSize");
    bool debugPrint = false, useMI = false;
    const string Print = confpara.getData<string>("debugPrint");
    if(Print == "true") debugPrint = true;
    const string MI = confpara.getData<string>("useMI");
    if(MI == "true") useMI = true;

    cout << "Configuration params: ==================================================================" << endl
         << "startIndex=" << startIndex << " datasize=" << dataSize << endl
         << "useMI=" << useMI << " debugPrint=" << Print << endl
         << "normTH=" << normTH << " ratioTH=" << ratioTH << " disparityTH=" << disparityTH << endl;

    //------------------------------datasets reader---------------------------------
    ParameterReader para;
    FrameReader freader(para);

    //------------------------------frame/keyframe vectors-----------------------
    vector<FrameShell*> allFramesHistory;
    vector<FrameShell*> allKeyFramesHistory;
    vector<Frame*> activeKeyFrames;
    vector<int> indexOfKeyFrames;

    //------------------------------statictical variables-----------------------
    double squaError = 0;
    double meanError = 0;
    double error = 0;

    //------------------------------initialize tracker---------------------------
    Tracker* tracker = new Tracker(width, height, "./Tracker.txt");

    //------------------------------track frames---------------------------------
//    const int totalSize = freader.getSizeOfData();
    const int totalSize = dataSize;
    const int index = startIndex;
    Frame* lastsuccessFrame = new Frame();     // store the last successfully tracked frame
    FrameInfo InfoBackup;  // store the info of the lastsucecessFrame

    SE3 refToNew = SE3(), refToNew_MI = SE3();
    SE3 NewToRef = SE3(), NewToRef_MI = SE3();
    SE3 refToNew_backup = SE3();
    Vector5f MInfoTH = Vector5f::Zero();
    Vector5f EnergyTH = Vector5f::Constant(1e6);
    Vector7d PoseOfFirstFrame = Vector7d::Zero();
    Vector7d PoseOfCurrtFrame = Vector7d::Zero();
    cout << "Start to tracking ************************************************************************" << endl;
    for(int i = index; i < index + totalSize; ++i)
    {
        MInfoTH = Vector5f::Zero();
        EnergyTH = Vector5f::Constant(1e6);

        Frame* frame = freader.getSpecificFrame(i);
        if(i == index)
        {
            PoseOfFirstFrame = freader.getSpecificPose(i);
            if(!(width == frame->width[0] && height == frame->height[0]))
            {
                delete tracker;
                tracker = new Tracker(frame->width[0], frame->height[0], "./Tracker.txt");
            }
        }
        else PoseOfCurrtFrame = freader.getSpecificPose(i);

        FrameShell* shell = new FrameShell();
        shell->timestamp = frame->timestamp;
        shell->id = shell->marginalizedAt = allFramesHistory.size();
        shell->next_id = shell->id + 1;

        frame->shell = shell;
        allFramesHistory.push_back(shell);

        if(tracker->lastRef == 0)              //uninitialized
        {
            tracker->makeTrackPointsbyFrame(frame);        // set the first keyframe
            for(int lvl = 0; lvl < PYR_LEVELS; lvl++)
            {
                cout << "Num of points in level " << lvl << ": " << tracker->NumPnts[lvl];
                cout << endl;
            }

            frame->frameID = allKeyFramesHistory.size();
            allKeyFramesHistory.push_back(frame->shell);
            frame->idx = activeKeyFrames.size();
            activeKeyFrames.push_back(frame);
            indexOfKeyFrames.push_back(i);

            lastsuccessFrame = frame;

            frame->shell->trackingRef = frame->shell;
            cout << "Frame " << i << " >>>>>" << "The 1st keyframe: " << "PoseToWorld=" << frame->shell->FrameToWorld.translation().transpose() << endl;
        }
        else
        {
            frame->shell->trackingRef = tracker->lastRef->shell;                         //set tracking refFrame
            tracker->trackNewFrame(frame, refToNew, PYR_LEVELS-1, Vector5f::Zero());
            NewToRef = refToNew.inverse();
            if(useMI)                                              //further do MI optimization
            {
                {
                    refToNew_backup = refToNew;
                    bool flagMI = tracker->OptimizMI(refToNew_MI, MInfoTH, EnergyTH);
                    if(flagMI)
                    {
                        MInfoTH = tracker->lastMInfo;
                        EnergyTH = tracker->lastEnergy;
                    }
/*
                    bool flagSSD = tracker->OptimizMI(refToNew, MInfoTH, EnergyTH, 2);
                    if(flagSSD) refToNew_MI = refToNew;
                    else if(flagMI) refToNew = refToNew_MI;
                    else {
                        refToNew = refToNew_backup;
                        refToNew_MI = refToNew_backup;
                    }
*/
               }
/*
               {
//                    refToNew_backup = refToNew_MI;
                    refToNew_backup = refToNew;
                    bool flagSSD = tracker->OptimizMI(refToNew, MInfoTH, EnergyTH, 3);
                    if(flagSSD)
                    {
                        MInfoTH = tracker->lastMInfo;
                        EnergyTH = tracker->lastEnergy;
                    }

                    bool flagMI = tracker->OptimizMI(refToNew_MI, MInfoTH, EnergyTH);
                    if(flagMI) refToNew = refToNew_MI;
                    else if(flagSSD) refToNew_MI = refToNew;
                    else{
                        refToNew = refToNew_backup;
                        refToNew_MI = refToNew_backup;
                    }
               }
*/
                NewToRef_MI = refToNew_MI.inverse();
            }

            //------------------information in tracking------------------------------------
            IterationPrintInfo info = tracker->IterationStatus.IteInfo[0].back();
            float energy = info.leftEnergy;
            float numOfGoodRes = (float)info.num_GoodRes;
            float totalSampledPnts = (float)tracker->NumPnts[0];
            float ratio = numOfGoodRes/totalSampledPnts;
            Vector3 flows = tracker->lastFlowIndicators;
            if(!isfinite(energy) || !isfinite(flows(0)) || !isfinite(flows(1)) || !isfinite(flows(2)) || ratio < 0.2)
            {
                delete frame->shell;
                delete frame;
                allFramesHistory.pop_back();              // delete the failed frame pointer && shell pointer

                if(lastsuccessFrame->frameID == -1)       // not keyframe before, needs to add to keyframesHistory
                {
                    lastsuccessFrame->frameID = allKeyFramesHistory.size();
                    allKeyFramesHistory.push_back(lastsuccessFrame->shell);
                    lastsuccessFrame->idx = activeKeyFrames.size();
                    activeKeyFrames.push_back(lastsuccessFrame);
                    indexOfKeyFrames.push_back(lastsuccessFrame->abIDx);     // approximated...

                    tracker->makeTrackPointsbyFrame(lastsuccessFrame);    // switch the last sucessfully-tracked frame to the new keyframe
                    refToNew = SE3(); refToNew_MI = SE3();
/*
                    cout << std::fixed;
                    cout << "Frame " << lastsuccessFrame->abIDx << " >>>>>" << "Swithched to new keyframe: " << setprecision(6)
                                                        << "timeStamp=" << lastsuccessFrame->shell->timestamp << endl;
                    cout.unsetf(ios_base::fixed);
*/
                    cout << "Frame " << lastsuccessFrame->abIDx << " >>>>>" << "Swithched to new keyframe: " << InfoBackup;

                    i--;                               // try to track this frame once with new keyframe
                    continue;
                }

                refToNew = SE3(); refToNew_MI = SE3();
                cout << "Frame " << i << "====>tracking failed: Lost! " << "energy=" << energy << ", numOfGoodres=" << numOfGoodRes << endl;
                continue;
            }
            else
            {
                if(useMI)                                               // MI optimization
                {
                     frame->shell->FrameToTrackingRef = NewToRef_MI;
                     frame->shell->FrameToWorld = frame->shell->trackingRef->FrameToWorld * NewToRef_MI;
   //                frame->shell->FrameToTrackingRef = NewToRef;
   //                 frame->shell->FrameToWorld = frame->shell->trackingRef->FrameToWorld * NewToRef;
                }
                else                                                    // SSD optimization
                {
                    frame->shell->FrameToTrackingRef = NewToRef;
                    frame->shell->FrameToWorld = frame->shell->trackingRef->FrameToWorld * NewToRef;
                }

                Eigen::Quaterniond q0(PoseOfFirstFrame(6), PoseOfFirstFrame(3), PoseOfFirstFrame(4), PoseOfFirstFrame(5));
                Eigen::Quaterniond q1(PoseOfCurrtFrame(6), PoseOfCurrtFrame(3), PoseOfCurrtFrame(4), PoseOfCurrtFrame(5));
                SE3 PoseFirst(q0, PoseOfFirstFrame.head(3));
                SE3 PoseCurrt(q1, PoseOfCurrtFrame.head(3));
                SE3 firstToCurrt = PoseCurrt.inverse() *  PoseFirst;   // relative transformation in grouthtruth

                SE3 CurrtTofirst = firstToCurrt.inverse();
                error = computeDev(CurrtTofirst, frame->shell->FrameToWorld);

            }

            //------------------define keyframe selection criteria-------------------------------------
            // geometry criterion
            Vector3 tvect = Vector3::Zero(), rvect = Vector3::Zero();
            SO3 rotation = SO3();
            if(useMI)
            {
                tvect = NewToRef_MI.translation().cast<float>();
                rotation = SO3(NewToRef_MI.rotationMatrix());
            }
            else
            {
                tvect = NewToRef.translation().cast<float>();
                rotation = SO3(NewToRef.rotationMatrix());
            }
            rvect = rotation.log().cast<float>();
            float Norm = tvect.norm() + 0.1 * rvect.norm();
            bool GeometryCheck = (Norm > normTH);    // translation exceeds 30cm...

            // brightness/visibility criterion
            float disparity = setting_kfGlobalWeight*setting_maxShiftWeightT *  sqrtf((double)flows[0]) / (width+height) +
                              setting_kfGlobalWeight*setting_maxShiftWeightR *  sqrtf((double)flows[1]) / (width+height) +
                              setting_kfGlobalWeight*setting_maxShiftWeightRT * sqrtf((double)flows[2]) / (width+height);

            bool BrightnessCheck = allFramesHistory.size()== 1 || ratio < ratioTH || disparity > disparityTH;

            // relative distance criterion, assuming smooth motion
            double RelativeDistance = (lastsuccessFrame->shell->FrameToWorld.inverse()*frame->shell->FrameToWorld).translation().norm();
            bool RelativeNormCheck = (RelativeDistance >= 8.0*InfoBackup.distance) && (lastsuccessFrame->frameID != 0);

            if(debugPrint)
            {
                cout << "Frame " << i << "---->tracking result: "  << "energy=" << energy << ", norm=" << Norm << ", ratio=" << ratio
                     << ", disparity=" << disparity << ", error=" << error << endl;
            }

            if(GeometryCheck || BrightnessCheck || RelativeNormCheck)             // decide to accept the current frame as a new keyframe
            {
 /*
                cout << "Frame " << i << " >>>>>" << "Add new keyframe: " << "GeometryCheck=" << GeometryCheck << ", BrightnessCheck=" << BrightnessCheck
                                                                  << ", energy=" << energy << ", ratio=" << ratio << ", disparity=" << disparity << ", error=" << error << endl;
*/
                if(lastsuccessFrame->frameID == -1)                               // Add new keyframe
                {
                    cout << "Frame " << lastsuccessFrame->abIDx << " >>>>>" << "Add new keyframe: " << InfoBackup;

                    tracker->makeTrackPointsbyFrame(lastsuccessFrame);
                    refToNew = SE3(); refToNew_MI = SE3();

                    lastsuccessFrame->frameID = allKeyFramesHistory.size();
                    allKeyFramesHistory.push_back(lastsuccessFrame->shell);
                    lastsuccessFrame->idx = activeKeyFrames.size();
                    activeKeyFrames.push_back(lastsuccessFrame);
                    indexOfKeyFrames.push_back(lastsuccessFrame->abIDx);                             // approximated...

                    if( ratio <= 1 )     // 0.45                     // bad tracking quality, try to track again with new keyframe
                    {
                        delete frame->shell;
                        delete frame;
                        allFramesHistory.pop_back();              // delete the bad-quality frame

                        i--;
                        continue;
                    }
                }
                else if( Norm >= 1.1*normTH || ratio <= 0.85*ratioTH )
                {
                    delete frame->shell;
                    delete frame;
                    allFramesHistory.pop_back();
                    refToNew = SE3(); refToNew_MI = SE3();

                    cout << "Frame " << i << "====>tracking failed_2: Lost! " << "energy=" << energy << ", norm=" << Norm << ", ratio=" << ratio << endl;
                    continue;
                }
               /*
                frame->frameID = allKeyFramesHistory.size();
                allKeyFramesHistory.push_back(frame->shell);
                frame->idx = activeKeyFrames.size();
                activeKeyFrames.push_back(frame);
                indexOfKeyFrames.push_back(i);

                tracker->makeTrackPointsbyFrame(frame);         // change the reference frame for tracking
                refToNew = SE3();                               // reset relative transformation
               */
            }
            meanError += error; squaError += error*error;

            double RelativeNorm = (lastsuccessFrame->shell->FrameToWorld.inverse()*frame->shell->FrameToWorld).translation().norm();
            InfoBackup.distance = RelativeNorm;
            InfoBackup.BrightnessCheck = BrightnessCheck;
            InfoBackup.GeometryCheck   = GeometryCheck;
            InfoBackup.energy = energy; InfoBackup.ratio = ratio; InfoBackup.disparity = disparity; InfoBackup.abDevation = error;

            if(lastsuccessFrame->frameID == -1)  delete lastsuccessFrame;   // delete the memory for non-keyframes
            lastsuccessFrame = frame;
        }

    }
    cout << "Tracking done ******************************************************************************" << endl;
    cout << "statistics result: =======================================================================" << endl
         << "Num_of_Datasets  = " << totalSize << endl
         << "Num_of_Frames    = " << allFramesHistory.size() << endl
         << "Num_of_KeyFrames = " << allKeyFramesHistory.size() << endl
         << "Mean_Devation = "    << meanError/allFramesHistory.size() << endl
         << "RMSE = "             << sqrtf(squaError/allFramesHistory.size()) << endl;

    cout << "saving tracjectory...." << endl;
    string dir = "./trajectory";
    if(!useMI)
        dir += "_SSD";
    else if(!tracker->IsWeighted())
        dir += "_NI";
    else dir += "_NI_W";

    dir += ".txt";
    saveTrajectory(dir, allFramesHistory);

    cout << "saving completed!" << endl;
}

double computeDev(const SE3& Ref_Pose, const SE3& Est_Pose)
{
    Vector3 OptCenter_ref = Ref_Pose.translation().cast<float>();
    Vector3 OptCenter_est = Est_Pose.translation().cast<float>();

    return Vector3(OptCenter_ref - OptCenter_est).norm();
}

void saveTrajectory(const string dir, const vector<FrameShell*> &results)
{
    ofstream fout(dir);
    ostringstream os;

    for(FrameShell* fr : results)
    {
        os.str("");

        Eigen::Vector3d tVec = fr->FrameToWorld.translation();
        Eigen::Quaterniond rVec = Eigen::Quaterniond(fr->FrameToWorld.rotationMatrix());

        os << std::fixed << fr->timestamp << " ";
        os << std::setprecision(9) << tVec[0] << " " << tVec[1] << " " << tVec[2] << " "
                               << rVec.x() << " " << rVec.y() << " " << rVec.z() << " " << rVec.w();

        fout << os.str() << "\n";
    }
}
