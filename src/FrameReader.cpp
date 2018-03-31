#include <FrameReader.h>

namespace DSLAM
{
    void FrameReader::initia_Datasets()
    {
        dataset_dir = paraReader.getData<string>("data_source");
        string associate_file = dataset_dir + "associate_with_groundtruth.txt";

        ifstream fin(associate_file.c_str());
        if(!fin)
        {
            cout << "cannot find associate.txt !" << endl;
            return;
        }

        while( !fin.eof() )
        {
            string rgbTime, rgbFile, depthTime, depthFile;
            string truTime, t0, t1, t2, q0, q1, q2, q3;
            fin >> rgbTime >> rgbFile >> depthTime >> depthFile >>
                    truTime >> t0 >> t1 >> t2 >>
                     q0 >> q1 >> q2 >> q3;

            if(!fin.good())
            {
                break;
            }

            double t11 = boost::lexical_cast<double>(rgbTime);
            double t22 = boost::lexical_cast<double>(depthTime);
    //        double timeStamp = 0.5 * (t11 + t22);
            double timeStamp = t11;

            rgbfiles.push_back(rgbFile);
            depthfiles.push_back(depthFile);
            TimeStamps.push_back(timeStamp);

            Vector7d ref = Vector7d::Zero();
            ref[0] = boost::lexical_cast<double>(t0);
            ref[1] = boost::lexical_cast<double>(t1);
            ref[2] = boost::lexical_cast<double>(t2);
            ref[3] = boost::lexical_cast<double>(q0);
            ref[4] = boost::lexical_cast<double>(q1);
            ref[5] = boost::lexical_cast<double>(q2);
            ref[6] = boost::lexical_cast<double>(q3);

            truth.push_back(ref);
        }

        cout << "found " << rgbfiles.size() << " images" << endl;
        camera = paraReader.getCamera();
        start_idx = paraReader.getData<int>("start_index");
        current_idx = start_idx;
    }

    Frame* FrameReader::getNextFrame()
    {
        if(current_idx < start_idx || current_idx >= rgbfiles.size())
            return NULL;

        cv::Mat rgb = cv::imread(dataset_dir + rgbfiles[current_idx], CV_LOAD_IMAGE_GRAYSCALE);
        cv::Mat depth = cv::imread(dataset_dir + depthfiles[current_idx], -1);
        double TimeStamp = TimeStamps[current_idx];

        if(rgb.empty() || depth.empty())
        {
            return NULL;
        }

        Frame* frame = new Frame(rgb, depth, TimeStamp, this->camera);
        frame->abIDx = current_idx;

        current_idx++;
        return frame;
    }

    Frame* FrameReader::getSpecificFrame(const int idx)
    {
        current_idx = idx;
        return getNextFrame();
    }
}
