#ifndef PARAMETER_READER_H_
#define PARAMETER_READER_H_

#include <common.h>
#include <CameraIntrinsic.h>

namespace DSLAM
{

class ParameterReader
{

public:

    ParameterReader(const string& filename = "./Parameters.txt")
    {
        ifstream fin(filename.c_str());
        if(!fin)
        {
            fin.open("."+filename);
            if(!fin)
            {
                cout << "cannot find specific file: " << filename << endl;
                return;
            }
        }


        // read information from this parameters file
        while(!fin.eof())
        {
            string str;
            getline(fin, str);

            if(str[0] == '#')    // "#" means this line is only a comment!
                continue;

            int pos = str.find('#');   // The content after "#" is a comment
            if(pos != -1)
            {
                str = str.substr(0, pos);
            }

            pos = str.find('=');    // find "=" in this line
            if(pos == -1)
                continue;

            string key = str.substr(0, pos);    // store information in a map
            string value = str.substr(pos+1, str.length());
            data[key] = value;

            if(!fin.good())
                break;
        }

    }


    // get data from the map structure
    // template function, cause not sure which datatype the string wouble be transferd to
    template<class T>
    T getData(const string& key) const
    {
        auto iter = data.find(key);
        if(iter == data.end())
        {
            cout << "Parameter name: " << key << " not found !" << endl;
            return boost::lexical_cast<T>( "" );
        }

        return boost::lexical_cast<T>( iter->second );
    }

    CameraIntrinsic getCamera() const
    {
        Eigen::Vector4f intrinsic;
        Vector5f distortion;

        intrinsic[0] = this->getData<float>("camera.fx");
        intrinsic[1] = this->getData<float>("camera.fy");
        intrinsic[2] = this->getData<float>("camera.cx");
        intrinsic[3] = this->getData<float>("camera.cy");

        distortion[0] = this->getData<float>("camera.d0");
        distortion[1] = this->getData<float>("camera.d1");
        distortion[2] = this->getData<float>("camera.d2");
        distortion[3] = this->getData<float>("camera.d3");
        distortion[4] = this->getData<float>("camera.d4");

        return CameraIntrinsic(intrinsic, distortion);
    }

protected:
    map<string, string> data;
};

}

#endif
