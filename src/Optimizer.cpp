#include <Optimizer.h>

#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/types/sba/types_sba.h>

using namespace g2o;

namespace DSLAM
{
struct Measurement
{
    Vector3f data;
    float grayvalue;

    Measurement(Eigen::Vector3f p, float g) : data(p), grayvalue(g) {}
    Measurement() {data = Vector3f::Zero(), grayvalue = 0;}
};

class EdgeSE3ProjectDirect: public BaseBinaryEdge<1, double, VertexSE3Expmap, VertexSE3Expmap>
{

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeSE3ProjectDirect() : _fx(0.0), _cx(0.0), _fy(0.0), _cy(0.0) {}
    EdgeSE3ProjectDirect( Vector3f data, float fx, float fy, float cx, float cy, const cv::Mat &img) :
        _data(data), _fx(fx), _fy(fy), _cx(cx), _cy(cy), _img(img)
    {}

    virtual void computeError()
    {
        const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[0]);
        const VertexSE3Expmap* v2 = static_cast<const VertexSE3Expmap*>(_vertices[1]);
        SE3Quat transform = v2->estimate() * v1->estimate().inverse();

        Mat3x3 K, Ki;
        K << _fx, 0, _cx, 0, _fy, _cy, 0, 0, 1;
        Ki = K.inverse();
        Vector3f Pt1 = _data[2] * ( Ki * Vector3(_data[0], _data[1], 1.0f) );
        Vector3d Pt2 = transform.map(Pt1.cast<double>());
        float Ku = _fx * Pt2[0]/Pt2[2] + _cx;
        float Kv = _fy * Pt2[1]/Pt2[2] + _cy;

        if(!(Ku > 3 && Kv > 3 && Ku < _img.cols-3 && Kv < _img.rows-3))
        {
            _error(0,0) = 0.0;     // ??
            this->setLevel(1);
        }
        else
        {
            float hitData = bilinearInterpolation( _img, Ku, Kv);
            _error(0,0) = hitData - _measurement;
        }

    }

    virtual void linearizeOplus()
    {
        if(this->level() == 1)
        {
            _jacobianOplusXi = Eigen::Matrix<double, 1, 6>::Zero();
            _jacobianOplusXj = Eigen::Matrix<double, 1, 6>::Zero();

            return;
        }

        const VertexSE3Expmap* vi = static_cast<const VertexSE3Expmap*>(_vertices[0]);
        const VertexSE3Expmap* vj = static_cast<const VertexSE3Expmap*>(_vertices[1]);
        SE3Quat transform = vj->estimate() * vi->estimate().inverse();

        Mat3x3 K, Ki;
        K << _fx, 0, _cx, 0, _fy, _cy, 0, 0, 1;
        Ki = K.inverse();
        Vector3f Pt1 = _data[2] * ( Ki * Vector3(_data[0], _data[1], 1.0f) );
        Vector3d Pt2 = transform.map(Pt1.cast<double>());
        float Ku = _fx * Pt2[0]/Pt2[2] + _cx;
        float Kv = _fy * Pt2[1]/Pt2[2] + _cy;

        float u = Pt2[0] / Pt2[2];
        float v = Pt2[1] / Pt2[2];
        float idepth = 1.0 / Pt2[2];

        Eigen::Matrix<double, 2, 6> jacobian_uv_ksai;
        jacobian_uv_ksai(0,0) = -u * v * _fx;
        jacobian_uv_ksai(0,1) = (1 + u*u) * _fx;
        jacobian_uv_ksai(0,2) = -v * _fx;
        jacobian_uv_ksai(0,3) = idepth * _fx;
        jacobian_uv_ksai(0,4) = 0;
        jacobian_uv_ksai(0,5) = -idepth * u * _fx;

        jacobian_uv_ksai(1,0) = -(1 + v*v) * _fy;
        jacobian_uv_ksai(1,1) = u * v * _fy;
        jacobian_uv_ksai(1,2) = u * _fy;
        jacobian_uv_ksai(1,3) = 0;
        jacobian_uv_ksai(1,4) = idepth * _fy;
        jacobian_uv_ksai(1,5) = -idepth * v * _fy;

        Eigen::Matrix<double, 1, 2> jacobian_gray_uv;
        jacobian_gray_uv(0,0) = 0.5 * (bilinearInterpolation(_img, Ku+1, Kv) - bilinearInterpolation(_img, Ku-1, Kv));
        jacobian_gray_uv(0,1) = 0.5 * (bilinearInterpolation(_img, Ku, Kv+1) - bilinearInterpolation(_img, Ku, Kv-1));

        _jacobianOplusXj = jacobian_gray_uv * jacobian_uv_ksai;
        _jacobianOplusXi = -1 * _jacobianOplusXj * transform.adj();
    }

    virtual bool read(std::istream& in) {}
    virtual bool write(std::ostream& out) const {}

public:
    Vector3f _data;               // pixel coordinates: <u, v> And pixel depth <d>
    float _cx, _cy, _fx, _fy;     // camera intrinsics
    cv::Mat _img;                 // image of target frame
};

class EdgeSE3ProjectDirectUnary: public BaseUnaryEdge<1, double, VertexSE3Expmap>
{

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeSE3ProjectDirectUnary() : _fx(0.0), _cx(0.0), _fy(0.0), _cy(0.0) {}
    EdgeSE3ProjectDirectUnary( Vector3f data, float fx, float fy, float cx, float cy, const cv::Mat &img ) :
        _data(data), _fx(fx), _fy(fy), _cx(cx), _cy(cy), _img(img)
    {}

    virtual void computeError()
    {
        const VertexSE3Expmap* v1 = static_cast<const VertexSE3Expmap*>(_vertices[0]);
        SE3Quat transform = v1->estimate();

        Mat3x3 K, Ki;
        K << _fx, 0, _cx, 0, _fy, _cy, 0, 0, 1;
        Ki = K.inverse();
        Vector3f Pt1 = _data[2] * ( Ki * Vector3(_data[0], _data[1], 1.0f) );
        Vector3d Pt2 = transform.map(Pt1.cast<double>());
        float Ku = _fx * Pt2[0]/Pt2[2] + _cx;
        float Kv = _fy * Pt2[1]/Pt2[2] + _cy;

        if(!(Ku > 3 && Kv > 3 && Ku < _img.cols-3 && Kv < _img.rows-3))
        {
            _error(0,0) = 0.0;     // ??
            this->setLevel(1);
        }
        else
        {
            float hitData = bilinearInterpolation( _img, Ku, Kv);
            _error(0,0) = hitData - _measurement;
        }
    }

    virtual void linearizeOplus()
    {
        if(this->level() == 1)
        {
            _jacobianOplusXi = Eigen::Matrix<double, 1, 6>::Zero();
            return;
        }

        const VertexSE3Expmap* vi = static_cast<const VertexSE3Expmap*>(_vertices[0]);
        SE3Quat transform = vi->estimate();

        Mat3x3 K, Ki;
        K << _fx, 0, _cx, 0, _fy, _cy, 0, 0, 1;
        Ki = K.inverse();
        Vector3f Pt1 = _data[2] * ( Ki * Vector3(_data[0], _data[1], 1.0f) );
        Vector3d Pt2 = transform.map(Pt1.cast<double>());
        float Ku = _fx * Pt2[0]/Pt2[2] + _cx;
        float Kv = _fy * Pt2[1]/Pt2[2] + _cy;

        float u = Pt2[0] / Pt2[2];
        float v = Pt2[1] / Pt2[2];
        float idepth = 1.0 / Pt2[2];

        Eigen::Matrix<double, 2, 6> jacobian_uv_ksai;
        jacobian_uv_ksai(0,0) = -u * v * _fx;
        jacobian_uv_ksai(0,1) = (1 + u*u) * _fx;
        jacobian_uv_ksai(0,2) = -v * _fx;
        jacobian_uv_ksai(0,3) = idepth * _fx;
        jacobian_uv_ksai(0,4) = 0;
        jacobian_uv_ksai(0,5) = -idepth * u * _fx;

        jacobian_uv_ksai(1,0) = -(1 + v*v) * _fy;
        jacobian_uv_ksai(1,1) = u * v * _fy;
        jacobian_uv_ksai(1,2) = u * _fy;
        jacobian_uv_ksai(1,3) = 0;
        jacobian_uv_ksai(1,4) = idepth * _fy;
        jacobian_uv_ksai(1,5) = -idepth * v * _fy;

        Eigen::Matrix<double, 1, 2> jacobian_gray_uv;
        jacobian_gray_uv(0,0) = 0.5 * (bilinearInterpolation(_img, Ku+1, Kv) - bilinearInterpolation(_img, Ku-1, Kv));
        jacobian_gray_uv(0,1) = 0.5 * (bilinearInterpolation(_img, Ku, Kv+1) - bilinearInterpolation(_img, Ku, Kv-1));

        _jacobianOplusXi = jacobian_gray_uv * jacobian_uv_ksai;
    }
    virtual bool read(std::istream& in) {}
    virtual bool write(std::ostream& out) const {}

public:
    Vector3f _data;               // pixel coordinates: <u, v> And pixel depth <d>
    float _cx, _cy, _fx, _fy;     // camera intrinsics
    cv::Mat _img;                 // image of target frame
};

//--------------------------------------------optimize global poses of 2 frames------------------------------
//--------------------------------------------use binary-direct edge-----------------------------------------
void Optimizer::DirectOptimization(Frame *pKF1, Frame *pKF2, int nIterations, const bool bRobust)
{
    int w = pKF1->intensity[0].cols;
    int h = pKF1->intensity[0].rows;

    CameraIntrinsic* cam = pKF1->intrinsic;
    Mat3x3 K = cam->ConvertToMatrix();

    float* statusMap = new float[w*h];
    float density = 0.03;

    PixelSelector selector(w,h);                                             // initialize point selector

    int npts1 = selector.makeMaps(pKF1, statusMap, w*h*density, 1, true, 2);
    vector<Measurement> Measures_1;
    const float* depth = pKF1->depth[0].ptr<float>();
    const float* color = pKF1->intensity[0].ptr<float>();
    for(int y = 3; y < h - 3; y++)
        for(int x = 3; x < w - 3; x++)
        {
            int idx = x + y*w;
            if(statusMap[idx] != 0)
            {
                if(!std::isfinite(color[idx]) || depth[idx] < 0.4 || depth[idx] > 10.0)    // 0.4~10
                   continue;

                float u = x + 0.1;
                float v = y + 0.1;

                Measurement measure(Vector3f(u,v,depth[idx]), color[idx]);
                Measures_1.push_back(measure);
            }
        }

    memset(statusMap, sizeof(float)*w*h, 0);
    int npts2 = selector.makeMaps(pKF2, statusMap, w*h*density, 1, true, 2);
    vector<Measurement> Measures_2;
    const float* depth2 = pKF2->depth[0].ptr<float>();
    const float* color2 = pKF2->intensity[0].ptr<float>();
    for(int y = 3; y < h - 3; y++)
        for(int x = 3; x < w - 3; x++)
        {
            int idx = x + y*w;
            if(statusMap[idx] != 0)
            {
                if(!std::isfinite(color2[idx]) || depth2[idx] < 0.4 || depth2[idx] > 10.0)    // 0.4~10
                   continue;

                float u = x + 0.1;
                float v = y + 0.1;

                Measurement measure(Vector3f(u,v,depth2[idx]), color2[idx]);
                Measures_2.push_back(measure);
            }
        }

    typedef g2o::BlockSolver<g2o::BlockSolverTraits<12,1>> DirectBlock;
    DirectBlock::LinearSolverType* LinearSolver = new g2o::LinearSolverDense<DirectBlock::PoseMatrixType>();
    DirectBlock* solver_ptr = new DirectBlock(LinearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    g2o::VertexSE3Expmap* pose1 = new g2o::VertexSE3Expmap();
    SE3 w2f_1 = pKF1->shell->FrameToWorld.inverse();
    pose1->setEstimate(g2o::SE3Quat(w2f_1.rotationMatrix(), w2f_1.translation()));
    pose1->setId(0);
    if(pKF1->shell->id == 0) pose1->setFixed(true);
    optimizer.addVertex(pose1);

    g2o::VertexSE3Expmap* pose2 = new g2o::VertexSE3Expmap();
    SE3 w2f_2 = pKF2->shell->FrameToWorld.inverse();
    pose2->setEstimate(g2o::SE3Quat(w2f_2.rotationMatrix(), w2f_2.translation()));
    pose2->setId(1);
    optimizer.addVertex(pose2);

    float thHuber2D = 10;
    int id = 1;

    for(Measurement m : Measures_1)
    {
        EdgeSE3ProjectDirect* edge = new EdgeSE3ProjectDirect(m.data, K(0,0), K(1,1), K(0,2), K(1,2), pKF2->intensity[0]);

        edge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        edge->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(1)));
        edge->setMeasurement(m.grayvalue);
        edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
        edge->setId(id++);

        if(bRobust)
        {
            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            edge->setRobustKernel(rk);
            rk->setDelta(thHuber2D);
        }

        optimizer.addEdge(edge);
    }

    for(Measurement m : Measures_2)
    {
        EdgeSE3ProjectDirect* edge = new EdgeSE3ProjectDirect(m.data, K(0,0), K(1,1), K(0,2), K(1,2), pKF1->intensity[0]);

        edge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(1)));
        edge->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        edge->setMeasurement(m.grayvalue);
        edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
        edge->setId(id++);

        if(bRobust)
        {
            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            edge->setRobustKernel(rk);
            rk->setDelta(thHuber2D);
 //           rk->setDelta(0);
        }

        optimizer.addEdge(edge);
    }

//    cout << "edges in graph: " << optimizer.edges().size() << endl;
    const float chi2TH[4] = {20, 18, 15, 15};
    const int its[4] = {100, 50, 50, 50};

    int nBad = 0;
    for(size_t it = 0; it < 4; it++)
    {
//        pose1->setEstimate(g2o::SE3Quat(w2f_1.rotationMatrix(), w2f_1.translation()));
//        pose2->setEstimate(g2o::SE3Quat(w2f_2.rotationMatrix(), w2f_2.translation()));
        optimizer.initializeOptimization(0);
        optimizer.optimize(its[it]);

        nBad = 0;
        for(g2o::HyperGraph::Edge* e : optimizer.edges())
        {
            EdgeSE3ProjectDirect* edge = static_cast<EdgeSE3ProjectDirect*>(e);

            if(edge->level() == 1)
            {
                nBad++;
                continue;
            }

            const double err = edge->errorData()[0];
            if(err > chi2TH[it])
            {
                edge->setLevel(1);
                nBad++;
            }

        }
        cout << nBad << "  ";
    }
    cout << endl;
/*
    optimizer.initializeOptimization(0);
    optimizer.optimize(200);
*/
    g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
    g2o::SE3Quat SE3quat = vSE3->estimate();
    w2f_1 = SE3(SE3quat.rotation(), SE3quat.translation());
    pKF1->shell->FrameToWorld = w2f_1.inverse();

    vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(1));
    SE3quat = vSE3->estimate();
    w2f_2 = SE3(SE3quat.rotation(), SE3quat.translation());
    pKF2->shell->FrameToWorld = w2f_2.inverse();
}

//---------------------------------------optimize relative tranformation between 2 frames------------------------------------
//---------------------------------------use unary-direct edge---------------------------------------------------------------
void Optimizer::DirectUnaryOptimization(const Frame *pKF1, const Frame *pKF2, const bool bRobust)
{
    int w = pKF1->intensity[0].cols;
    int h = pKF1->intensity[0].rows;

    CameraIntrinsic* cam = pKF1->intrinsic;
    Mat3x3 K = cam->ConvertToMatrix();

    float* statusMap = new float[w*h];
    float density = 0.03;

    PixelSelector selector(w,h);                                             // initialize point selector

    int npts1 = selector.makeMaps(pKF1, statusMap, w*h*density, 1, true, 2);
    vector<Measurement> Measures_1;
    const float* depth = pKF1->depth[0].ptr<float>();
    const float* color = pKF1->intensity[0].ptr<float>();
    for(int y = 3; y < h - 3; y++)
        for(int x = 3; x < w - 3; x++)
        {
            int idx = x + y*w;
            if(statusMap[idx] != 0)
            {
                if(!std::isfinite(color[idx]) || depth[idx] < 0.4 || depth[idx] > 10.0)    // 0.4~10
                   continue;

                float u = x + 0.1;
                float v = y + 0.1;

                Measurement measure(Vector3f(u,v,depth[idx]), color[idx]);
                Measures_1.push_back(measure);
            }
        }

    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,1>> DirectBlock;
    DirectBlock::LinearSolverType* LinearSolver = new g2o::LinearSolverDense<DirectBlock::PoseMatrixType>();
    DirectBlock* solver_ptr = new DirectBlock(LinearSolver);
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(false);

    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
    SE3 transforms = pKF2->shell->FrameToWorld.inverse() * pKF1->shell->FrameToWorld;
    pose->setEstimate(g2o::SE3Quat(transforms.rotationMatrix(), transforms.translation()));
    pose->setId(0);
    if(pKF2->shell->id == 0) pose->setFixed(true);
    optimizer.addVertex(pose);

    float thHuber2D = 10;
    int id = 1;
    for(Measurement m : Measures_1)
    {
        EdgeSE3ProjectDirectUnary* edge = new EdgeSE3ProjectDirectUnary(m.data, K(0,0), K(1,1), K(0,2), K(1,2), pKF2->intensity[0]);

        edge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        edge->setMeasurement(m.grayvalue);
        edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
        edge->setId(id++);

        if(bRobust)
        {
            g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
            edge->setRobustKernel(rk);
            rk->setDelta(thHuber2D);
        }

        optimizer.addEdge(edge);
    }

    optimizer.initializeOptimization(0);
    optimizer.optimize(50);

    g2o::VertexSE3Expmap* vSE3 = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
    g2o::SE3Quat SE3quat = vSE3->estimate().inverse();
    pKF2->shell->FrameToTrackingRef = SE3(SE3quat.rotation(), SE3quat.translation());
    pKF2->shell->FrameToWorld = pKF1->shell->FrameToWorld * pKF2->shell->FrameToTrackingRef;
}

}

