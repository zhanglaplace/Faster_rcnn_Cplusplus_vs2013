#ifndef _CAFFE_NET_
#define _CAFFE_NET_
#include <caffe/caffe.hpp>
#include <boost/algorithm/string.hpp>
#include <opencv2/opencv.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include "config.h"
#include "register.h"

using namespace cv;
using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::Solver;
using caffe::shared_ptr;
using caffe::string;
using caffe::Timer;
using caffe::vector;
using caffe::ostringstream;


class Caffe_Net{
private:
	Mat src_image, mean_image, net_input_image;
	Size feature_map_size_, net_input_image_size_;
	shared_ptr<Net<float> >rpn_net_, fcnn_net_;
	double im_scale_;

public:
	config conf;

public:
	Caffe_Net(const string rpn_prototxt, const string rpn_model, const string fcnn_prototxt, const string fcnn_model, const string mean_image_file);
	Mat fasterrcnn(Mat src);
	void process(Mat cv_image);
	double prep_im_for_blod_size(int height, int width, int target_size, int max_size);
	vector<aboxes> forward();
	Mat proposal_local_anchor();
	Mat proposal_local_anchor1();
	Mat bbox_tranform_inv(Mat anchors, Mat boxs_delta, string type);
	Mat get_rpn_score(Blob<float>*result, int w, int h);
	void filter_boxes(Mat& pre_box, Mat& score, vector<aboxes>& box);
	void nms(vector<aboxes>input_boxes, double overlap, vector<int>&vPick, int&nPick);
	void boxes_filter(vector<aboxes>&box_, int nPick, vector<aboxes>box, vector<int>vPick);
	std::map<int, vector<Rect> > fcnn_detect(vector<aboxes>&box);
};



#endif