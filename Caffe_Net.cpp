#include "Caffe_Net.h"
#include <fstream>

cv::Scalar colortable[20] = { cv::Scalar(0, 0, 0), cv::Scalar(0, 0, 125),
cv::Scalar(0, 125, 125), cv::Scalar(125, 125, 125), cv::Scalar(125, 0, 0), cv::Scalar(125, 125, 0), cv::Scalar(0, 125, 0), cv::Scalar(125, 0, 125),
cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 255), cv::Scalar(255, 255, 255), cv::Scalar(255, 0, 0), cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 0),
cv::Scalar(255, 0, 255), cv::Scalar(0, 255, 100), cv::Scalar(0, 0, 100),
cv::Scalar(255, 0, 100), cv::Scalar(255, 255, 100), cv::Scalar(100, 100, 100) };
string classname[20] = { "aeroplane", "bike", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
"diningtable", "dog", "horse", "motobike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor" };



/*************************************************
// Method: Caffe_Net
// Description: Constructor function .init faster-rcnn Net
// Author: zhanglaplace
// Date: 2017/02/25
// Returns: 
// Parameter: string，sting，sting，sting，sting
// History:
*************************************************/
Caffe_Net::Caffe_Net(const string rpn_prototxt, const string rpn_model, const string fcnn_prototxt, const string fcnn_model, const string mean_image_file){
	Caffe::SetDevice(0);
	Caffe::set_mode(Caffe::CPU);
	rpn_net_ = shared_ptr<Net <float> >(new Net<float>(rpn_prototxt, caffe::TEST));
	rpn_net_->CopyTrainedLayersFrom(rpn_model);
	fcnn_net_ = shared_ptr<Net <float> >(new Net<float>(fcnn_prototxt, caffe::TEST));
	fcnn_net_->CopyTrainedLayersFrom(fcnn_model);
	CHECK_EQ(rpn_net_->num_inputs(), 1) << " Network should have exactly one input.";
	CHECK_EQ(rpn_net_->num_outputs(), 2) << "Network should have exactly two inputs.";
	Blob<float>* input_layer = rpn_net_->input_blobs()[0];
	CHECK_EQ(input_layer->channels(), 3) << "Input layer should have 3 channels.";
	mean_image = imread(mean_image_file);//bgr
}

/*************************************************
// Method: fasterrcnn  
// Description: fasterrcnn interface function
// Author: zhanglaplace 
// Date: 2017/02/25
// Returns: Mat
// Parameter: Mat src
// History:
************************************************/
Mat  Caffe_Net::fasterrcnn(Mat src){
	src_image = src.clone();
	process(src);
	vector<aboxes>abox = forward();
	std::map<int, vector<aboxes> >result = fcnn_detect(abox);
	return src_image;
}



/*************************************************
// Method: fcnn_detect
// Description: fcnn_net and test img
// Author: zhanglaplace
// Date: 2017/02/28
// Returns: fg index and location
// Parameter: 300 proposal rect and score
// History:2017/02/26
*************************************************/
std::map<int, vector<aboxes> > Caffe_Net::fcnn_detect(vector<aboxes>&box){
	float scales = im_scale_;
	Mat anchors(box.size(), 4, CV_32FC1);
	fcnn_net_->blob_by_name("data")->CopyFrom(*rpn_net_->blob_by_name("conv5").get(), false, true);
	Blob<float>*input_layer = fcnn_net_->input_blobs()[1];//[1]==rois ,[0]==data
	int num_rois = box.size();
	input_layer->Reshape(num_rois, 5, 1, 1);//score rect;
	float* input_data = input_layer->mutable_cpu_data();
	int num = 0;
	vector<aboxes>::iterator iter;
	for (iter = box.begin(); iter != box.end(); iter++){

		//input image rect
		anchors.at<float>(num, 0) = iter->x1;
		anchors.at<float>(num, 1) = iter->y1;
		anchors.at<float>(num, 2) = iter->x2;
		anchors.at<float>(num, 3) = iter->y2;
		iter->score = 0;


		//map to net_input_size
		iter->x1 = (iter->x1 - 1) * scales;
		iter->y1 = (iter->y1 - 1) * scales;
		iter->x2 = (iter->x2 - 1) * scales;
		iter->y2 = (iter->y2 - 1) * scales;
		input_data[num * 5 + 0] = iter->score;
		input_data[num * 5 + 1] = iter->x1;
		input_data[num * 5 + 2] = iter->y1;
		input_data[num * 5 + 3] = iter->x2;
		input_data[num * 5 + 4] = iter->y2;
		num++;
	}
	const vector<Blob<float>*>& result = fcnn_net_->Forward();
	Blob<float>* box_deltas = result[0];//reg 84
	Blob<float>* scores = result[1];//score 21  300*21*1*1
	Mat box_delta(box_deltas->num(), box_deltas->channels(), CV_32FC1);//300*84*1*1
	//std::cout << box_deltas->num() << " " << box_deltas->channels() << " " << box_deltas->height() << " " << box_deltas->width() << std::endl;
	for (int i = 0; i < box_deltas->num(); i++)
	{

		for (int j = 0; j < box_deltas->channels(); j++)
		{
			box_delta.at<float>(i, j) = box_deltas->data_at(i, j, 0, 0);
		}
	}
	Mat output_score(scores->num(), scores->channels() - 1, CV_32FC1);
	for (int i = 0; i < scores->num(); i++){
		for (int j = 1; j < scores->channels(); j++){
			output_score.at<float>(i, j - 1) = scores->data_at(i, j, 0, 0);
		}
	}
	Mat pred_boxes = bbox_tranform_inv(anchors, box_delta, "rcnn");

	std::map<int, vector<aboxes> > classer;
	////////20为类别数///////
	for (int i = 0; i < 20; i++)
	{
		vector<aboxes> final_box;
		vector<aboxes> output_box;
		for (int j = 0; j < pred_boxes.rows; j++)
		{
			aboxes tmp;
			tmp.x1 = pred_boxes.at<float>(j, i * 4);
			tmp.y1 = pred_boxes.at<float>(j, i * 4 + 1);
			tmp.x2 = pred_boxes.at<float>(j, i * 4 + 2);
			tmp.y2 = pred_boxes.at<float>(j, i * 4 + 3);
			tmp.score = output_score.at<float>(j, i);
			output_box.push_back(tmp);
		}
		vector<int> vPick(box.size());
		int nPick;
		nms(output_box, conf.num_overlap_thres_fcnn, vPick, nPick);
		for (int i = 0; i < nPick; i++)
		{
			if (output_box[vPick[i]].score > conf.test_fcnn_thres_score){
				final_box.push_back(output_box[vPick[i]]);
			}
		}
		classer.insert(std::pair<int, vector<aboxes>>(i, final_box));
	}
	char strTemp[100];
	for (int i = 0; i < 20; i++)
	{
		if (!classer[i].empty())
		{
			for (auto ite = classer[i].begin(); ite != classer[i].end(); ite++)
			{
				aboxes test = *ite;
				rectangle(src_image, Rect(test.x1, test.y1, test.x2 - test.x1 + 1, test.y2 - test.y1 + 1), colortable[i], 2 + test.score * 2, 4, 0);
				sprintf_s(strTemp, "%s:%.3f", classname[i], test.score);
				putText(src_image, strTemp, Point(test.x1 + 2, min((int)test.y1 + 2, src_image.rows - 1)), FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(0, 0, 0)/*colortable[i]*/, 1, 2);
			}
		}
	}
	return classer;
}

/*************************************************
// Method: aboxcmp
// Description:
// Author: zhanglaplace
// Date: 2017/02/25
// Returns: 
// Parameter: 
// History:
*************************************************/
bool aboxcomp(aboxes&b1, aboxes&b2)
{
	return b1.score > b2.score;
}


/*************************************************
// Method: forward
// Description: rpn_net forward function . 
// Author: zhanglaplace
// Date: 2017/02/25
// Returns: vector<aboxes>  300*4
// Parameter: 
// History:
*************************************************/
vector<aboxes> Caffe_Net::forward(){
	const vector<Blob<float>*>& result = rpn_net_->Forward();
	Blob<float>*result0 = result[0];//reg
	Blob<float>*result1 = result[1];//bg or fg
	int reg_num = result0->num()*result0->channels()*result0->width()*result0->height() / 4;
	Mat box_deltas(reg_num, 4, CV_32FC1);
	float* p = result0->mutable_cpu_data();
	int num = 0;
	for (int i = 0; i < result0->width(); i++){
		for (int j = 0; j < result0->height(); j++){
			for (int k = 0; k < result0->channels() / 4; k++){
				for (int t = 0; t < 4; t++){
					box_deltas.at<float>(num, t) = result0->data_at(0, k * 4 + t, j, i);
				}
				num++;
			}
		}
	}

	feature_map_size_ = Size(result0->width(), result0->height());
	Mat anchors = proposal_local_anchor();
	//Mat anchors1 = proposal_local_anchor1();//dengjia 
	Mat pre_box = bbox_tranform_inv(anchors, box_deltas, "rpn");
	Mat score = get_rpn_score(result1, result0->width(), result0->height());
	vector<aboxes>boxs;
	filter_boxes(pre_box, score, boxs);
	std::sort(boxs.begin(), boxs.end(), aboxcomp);
	vector<int>vPick(boxs.size());
	int nPick = 0;
	nms(boxs, conf.nms_overlap_thres_proposal, vPick, nPick);
	vector<aboxes>boxs_;
	boxes_filter(boxs_, nPick, boxs, vPick);
	return boxs_;
}

/*************************************************
// Method: boxes_filter
// Description: select the 300(n) candidates box with the highest score 
// Author: zhanglaplace
// Date: 2017/02/26
// Returns:  filter boxes
// Parameter: 
// History:
*************************************************/
void Caffe_Net::boxes_filter(vector<aboxes>&box_, int nPick, vector<aboxes>box, vector<int>vPick){
	int n = min(nPick, conf.after_nms_topN);
	for (int i = 0; i < n;i++){
		box_.push_back(box[vPick[i]]);
	}
}


/*************************************************
// Method: nms
// Description: non maximum suppression
// Author: zhanglaplace
// Date: 2017/02/26
// Returns: void 
// Parameter: input_box,Iouratio,Index of input_box,count 
// History:
*************************************************/
void Caffe_Net::nms(vector<aboxes>input_boxes, double overlap, vector<int>&vPick, int&nPick){
	int nSample = min(int(input_boxes.size()), conf.per_nms_topN);
	vector<double> vArea(nSample);
	for (int i = 0; i < nSample; ++i)
	{
		vArea[i] = double(input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
			* (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
	}
	std::multimap<double, int> scores;
	for (int i = 0; i < nSample; ++i)
		scores.insert(std::pair<double, int>(input_boxes.at(i).score, i));
	nPick = 0;
	do
	{
		int last = scores.rbegin()->second;
		vPick[nPick] = last;
		nPick += 1;

		for (std::multimap<double, int>::iterator it = scores.begin(); it != scores.end();)
		{
			int it_idx = it->second;
			double xx1 = max(input_boxes.at(last).x1, input_boxes.at(it_idx).x1);
			double yy1 = max(input_boxes.at(last).y1, input_boxes.at(it_idx).y1);
			double xx2 = min(input_boxes.at(last).x2, input_boxes.at(it_idx).x2);
			double yy2 = min(input_boxes.at(last).y2, input_boxes.at(it_idx).y2);

			double w = max(double(0.0), xx2 - xx1 + 1), h = max(double(0.0), yy2 - yy1 + 1);

			double ov = w*h / (vArea[last] + vArea[it_idx] - w*h);

			if (ov > overlap)
			{
				it = scores.erase(it);
			}
			else
			{
				it++;
			}
		}
	} while (scores.size() != 0);
}


/*************************************************
// Method: filter_boxes
// Description: filter_boxes
// Author: zhanglaplace
// Date: 2017/02/26
// Returns: box
// Parameter: pre_box , score 
// History:
*************************************************/
void Caffe_Net::filter_boxes(Mat& pre_box, Mat& score, vector<aboxes>& box){
	box.clear();
	for (int i = 0; i < pre_box.rows; i++){
		int width = pre_box.at<float>(i, 2) - pre_box.at<float>(i, 0) + 1;
		int height = pre_box.at<float>(i, 3) - pre_box.at<float>(i, 1) + 1;
		if (width < conf.test_min_box_size || height < conf.test_min_box_size)
		{
			pre_box.at<float>(i, 0) = 0;
			pre_box.at<float>(i, 1) = 0;
			pre_box.at<float>(i, 2) = 0;
			pre_box.at<float>(i, 3) = 0;
			score.at<float>(i, 0) = 0;
		}
		aboxes tmp;
		tmp.x1 = pre_box.at<float>(i, 0);
		tmp.y1 = pre_box.at<float>(i, 1);
		tmp.x2 = pre_box.at<float>(i, 2);
		tmp.y2 = pre_box.at<float>(i, 3);
		tmp.score = score.at<float>(i, 0);
		box.push_back(tmp);
	}
}


/*************************************************
// Method: get_rpn_score
// Description: get rpn_net output score
// Author: zhanglaplace
// Date: 2017/02/26
// Returns: Mat score	
// Parameter: Blob output score,w,h;
// History:
*************************************************/
Mat Caffe_Net::get_rpn_score(Blob<float>*result, int w, int h){
	int channels = result->width()*result->height() / (w*h);
	Mat score(result->width()*result->height()*channels, 1, CV_32FC1);
	int num = 0;
	for (int i = 0; i < w; i++){
		for (int j = 0; j < h; j++){
			for (int k = 0; k < channels; k++){
				score.at<float>(num++, 0) = result->data_at(0, 1, h*k + j, i);
			}
		}
	}
	return score;
}


/*************************************************
// Method: bbox_tranform_inv
// Description: according the anchors and boxs_delta to get the real rect in the image;
// Author: zhanglaplace
// Date: 2017/02/25
// Returns: Mat real rect 
// Parameter: anchors  bbox_delta and type
// History:
*************************************************/
Mat Caffe_Net::bbox_tranform_inv(Mat anchors, Mat boxs_delta, string type){
	float width, height, ctr_x, ctr_y, dx, dy, dw, dh, pred_ctr_x, pred_ctr_y, pred_w, pred_h;
	if (type == "rpn"){
		Mat pre_box(anchors.rows, anchors.cols, CV_32FC1);
		for (int i = 0; i < anchors.rows; i++){
			width = anchors.at<float>(i, 2) - anchors.at<float>(i, 0) + 1.0;
			height = anchors.at<float>(i, 3) - anchors.at<float>(i, 1) + 1.0;
			ctr_x = anchors.at<float>(i, 0) + 0.5*(width - 1);
			ctr_y = anchors.at<float>(i, 1) + 0.5*(height - 1);

			dx = boxs_delta.at<float>(i, 0);
			dy = boxs_delta.at<float>(i, 1);
			dw = boxs_delta.at<float>(i, 2);
			dh = boxs_delta.at<float>(i, 3);

			pred_ctr_x = dx*width + ctr_x;
			pred_ctr_y = dy*height + ctr_y;
			pred_w = exp(dw)*width;
			pred_h = exp(dh)*height;

			pre_box.at<float>(i, 0) = ((pred_ctr_x - 0.5*(pred_w - 1)) - 1)*(float)(src_image.cols - 1) / (net_input_image_size_.width - 1) + 1;
			pre_box.at<float>(i, 1) = ((pred_ctr_y - 0.5*(pred_h - 1)) - 1)*(float)(src_image.rows - 1) / (net_input_image_size_.height - 1) + 1;
			pre_box.at<float>(i, 2) = ((pred_ctr_x + 0.5*(pred_w - 1)) - 1)*(float)(src_image.cols - 1) / (net_input_image_size_.width - 1) + 1;
			pre_box.at<float>(i, 3) = ((pred_ctr_y + 0.5*(pred_h - 1)) - 1)*(float)(src_image.rows - 1) / (net_input_image_size_.height - 1) + 1;;

			if (pre_box.at<float>(i, 0) < 0)
				pre_box.at<float>(i, 0) = 0;
			if (pre_box.at<float>(i, 0) > (src_image.cols - 1))
				pre_box.at<float>(i, 0) = src_image.cols - 1;
			if (pre_box.at<float>(i, 2) < 0)
				pre_box.at<float>(i, 2) = 0;
			if (pre_box.at<float>(i, 2) > (src_image.cols - 1))
				pre_box.at<float>(i, 2) = src_image.cols - 1;

			if (pre_box.at<float>(i, 1) < 0)
				pre_box.at<float>(i, 1) = 0;
			if (pre_box.at<float>(i, 1) > (src_image.rows - 1))
				pre_box.at<float>(i, 1) = src_image.rows - 1;
			if (pre_box.at<float>(i, 3) < 0)
				pre_box.at<float>(i, 3) = 0;
			if (pre_box.at<float>(i, 3) > (src_image.rows - 1))
				pre_box.at<float>(i, 3) = src_image.rows - 1;

		}
		return pre_box;
	}
	else if (type == "rcnn")
	{
		Mat pre_box(boxs_delta.rows, boxs_delta.cols, CV_32FC1);
		for (int i = 0; i < boxs_delta.rows; i++)
		{
			for (int j = 1; j < boxs_delta.cols / 4; j++)
			{
				width = anchors.at<float>(i, 2) - anchors.at<float>(i, 0) + 1;
				height = anchors.at<float>(i, 3) - anchors.at<float>(i, 1) + 1;
				ctr_x = anchors.at<float>(i, 0) + 0.5 * (width - 1);
				ctr_y = anchors.at<float>(i, 1) + 0.5 * (height - 1);

				dx = boxs_delta.at<float>(i, 4 * j + 0);
				dy = boxs_delta.at<float>(i, 4 * j + 1);
				dw = boxs_delta.at<float>(i, 4 * j + 2);
				dh = boxs_delta.at<float>(i, 4 * j + 3);

				pred_ctr_x = dx*width + ctr_x;
				pred_ctr_y = dy*height + ctr_y;
				pred_w = exp(dw) * width;
				pred_h = exp(dh) * height;

				pre_box.at<float>(i, 4 * (j - 1) + 0) = ((pred_ctr_x - 0.5*(pred_w - 1)) - 1);
				pre_box.at<float>(i, 4 * (j - 1) + 1) = ((pred_ctr_y - 0.5*(pred_h - 1)) - 1);
				pre_box.at<float>(i, 4 * (j - 1) + 2) = ((pred_ctr_x + 0.5*(pred_w - 1)) - 1);
				pre_box.at<float>(i, 4 * (j - 1) + 3) = ((pred_ctr_y + 0.5*(pred_h - 1)) - 1);
				if (pre_box.at<float>(i, 4 * (j - 1) + 0) < 0)
					pre_box.at<float>(i, 4 * (j - 1) + 0) = 0;
				if (pre_box.at<float>(i, 4 * (j - 1) + 0) > (src_image.cols - 1))
					pre_box.at<float>(i, 4 * (j - 1) + 0) = src_image.cols - 1;
				if (pre_box.at<float>(i, 4 * (j - 1) + 2) < 0)
					pre_box.at<float>(i, 4 * (j - 1) + 2) = 0;
				if (pre_box.at<float>(i, 4 * (j - 1) + 2) > (src_image.cols - 1))
					pre_box.at<float>(i, 4 * (j - 1) + 2) = src_image.cols - 1;

				if (pre_box.at<float>(i, 4 * (j - 1) + 1) < 0)
					pre_box.at<float>(i, 4 * (j - 1) + 1) = 0;
				if (pre_box.at<float>(i, 4 * (j - 1) + 1) > (src_image.rows - 1))
					pre_box.at<float>(i, 4 * (j - 1) + 1) = src_image.rows - 1;
				if (pre_box.at<float>(i, 4 * (j - 1) + 3) < 0)
					pre_box.at<float>(i, 4 * (j - 1) + 3) = 0;
				if (pre_box.at<float>(i, 4 * (j - 1) + 3) > (src_image.rows - 1))
					pre_box.at<float>(i, 4 * (j - 1) + 3) = src_image.rows - 1;
			}

		}
		return pre_box;
	}
}


/*************************************************
// Method: proposal_local_anchor
// Description: generate the proposal local anchor
// Author: zhanglaplace
// Date: 2017/02/25
// Returns: Mat   proposal local anchors
// Parameter: 
// History:
*************************************************/
Mat Caffe_Net::proposal_local_anchor(){
	Mat anchors(9 * feature_map_size_.width*feature_map_size_.height, 4, CV_32FC1);
	int index = 0;
	for (int w = 0; w < feature_map_size_.width; w++){
		for (int h = 0; h < feature_map_size_.height; h++){
			for (int i = 0; i < 9; i++){
				for (int j = 0; j < 2; j++)
				{
					anchors.at<float>(index, 2 * j) = conf.feat_stride*w + conf.anchors[i][2 * j];
					anchors.at<float>(index, 2 * j + 1) = conf.feat_stride*h + conf.anchors[i][2 * j + 1];
				}
				index++;
			}
		}
	}
	return anchors;
}


/*************************************************
// Method: proposal_local_anchor1
// Description: generate the proposal local anchor  method2
// Author: zhanglaplace
// Date: 2017/02/25
// Returns: Mat   proposal local anchors
// Parameter:
// History:
*************************************************/
Mat Caffe_Net::proposal_local_anchor1(){
	Blob<float> shift;
	Mat shitf_x(feature_map_size_.height, feature_map_size_.width, CV_32SC1);
	Mat shitf_y(feature_map_size_.height, feature_map_size_.width, CV_32SC1);
	for (size_t i = 0; i < feature_map_size_.width; i++)
	{
		for (size_t j = 0; j < feature_map_size_.height; j++)
		{
			shitf_x.at<int>(j, i) = i * conf.feat_stride;
			shitf_y.at<int>(j, i) = j * conf.feat_stride;
		}
	}
	shift.Reshape(9, feature_map_size_.width*feature_map_size_.height, 4, 1);
	float *p = shift.mutable_cpu_diff(), *a = shift.mutable_cpu_data();
	for (int i = 0; i < feature_map_size_.width*feature_map_size_.height; i++)
	{
		for (int j = 0; j < 9; j++)
		{
			{
				size_t num = i * 4 + j * 4 * feature_map_size_.width*feature_map_size_.height;
				p[num + 0] = -shitf_x.at<int>(i % shitf_x.rows, i / shitf_x.rows);
				p[num + 2] = -shitf_x.at<int>(i % shitf_x.rows, i / shitf_x.rows);
				p[num + 1] = -shitf_y.at<int>(i % shitf_y.rows, i / shitf_y.rows);
				p[num + 3] = -shitf_y.at<int>(i % shitf_y.rows, i / shitf_y.rows);
				a[num + 0] = conf.anchors[j][0];
				a[num + 1] = conf.anchors[j][1];
				a[num + 2] = conf.anchors[j][2];
				a[num + 3] = conf.anchors[j][3];
			}
		}
	}
	shift.Update();
	Mat anchors(9 * feature_map_size_.width*feature_map_size_.height, 4, CV_32FC1);
	size_t num = 0;
	for (int i = 0; i < anchors.cols; i++)
	{
		for (int j = 0; j < anchors.rows; j++)
		{
			anchors.at<float>(j, i) = shift.data_at(num%shift.num(),
				(num - num / (shift.num() * shift.channels())*shift.num() * shift.channels()) / shift.num(),
				num / (shift.num() * shift.channels()), 0);
			num++;
		}
	}
	return anchors;
}



/*************************************************
// Method: process
// Description: mean , init rpn_net input data;
// Author: zhanglaplace
// Date: 2017/02/25
// Returns:  void
// Parameter: Mat cv_image 
// History:
*************************************************/
void Caffe_Net::process(Mat cv_image){
	Mat img_float;
	cv_image.convertTo(img_float, CV_32FC3);

	Mat cv_new(cv_image.rows, cv_image.cols, CV_32FC3);
	int height = cv_image.rows;
	int width = cv_image.cols;

	im_scale_ = prep_im_for_blod_size(width, height, conf.target_size, conf.max_size);
	net_input_image_size_ = Size(round(width*im_scale_), round(height*im_scale_));
	if (Caffe::mode() == Caffe::CPU){
		Mat temp_mean = mean_image.clone();
		temp_mean.convertTo(temp_mean, CV_32FC3);
	    cv:resize(temp_mean, temp_mean, Size(width, height), 0.0, 0.0, CV_INTER_LINEAR);
		subtract(img_float, temp_mean, cv_new);

	}
	else if (Caffe::mode() == Caffe::GPU){
		double im_means_scale = max(double(cv_image.rows) / mean_image.rows, double(cv_image.cols) / mean_image.cols);
		Mat temp_mean = mean_image.clone();
		cv::resize(mean_image, temp_mean, Size(round(mean_image.cols*im_means_scale), round(mean_image.rows*im_means_scale)));

		int y_start = floor((temp_mean.rows - cv_image.rows) / 2);
		int x_start = floor((temp_mean.cols - cv_image.cols) / 2);
		//select roi 
		cv::Rect roi(x_start, y_start, cv_image.cols, cv_image.rows);
		temp_mean = temp_mean(roi);
		temp_mean.convertTo(temp_mean, CV_32FC3);
		subtract(img_float, temp_mean, cv_new);
	}
	Mat cv_resized;
	cv::resize(cv_new, cv_resized, Size(net_input_image_size_.width, net_input_image_size_.height));
	cvtColor(cv_resized, cv_resized, CV_BGR2RGB);//add or not add is a question

	Blob<float>* input_layers = rpn_net_->input_blobs()[0];
	input_layers->Reshape(1, cv_resized.channels(), cv_resized.rows, cv_resized.cols);
	rpn_net_->Reshape();
	float* input_data = input_layers->mutable_cpu_data();
	vector<cv::Mat> input_channels;
	for (int i = 0; i < input_layers->channels(); ++i) {
		cv::Mat channel(cv_resized.rows, cv_resized.cols, CV_32FC1, input_data);
		input_channels.push_back(channel);
		input_data += cv_resized.rows * cv_resized.cols;
	}
	cv::split(cv_resized, input_channels);

	CHECK(reinterpret_cast<float*>(input_channels.at(0).data)== rpn_net_->input_blobs()[0]->cpu_data())<< "Input channels are not wrapping the input layer of the network.";
	/*
	for (int h = 0; h < cv_resized.rows; h++){
		uchar* data = cv_resized.ptr<uchar>(h);
		int image_index = 0;
		for (int w = 0; w < cv_resized.cols; w++){
			for (int c = 0; c < nchannels; c++){
				input_data[(c*height_resized + h)*width_resized + w] = float(data[image_index++]);
				//input_data[(c*height_resized + h)*width_resized + w] = float(cv_resized.at<cv::Vec3b>(cv::Point(w, h))[c]);
			}
		}
	}
	*/
}


/*************************************************
// Method: prep_im_for_blod_size
// Description: get the scale of  the input image and the rpn_net input ;
// Author: zhanglaplace
// Date: 2017/02/25
// Returns: double im_scale_
// Parameter: image height ,width ,target size and max size
// History:
*************************************************/
double Caffe_Net::prep_im_for_blod_size(int height, int width, int target_size, int max_size){
	int im_size_min = min(height, width);
	int im_size_max = max(height, width);
	double im_scale = double(target_size) / im_size_min;

	//prevent the biggest axis from being more than max_size
	if (round(im_scale*im_size_max)> max_size){
		im_scale = (double)max_size / im_size_max;
	}
	return im_scale;
}