#ifndef _CONFIG_H_
#define _CONFIG_H_


#include <map>
#include <vector>

typedef struct config{
	int per_nms_topN;
	int after_nms_topN;
	int target_size;
	float nms_overlap_thres_proposal;
	float num_overlap_thres_fcnn;
	int feat_stride;
	int max_size;
	int anchors[9][4];
	int test_min_box_size;
	double test_fcnn_thres_score;
	config(){
		per_nms_topN = 6000;
		after_nms_topN = 300;
		target_size = 600;
		num_overlap_thres_fcnn = 0.3;
		max_size = 1000;
		feat_stride = 16;
		test_min_box_size = 16;
		test_fcnn_thres_score = 0.6;
		nms_overlap_thres_proposal = 0.70;
		int temp[9][4] = {
			{ -83, -39, 100, 56 },
			{ -175, -87, 192, 104 },
			{ -359, -183, 376, 200 },
			{ -55, -55, 72, 72 },
			{ -119, -119, 136, 136 },
			{ -247, -247, 264, 264 },
			{ -35, -79, 52, 96 },
			{ -79, -167, 96, 184 },
			{ -167, -343, 184, 360 }
		};
		memcpy(anchors, temp, 9 * 4 * sizeof(int));
	}

}config;

typedef struct aboxes{
	float x1;
	float y1;
	float x2;
	float y2;
	float score;

}aboxes;




#endif