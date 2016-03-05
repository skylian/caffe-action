#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#ifdef USE_MPI
#include "mpi.h"
#endif
#include <boost/filesystem.hpp>
using namespace boost::filesystem;

namespace caffe{
template <typename Dtype>
VideoDataLayer<Dtype>:: ~VideoDataLayer<Dtype>(){
	this->JoinPrefetchThread();
}

template <typename Dtype>
void VideoDataLayer<Dtype>:: DataLayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	const VideoDataParameter& video_data_param = this->layer_param_.video_data_param();

	new_length_.clear();
	std::copy(video_data_param.new_length().begin(), video_data_param.new_length().end(), std::back_inserter(new_length_));
	if (video_data_param.modality() == VideoDataParameter_Modality_BOTH)
		CHECK_EQ(new_length_.size(),2) << "Two new_length has to be specified for modality BOTH.";
	else
		CHECK_EQ(new_length_.size(), 1) << "One new_length should be specified for modality FLOW or RGB.";
	
    root_folders_.clear();
	std::copy(video_data_param.root_folder().begin(), video_data_param.root_folder().end(),
			std::back_inserter(root_folders_));
	if (video_data_param.modality() == VideoDataParameter_Modality_BOTH)
		CHECK_EQ(root_folders_.size(), 2) << "Two root folders should be specified for modality BOTH.";
	else
		CHECK_EQ(root_folders_.size(), 1) << "One root folder should be specified for modality FLOW or RGB.";
	const int interval = video_data_param.interval();
	if (video_data_param.modality() != VideoDataParameter_Modality_RGB)
		CHECK_GT(interval, 0) << "Flow data must have interval > 0.";
	
    num_labels_ = video_data_param.num_labels();
	LOG(INFO) << "number of labels: " << num_labels_;
	const string& source = video_data_param.source();
    LOG(INFO) << "Opening file: " << source;
	std:: ifstream infile(source.c_str());
	string filename;
	vector<int> label(num_labels_);
	int length;
	while (infile >> filename >> length){
		for (int i = 0; i < num_labels_; ++i)
			infile >> label[i];
		lines_.push_back(std::make_pair(filename,label));
		lines_duration_.push_back(length-interval);
	}
	if (video_data_param.shuffle()){
		const unsigned int prefectch_rng_seed = caffe_rng_rand();
		prefetch_rng_1_.reset(new Caffe::RNG(prefectch_rng_seed));
		prefetch_rng_2_.reset(new Caffe::RNG(prefectch_rng_seed));
		ShuffleVideos();
	}
	LOG(INFO) << "A total of " << lines_.size() << " videos.";
	lines_id_ = 0;

	const int new_height  = video_data_param.new_height();
	const int new_width  = video_data_param.new_width();
	const int num_segments = video_data_param.num_segments();
	int max_length = *std::max_element(new_length_.begin(), new_length_.end());
	Datum datum;
	const unsigned int frame_prefectch_rng_seed = caffe_rng_rand();
	frame_prefetch_rng_.reset(new Caffe::RNG(frame_prefectch_rng_seed));
	int average_duration = (int) lines_duration_[lines_id_]/num_segments;
	vector<int> offsets;
	for (int i = 0; i < num_segments; ++i){
		caffe::rng_t* frame_rng = static_cast<caffe::rng_t*>(frame_prefetch_rng_->generator());
		int offset = (*frame_rng)() % (average_duration - max_length + 1);
		offsets.push_back(offset+i*average_duration);
	}
	if (this->layer_param_.video_data_param().modality() == VideoDataParameter_Modality_FLOW)
		CHECK(ReadSegmentFlowToDatum(root_folders_[0]+lines_[lines_id_].first, (lines_[lines_id_].second)[0], offsets, new_height, new_width, new_length_[0], &datum));
	else if (this->layer_param_.video_data_param().modality() == VideoDataParameter_Modality_RGB)
		CHECK(ReadSegmentRGBToDatum(root_folders_[0]+lines_[lines_id_].first, (lines_[lines_id_].second)[0], offsets, new_height, new_width, new_length_[0], &datum, true));
	else
		CHECK(ReadSegmentRGBFlowToDatum(root_folders_, lines_[lines_id_].first, (lines_[lines_id_].second)[0], offsets, new_height, new_width, new_length_, &datum, true));
	const int crop_size = this->layer_param_.transform_param().crop_size();
	const int batch_size = this->layer_param_.video_data_param().batch_size();
	if (crop_size > 0){
		top[0]->Reshape(batch_size, datum.channels(), crop_size, crop_size);
		this->prefetch_data_.Reshape(batch_size, datum.channels(), crop_size, crop_size);
	} else {
		top[0]->Reshape(batch_size, datum.channels(), datum.height(), datum.width());
		this->prefetch_data_.Reshape(batch_size, datum.channels(), datum.height(), datum.width());
	}
	LOG(INFO) << "output data size: " << top[0]->num() << "," << top[0]->channels() << "," << top[0]->height() << "," << top[0]->width();

	top[1]->Reshape(batch_size, num_labels_, 1, 1);
	this->prefetch_label_.Reshape(batch_size, num_labels_, 1, 1);

	this->num_rois_ = this->layer_param_.video_data_param().num_rois();
    if (this->num_rois_) {
        CHECK_EQ(top.size(), 3) << "There should be 3 tops when ROIs are given.";
        CHECK_EQ(num_segments, 1) << "Number of segments per video should be one when ROIs are given.";
        vector<int> shape(2);
        shape[0] = batch_size*this->num_rois_;
        shape[1] = 5;
        top[2]->Reshape(shape);
        this->prefetch_roi_.Reshape(shape);
    }
    else {
    	CHECK_EQ(top.size(), 2) << "There should be 2 tops.";
    }

	vector<int> top_shape = this->data_transformer_->InferBlobShape(datum);
	this->transformed_data_.Reshape(top_shape);
}

template <typename Dtype>
void VideoDataLayer<Dtype>::ShuffleVideos(){
	caffe::rng_t* prefetch_rng1 = static_cast<caffe::rng_t*>(prefetch_rng_1_->generator());
	caffe::rng_t* prefetch_rng2 = static_cast<caffe::rng_t*>(prefetch_rng_2_->generator());
	shuffle(lines_.begin(), lines_.end(), prefetch_rng1);
	shuffle(lines_duration_.begin(), lines_duration_.end(),prefetch_rng2);
}

template <typename Dtype>
void VideoDataLayer<Dtype>::InternalThreadEntry(){
	const VideoDataParameter& video_data_param = this->layer_param_.video_data_param();
	const int batch_size = video_data_param.batch_size();
	const int new_height = video_data_param.new_height();
	const int new_width = video_data_param.new_width();
	const int num_segments = video_data_param.num_segments();
	string roi_folder = this->layer_param_.video_data_param().roi_folder();
	num_labels_ = video_data_param.num_labels();

	Datum datum;
	CHECK(this->prefetch_data_.count());
	Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
	Dtype* top_label = this->prefetch_label_.mutable_cpu_data();

	new_length_.clear();
	std::copy(video_data_param.new_length().begin(), video_data_param.new_length().end(), std::back_inserter(new_length_));
	if (video_data_param.modality() == VideoDataParameter_Modality_BOTH)
		CHECK_EQ(new_length_.size(),2) << "Two new_length has to be specified for modality BOTH.";
	else
		CHECK_EQ(new_length_.size(), 1) << "One new_length should be specified for modality FLOW or RGB.";

	root_folders_.clear();
	std::copy(video_data_param.root_folder().begin(), video_data_param.root_folder().end(),
			std::back_inserter(root_folders_));

	Blob<Dtype> rois;
	const int max_length = *std::max_element(new_length_.begin(), new_length_.end());
	const int lines_size = lines_.size();
	if (this->num_rois_) {
		vector<int> shape(2);
		shape[0] = batch_size * this->num_rois_;
		shape[1] = 5;
		this->prefetch_roi_.Reshape(shape);
	}
	for (int item_id = 0; item_id < batch_size; ++item_id){
		CHECK_GT(lines_size, lines_id_);
		vector<int> offsets;
		int average_duration = (int) lines_duration_[lines_id_] / num_segments;
		for (int i = 0; i < num_segments; ++i) {
			if (this->phase_==TRAIN){
				caffe::rng_t* frame_rng = static_cast<caffe::rng_t*>(frame_prefetch_rng_->generator());
				int offset = (*frame_rng)() % (average_duration - max_length + 1);
				offsets.push_back(offset+i*average_duration);
			} else{
				offsets.push_back(int((average_duration-max_length+1)/2 + i*average_duration));
			}
		}
		if (this->layer_param_.video_data_param().modality() == VideoDataParameter_Modality_FLOW){
			if(!ReadSegmentFlowToDatum(root_folders_[0]+lines_[lines_id_].first, (lines_[lines_id_].second)[0], offsets, new_height, new_width, new_length_[0], &datum)) {
				continue;
			}
		} else if (this->layer_param_.video_data_param().modality() == VideoDataParameter_Modality_RGB){
			if(!ReadSegmentRGBToDatum(root_folders_[0]+lines_[lines_id_].first, (lines_[lines_id_].second)[0], offsets, new_height, new_width, new_length_[0], &datum, true)) {
				continue;
			}
		} else {
			if(!ReadSegmentRGBFlowToDatum(root_folders_, lines_[lines_id_].first, (lines_[lines_id_].second)[0], offsets, new_height, new_width, new_length_, &datum, true))
				continue;
		}

		for (int l = 0; l < num_labels_; ++l) {
			top_label[item_id*num_labels_+l] = (lines_[lines_id_].second)[l];
		}

		this->transformed_data_.set_cpu_data(top_data + this->prefetch_data_.offset(item_id));
        if (this->num_rois_) {
        	boost::filesystem::path roi_path(lines_[lines_id_].first);
            string roi_file = roi_folder + roi_path.stem().string() + ".mat";
            if (!ReadROI(roi_file, rois, offsets[0])) {
                LOG(ERROR) << "Error reading ROI file " << roi_file
            		       << "at index " << offsets[0];
                item_id--;
            } else {
                this->data_transformer_->Transform(datum, &(this->transformed_data_), &rois);

                Dtype *roi_top = this->prefetch_roi_.mutable_cpu_data() + this->prefetch_roi_.offset(item_id*this->num_rois_);
                const Dtype *roi_data = rois.cpu_data();
                int n = 0;
                for (int p = 0; p < rois.count() && n < this->num_rois_-1; p+=rois.shape(1)) {
                	int x1 = roi_data[p], y1 = roi_data[p+1], x2 = roi_data[p+2], y2 = roi_data[p+3];
	    			if (std::min(x2-x1, y2-y1) >= 50) {
		    			int c = this->prefetch_roi_.shape(1) * n;
			    		roi_top[c] = item_id;
				    	for (int i = 0; i < 4; ++i)
					    	roi_top[c+i+1] = roi_data[p+i];
    					n++;
	    			}
                }
                for (; n < this->num_rois_; ++n) {
            	    int c = this->prefetch_roi_.shape(1) * n;
                	roi_top[c] = item_id;
	    			roi_top[c+1] = 0;
		    		roi_top[c+2] = 0;
			    	roi_top[c+3] = this->transformed_data_.shape(3)-1;
				    roi_top[c+4] = this->transformed_data_.shape(2)-1;
                }
            }
        } else {
        	this->data_transformer_->Transform(datum, &(this->transformed_data_), NULL);
        }

     	//next iteration
		lines_id_++;
		if (lines_id_ >= lines_size) {
			DLOG(INFO) << "Restarting data prefetching from start.";
			lines_id_ = 0;
			if(this->layer_param_.video_data_param().shuffle()){
				ShuffleVideos();
			}
		}
	}
}

INSTANTIATE_CLASS(VideoDataLayer);
REGISTER_LAYER_CLASS(VideoData);
}
