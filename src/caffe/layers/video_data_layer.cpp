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
#include <boost/filesystem.hpp>
using namespace boost::filesystem;
#endif

namespace caffe{
template <typename Dtype>
VideoDataLayer<Dtype>:: ~VideoDataLayer<Dtype>(){
	this->JoinPrefetchThread();
}

template <typename Dtype>
void VideoDataLayer<Dtype>:: DataLayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	const VideoDataParameter& video_data_param = this->layer_param_.video_data_param();
	const int new_height  = video_data_param.new_height();
	const int new_width  = video_data_param.new_width();
	new_length_.clear();
	std::copy(video_data_param.new_length().begin(), video_data_param.new_length().end(), std::back_inserter(new_length_));
	if (video_data_param.modality() == VideoDataParameter_Modality_BOTH) {
		if (new_length_.size() == 1)
			new_length_.push_back(new_length_[0]);
	}
	else
		CHECK_EQ(new_length_.size(), 1) << "One new length should be specified for modality FLOW or RGB.";
	int max_length = *std::max_element(new_length_.begin(), new_length_.end());
	const int num_segments = video_data_param.num_segments();
	const string& source = video_data_param.source();
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
	LOG(INFO) << "number of labels: " << num_labels_	;
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

	Datum datum;
	CHECK(this->prefetch_data_.count());
	Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
	Dtype* top_label = this->prefetch_label_.mutable_cpu_data();
	const VideoDataParameter& video_data_param = this->layer_param_.video_data_param();
	const int batch_size = video_data_param.batch_size();
	const int new_height = video_data_param.new_height();
	const int new_width = video_data_param.new_width();
	const int num_segments = video_data_param.num_segments();
	const int lines_size = lines_.size();
	new_length_.clear();
	std::copy(video_data_param.new_length().begin(), video_data_param.new_length().end(), std::back_inserter(new_length_));
	if (video_data_param.modality() == VideoDataParameter_Modality_BOTH) {
		if (new_length_.size() == 1)
			new_length_.push_back(new_length_[0]);
	}
	else
		CHECK_EQ(new_length_.size(), 1) << "One new length should be specified for modality FLOW or RGB.";
	int max_length = *std::max_element(new_length_.begin(), new_length_.end());
	root_folders_.clear();
	std::copy(video_data_param.root_folder().begin(), video_data_param.root_folder().end(),
			std::back_inserter(root_folders_));
	num_labels_ = video_data_param.num_labels();

	for (int item_id = 0; item_id < batch_size; ++item_id){
		CHECK_GT(lines_size, lines_id_);
		vector<int> offsets;
		int average_duration = (int) lines_duration_[lines_id_] / num_segments;
		for (int i = 0; i < num_segments; ++i){
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
		int offset1 = this->prefetch_data_.offset(item_id);
    	this->transformed_data_.set_cpu_data(top_data + offset1);
		this->data_transformer_->Transform(datum, &(this->transformed_data_));
		for (int l = 0; l < num_labels_; ++l) {
			top_label[item_id*num_labels_+l] = (lines_[lines_id_].second)[l];
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
