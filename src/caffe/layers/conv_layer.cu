#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	bool dynamic_conv = this->layer_param_.convolution_param().dynamic_conv();
	if (!dynamic_conv) {
		const Dtype* weight = this->blobs_[0]->gpu_data();
		for (int i = 0; i < bottom.size(); ++i) {
			const Dtype* bottom_data = bottom[i]->gpu_data();
			Dtype* top_data = top[i]->mutable_gpu_data();
			for (int n = 0; n < this->num_; ++n) {
				this->forward_gpu_gemm(bottom_data + bottom[i]->offset(n), weight,
						top_data + top[i]->offset(n));
				if (this->bias_term_) {
					const Dtype* bias = this->blobs_[1]->gpu_data();
					this->forward_gpu_bias(top_data + top[i]->offset(n), bias);
				}
			}
		}
	} else {
		const Dtype* weight = bottom[0]->gpu_data();
		for (int i = 1; i < bottom.size(); ++i) {
			const Dtype* bottom_data = bottom[i]->gpu_data();
			Dtype* top_data = top[i-1]->mutable_gpu_data();
			for (int n = 0; n < this->num_; ++n) {
				this->forward_gpu_gemm(bottom_data + bottom[i]->offset(n), weight+bottom[0]->offset(n),
						top_data + top[i-1]->offset(n));
			}
		}
	}
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	bool dynamic_conv = this->layer_param_.convolution_param().dynamic_conv();
	if (!dynamic_conv) {
		const Dtype* weight = this->blobs_[0]->gpu_data();
		Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
		for (int i = 0; i < top.size(); ++i) {
			const Dtype* top_diff = top[i]->gpu_diff();
			// Bias gradient, if necessary.
			if (this->bias_term_ && this->param_propagate_down_[1]) {
				Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
				for (int n = 0; n < this->num_; ++n) {
					this->backward_gpu_bias(bias_diff, top_diff + top[i]->offset(n));
				}
			}
			if (this->param_propagate_down_[0] || propagate_down[i]) {
				const Dtype* bottom_data = bottom[i]->gpu_data();
				Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
				for (int n = 0; n < this->num_; ++n) {
					// gradient w.r.t. weight. Note that we will accumulate diffs.
					if (this->param_propagate_down_[0]) {
						this->weight_gpu_gemm(bottom_data + bottom[i]->offset(n),
								top_diff + top[i]->offset(n), weight_diff);
					}
					// gradient w.r.t. bottom data, if necessary.
					if (propagate_down[i]) {
						this->backward_gpu_gemm(top_diff + top[i]->offset(n), weight,
								bottom_diff + bottom[i]->offset(n));
					}
				}
			}
		}
	} else {
		const Dtype* weight = bottom[0]->gpu_data();
		Dtype* weight_diff = bottom[0]->mutable_gpu_diff();
		for (int i = 0; i < top.size(); ++i) {
			const Dtype* top_diff = top[i]->gpu_diff();
			if (propagate_down[0] || propagate_down[i+1]) {
				const Dtype* bottom_data = bottom[i+1]->gpu_data();
				Dtype* bottom_diff = bottom[i+1]->mutable_gpu_diff();
				for (int n = 0; n < this->num_; ++n) {
					// gradient w.r.t. weight. Note that we will accumulate diffs.
					if (propagate_down[0]) {
						this->weight_gpu_gemm(bottom_data + bottom[i+1]->offset(n),
								top_diff + top[i]->offset(n), weight_diff+bottom[0]->offset(n));
					}
					// gradient w.r.t. bottom data, if necessary.
					if (propagate_down[i+1]) {
						this->backward_gpu_gemm(top_diff + top[i]->offset(n), weight+bottom[0]->offset(n),
								bottom_diff + bottom[i+1]->offset(n));
					}
				}
			}
		}
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionLayer);

}  // namespace caffe
