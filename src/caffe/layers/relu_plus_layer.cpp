#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ReLUPlusLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	LOG(INFO) << "calling ReLUPlus Layer Setup";
	NeuronLayer<Dtype>::LayerSetUp(bottom, top);
	activations_.Reshape(bottom[0]->num(), bottom[0]->channels(),
			bottom[0]->height(), bottom[0]->width());
	caffe_set(activations_.count(), Dtype(1), activations_.mutable_cpu_data());
}

template <typename Dtype>
void ReLUPlusLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();
	const int count = bottom[0]->count();
	for (int i = 0; i < count; ++i) {
		top_data[i] = std::max(bottom_data[i], Dtype(0));
		top_data[i] = activations_.cpu_data()[i] * top_data[i];
	}
}

template <typename Dtype>
void ReLUPlusLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
	if (propagate_down[0]) {
		const Dtype* bottom_data = bottom[0]->cpu_data();
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		const int count = bottom[0]->count();
		for (int i = 0; i < count; ++i) {
			bottom_diff[i] = top_diff[i] * (bottom_data[i] > 0);
			activations_.mutable_cpu_data()[i] = top_diff[i] > 0 ? Dtype(1) : Dtype(0);
			bottom_diff[i] = activations_.cpu_data()[i] * bottom_diff[i];
		}
	}
}


#ifdef CPU_ONLY
STUB_GPU(ReLUPlusLayer);
#endif

INSTANTIATE_CLASS(ReLUPlusLayer);
REGISTER_LAYER_CLASS(ReLUPlus);


}  // namespace caffe
