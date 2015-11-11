#ifndef CAFFE_PYTHON_LAYER_HPP_
#define CAFFE_PYTHON_LAYER_HPP_

#include <boost/python.hpp>
#include <vector>

#include "caffe/layer.hpp"

namespace bp = boost::python;
#include <boost/thread.hpp>

namespace caffe {

template <typename Dtype>
class PythonLayer : public Layer<Dtype> {
 public:
  PythonLayer(PyObject* self, const LayerParameter& param)
      : Layer<Dtype>(param), self_(bp::handle<>(bp::borrowed(self))) { }

  virtual ~PythonLayer(){

  }
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    //ensure the GIL
    PyGILState_STATE state;
    state = PyGILState_Ensure();
    self_.attr("param_str") = bp::str(
            this->layer_param_.python_param().param_str());
    self_.attr("phase") = bp::str(
        (this->phase_ == TRAIN)?"train":"test"
    );
    self_.attr("_prefetch") = false;
    try {
      self_.attr("setup")(bottom, top);
      prefetch_ = self_.attr("_prefetch");
    } catch (bp::error_already_set) {
      PyErr_Print();
      throw;
    }
    PyGILState_Release(state);
    MaybeStartPrefetchThread();
  }

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    PyGILState_STATE state;
    state = PyGILState_Ensure();
    try {
      self_.attr("reshape")(bottom, top);
    } catch (bp::error_already_set) {
      PyErr_Print();
      throw;
    }
    PyGILState_Release(state);
  }

  virtual inline const char* type() const { return "Python"; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    WaitForPrefetchThread();
    PyGILState_STATE state;
    state = PyGILState_Ensure();
    try {
      self_.attr("forward")(bottom, top);
    } catch (bp::error_already_set) {
      PyErr_Print();
      throw;
    }
    PyGILState_Release(state);
    MaybeStartPrefetchThread();
  }
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    PyGILState_STATE state;
    state = PyGILState_Ensure();
    try {
      self_.attr("backward")(top, propagate_down, bottom);
    } catch (bp::error_already_set) {
      PyErr_Print();
      throw;
    }
    PyGILState_Release(state);
  }

  void PrefetchThread(){
    PyGILState_STATE state;
    state = PyGILState_Ensure();
    try {
      self_.attr("prefetch")();
    } catch (bp::error_already_set) {
      PyErr_Print();
      throw;
    }
    PyGILState_Release(state);
  }

  void MaybeStartPrefetchThread(){
    if (prefetch_){
      thread_.reset(
        new boost::thread(&PythonLayer::PrefetchThread, this));
    }

  }

  void WaitForPrefetchThread(){
    if ((thread_.get() != NULL) && thread_->joinable()){
      try {
        thread_->join();
      } catch (...) {
        throw;
      }
    }
  }

 private:
  bp::object self_;
  bool prefetch_;
  shared_ptr<boost::thread> thread_;
};

}  // namespace caffe

#endif
