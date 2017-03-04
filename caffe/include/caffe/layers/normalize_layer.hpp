#ifndef CAFFE_NORMALIZE_LAYER_HPP_
#define CAFFE_NORMALIZE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {


template<typename Dtype>
class NormalizeLayer: public Layer<Dtype> {
 public:
    explicit NormalizeLayer(const LayerParameter& param) :
            Layer<Dtype>(param) {
    }
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const {
        return "Normalize";
    }
    virtual inline int ExactNumBottomBlobs() const {
        return 1;
    }
    virtual inline int ExactNumTopBlobs() const {
        return 1;
    }

 protected:
    /// @copydoc NormalizeLayer
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);

    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
            const vector<bool>& propagate_down,
            const vector<Blob<Dtype>*>& bottom);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    
    bool channel_shared_;  // cached for backward pass
    Blob<Dtype> sum_spatial_multiplier_;
    Blob<Dtype> sum_channel_multiplier_;
    Blob<Dtype> norm_;
    Blob<Dtype> buffer_;
    bool across_spatial_;
    Blob<Dtype> buffer_channel_;
    Blob<Dtype> buffer_spatial_;
    float eps_;
};

}  // namespace caffe

#endif  //CAFFE_NORMALIZE_LAYER_HPP_
