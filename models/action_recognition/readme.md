##Spatial net (RGB input) models:
| name | caffemodel | caffemodel_url | license | caffe_commit |
| --- | --- | --- | --- | --- |
| CUHK Action Recognition Spatial Model (UCF101 Split1) | cuhk_action_spatial_vgg_16_split1.caffemodel | http://mmlab.siat.ac.cn/very_deep_two_stream_model/cuhk_action_spatial_vgg_16_split1.caffemodel | license: non-commercial | d26b3b8b8eec182a27ce9871752fedd374b63650 
| CUHK Action Recognition Spatial Model (UCF101 Split2) | cuhk_action_spatial_vgg_16_split2.caffemodel | http://mmlab.siat.ac.cn/very_deep_two_stream_model/cuhk_action_spatial_vgg_16_split2.caffemodel | license: non-commercial | d26b3b8b8eec182a27ce9871752fedd374b63650
| CUHK Action Recognition Spatial Model (UCF101 Split3) | cuhk_action_spatial_vgg_16_split3.caffemodel | http://mmlab.siat.ac.cn/very_deep_two_stream_model/cuhk_action_spatial_vgg_16_split3.caffemodel | license: non-commercial | d26b3b8b8eec182a27ce9871752fedd374b63650

##Temporal net (optical flow input) models:
| name | caffemodel | caffemodel_url | license | caffe_commit |
| --- | --- | --- | --- | --- |
| CUHK Action Recognition Temporal Model (UCF101 Split1) | cuhk_action_temporal_vgg_16_split1.caffemodel | http://mmlab.siat.ac.cn/very_deep_two_stream_model/cuhk_action_temporal_vgg_16_split1.caffemodel | license: non-commercial | d26b3b8b8eec182a27ce9871752fedd374b63650 
| CUHK Action Recognition Temporal Model (UCF101 Split2) | cuhk_action_temporal_vgg_16_split2.caffemodel | http://mmlab.siat.ac.cn/very_deep_two_stream_model/cuhk_action_temporal_vgg_16_split2.caffemodel | license: non-commercial | d26b3b8b8eec182a27ce9871752fedd374b63650
| CUHK Action Recognition Temporal Model (UCF101 Split3) | cuhk_action_temporal_vgg_16_split3.caffemodel | http://mmlab.siat.ac.cn/very_deep_two_stream_model/cuhk_action_temporal_vgg_16_split3.caffemodel | license: non-commercial | d26b3b8b8eec182a27ce9871752fedd374b63650

These models are trained using the strategy described in 
[the tech report](http://arxiv.org/abs/1507.02159). Model and training configurations are set according to the original report. 

The model parameters are initialized with the public available VGG-16 model and trained on the UCF-101 dataset. 

The bundled models are the iteration 15,000 snapshots using corresponding solvers.

For more details, please see the [project page]().

These models were trained by Limin Wang @wanglimin and Yuanjun Xiong @yjxiong.

## License

The models are released for non-commercial use.
