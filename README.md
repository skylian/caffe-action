# Action Recognition with Deep Learning

This branch hosts the code for the technical report ["Towards Good Practices for Very Deep Two-stream ConvNets"](http://arxiv.org/abs/1507.02159).

### Features:
- `VideoDataLayer` for inputing video data
- Training on optical flow data. [Optical flow extraction](https://github.com/wanglimin/dense_flow).
- Data augmentation with fixed corner cropping and multi-scale cropping
- Parallel training with multiple GPUs.

### Usage
Generally it's the same as the original caffe. Please see the original README. 
Please see following instruction for accessing features above. More detailed documentation is on the way.

- Video/optic flow data
  - A new data layer call "VideoDataLayer" has been added to support mutiple frame input
- Fixed corner cropping augmentation
  - Set `fix_crop` to `true` in `tranform_param` of network's protocol buffer definition.
- "Multi-scale" cropping augmentation
  - Set `multi_scale` to `true` in `transform_param`
- Training with multiple GPUs
  - Requires OpenMPI > 1.8.5 ([Why?](https://www.open-mpi.org/faq/?category=runcuda#mpi-apis-no-cuda))
  - Specify list of GPU IDs to be used for training, in the solver protocol buffer definition, like `device_id: [0,1,2,3]`
  - Compile using cmake and use `mpirun` to launch caffe executable, like 
```bash
mkdir build && cd build
cmake .. -DUSE_MPI=ON
make && make install
mpirun -np 4 ./install/bin/caffe train --solver=<Your Solver File>'
```

**Note**: actual batch_size will be `num_device` times `batch_size` specified in network's prototxt.

### Working Examples
- Action recognition on UCF101
  - Coming soon...

### Extension
Currently all existing data layers sub-classed from `BasePrefetchLayer` support parallel training. If you have newly added layer which is also sub-classed `BasePrefetchLayer`, simply override the virtual method 
```C++
inline virtual void advance_cursor();
```
It's function should be forwarding the "cursor" in your data layer for one step. 

### Questions
Contact 
- [Limin Wang](http://wanglimin.github.io/)
- [Yuanjun Xiong](http://personal.ie.cuhk.edu.hk/~xy012/)

----
Following is the original README of Caffe.

# Caffe

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by the Berkeley Vision and Learning Center ([BVLC](http://bvlc.eecs.berkeley.edu)) and community contributors.

Check out the [project site](http://caffe.berkeleyvision.org) for all the details like

- [DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)
- [Tutorial Documentation](http://caffe.berkeleyvision.org/tutorial/)
- [BVLC reference models](http://caffe.berkeleyvision.org/model_zoo.html) and the [community model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

and step-by-step examples.

[![Join the chat at https://gitter.im/BVLC/caffe](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/BVLC/caffe?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Please join the [caffe-users group](https://groups.google.com/forum/#!forum/caffe-users) or [gitter chat](https://gitter.im/BVLC/caffe) to ask questions and talk about methods and models.
Framework development discussions and thorough bug reports are collected on [Issues](https://github.com/BVLC/caffe/issues).

Happy brewing!

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }
