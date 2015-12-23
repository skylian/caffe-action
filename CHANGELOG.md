## Dec 23, 2015

Features:

  - Implemented a planning algorithm to globally optimize the cudnn workspace consumption and speed trade-off.
  - Now `richness` parameter specifies the total memory in MBs available to cudnn for convolution workspaces.
  - Now the framework will try to find the best convolution algorithm combinations under memory limit.
  
## Dec 17, 2015

Features:

  - cuDNN v4 support
  - 20% overall speed gain with faster convolution and batch normalization
  - the native batch normalization is changed to comply with cuDNN. Use the script in `python/` to upgrade your models.
  
## Nov 22, 2015

Features:
  - python layer can expose a prefetch() method, which will be run in parallel with network processing.
  
## Oct 13, 2015

Features:
  - Improved cuDNN wrapper to use less GPU memory. 
  - Now there is a new parameter `richness` which controls the limit of workspace for cuDNN.
  
## Sep 30, 2015

Features:
  - Support for cuDNN v3.
  
## Sep. 7, 2015

Features:
  - New mechanism for parallel comminucation reduced parallel overhead.
  - Batch normalization, courtesy of @Cysu.
  
## Jul, 2015

Features:
  - Action recognition tools, scripts, and examples.
  - Basic parallel training support
  - Various extra data augmentations
