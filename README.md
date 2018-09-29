## Fully-Convolutional Siamese Networks for Object Tracking


- - - -
The code in this repository enables you to reproduce the experiments of our paper.
It can be used in two ways: **(1) tracking only** and **(2) training and tracking**.

Project page: <http://www.robots.ox.ac.uk/~luca/siamese-fc.html>
- - - -

![pipeline image][logo]

[logo]: http://www.robots.ox.ac.uk/~luca/stuff/siamesefc_conv-explicit_small.jpg "Pipeline image"
- - - -
If you find our work and/or curated dataset useful, please cite:
```
@article{bertinetto2016fully,
  title={Fully-Convolutional Siamese Networks for Object Tracking},
  author={Bertinetto, Luca and Valmadre, Jack and Henriques, Jo{\~a}o F and Vedaldi, Andrea and Torr, Philip},
  journal={arXiv preprint arXiv:1606.09549},
  year={2016}
}
```
- - - -

1. [ **Tracking only** ] If you don't care much about training, simply plug one of our pretrained networks to our basic tracker and see it in action.
  1. Prerequisites: GPU, CUDA drivers, [cuDNN](https://developer.nvidia.com/cudnn), Matlab (we used 2015b), [MatConvNet](http://www.vlfeat.org/matconvnet/install/) (we used `v1.0-beta20`).
  2. Clone the repository.
  3. Download one of the pretrained networks from <http://www.robots.ox.ac.uk/~luca/siamese-fc.html>
  4. Go to `siam-fc/tracking/` and remove the trailing `.example` from `env_paths_tracking.m.example`, `startup.m.example` and `run_tracking.m.example`, editing the files as appropriate.
  5. Be sure to have at least one video sequence in the appropriate format. You can find an example here in the repository (`siam-fc/demo-sequences/vot15_bag`).
  6. `siam-fc/tracking/run_tracking.m` is the entry point to execute the tracker, have fun!

2. [ **Training and tracking** ] Well, if you prefer to train your own network, the process is slightly more involved (but also more fun).
  1. Prerequisites: GPU, CUDA drivers, [cuDNN](https://developer.nvidia.com/cudnn), Matlab (we used 2015b), [MatConvNet](http://www.vlfeat.org/matconvnet/install/) (we used `v1.0-beta20`).
  2. Clone the repository.
  3. Follow these [step-by-step instructions](https://github.com/bertinetto/siamese-fc/tree/master/ILSVRC15-curation), which will help you generating a curated dataset compatible with the rest of the code.  
  4. If you did not generate your own, download the [imdb_video.mat](http://bit.ly/imdb_video) (6.7GB) with all the metadata and the [dataset stats](http://bit.ly/imdb_video_stats).
  5. Go to `siam-fc/training/` and remove the trailing `.example` from `env_paths.m.example`, `startup.m.example` and `run_experiment.m.example` editing the files as appropriate.
  6. `siam-fc/training/run_experiment.m` is the entry point to start training. Default hyper-params are at the start of `experiment.m` and can be overwritten by custom ones specified in `run_experiment.m`.
  7. By default, training plots are saved in `siam-fc/training/data/`. When you are happy, grab a network snapshot (`net-epoch-X.mat`) and save it somewhere convenient to use it for tracking.
  8. Go to point 1.iv. and enjoy the result of the labour of your own GPUs!
	
