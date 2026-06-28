# Introduction
- This paper is one of the fundamental papers that initialised the AI growth in popularity in the present.
- It looked to improve on the image classification using a Neural Network, which at the time was not a very popular approach due to its difficulty in optimisation.
- They talk about how many of the datasets used in benchmarking are very small compared to what would be needed
- Instead of small datasets like CIFAR of NORB, they are interested in working with millions of images and thousands of categories to improve image recognition in the real world - this uses images from the real world with real world noise
- They chose CNNs over feed forward NNs because they are easier to train on image datasets
- Their model achieved the best results on the ImageNet dataset yet --- they have a publicly available gpu implementation of a 2d cnn on top of page 1 (use for repeating exp)
- Their final model was made of five convolutional and three fully-connected layers (combining CNN and feed forward networks)
- They took between five to six days to train the model on two GTX 580 3GB GPUs 
# The Dataset
- Since their model only took one dimension of inputs, the images had to all be the same resolution
- In the case of high resolution images they down scaled them to a standard 256x256 image. For a rectangular image, they had to downscale so the shorter side would be 256 and then cropped out the edges to make a 256x256 image
- The model performance is also measured using top-1 and top-5 (weather the model considers the actual category in the top 1 or top 5 categories for a given image)
# The Architecture
## ReLU Nonlinearity
- They used the ReLU activation function to optimise the network.
- They go on to explain that using ReLU is around six times faster than the tanh function at optimisation
- The other functions mentioned were tanh and also f(x) = |tanh(x)| which introduced non linear functions
## Training on Multiple GPUs
- They only had access to GTX 580s at the time so they had to train the model on two of them since one didn't have enough memory to train the model
- They used a type of parallelisation where each GPU contained half the kernels of the network but each layer of kernels only had access to certain amount of kernels in the other GPU
- I don't really understand this but they say it all allowed them the precisely tune the amount of communication until it's an acceptable fraction of the amount of computation (???)
- Ok, I figured it out, basically they only let some of the layers to access all the layer maps on both GPUs but other layers can only access layer maps on one GPU so it is computationally less expensive :)
- This communication improved they error rates by 1.7% and 1.2%