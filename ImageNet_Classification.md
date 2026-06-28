**Introduction**
- This paper is one of the fundamental papers that initialised the AI growth in popularity in the present.
- It looked to improve on the image classification using a Neural Network, which at the time was not a very popular approach due to its difficulty in optimisation.
- They talk about how many of the datasets used in benchmarking are very small compared to what would be needed
- Instead of small datasets like CIFAR of NORB, they are interested in working with millions of images and thousands of categories to improve image recognition in the real world - this uses images from the real world with real world noise
- They chose CNNs over feed forward NNs because they are easier to train on image datasets
- Their model achieved the best results on the ImageNet dataset yet --- they have a publicly available gpu implementation of a 2d cnn [1] on top of page 1 (use for repeating exp)
- Their final model was made of five convolutional and three fully-connected layers (combining CNN and feed forward networks)
- They took between five to six days to train the model on two GTX 580 3GB GPUs 
