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
## Local Response Normalisation
- Hyperparameters set are: 
$$
k=2; n=5; \alpha = 10^{-4}; \beta = 0.75
$$
- remember these for experiment reconstruction
## Overlapping Pooling
- They used overlapping pooling layers which means pooling units are allowed to share pixels in the image, they are not mutually exclusive to one pooling unit![[Pasted image 20260629214559.png|438]]
- This reduced the top-1 and top-5 error rates by 0.4% and 0.3% respectively
## Overall Architecture
- Eight weighted layers: five convolutional, three full-connected layers
- Final layer is fed into a 1000-way softmax a prediction probability for each of the 1000 possible labels for a given input
- Kernels of the $2^{nd}$, $4^{th}$ and $5^{th}$ convolutional layers are connected the the kernel maps in the previous layer on the same GPU
- The kernel on the $3^{rd}$ layer are connected to all kernel maps in the $2^{nd}$ 
- All the full-connected layers are connected to all neurons in the previous layer
- [[#Local Response Normalisation|Response Normalisation]]  come after the first and second convolutional layers
- Max-pooling layers come aftter the response normalization layer and the fifth convoultional layer
![CNN Architecture](2026-06-29-220805_hyprshot.png)
- First conv. layer filters 225x224x3 with 96 kernels of size 11x11x3 with stride of 4 pixels
- Second conv. layer has 256 kernels of size 5x5x48
- Third conv. layer has 384 kernels of size 3x3x256
- Fourth conv. layer has 384 kernels of size 3x3x192
- Fifth conv. layer has 256 kernels of size 3x3x192
- All fully-connected layers have 4096 neurons each
# Reducing Overfitting
- Model has 60 mil params -> prone to overfitting 
- They use two different ways to avoid this
## Data Augmentation
- If there is not enough data to learn all of the parameters, we can just artificially enlarge the images giving us more data to train
- Use the CPU to enlarge the images
- They get random patches of the larger images and their horizontal reflection to increase the training set by a factor of 2048
- In test time, they take the test image and get five patches of it and average the overall prediction on the prediction of each of the patches
- They also alter the intensities of the RGB channels (PCA on a set of pixel values)
- To each RGB image pixel $I_{x y} = [I_{xy}^R, I_{xy}^G, I_{xy}^B]^T$ we add
 $$
 [\textbf{p}_1, \textbf{p}_2, \textbf{p}_3][\alpha_1\lambda_1, \alpha_2\lambda_2, \alpha_3\lambda_3]^T
 $$

 - don't fully understand this but it's just augmenting the RGB values according to the vector's eigenvalues and a random number
 - $\textbf{p}_i$ and $\lambda_i$  are the $i$th eigenvector and eigenvalue of the 3x3 covariance matrix of the pixels and $\alpha_i$ is the random number on a normal distribution
 - This creates diversity within the same image allowing the network to generalise well without needing completely new data
## Dropout
- Technique of combining predictions of different models in a more effective way. This will allow the model to reduce test error
- Sets the output of each hidden neuron to 0 with probability 0.5 during training
- During training, these neurons set to 0 don't contribute to the model learning so when an input is presented, the network samples a different architecture. This reduced co-dependence of neurons since, if neurons have a 50% chance of being put to 0, other neurons can't depend on others when making a prediction
- This makes the neurons learn more complex patterns in the data, hence making it more effective
- During testing we half each neuron output by 0.5
- They use dropout in the first two fully-connected layers. With this enabled they saw a lot of overfitting
# Details of Learning
- Stochastic gradient descent, batch size = 128, momentum = 0.9, weight decay = 0.0005
- Weight decay improves the model performance on top of regularising the network
- The update rule:
$$
\begin{gathered}
v_{i+1} := 0.9\cdot v_i = 0.0005 \cdot \epsilon \cdot w_i - \epsilon \cdot \langle \frac{\partial L}{\partial w}|_{w_i}\rangle_{D_i}\\
w_{i+1} := w_i + v_{i+1}
\end{gathered}
$$
where $i$ is the iteration index
$v$ is the momentum variable
$\epsilon$ is the learning rate
and $\langle \frac{\partial L}{\partial w}|_{w_i}\rangle_{D_i}$ is the average over the $i$th batch $D_i$ of the derivative of the objective function with respect to $w$ evaluated at $w_i$ 

- They used Gausian weight initialisation with $\mu=0$ and $\sigma=0.01$ 
- They initialised the neuron biases in the second, fourth and fifth convolutional layers and the fully-connected hidden layers with constant 1
- Then they initialised the remaining neuron biases with constant 0
- The positive biases given to the ReLUs allowed the network to learn faster during the early stages
- Each layer has the same learning rate (initialised to 0.01)-> adjusted this manually throughout training. They divided the learning rate by 10 when the validation error rate stopped improving (this happened three times during training)
- It was trained for around 90 cycles through the 1.2 mil images (this took 5-6 days on their hardware)
# Results
- They achieved a top-1 test error rate of 37.5% and a top-5 test error rate pf 17% tested on the ILSVRC-2010 dataset. Others (Sparse coding and SIFT+FVs) are shown to be outperformed by the CNN

| Model         | Top-1     | Top-5     |
| ------------- | --------- | --------- |
| Sparse coding | 47.1%     | 28.2%     |
| SIFT + FVs    | 45.7%     | 25.7%     |
| CNN           | **37.5%** | **17.0%** |

- They also tested it on the ImageNet dataset (Fall 2009) -> here they were only given half the dataset and the other half to test it. They got top-1 and top-5 error rates of 67.4% and 40.9%. In this they also used a sixth convolutional layer over the last pooling layer

| Model               | Top-1 (val) | Top-5 (val) | Top-5 (test) |
| ------------------- | ----------- | ----------- | ------------ |
| *SIFT + FVs*        | -           | -           | *26.2%*      |
| 1 CNN               | 40.7%       | 18.2%       | -            |
| 5 CNNs              | 38.1%       | 16.4%       | **16.4%**    |
| 1 CNN (pretrained)  | 39.0%       | 16.6%       | -            |
| 7 CNNs (pretrained) | 36.7%       | 15.4%       | **15.3%**    |
## Qualitative Evaluations
- Because of the limited connectivity of the GPUs, they have become specialised in different aspects of classification; GPU1 mostly only learned shaped and patterns while GPU2 is more focused on the colours
![[Pasted image 20260704193134.png|655]]

