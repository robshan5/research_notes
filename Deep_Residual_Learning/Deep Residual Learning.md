# Abstract
- They are presenting an easier way to optimise deep neural networks
- They reformulate the layers as learning residual functions with reference to layer inputs instead of learning unreferenced functions
- They achieved a 3.57% error rate (top-5) on the ImageNet test set which was just three years after [[ImageNet_Classification#Results]] which was at around 17% error (top-5)
- Deep residual nets are the focus of this paper
# Introduction
- Depth is very important for the performance of a neural network - works done on vision models up to that point used deep networks
- When a network deepens, accuracy reaches a peak and then rapidly descends
- They talk about constructing a deep network from a shallower network by taking the layers from the shallow network and then the added layers are identity mapping (using skip connections to pass information between layers not adjacent to each other - hence *skipping* over different layers)
- Because this construction solution exists deeper models should not be able to outperform the shallower model since it is just a derivative of it but they say they have empirical evidence of the opposite 
-  They address the degradation problem by letting stacked layers fit a residual mapping\; denoting the desired mapping as $H(\textbf{x})$ and let the stacked layers fit $F(\textbf{x}) := H(\textbf{x})- \textbf{x}$, the original mapping is recast into $F(\textbf{x})+\textbf{x}$ 
- They hypothesise that it is easier to optimise the residual mapping than the original
- In this paper, they show that 
 1.  the residual networks are easier to optimise while also having a lower training error compared to normal networks
 2. the residual networks gain more accuracy the deeper they become, while normal networks don't
# Related Work
- Residual networks have already been done (VLAD and Fischer Vector) and have proven to simplify the optimisation process
- Shortcut connections have also been implemented before: "highway connections" are similar to the identity mapping used in this paper, but they were data-dependent. They were also sometimes closed off to represent *non-residual* functions, whereas in this paper the connections are never closed.
- These previous attempts at skip connections also never showed any increase in accuracy with increased depth (which this paper is trying to prove)
# Deep Residual Learning
## Residual Learning
- If non linear layers of the network can asymptotically approximate complicated functions then they should be able to approximate the residual functions $H(\textbf{x}) - \textbf{x}$, so we let the stacked layers approximate $F(\textbf{x}) := H(\textbf{x}) - \textbf{x}$, thus the original function becomes $F(\textbf{x}) + \textbf{x}$.
- They hypothesise that the ease of learning this residual function may be different to learning $H(\textbf{x})$ 
- They think that the degradation problem may be caused from solvers finding it difficult to approximate the residual function, so by changing the function, the solvers may find it easier to approach it 
## Identity Mapping by Shortcuts
- They adopt residual learning to every few stacked layers
![[Pasted image 20260712215116.png|490]] 
- A building block is defined as 
 $$
 \textbf{y} = F(\textbf{x}, \{W_i\}) + \textbf{x}
 $$
 where $\textbf{x}$ and $\textbf{y}$ are the input and output variables and the function $F(\textbf{x}, \{W_i\})$ 
 - This equation avoids adding extra computation complexity and more parameters (nodes)
 - For this to work $\textbf{x}$ and $F$ need to be the same, but if they aren't, a [[linear projection]] just needs to be performed on $\textbf{x}$ to match the dimensions of $F$ ^mapping
 - While this is also just for a simple fully-connected layer, this can also work for a convolutional layer
# Architectures
### Plain Networks
- 34 weighted layers
- 3x3 filters in convolutional layers
- the layers have the same number of filters for the same size feature map
- if the feature map size is halved, the filter count is doubled
- Used as a control for the [[#Experiments]]
### Residual Network
- Same as the plain network but this shortcut connections (always skips two layers)
- Some layers have different dimensions so the [[#^mapping|mapping]] is used or the extra dimensions are padded with 0s ^0padding
## Implementation
- Image is randomly cropped (or it's horizontal flip) to fit the 224x224 size
- Standard colour augmentation is used
- Use batch normalisation after each convolution and before each activation
# Experiments
## ImageNet Classification
- ImageNet 2012 is used, same as [[ImageNet_Classification|AlexNet]]
### Plain Networks
- 18-layer and then 34-layers
- deeper network has a higher validation error (28.54% compared to 27.94% in top-1 error) & higher training error
### Residual Networks
- the 34-layer network outperformed the 18-layer

| Network   | ResNet    | Plain |
| --------- | --------- | ----- |
| 18 layers | 27.88     | 27.94 |
| 34 layers | **25.03** | 25.03 |

- Used identity mapping for the shortcuts and [[#^0padding|0 padding]] for the extra dimensions
#### Identity vs. Projection Shortcuts
- Using shortcuts improves the performance of the model although the type of projection shortcut doesn't really matter
#### Deeper Bottleneck Architectures
- parameter-free identity shortcuts are important for the bottleneck architectures (skips 1x1, 3x3, 1x1 layers) -> keeps time complexity low

#### 50-layer ResNet
- replace 2-layer block with the bottleneck block in the 34-layer net -> creates a 50-layer net
- They created 101-layer and 152 layer that each improved accuracy meaning the deeper a ResNet goes, the better it will perform