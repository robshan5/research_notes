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
 - For this to work $\textbf{x}$ and $F$ need to be the same, but if they aren't, a linear projection just needs to be performed on $\textbf{x}$ to match the dimensions of $F$
 - While this is also just for a simple fully-connected layer, this can also work for a convolutional layer