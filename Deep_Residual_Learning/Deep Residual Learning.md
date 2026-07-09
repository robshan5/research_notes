# Abstract
- They are presenting an easier way to optimise deep neural networks
- They reformulate the layers as learning residual functions with reference to layer inputs instead of learning unreferenced functions
- They achieved a 3.57% error rate (top-5) on the ImageNet test set which was just three years after AlexNet which was at around 12% error (top-5)
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