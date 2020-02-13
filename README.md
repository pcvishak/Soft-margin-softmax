
Based on the paper http://www.cbsr.ia.ac.cn/users/xiaobowang/papers/SM-Softmax.pdf

## Why
Most of the classifiers work on softmax loss. We need a loss that discriminate the features more.

## Related Work

Contrastive loss, Triplet loss, Center loss, L-Softmax loss

## Idea

For usual Softmax, we have the following equation:
    $L_{Softmax} = -log(\frac{e^{W^T_{y_i}x_i}}{\sum_{j}^{K}e^{W^T_{j}x_i}}) = -\log (\frac{e^{\|{W_{y_i}}\|\|{x_i}\|\cos(\theta_{y_i})}}{\sum_{j}^{K}e^{\|{W_{j}}\|\|{x_i}\|\cos(\theta_{j})}})$
    
L-Softmax loss employed a hard angle margin constraint on the softmax loss:
$L_{L-Softmax} = -\log (\frac{e^{\|{W_{y_i}}\|\|{x_i}\|\cos(a\theta_{y_i})}}{e^{\|{W_{y_i}}\|\|{x_i}\|\cos(a\theta_{y_i})} + \sum_{j\neq y_i}^{K}e^{\|{W_{j}}\|\|{x_i}\|\cos(\theta_{j})}})$

To make the classification more rigorous, L-Softmax introduces angle margin as:
$\|W_1\|\|x\|cos(\theta_1) \geq \|W_1\|\|x\|cos(a\theta_1) \geq \|W_1\|\|x\|cos(\theta_2)$

For soft margin we have:
$W^T_1x \geq W^T_1x - m \geq W^T_2x$

Soft margin softmax loss is given by:
$L_{Softmax} = -log(\frac{e^{W^T_{y_i}x_i - m}}{e^{W^T_{y_i}x_i - m} + \sum_{j\neq_i}e^{W^T_{j}x_i}})$

## Algorithm

Algorithm 1: Training a L-layers CNN supervised by SM-softmax loss.

Input: Training data ${x_i}$. Initialized parameters $\theta$ in convolution layers.

Parameters W in SM-Softmax loss layer. Hyperparameter m.
        
while not converged do

    Compute the forward propagation by the modified soft-margin Softmax ;
    Compute the standard backward propagation;
    Update the parameters W;
    Update the parameters $\theta$.
    
end

Output: The parameters $\theta$ and the weight W


```python

```
