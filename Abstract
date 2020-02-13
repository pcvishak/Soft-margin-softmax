{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Based on the paper http://www.cbsr.ia.ac.cn/users/xiaobowang/papers/SM-Softmax.pdf"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Why\n",
    "Most of the classifiers work on softmax loss. We need a loss that discriminate the features more."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Related Work\n",
    "\n",
    "Contrastive loss, Triplet loss, Center loss, L-Softmax loss"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Idea\n",
    "\n",
    "For usual Softmax, we have the following equation:\n",
    "    $L_{Softmax} = -log(\\frac{e^{W^T_{y_i}x_i}}{\\sum_{j}^{K}e^{W^T_{j}x_i}}) = -\\log (\\frac{e^{\\|{W_{y_i}}\\|\\|{x_i}\\|\\cos(\\theta_{y_i})}}{\\sum_{j}^{K}e^{\\|{W_{j}}\\|\\|{x_i}\\|\\cos(\\theta_{j})}})$\n",
    "    \n",
    "L-Softmax loss employed a hard angle margin constraint on the softmax loss:\n",
    "$L_{L-Softmax} = -\\log (\\frac{e^{\\|{W_{y_i}}\\|\\|{x_i}\\|\\cos(a\\theta_{y_i})}}{e^{\\|{W_{y_i}}\\|\\|{x_i}\\|\\cos(a\\theta_{y_i})} + \\sum_{j\\neq y_i}^{K}e^{\\|{W_{j}}\\|\\|{x_i}\\|\\cos(\\theta_{j})}})$\n",
    "\n",
    "To make the classification more rigorous, L-Softmax introduces angle margin as:\n",
    "$\\|W_1\\|\\|x\\|cos(\\theta_1) \\geq \\|W_1\\|\\|x\\|cos(a\\theta_1) \\geq \\|W_1\\|\\|x\\|cos(\\theta_2)$\n",
    "\n",
    "For soft margin we have:\n",
    "$W^T_1x \\geq W^T_1x - m \\geq W^T_2x$\n",
    "\n",
    "Soft margin softmax loss is given by:\n",
    "$L_{Softmax} = -log(\\frac{e^{W^T_{y_i}x_i - m}}{e^{W^T_{y_i}x_i - m} + \\sum_{j\\neq_i}e^{W^T_{j}x_i}})$"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Algorithm\n",
    "\n",
    "Algorithm 1: Training a L-layers CNN supervised by SM-softmax loss.\n",
    "\n",
    "Input: Training data ${x_i}$. Initialized parameters $\\theta$ in convolution layers.\n",
    "\n",
    "Parameters W in SM-Softmax loss layer. Hyperparameter m.\n",
    "        \n",
    "while not converged do\n",
    "\n",
    "    Compute the forward propagation by the modified soft-margin Softmax ;\n",
    "    Compute the standard backward propagation;\n",
    "    Update the parameters W;\n",
    "    Update the parameters $\\theta$.\n",
    "    \n",
    "end\n",
    "\n",
    "Output: The parameters $\\theta$ and the weight W"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 2
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython2",
   "version": "2.7.15rc1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
