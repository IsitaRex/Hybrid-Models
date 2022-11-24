# Hybrid Models using Pytorch
 
This practice will focus on supervised hybrid methods from fuzzy inference systems, passing through traditional unsupervised learning machines and supervised learning plus techniques of convolutional neural networks (CNN'S) for pattern recognition in high dimensions or patterns of high and low spatial frequencies. All this concepts will be applied to the MNIST dataset, a widely known dataset in the field of Machine Learning composed by  handwritten digits and commonly used for training various image processing systems.

## Installation instructions :computer:
To replicate the results follow this steps:

Clone the repository:
```
git clone https://github.com/IsitaRex/Hybrid-Models.git
```

Install the requirements list on your environment 
```
pip install -r requirements.txt
```
## Tasks  :crystal_ball:

### CNN (LeNet5):
The base code for this implementation can be found on this [link](https://towardsdatascience.com/implementing-yann-lecuns-lenet-5-in-pytorch-5e05a0911320)

To train a CNN run:

```
python main.py --epochs 10 --task Hybrid-Models-CNN --lr 0.001
```
### GAN:
The base code for this implementation can be found on this [link](https://debuggercafe.com/vanilla-gan-pytorch/)

To train a GAN run:

