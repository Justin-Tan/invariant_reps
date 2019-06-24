# Invariant Representations
Learning invariant representations with mutual information regularization.

<img src="show/decision_no_pen.png" width="425"/> <img src="show/decision_with_pen.png" width="425"/> 

* [Toy Example](https://nbviewer.jupyter.org/github/Justin-Tan/invariant_reps/blob/master/notebooks/toy_MI.ipynb)
* [Toy Example on Binder](https://hub.mybinder.org/user/justin-tan-invariant_reps-iqsxl56t/notebooks/notebooks/toy_MI.ipynb)

## Usage
The code depends on [Tensorflow 1.13](https://github.com/tensorflow/tensorflow)
```
# Check command line arguments
$ python3 train.py -h
# Run, e.g.
$ python3 train.py -i /my/training/data -test /my/testing/data --name my_model -lambda 10 -MI -kl
```
To enable adversarial training mode based on a variant of the method proposed in [Louppe et. al.](https://arxiv.org/abs/1611.01046), use `adv_train.py` in place of `train.py` and enable `use_adverary = True` in the `config` file. 

## Regularization Methods
This method is essentially based around adding penalties to the objective function that penalize some sort of divergence between the joint distribution of an intermediate representation of the data found by passing the data through a neural network, and the sensitive variables. See the presentation below for further details. There are several different penalties implemented, the recommended one that has been found to be the most effective is the `kl_update` penalty that is analogous to the generator update rule proposed in [`arXiv:1610.04490`](https://arxiv.org/abs/1610.04490). You can also enable adversarial training, proposed in [`arXiv:1611.01046`](https://arxiv.org/abs/1611.01046) to attempt decorrelation.

## Extensions
The network architecture is kept modular from the remainder of the computational graph. For ease of experimentation, the codebase will support any arbitrary architecture that yields logits in the context of binary classification. In addition, the adversarial training procedure can interface with any arbitrary network architecture. To swap out the network for your custom one, create a `@staticmethod` under the `Network` class in `network.py`:

```python
@staticmethod
def my_network(x, config, **kwargs):
    """
    Inputs:
    x: example data
    config: class defining hyperparameter values

    Returns:
    network logits
    """

    # To prevent overfitting, we don't even look at the inputs!
    return tf.random_normal([x.shape[0], config.n_classes], seed=42)
```
Now open model.py and edit one of the first lines under the Model init:
```python
class Model():
    def __init__(self, **kwargs):

        arch = Network.my_network
        # The rest of computational graph defined here
        # (You shouldn't need to do anything else)
```
### Monitoring / checkpoints
Tensorboard summaries are written periodically to `tensorboard/` and checkpoints are saved to `checkpoints/` every epoch.

### Dependencies
* Python 3.6 + Certain packages available via the [standard Anaconda distribution](https://www.anaconda.com/distribution/)
* [Pandas](https://pandas.pydata.org/)
* [TensorFlow 1.13](https://github.com/tensorflow/tensorflow)
* [Tensorflow Probability](https://www.tensorflow.org/probability)

### Resources / Related Work

* [Slides describing the method](https://indico.cern.ch/event/766872/contributions/3357989/)
* [Learning to pivot with adversarial networks](https://arxiv.org/abs/1611.01046)

### Future Work
* Increase stability of MI estimator.
* Include HEP example.

### Contact
Feel free to open an issue or ping [firstname.lastname@coepp.org.au] for any questions.

