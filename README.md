# 👋 CRAFT: Concept Recursive Activation FacTorization for Explainability (CVPR 2023) - update 2025, Master AI at Telecom Paris

This reproductivity project was realized for XAI courses of the specialized Master’s AI Data expert & MLops at Telecom Paris. 
This repository contains our works based on CRAFT's github and the code for paper : 

*Extending CRAFT: Cross-Architecture and Cross-Modal Concept Factorization*

*[begin update]*  

Voici la structure de dépôt:
- /test : notebooks de test pour différentes dataset (chat, bébé, daisy et voiture)
- craft_pytorch_DINOv2ClassifierHead.ipynb: Experimentation on ViT model (DINOv2), transformers architecture, dataset with images so no modification of the Craft class
- craft_pytorch_RoBERTaClassifer.ipynb: Experimentation on the language model RoBERTa fine-tuned with specifique dataset (classification of products reviews from e-commerce website) 
 
The Craft class was modified in this way :
- Explicit support for multiple input types for fit(): Images (4D tensor), re-calculated activations (2D), Text sequences (list of str)
- _batch_inference() adapted for texts sequences too
- estimate_importance_vector(): vectorized, faster version (used only for texts, _loop for images, existing method)
- Added the token_heatmap() method: Allows you to associate concept weights with text tokens, useful for NLP interpretability.
This new version is backward compatible

*[end update]*

---

This repository contains code for the paper:

*CRAFT: Concept Recursive Activation FacTorization for Explainability*, Thomas Fel*, Agustin Picard*, Louis Bethune*, Thibaut Boissin*, David Vigouroux, Julien Colin, Rémi Cadène, Thomas Serre. CVPR 2023, [[arXiv]](https://arxiv.org/abs/2211.10154).

The code is implemented and available **for Pytorch & Tensorflow**. A notebook for each of them is available: 
- <img src="https://pytorch.org/assets/images/pytorch-logo.png" width="24px">[ Notebook for Pytorch](./craft_pytorch.ipynb)
- <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Tensorflow_logo.svg/230px-Tensorflow_logo.svg.png" width="24px"> [ Notebook for Tensorflow](./craft_tensorflow.ipynb).


# 🚀 Quick Start

Craft requires a version of python higher than 3.6 and several libraries like Numpy, also you will need either Tensorflow or Torch. Installation can be done using Pypi:

```bash
pip install Craft-xai
```

Now that Craft is installed, here is the basic example of what you can do. 
The API, whether for Tensorflow or Pytorch, is similar and only requires two hyperparameters. First, you need to load your models and a set of images from a class you want to explain (generally, try to have at least 500 images).

Once you have that, split your model into two parts (see the notebooks if necessary) to have two functions: $g$, which maps from the input to the feature space, and $h$, which maps from the feature space to your logits. Once you have done this, you are ready to instantiate CRAFT.

```python
from craft.craft_torch import Craft
# or
#from craft.craft_tf import Craft


craft = Craft(input_to_latent=g,
              latent_to_logit=h,
              number_of_concepts=10,
              patch_size=64,
              batch_size=64)
```

Now, you can fit CRAFT with your preprocessed images (make sure they are preprocessed according to your model).

```python
crops, crops_u, w = craft.fit(images_preprocessed)
importances = craft.estimate_importance(images_preprocessed, class_id=class_id) # the logit you want to explain
```

That's it! To learn how to visualize the results, refer to the notebooks that explain how to make the most of all the information returned by CRAFT.


<img src="./assets/craft.jpg" width="800px">
<img src="./assets/craft_core.jpg" width="800px">
<img src="./assets/craft_results.jpg" width="800px">

# Citation

```
@inproceedings{fel2023craft,
      title={CRAFT: Concept Recursive Activation FacTorization for Explainability},
      author={Thomas, Fel and Agustin, Picard and Louis, Bethune and Thibaut, Boissin and David, Vigouroux and Julien, Colin and Rémi, Cadène and Thomas, Serre},
      year={2023},
      booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}
}
```

# More about CRAFT

The code for the metrics and the other attribution methods used in the paper come from the [Xplique toolbox](https://github.com/deel-ai/xplique).

<a href="https://github.com/deel-ai/xplique">
    <img src="https://github.com/deel-ai/xplique/blob/master/docs/assets/banner.png?raw=true" width="500px">
</a>


Additionally, we have created a website called the [LENS Project](https://github.com/serre-lab/Lens), which features the 1000 classes of ImageNet.

<a href="https://github.com/serre-lab/Lens">
    <img src="https://serre-lab.github.io/Lens/assets/lens_intro.jpg" width="500px">
</a>


# Authors of the code

- [Thomas Fel](https://thomasfel.fr) - thomas_fel@brown.edu, PhD Student DEEL (ANITI), Brown University
- [Agustin Picard]() - agustin-martin.picard@irt-saintexupery.com, IRT Saint-exupéry, DEEL
- [Louis Béthune]() - louis.bethune@univ-toulouse.fr, PhD Student DEEL (ANITI)
- [Thibaut Boissin]() - thibaut.boissin@irt-saintexupery.com, IRT Saint-exupéry,
