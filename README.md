### Dataset Preparation

There are two datasets used in this project:  [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) and [TinyStories-Instruct](https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct). Both datasets can be found on Hugging Face. To download them to your local disk, please refer to the `model/preprocess.ipynb` file for more details.

### Model Selection

[The paper](https://arxiv.org/abs/2305.07759) offers 14 different models with varying pretrain datasets and model sizes. The GPT-Neo architecture is used in the paper, with a window size of 256 and a context length of 512. The GPT-Neo tokenizer is used, but only the top 10K most common tokens are kept. 

In this repository, we use the GPT2 architecture and its tokenizer, which does not use local attention and sets the context length to 1024. For more details on the model sizes, please refer to the `model/model_size` file.

### Pretraining

To train the model using the TinyStories dataset, run the `model/train.sh` script. [GPT3](https://arxiv.org/pdf/2005.14165.pdf) uses the Adam optimizer with $\beta_1 = 0.9, \beta_2 = 0.95, \epsilon = 10^{−8}$. This repository follows the same optimization settings and sets the learning rate to $lr = 10^{−3}$. The training logs are saved in `log`, you can use `log/visualization.ipynb` to plot loss curve.


TODO : For training using the TinyStories-Instruct dataset, [The paper](https://arxiv.org/abs/2305.07759) states that a random subset of [Features, Summary, Sentence, Words] is chosen and combined into an instruction, followed by the story itself. However, this part has not been accomplished yet in this repository.

