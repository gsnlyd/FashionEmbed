## FashionEmbed

Project for training embedding networks based on [Learning-Similarity-Conditions](https://github.com/rxtan2/Learning-Similarity-Conditions).

### Usage

Train model: `python trainembed.py -h`

Output embeddings: `python savemasks.py -h`


### Custom triplet datasets

The training script expects a dataset directory which contains the following:

- `/images` - A directory containing JPG images.
- `triplets.txt` - A file containing three image names (without extension) per line, separated by spaces.
