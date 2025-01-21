# GPT vs BERT Tokenization Comparison

The purpose of this repository is to  compare the tokenization performance of Open AI's GPT and Google's BERT tokenizers, both available from [HuggingFace](https://huggingface.co/docs/transformers/v4.48.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer).
 
Parts of this code (the transformer architecture, training, and evaluation) are borrowed from [harvardnlp/annotated-transformer](https://github.com/harvardnlp/annotated-transformer). Below is a breakdown of the files included in this repository and example usage:

# Files
There are 5 files in this repositiory, each of which is described below.

**1. mainTokenComparison.ipynb:** jupyter notebook that outputs the English translation of the same German sentence by the two encoder-decoder transfomer models trained using Bert or GPT tokenizers.

**2. mainTrainTransformer.py:** main file that trains and saves the transformer model for the German To English task on the WMT-2014 dataset, for either the GPT or BERT tokenizer. The vocabularies for theset tokenizers are also saved.

**3. utilsTokenComparison.py:** This file contains the helper functions used in mainTokenComparison.ipynb, described above.

**4. utilsTrainTransformer.py:** This file contains the helper functions used in mainTrainTransformer.py, described above.

**5. utilsTransformer.py:** This file contains the helper functions (mostly tansformer architecture declaration related) that are used in transformer training and tokenization comparison. 

# Example Usage
If trained models and libraries do not exist, run mainTrainTrnasformer.py once with tokenizer = BERT and once again with tokenizer = GPT. These runs will save the trained transformer models and the corresponding vocabularies. After the models and vocabularies are saved, run mainTokenComparison.ipynb
