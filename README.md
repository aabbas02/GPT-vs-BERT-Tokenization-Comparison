# GPT vs BERT Tokenization Comparison [Tutorial]

The purpose of this repository is to  compare the tokenization performance of [Open AI's GPT](https://platform.openai.com/tokenizer) and Google's BERT tokenizers, both available from [HuggingFace](https://huggingface.co/docs/transformers/v4.48.0/en/main_classes/tokenizer#transformers.PreTrainedTokenizer).
 
Parts of this code (the transformer architecture, training, and evaluation) are borrowed from [harvardnlp/annotated-transformer](https://github.com/harvardnlp/annotated-transformer). Below is a breakdown of the files included in this repository and example usage.

# Files and Description
There are 5 files in this repositiory, each of which is described below.

**1. mainTokenComparison.ipynb:** jupyter notebook that outputs the two English translations of the same German sentence by the two encoder-decoder transfomer models trained using Bert or GPT tokenizers.

**2. mainTrainTransformer.py:**  python file that trains and saves the transformer models for the  for either the GPT or BERT tokenizers for the German To English translation task on the WMT-2014 dataset. The vocabularies for these tokenizers are also saved.

**3. utilsTokenComparison.py:** python file  containing the helper functions used in mainTokenComparison.ipynb, described above.

**4. utilsTrainTransformer.py:** python file containing the helper functions used in mainTrainTransformer.py, described above.

**5. utilsTransformer.py:** python file containing the helper functions (mostly tansformer architecture declaration related) that are used in transformer training and tokenization comparison. 

# Example Usage
If trained models and libraries do not exist, run **mainTrainTrnasformer.py** once with `tokenizer = BERT` and once again with `tokenizer = GPT`. These runs will save the trained transformer models and the corresponding vocabularies. After the models and vocabularies are saved, run **mainTokenComparison.ipynb**

# Expected Output

Preparing Data ...
Comparing Model Outputs:

Example 0 ========

Source Bert Text (Input)        : <s> [CLS] Frauen gehen durch den Tie ##fs ##chn ##ee einen st ##eile ##n Ab ##hang hin ##unter . [SEP] </s>
Target Bert Text (Ground Truth) : <s> [CLS] Women walking through deep snow and down a steep hill . [SEP] </s>
Model Bert Output: <s> [CLS] Women walking down the br ##ush ##ing down a steep slope . [SEP] </s>
========

Source GPT Text (Input)        : <s> frau en ge hen dur ch den ti ef sch nee ein en ste il en ab hang hin un ter . </s>
Target GPT Text (Ground Truth) : <s> women walking through deep snow and down a steep hill . </s>
Model Gpt Output: <s> women walking down the deep snow on the top of a steep incline . </s>
========


Example 1 ========

Source Bert Text (Input)        : <s> [CLS] Eine Band spielt auf einer Frei ##licht ##b ##ühne am Fluss . [SEP] </s>
Target Bert Text (Ground Truth) : <s> [CLS] A band playing in an outdoor theater , along the river . [SEP] </s>
Model Bert Output: <s> [CLS] A band plays on a stage , a stage by the river . [SEP] </s>
========

Source GPT Text (Input)        : <s> eine band spi elt au f e iner fre ili ch t bu h ne am flu ss . </s>
Target GPT Text (Ground Truth) : <s> a band playing in an outdoor theater , along the river . </s>
Model Gpt Output: <s> a band is playing on an outdoor stage by the river . </s>
========


Example 2 ========

Source Bert Text (Input)        : <s> [CLS] Da sind viele Menschen in einem Gebäude und ein paar Menschen ko ##chen etwas zu esse ##n . [SEP] </s>
Target Bert Text (Ground Truth) : <s> [CLS] There are many people in a building with some people cooking food . [SEP] </s>
Model Bert Output: <s> [CLS] There is a lot of people cooking some food and some people cooking . [SEP] </s>
========

Source GPT Text (Input)        : <s> da s ind vi ele men s chen in e ine m ge bau de und e in pa ar men s chen ko chen et was zu ess en . </s>
Target GPT Text (Ground Truth) : <s> there are many people in a building with some people cooking food . </s>
Model Gpt Output: <s> there are many people in a building and some people cooking in a building . </s>
========


Example 3 ========

Source Bert Text (Input)        : <s> [CLS] Zwei Frauen bli ##cken auf viele Häuser hin ##ab . [SEP] </s>
Target Bert Text (Ground Truth) : <s> [CLS] Two women look out at many houses below . [SEP] </s>
Model Bert Output: <s> [CLS] Two women look down at many houses . [SEP] </s>
========

Source GPT Text (Input)        : <s> z wei frau en bli cken au f vi ele hau ser hin ab . </s>
Target GPT Text (Ground Truth) : <s> two women look out at many houses below . </s>
Model Gpt Output: <s> two women are looking down at many houses . </s>
========


Example 4 ========

Source Bert Text (Input)        : <s> [CLS] Zwei Jungen innerhalb eines Za ##unes spring ##en in die Luft und halten dabei einen Basketball . [SEP] </s>
Target Bert Text (Ground Truth) : <s> [CLS] Two boys inside a fe ##nce jump in the air while holding a basketball . [SEP] </s>
Model Bert Output: <s> [CLS] Two boys jump in the air while holding a basketball in the air . [SEP] </s>
========

Source GPT Text (Input)        : <s> z wei jun gen inner hal b e ines za un es spring en in die lu ft und hal ten da be i ein en basketball . </s>
Target GPT Text (Ground Truth) : <s> two boys inside a fence jump in the air while holding a basketball . </s>
Model Gpt Output: <s> two young boys holding a ball in the air while holding a basketball . </s>
========


Example 5 ========

Source Bert Text (Input)        : <s> [CLS] Zwei Frauen , eine aus Deutschland und eine aus China , treten bei einem Ring ##kampf auf einer Matt ##e gegen ##einander an . [SEP] </s>
Target Bert Text (Ground Truth) : <s> [CLS] 2 females , 1 from german ##y and 1 from China , compete in a wrestling match on a mat . [SEP] </s>
Model Bert Output: <s> [CLS] Two women , one of and a St ##itch is performing a wrestling on a mat . [SEP] </s>
========

Source GPT Text (Input)        : <s> z wei frau en , eine a us deu t sch land und eine a us china , tre ten be i e ine m ring ka mp f au f e iner mat te ge gene in ander an . </s>
Target GPT Text (Ground Truth) : <s> 2 females , 1 from germany and 1 from china , compete in a wrestling match on a mat . </s>
Model Gpt Output: <s> two women , one out of dirt and one in china are doing a match on a mat by a match . </s>
========





Email at aabbasi1@iastate.edu for any questions.
