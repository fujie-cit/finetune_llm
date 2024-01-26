import pandas as pd
from datasets import Dataset, DatasetDict

from transformers import AutoTokenizer
model_name = "cyberagent/open-calm-3b"
tokenizer = AutoTokenizer.from_pretrained(model_name)

CUTOFF_LENGTH = 512

def tokenize(prompt, tokenizer: AutoTokenizer, cutoff_length=CUTOFF_LENGTH):
    result = tokenizer(prompt + tokenizer.eos_token,
                       truncation=True, 
                       max_length=cutoff_length,
                       padding=True)
    return {
        'input_ids': result['input_ids'],
        'attention_mask': result['attention_mask'],
    }

def generate_prompt(data_point):
    return f"""{data_point['text']} 

### Response:
{data_point['output']}"""

dataset_dict = DatasetDict()
for split in ['train', 'valid']:
    df = pd.read_excel(f'./{split}.xlsx')
    # df.columns = ['text', 'output']
    dataset = Dataset.from_pandas(df)
    dataset = dataset.shuffle().map(lambda x: tokenize(generate_prompt(x), tokenizer))
    dataset_dict[split] = dataset

dataset_dict.save_to_disk('./dataset_tokenized')



