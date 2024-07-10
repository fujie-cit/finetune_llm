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
    if data_point['関係条件'].startswith('先生'):
        role = "先生"
    else:
        role = "後輩"
    if data_point['口調条件'].startswith('敬語'):
        tone = "敬語"
    else:
        tone = "タメ口"

    # roleによって少し様子が変わりそうなので入れることにしました．
    # toneは使わない方がよさそうなので外しました．

    prompt = f"""role: {role}
content: {data_point['内容']} 

### Response:
{data_point['添削']}"""
    print(prompt)
    return prompt

dataset_dict = DatasetDict()
for split in ['train', 'valid']:
    df = pd.read_excel(f'./{split}.xlsx')
    # dfから話者が"S"のもののみ抽出
    df = df[df['話者'] == 'S']
    dataset = Dataset.from_pandas(df)
    dataset = dataset.shuffle().map(lambda x: tokenize(generate_prompt(x), tokenizer))
    dataset_dict[split] = dataset

dataset_dict.save_to_disk('./dataset_tokenized_simple')



