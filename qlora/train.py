import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from peft import PeftModel, PeftConfig
from peft import get_peft_model, LoraConfig, TaskType
from peft import prepare_model_for_kbit_training

model_id = "cyberagent/open-calm-7b"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=bnb_config, device_map="auto"
)

model.gradient_checkpointing_enable()

# prepare_model_for_kbit_trainingではパラメータのfreezeを行いながら以下の3つの設定も行う
# 1. レイヤーノームをfp32にキャスト
# 2. 出力埋め込みレイヤーが勾配を必要とするように設定
# 3. 言語モデルのヘッドをfp32にアップキャスト
model = prepare_model_for_kbit_training(model)

# 以降はLoRAのときとほとんど同じ
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["query_key_value"],
    lora_dropout=0.05,
    bias="none",
    fan_in_fan_out=False,
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 4194304 || all params: 3654950912 || trainable%: 0.11475678062403483

training_args = TrainingArguments(
        output_dir='./output/qlora',
        save_total_limit=1,
        per_device_train_batch_size=8,
        num_train_epochs=1,
        remove_unused_columns=False,
        logging_steps=20,
        bf16=True,
        dataloader_num_workers=16,
        report_to="none",
)

trainer = Trainer(
        model=model,
        data_collator=collator,
        args=training_args,
        train_dataset=train_dataset,
    )
model.config.use_cache = False

trainer.train()
model.save_pretrained('./output/qlora')
