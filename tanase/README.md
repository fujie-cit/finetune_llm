1. 下記の内容を data/ 以下に置く．

https://drive.google.com/drive/folders/1mMXl18AHy-BD9--hn1Hf9kQ7QL994RsC?usp=sharing

2. ```make_data.ipynb``` を実行し， ```data_all.xlsx``` を作成する．

3. ```make_dataset_tokenized_simple.py``` を実行し```dataset_tokenized_simple``` ディレクトリを作成する．

4. ```train_lora.py``` を実行し，学習を行う．
結果は評価値が高いチェックポイントが3つ，```lora-calm-3b-results``` ディレクトリ内に保存される．
また，最終的な結果は，```lora-calm-3d```に保存される．

5. ```demo_lora.py``` を実行し，言い換えを実行する．
結果は，```train-results.xlsx```，```valid-results.xlsx```に書き出される．



