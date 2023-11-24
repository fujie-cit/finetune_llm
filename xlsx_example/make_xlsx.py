import json

# 生データの読み込み
json_path='/autofs/diamond2/share/users/fujie/share/twitterJSA_data.json'
data_raw = json.load(open(json_path))

# データ数の確認
print(f"データ総数: {len(data_raw)}")



# id はツイートのID
# topic はトピックID
#   10000: エクスペリア，Xperia
#   10001: アクオス，Aquos
#   10002: アクオス，Aquos
#   10020: ココロボ（シャープが開発した自動掃除機）
#   10021: iPhone
#   10022: パナソニック，Panasonic
#   10024: コンビニにあるコピー機
#   10025: ルンバ，Rommba
#   10026: シャープ
# label はタグ付けされたラベル
#   1列目: ポジティブ&ネガティブ，このツイートは（ジャンル）についてポジティブなこともネガティブなことも書かれている
#   2列目: ポジティブ，このツイートは（ジャンル）についてポジティブなことが書かれている
#   3列目: ネガティブ，このツイートは（ジャンル）についてネガティブなことが書かれている
#   4列目: ニュートラル，このツイートは（ジャンル）についてポジティブなこともネガティブなことも書かれていない
#   5列目: 無関係，このツイートは（ジャンル）に関係が無い
#  0は該当しない，1は該当するという意味です．  
#  二つの列で該当するというのは，作業者の投票が同率で1位になった場合です．
# text はツイートの本文

# ツイートの本文が無い，または本文の長さが0のときには，データから除外する
data = []
for d in data_raw:
    if 'text' in d and len(d['text']) > 0:
        data.append(d)

# データ数の確認
print(f"フィルタ済データ総数: {len(data)}")

# ラベルの決定
#   ポジティブ&ネガティブが 1，もしくは，ポジティブとネガティブが両方1のものは positive and negative
#   ポジティブが 1 のものは positive
#   ネガティブが 1 のものは negative
#   ニュートラルが 1 のものは neutral
#   その他は unrelated
# とする
for d in data:
    label = d['label']
    output = 'unrelated'
    if label[0] == 1 or (label[1] == 1 and label[2] == 1):
        output = 'positive and negative'
    elif label[1] == 1:
        output = 'positive'
    elif label[2] == 1:
        output = 'negative'
    elif label[3] == 1:
        output = 'neutral'
    d['output'] = output

# 必要なキーを取り出して名前を変更する
data_all = []
for d in data:
    d_new = {
        '入力': d['text'],
        '出力': d['output'],
    }
    data_all.append(d_new)

# データをシャッフルする
import random
random.shuffle(data_all)

# データを分割する（学習用:検証用:テスト用 = 8:1:1）
n = len(data_all)
n_train = int(n * 0.8)
n_valid = int(n * 0.1)
n_test = n - n_train - n_valid
data_train = data_all[:n_train]
data_valid = data_all[n_train:n_train+n_valid]
data_test = data_all[n_train+n_valid:]

# データを保存する
import pandas as pd
df_train = pd.DataFrame(data_train)
df_valid = pd.DataFrame(data_valid)
df_test = pd.DataFrame(data_test)

df_train.to_excel('train.xlsx', index=False)
df_valid.to_excel('valid.xlsx', index=False)
df_test.to_excel('test.xlsx', index=False)



