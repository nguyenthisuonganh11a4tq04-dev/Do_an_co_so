import pandas as pd
import json
import math

def clean_str(val):
    if pd.isna(val): return ""
    return str(val).strip()

def clean_int(val):
    if pd.isna(val): return 1
    try:
        return int(float(val))
    except:
        return 1

# Process Confession Data
df_conf = pd.read_excel('data/TONG_HOP .xlsx')
# drop na sentences
df_conf = df_conf.dropna(subset=['sentence'])

# Standardize sentiment labels
def std_sent(x):
    s = str(x).strip().lower()
    if s in ['positive', 'pos', 'tích cực', '1', '1.0']: return 'positive'
    if s in ['negative', 'neg', 'tiêu cực', '-1', '-1.0']: return 'negative'
    return 'neutral'

df_conf['sentiment'] = df_conf['sentiment'].apply(std_sent)

stats = df_conf['sentiment'].value_counts().to_dict()
# default missing keys to 0
for k in ['positive', 'negative', 'neutral']:
    if k not in stats: stats[k] = 0

print("Confession Stats:", stats)

# Take all samples
samples = []
for label in ['positive', 'negative', 'neutral']:
    df_label = df_conf[df_conf['sentiment'] == label]
    for _, row in df_label.iterrows():
        samples.append({
            'sentiment': label,
            'sentence': clean_str(row['sentence'])
        })

# Process Teencode Data
df_tc = pd.read_excel('data/teencode_final.xlsx')
tc_data = []
for _, row in df_tc.iterrows():
    if pd.isna(row['Teencode']): continue
    tc_data.append({
        'teencode': clean_str(row['Teencode']),
        'nghia': clean_str(row['Nghĩa']),
        'dang_chuan': clean_str(row['Dạng chuẩn hóa']),
        'cam_xuc': clean_str(row['Nhãn cảm xúc']),
        'do_manh': clean_int(row['Độ mạnh (1=Nhẹ, 2=TB, 3=Mạnh)']),
        'loai': clean_str(row['Loại teencode']),
        'ngu_canh': clean_str(row['Ngữ cảnh sử dụng'])
    })

print("Teencode count:", len(tc_data))

# Write to data.js
with open('data.js', 'w', encoding='utf-8') as f:
    f.write('const CONFESSION_STATS = ' + json.dumps(stats, ensure_ascii=False) + ';\n\n')
    f.write('const CONFESSION_SAMPLE = ' + json.dumps(samples, ensure_ascii=False) + ';\n\n')
    f.write('const TEENCODE_DATA = ' + json.dumps(tc_data, ensure_ascii=False) + ';\n')

print("Successfully wrote data.js")
