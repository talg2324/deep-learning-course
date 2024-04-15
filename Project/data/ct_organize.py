import os
from tqdm.auto import tqdm
import pandas as pd

TRAIN_TEST_SPLIT = 0.8
TRAIN_PATH = './data/ct-rsna/train'
VALIDATION_PATH = './data/ct-rsna/validation'

if not os.path.exists(TRAIN_PATH):
    os.mkdir(TRAIN_PATH)
if not os.path.exists(VALIDATION_PATH):
    os.mkdir(VALIDATION_PATH)

all_files = os.listdir('./data/ct-rsna')
df = pd.read_csv('./data/ct-rsna/stage_2_train.csv')

# Squash class labels into single row
df['class'] = df['ID'].apply(lambda x: x.split('_')[-1]) 
df['ID'] = df['ID'].apply(lambda x: '_'.join(x.split('_')[:-1]))
df = df.pivot_table(columns='class', index='ID', values='Label', aggfunc='sum')
df.reset_index(inplace=True)

# Only keep relevant entries
df = df[df['ID'].isin([f.rstrip('.npy') for f in all_files])]

# No multi-class entries
df['sum'] = df.drop(['any', 'ID'], axis=1).sum(axis=1)
df = df[df['sum'] < 2]

train_data = df.sample(frac=TRAIN_TEST_SPLIT)
test_data = df.drop(train_data.index)

train_data.to_csv(os.path.join(TRAIN_PATH, 'train_set.csv'))
test_data.to_csv(os.path.join(VALIDATION_PATH, 'validation_set.csv'))

for idx, row in tqdm(train_data.iterrows()):
    id = row['ID']
    src = f'./data/ct-rsna/{id}.npy'
    dst = f'{TRAIN_PATH}/{id}.npy'
    os.system(f'cp {src} {dst}')

for idx, row in tqdm(test_data.iterrows()):
    id = row['ID']
    src = f'./data/ct-rsna/{id}.npy'
    dst = f'{VALIDATION_PATH}/{id}.npy'
    os.system(f'cp {src} {dst}')
