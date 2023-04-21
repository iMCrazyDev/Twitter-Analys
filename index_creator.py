import os
import sqlite3
import sys
import random
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import random
import torch
from pmdarima.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from transformers import TextClassificationPipeline, AutoModelForSequenceClassification, AutoTokenizer
from sklearn.cluster import KMeans

if len(sys.argv) < 5:
    print('folder or db name  or username or posts number is not selected')
    exit(0)

n_posts_per_acc = int(sys.argv[4])
def get_acc_post_bert(acc):
    # Обход директорий и файлов внутри
    for root, dirs, files in os.walk(folder_path + '\\' + acc):
        for file in files:
            if file.endswith(".csv"):
                subdir_name = os.path.basename(root)
                path = os.path.join(root, file)
                print(path)
                with open(path, 'r', errors='ignore') as csv:
                    df = pd.read_csv(csv)
                    dates = list(df['date'])
                    raws = list(df['rawContent'])
                    return zip(dates, raws)


# Функция для получения всех чисел из info.txt файлов
def get_numbers_from_files(folder_path):
    numbers = []

    # Обход директорий и файлов внутри
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file == "info.txt":
                subdir_name = os.path.basename(root)
                with open(os.path.join(root, file), "r") as f:
                    number = int(f.read().strip())
                    numbers.append((subdir_name, number))

    return numbers


# Функция для кластеризации чисел на 3 кластера
def cluster_numbers(numbers):
    kmeans = KMeans(n_clusters=3)
    clustered_numbers = kmeans.fit_predict([[n[1]] for n in numbers])

    # Упорядочиваем кластеры по величине
    cluster_order = sorted(range(3), key=lambda x: kmeans.cluster_centers_[x])
    ordered_clustered_numbers = [cluster_order.index(x) for x in clustered_numbers]

    return ordered_clustered_numbers


# Основной код
folder_path = sys.argv[1]
numbers = get_numbers_from_files(folder_path)
clustered_numbers = cluster_numbers(numbers)
analyse_name = sys.argv[3]
clust_n = 0
# Вывод результатов кластеризации
for (subdir_name, number), cluster in zip(numbers, clustered_numbers):
    size = ["маленькие", "средние", "большие"][cluster]
    if subdir_name == analyse_name:
        clust_n = number
    print(f"Число {number} из поддиректории '{subdir_name}' принадлежит к кластеру {size}")


accs = []
for (subdir_name, number), cluster in zip(numbers, clustered_numbers):
    if cluster == clust_n:
        accs.append((subdir_name, number))

db = sqlite3.connect(sys.argv[2],  detect_types=sqlite3.PARSE_DECLTYPES)
cursor = db.cursor()
max_date = cursor.execute("SELECT max(date) FROM Charts").fetchall()[0][0]
max_date = datetime.strptime(max_date, "%Y-%m-%d %H:%M:%S")
max_date -= timedelta(days=2)
dates = cursor.execute("SELECT date, capitalizaiton FROM Charts").fetchall()
add_to_pd_data = []
for dt in dates:
    st = dt[0]
    dtime = datetime.strptime(st, "%Y-%m-%d %H:%M:%S")
    add_to_pd_data.append([dtime, 0, dt[1]])
df = pd.DataFrame(add_to_pd_data,
                  columns=['date', 'labelscore', 'value']).set_index('date')

df2 = df.copy()
df3 = df.copy()

print('Running NLP....')
model_name = "ElKulako/cryptobert"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = 3)
pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, max_length=64, truncation=True, padding = 'max_length')


for acc in accs:
    posts_ = list(get_acc_post_bert(acc[0]))
    posts_text = []
    posts = []
    for post in posts_:
        dt = datetime.strptime(post[0], "%Y-%m-%d %H:%M:%S%z")
        if dt.timestamp() >= max_date.timestamp():
            continue
        posts.append([dt, post[1]])
        break
    posts = random.choices(posts_, k=n_posts_per_acc)
    for x in posts:
        posts_text.append(x[1])

    print('pipe', len(posts_text))
    preds = pipe(posts_text)
    print(len(preds))

    for i in range(len(posts)):
        try:
            pred = preds[i]
            if pred['label'] == 'Neutral':
                continue
            next_date_index = df2.index.get_loc(posts[i][0],
                                                method='backfill')
            score = -1 * pred['score']
            if pred['label'] == 'Bullish':
                score = pred['score']

            df2.iloc[next_date_index, df.columns.get_loc('labelscore')] = score
            if posts[i] != analyse_name:
                df3.iloc[next_date_index, df.columns.get_loc('labelscore')] = score
        except Exception as e:
            print(e)

print('Running ML....')

def ml(data):
    train, test = train_test_split(data, test_size=0.2)

    model = SARIMAX(train["value"],
                    exog=train[["labelscore"]],
                    order=(2, 1, 1))

    fitted_model = model.fit()

    forecast = fitted_model.predict(start=test.index[0], end=test.index[-1], exog=test[["labelscore"]])
    mae = mean_absolute_error(test["value"], forecast)
    return mae

mae_with = ml(df2)
mae_without = ml(df3)
final_index = mae_with - mae_without
print(f'MAE: {mae_with} - {mae_without} = {final_index}')
print(f'Final index = {final_index}')