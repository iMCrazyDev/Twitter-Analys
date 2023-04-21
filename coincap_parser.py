import requests as requests
from datetime import datetime, timedelta, timezone
import time
from time import sleep
from pathlib import Path
import sys
import json
import shutil
import sqlite3


def get_data(currency, start, end):
    headers = {'Authorization': 'Bearer 52b289e2-26cb-44e5-8bbd-89de373b8cbc'}
    resp = requests.get(f'https://api.coincap.io/v2/assets/{currency}/history?interval=m5&start={start}000&end={end}000', headers=headers)
    return resp.text


if len(sys.argv) < 3:
    print('file or db name is not selected')
    exit(0)

source_filename = 'template.sqlite'
destination_filename = f'{sys.argv[2]}'
shutil.copy(source_filename, destination_filename)

conn = sqlite3.connect(destination_filename)
cursor = conn.cursor()

with open(sys.argv[1]) as file:
    for line in file:
        dt = datetime(2023, 1, 5)
        end = datetime(2023, 1, 10)
        ind = 0
        while dt < end:
            fut = dt + timedelta(days=5)
            res = get_data(line.strip(), str(int(dt.replace(tzinfo=timezone.utc).timestamp())), str(int(fut.replace(tzinfo=timezone.utc).timestamp())))
            # pth = f"{line.strip()}_charts/{ind} {dt.year}-{dt.month}-{dt.day} to {dt.year}-{dt.month}-{dt.day}/"
            # Path(pth).mkdir(parents=True, exist_ok=True)
            # with open(pth + 'data.txt', 'w') as the_file:
            #     the_file.write(res)
            # dt = fut
            # print(dt)

            data = json.loads(res)['data']

            for item in data:
                supply = float(item['circulatingSupply'])
                price = float(item['priceUsd'])
                capitalization = supply * price
                date = item['date'].replace('.000Z', '').replace('T', ' ')
                cursor.execute('SELECT capitalizaiton FROM Charts WHERE date = ?', (date,))
                result = cursor.fetchone()

                capitalization += result[0]
                cursor.execute('UPDATE Charts SET capitalizaiton = ? WHERE date = ?',
                               (capitalization, date))

            ind += 1
            sleep(0.001)

conn.commit()

conn.close()
print(f'parsed, saved to datebase {destination_filename}')


print('done')
