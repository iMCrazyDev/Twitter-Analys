import sys

import pandas as pd
import snscrape.modules.twitter as sntwitter
import itertools
from pathlib import Path


def parse(acc):
    query = "from:" + acc[0]
    scraped_tweets = sntwitter.TwitterSearchScraper(query).get_items()
    df = pd.DataFrame(scraped_tweets)
    pth = "twitterparsing/" + acc[0] + "/"
    Path(pth).mkdir(parents=True, exist_ok=True)
    with open(pth + 'info.txt', 'w') as the_file:
        the_file.write(acc[1])
    df.to_csv(pth + "tweets.csv")


if len(sys.argv) < 2:
    print('file is not selected')
    exit(0)

with open(sys.argv[1]) as file:
    print(f'parsing from file {sys.argv[1]}')
    for line in file:
        ln = line.strip().split()
        parse(ln)

print('done!')
