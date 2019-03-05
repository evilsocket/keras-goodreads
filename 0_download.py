import os
import pandas as pd 
from urllib.request import urlopen
from bs4 import BeautifulSoup

data = pd.read_csv('./goodreads_library_export.csv')

for book_id in data['Book Id']:
    filename = os.path.join('data', '%s.txt' % book_id)
    if not os.path.exists(filename):
        url = "https://www.goodreads.com/book/show/%d" % book_id
        print("downloading %s to %s ..." % (url, filename))
        html = urlopen(url).read().decode('utf-8')
        soup = BeautifulSoup(html, "html.parser")
        div = soup.find("div", {"id": "description"})
        if div is None:
            text = ""
        else:
            text = div.text.replace("...more", "").strip()
        with open(filename, 'wt') as fp:
            fp.write(text)
