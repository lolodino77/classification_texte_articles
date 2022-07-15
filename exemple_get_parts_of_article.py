import requests
from bs4 import BeautifulSoup
import html2text
from lib_scraping import *

url = "https://parlafoi.fr/2019/06/08/thomas-daquin-sur-la-puissance-de-dieu-2/"
page = requests.get(url=url)
soup = BeautifulSoup(page.content, 'html.parser')
text = str(soup)
text = text.replace("\n", " ")
text = text.replace("</p>", "</p>\n\n")
text = text.replace("<li>", "<p>")
text = text.replace("</li>", "</p>\n\n")
text = html2text.html2text(text)

output_filename = "corpus_philosophy.txt"
file_open_mode = "w"
get_parts_of_article(url, output_filename, file_open_mode)

