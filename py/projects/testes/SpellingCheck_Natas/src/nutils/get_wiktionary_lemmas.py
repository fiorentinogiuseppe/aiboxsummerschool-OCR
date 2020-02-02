import requests
from bs4 import BeautifulSoup
from mikatools import *
import json

url = "https://en.wiktionary.org/wiki/Category:Portuguese_lemmas"
base_url = "https://en.wiktionary.org/"


def get_pages(url):
	all_lemmas = []
	while True:
		r = requests.get(url)
		soup = BeautifulSoup(r.text, 'html.parser')
		base = soup.find("div", id="mw-pages")
		lemmas = base.find_all("li")
		for lemma in lemmas:
			text = lemma.get_text()
			all_lemmas.append(text)
		next_button = base.find("a", string="next page")
		if next_button is None:
			break
		url = base_url + next_button.get("href")
	return all_lemmas


data = get_pages(url)

with open('wiktionary_lemmas.json', 'w', encoding='utf-8') as f:
	json.dump(data, f, ensure_ascii=False, indent=4)
