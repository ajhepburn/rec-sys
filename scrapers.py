from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup
import json

def get_content(url):
    try:
        with closing(get(url, stream=True)) as resp:
            if is_good_response(resp):
                return resp.content
            else:
                return None

    except RequestException as e:
        log_error('Error during requests to {0} : {1}'.format(url, str(e)))
        return None


def is_good_response(resp):
    content_type = resp.headers['Content-Type'].lower()
    return (resp.status_code == 200 
            and content_type is not None 
            and content_type.find('html') > -1)


def log_error(e):
    print(e)


def scrape_slang():
    slang_terms = []

    def scrape_terms():
        terms = []
        asp_content = get_content('https://www.webopedia.com/quick_ref/textmessageabbreviations.asp')
        soup = BeautifulSoup(asp_content, 'lxml')
        content = soup.find("div", {"id": "article_main_column"})
        table = content.find("table")
        slang_terms = table.find_all("td", {"style": "width: 10%;"})
        
        for term in slang_terms:
            if term.find('p'):
                terms.append(term.find('p').text)
        return terms
    
    def write_to_utilities(path, terms):
        print("Writing to file...")
        with open(path+"slang.txt", 'w') as fp:
            json.dump(terms, fp)
        print("Write complete")
    
    slang_terms = scrape_terms()
    if slang_terms:
        write_to_utilities('./utilities/',slang_terms)

scrape_slang()
