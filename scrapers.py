from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup
import json

class SlangScraper:
    def __init__(self, url):
        self.url = url
        self.util_path = './utilities/'

    def get_content(self):
        try:
            with closing(get(self.url, stream=True)) as resp:
                if self.is_good_response(resp):
                    return resp.content
                else:
                    return None

        except RequestException as e:
            self.log_error('Error during requests to {0} : {1}'.format(self.url, str(e)))
            return None


    def is_good_response(self, resp):
        content_type = resp.headers['Content-Type'].lower()
        return (resp.status_code == 200 
                and content_type is not None 
                and content_type.find('html') > -1)


    def log_error(self, e):
        print(e)

    def scrape_terms(self):
        terms = []
        asp_content = self.get_content()
        soup = BeautifulSoup(asp_content, 'lxml')
        content = soup.find("div", {"id": "article_main_column"})
        table = content.find("table")
        slang_terms = table.find_all("td", {"style": "width: 10%;"})
        
        for term in slang_terms:
            if term.find('p'):
                terms.append(term.find('p').text)
        return terms

    def write_to_utilities(self, terms):
        print("Writing to file...")
        with open(self.util_path+"slang.txt", 'w') as fp:
            json.dump(terms, fp)
        print("Write complete")

if __name__ == '__main__':
    ss = SlangScraper(url='https://www.webopedia.com/quick_ref/textmessageabbreviations.asp')
    slang_terms = ss.scrape_terms()
    if slang_terms:
        ss.write_to_utilities(slang_terms)