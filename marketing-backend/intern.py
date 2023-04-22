import requests
import re
from urllib.request import Request, urlopen
import urllib.request
from bs4 import BeautifulSoup
from collections import deque
from html.parser import HTMLParser
from urllib.parse import urlparse
# import osd
import pandas as pd
import tiktoken
import openai
import os
import numpy as np
from openai.embeddings_utils import distances_from_embeddings, cosine_similarity
from selenium import webdriver
import time
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

openai.api_key= os.environ.get("OPENAI_API_KEY")

# Regex pattern to match a URL
HTTP_URL_PATTERN = r'^http[s]{0,1}://.+$'

# Define root domain to crawl (my personal app used)
domain = "celestia.org/"
full_url = "https://celestia.org/"

# # Create a class to parse the HTML and get the hyperlinks
# class HyperlinkParser(HTMLParser):
#     def __init__(self):
#         super().__init__()
#         # Create a list to store the hyperlinks
#         self.hyperlinks = []

#     # Override the HTMLParser's handle_starttag method to get the hyperlinks
#     def handle_starttag(self, tag, attrs):
#         attrs = dict(attrs)

#         # If the tag is an anchor tag and it has an href attribute, add the href attribute to the list of hyperlinks
#         if tag == "a" and "href" in attrs:
#             self.hyperlinks.append(attrs["href"])

# ################################################################################
# ### Step 2
# ################################################################################

# # Function to get the hyperlinks from a URL
# def get_hyperlinks(url):
    
#     # Try to open the URL and read the HTML
#     try:
#         # Open the URL and read the HTML
#         # with urllib.request.urlopen(url) as response:

#         #     # If the response is not HTML, return an empty list
#         #     if not response.info().get('Content-Type').startswith("text/html"):
#         #         return []
            
#         #     # Decode the HTML
#         #     html = response.read().decode('utf-8')
#         req = Request(
#             url=url, 
#             headers={'User-Agent': 'Mozilla/6.0'}
#         )
#         webpage = urlopen(req).read().decode('utf-8')

#     except Exception as e:
#         print("Error: ", e)
#         return []

#     # Create the HTML Parser and then Parse the HTML to get hyperlinks
#     parser = HyperlinkParser()
#     # parser.feed(html)
#     parser.feed(webpage)

#     return parser.hyperlinks

# ################################################################################
# ### Step 3
# ################################################################################

# # Function to get the hyperlinks from a URL that are within the same domain
# def get_domain_hyperlinks(local_domain, url):
#     clean_links = []
#     for link in set(get_hyperlinks(url)):
#         clean_link = None

#         # If the link is a URL, check if it is within the same domain
#         if re.search(HTTP_URL_PATTERN, link):
#             # Parse the URL and check if the domain is the same
#             url_obj = urlparse(link)
#             if url_obj.netloc == local_domain:
#                 clean_link = link

#         # If the link is not a URL, check if it is a relative link
#         else:
#             if link.startswith("/"):
#                 link = link[1:]
#             elif (
#                 link.startswith("#")
#                 or link.startswith("mailto:")
#                 or link.startswith("tel:")
#             ):
#                 continue
#             clean_link = "https://" + local_domain + "/" + link

#         if clean_link is not None:
#             if clean_link.endswith("/"):
#                 clean_link = clean_link[:-1]
#             clean_links.append(clean_link)

#     # Return the list of hyperlinks that are within the same domain
#     return list(set(clean_links))


# ################################################################################
# ### Step 4
# ################################################################################

# def crawl(url):
#     # Parse the URL and get the domain
#     local_domain = urlparse(url).netloc

#     # Create a queue to store the URLs to crawl
#     queue = deque([url])

#     # Create a set to store the URLs that have already been seen (no duplicates)
#     seen = set([url])

#     # Create a directory to store the text files
#     if not os.path.exists("text/"):
#             os.mkdir("text/")

#     if not os.path.exists("text/"+local_domain+"/"):
#             os.mkdir("text/" + local_domain + "/")

#     # Create a directory to store the csv files
#     if not os.path.exists("processed"):
#             os.mkdir("processed")

#     # While the queue is not empty, continue crawling
#     while queue:

#         # Get the next URL from the queue
#         url = queue.pop()
#         print(url) # for debugging and to see the progress

#         # Save text from the url to a <url>.txt file
#         with open('text/'+local_domain+'/'+url[8:].replace("/", "_") + ".txt", "w", encoding="UTF-8") as f:

#             # Get the text from URL using BeautifulSoup, while waiting for Javascript content to load
#             options = webdriver.ChromeOptions()
#             options.add_argument('--headless')
#             # executable_path param is not needed if you updated PATH
#             browser = webdriver.Chrome(options=options)
#             browser.get(url)
#             time.sleep(10)
#             html = browser.page_source
#             soup = BeautifulSoup(html, features="html.parser")
#             browser.quit()

#             # Get the text from the URL using BeautifulSoup --> Doesn't work on javascript 
#             #soup = BeautifulSoup(requests.get(url).text, "html.parser")

#             # Get the text but remove the tags
#             text = soup.get_text()

#             # If the crawler gets to a page that requires JavaScript, it will stop the crawl
#             if ("You need to enable JavaScript to run this app." in text):
#                 print("Unable to parse page " + url + " due to JavaScript being required")
            
#             # Otherwise, write the text to the file in the text directory
#             f.write(text)

#         # Get the hyperlinks from the URL and add them to the queue
#         for link in get_domain_hyperlinks(local_domain, url):
#             if link not in seen:
#                 queue.append(link)
#                 seen.add(link)

# crawl(full_url)

################################################################################
### Step 5
################################################################################

def remove_newlines(serie):
    serie = serie.str.replace('\n', ' ')
    serie = serie.str.replace('\\n', ' ')
    serie = serie.str.replace('  ', ' ')
    serie = serie.str.replace('  ', ' ')
    return serie

################################################################################
### Step 6
################################################################################

# Create a list to store the text files
texts=[]

# Get all the text files in the text directory
for file in os.listdir("text/" + domain + "/"):

    # Open the file and read the text
    with open("text/" + domain + "/" + file, "r", encoding="UTF-8") as f:
        text = f.read()

        # Omit the first 11 lines and the last 4 lines, then replace -, _, and #update with spaces.
        texts.append((file[11:-4].replace('-',' ').replace('_', ' ').replace('#update',''), text))

# Create a dataframe from the list of texts
df = pd.DataFrame(texts, columns = ['fname', 'text'])

# Set the text column to be the raw text with the newlines removed
df['text'] = df.fname + ". " + remove_newlines(df.text)
df.to_csv('processed/scraped.csv')
df.head()

