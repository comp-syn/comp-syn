#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import urllib.request
import datetime
import os
import json
import rapidjson
import io 
import hashlib
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from PIL import Image
from random_word import RandomWords
from nltk.corpus import wordnet as wn
import random
import pandas as pd
from textblob import Word
from textblob.wordnet import Synset
import numpy as np
from google.cloud import vision_v1p2beta1 as vision
from google.protobuf.json_format import MessageToDict
from google_images_download import google_images_download 
import pickle
import matplotlib.pyplot as plt
from numba import jit
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.manifold import TSNE
import gzip


#This client for the Google API needs to be set for the VISION classification
#but it is not necessary for the selenium scaper for image downloading 

#os.environ["GOOGLE_APPLICATION_CREDENTIALS"]='COMPSYN2-18e251c693df.json'
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]='compsyn3-8cf6580619a9.json'
client = vision.ImageAnnotatorClient() # authentification via environment variable

#See here for scraper details: 
#https://towardsdatascience.com/image-scraping-with-python-a96feda8af2d


# In[2]:

DRIVER_PATH = "/Users/bhargavvader/open_source/comp-syn/chromedriver"
wd = webdriver.Chrome(DRIVER_PATH) #incase you are chrome
wd.quit()


# In[3]:


def fetch_image_urls(query:str, max_links_to_fetch:int, wd:webdriver, sleep_between_interactions:int=1):
    def scroll_to_end(wd):
        wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(sleep_between_interactions)    
    
    # build the google query
    search_url = "https://www.google.com/search?safe=off&site=&tbm=isch&source=hp&q={q}&oq={q}&gs_l=img"

    # load the page
    wd.get(search_url.format(q=query))

    image_urls = set()
    image_count = 0
    results_start = 0
    while image_count < max_links_to_fetch:
        scroll_to_end(wd)

        # get all image thumbnail results
        thumbnail_results = wd.find_elements_by_css_selector("img.Q4LuWd")
        number_results = len(thumbnail_results)
        
        print(f"Found: {number_results} search results. Extracting links from {results_start}:{number_results}")
        
        for img in thumbnail_results[results_start:number_results]:
            # try to click every thumbnail such that we can get the real image behind it
            try:
                img.click()
                time.sleep(sleep_between_interactions)
            except Exception:
                continue

            # extract image urls    
            actual_images = wd.find_elements_by_css_selector('img.n3VNCb')
            for actual_image in actual_images:
                if actual_image.get_attribute('src') and 'http' in actual_image.get_attribute('src'):
                    image_urls.add(actual_image.get_attribute('src'))

            image_count = len(image_urls)

            if len(image_urls) >= max_links_to_fetch:
                print(f"Found: {len(image_urls)} image links, done!")
                break
        else:
            print("Found:", len(image_urls), "image links, looking for more ...")
            time.sleep(30)
            return
            load_more_button = wd.find_element_by_css_selector(".mye4qd")
            if load_more_button:
                wd.execute_script("document.querySelector('.mye4qd').click();")

        # move the result startpoint further down
        results_start = len(thumbnail_results)

    return image_urls


# In[4]:


def persist_image(folder_path:str,url:str):
    try:
        image_content = requests.get(url).content

    except Exception as e:
        print(f"ERROR - Could not download {url} - {e}")

    try:
        image_file = io.BytesIO(image_content)
        image = Image.open(image_file).convert('RGB')
        file_path = os.path.join(folder_path,hashlib.sha1(image_content).hexdigest()[:10] + '.jpg')
        with open(file_path, 'wb') as f:
            image.save(f, "JPEG", quality=85)
        print(f"SUCCESS - saved {url} - as {file_path}")
    except Exception as e:
        print(f"ERROR - Could not save {url} - {e}")
        pass


# In[5]:


def search_and_download(search_term:str,driver_path:str,home, target_path='./downloads',number_images=5):
    target_folder = os.path.join(target_path,'_'.join(search_term.lower().split(' ')))

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    with webdriver.Chrome(executable_path=driver_path) as wd:
        res = fetch_image_urls(search_term, number_images, wd=wd, sleep_between_interactions=0.5)
    
    if res: 
        for elem in res:
            try:
                persist_image(target_folder,elem)
            except Exception as e:
                pass

    wd.quit()
    os.chdir(home)
    
    return res


# In[6]:


def get_imgs(searchterms_list, home):
    os.chdir(home)
    
    img_dict = {}
    for term in searchterms_list:
        term_img_set = os.listdir('downloads/' + term)
        img_dict[term]=term_img_set
    return img_dict


# In[7]:


def run_google_vision(img_urls_dict):
    print("Classifying Imgs. w. Google Vision API...")
    
    client = vision.ImageAnnotatorClient()
    image = vision.types.Image()
    
    for search_term in img_urls_dict.keys(): 
        img_urls = img_urls_dict[search_term]
    
        img_classified_dict = {}
        img_classified_dict[search_term] = {}
        
        for image_uri in img_urls: 
            try: 
                image.source.image_uri = image_uri
                response = client.label_detection(image=image)
                img_classified_dict[image_uri]={}

                for label in response.label_annotations:
                    img_classified_dict[search_term][image_uri] = {}
                    img_classified_dict[search_term][image_uri][label.description] = label.score
                    
            except: pass

    return img_classified_dict


# In[8]:


def write_to_json(to_save, filename):
    with open(filename, 'a') as to_write_to:
        json.dump(to_save, to_write_to, indent=4)


# In[9]:


def write_img_classifications_to_file(home, search_terms, img_classified_dict):
    
    os.chdir(home + "/image_classifications")
    
    for term in search_terms:
        term_data = img_classified_dict[term]
        
        if term_data:
            filename = "classifications_" + term + '_' + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M") + ".json"
            file_exist = os.path.isfile(filename)

            if file_exist:
                print("File already exists! Appending to file.. ")

                with open(filename, encoding='utf-8') as f:
                    term_data_orig = json.load(f)
                    
                term_data_orig.update(term_data)
                os.remove(filename)
                write_to_json(term_data_orig, filename)

            else: 
                print("File new! Saving..")
                write_to_json(term_data, filename)

    os.chdir(home)          


# In[10]:


def get_branching_factor(wordlist):

    branching_dict={}
    
    for word in wordlist:  
        
        try: 
            wordsyn = wn.synsets(word)[0]
            branches = set([i for i in wordsyn.closure(lambda s:s.hyponyms())])
            branching_factor = len(branches)+1 #1 for the node itself 
            branching_dict[word] = branching_factor
        except: 
            branching_dict[word] = 0
        
    return branching_dict  


# In[11]:


def get_wordnet_data(wordlist,home):
    os.chdir(home)
    if not os.path.isdir('/tree_data/'): os.mkdir('/tree_data/')
            
    all_wordnet_data = pd.DataFrame(columns=['ref_term','new_term', 
                                             'role', 'synset', 
                                             'Branch_fact', 'Num_senses'])
    
    if wordlist: 
        for word in wordlist: 
            word_data = "wordnet_data_" + word + ".json"

            with open(word_data) as f:
                word_data_dict = json.load(f)

            word_data_df = pd.DataFrame.from_dict(word_data_dict)

            all_wordnet_data = all_wordnet_data.append(word_data_df, ignore_index=True)
        
        return all_wordnet_data
        
    else: 
        print("No wordlist to process")


# In[12]:


def expandTree(wordlist):
    treeneighbors = {}
    
    for word in wordlist:
        synsets = wn.synsets(word)
        
        all_hyponyms = []
        all_hypernyms = []
        
        for synset in synsets: 
            hyponyms = synset.hyponyms()
            hypernyms = synset.hypernyms()

            if hyponyms: hyponyms = [f.name() for f in hyponyms]
            if hypernyms: hypernyms = [f.name() for f in hypernyms]
                
            all_hyponyms.extend(hyponyms)
            
            all_hypernyms.extend(hypernyms)

        neighbors = {'hyponyms':all_hyponyms, 'hypernyms':all_hypernyms, 
                     'substanceMeronyms':[], 'partMeronyms':[]}
        
        treeneighbors[word] = neighbors
        
    return treeneighbors


# In[13]:


def get_tree_structure(tree, home):
    os.chdir(home)
    
    tree_data = pd.DataFrame(columns=['ref_term','new_term', 'role', 'synset', 
                                      'Branch_fact', 'Num_senses'])
    
    for term in tree.keys():
        hyponyms = tree[term]['hyponyms']
        hypernyms = tree[term]['hypernyms']
        substanceMeronyms = tree[term]['substanceMeronyms']
        partMeronyms = tree[term]['partMeronyms']
        
        
        for hypo in hyponyms: 
            hypo=wn.synset(hypo)
            new_term = [t.name() for t in hypo.lemmas()][0]
            term_branch = get_branching_factor([new_term])
            term_branch = term_branch[new_term]
            row = {'ref_term': term, 'new_term': new_term, 'role': 'hyponym', 
                   'synset': hypo, 'Branch_fact': term_branch, 
                   'Num_senses': len(wn.synsets(new_term))}
            
            tree_data = tree_data.append(row, ignore_index=True)
            
        for hyper in hypernyms: 
            hyper=wn.synset(hyper)
            new_term = [t.name() for t in hyper.lemmas()][0]
            term_branch = get_branching_factor([new_term])
            term_branch = term_branch[new_term]
            row = {'ref_term': term, 'new_term': new_term, 'role': 'hypernym', 
                   'synset': hyper, 
                   'Branch_fact': term_branch, 
                   'Num_senses': len(wn.synsets(new_term))}
            tree_data = tree_data.append(row, ignore_index=True)
        
        for subst in substanceMeronyms: 
            subst=wn.synset(subst)
            new_term = [t.name() for t in subst.lemmas()][0]
            term_branch = get_branching_factor([new_term])
            term_branch = term_branch[new_term]
            row = {'ref_term': term, 'new_term': new_term, 'role': 'substmeronym', 
                   'synset': subst, 
                   'Branch_fact': term_branch,  
                   'Num_senses': len(wn.synsets(new_term))}
            tree_data = tree_data.append(row, ignore_index=True)
            
        for part in partMeronyms: 
            part=wn.synset(part)
            new_term = [t.name() for t in part.lemmas()][0]
            term_branch = get_branching_factor([new_term])
            term_branch = term_branch[new_term]
            row = {'ref_term': term, 'new_term': new_term, 'role': 'partMeronyms', 
                   'synset': part, 'Branch_fact': term_branch, 
                   'Num_senses': len(wn.synsets(new_term))}
            tree_data = tree_data.append(row, ignore_index=True)
        
    new_searchterms = list(set(tree_data['new_term'].values)) 
    new_searchterms = [t.replace("_", " ") for t in new_searchterms]
    
    tree_data_path= 'tree_data/'
    try: os.mkdir(tree_data_path)
    except: pass
        
    os.chdir(tree_data_path)
    
    for term in tree.keys(): 
        tree_data_term = tree_data[tree_data['ref_term'] == term]
        tree_data_term.to_json(r'tree_data_' + term + '.json')
    
    return tree_data, new_searchterms


# In[14]:


def get_wordnet_tree_data(wordlist, home, get_trees=True): 
    
    if get_trees: 
        try:
            tree = expandTree(wordlist) #get tree for wordlist
            tree_data, new_searchterms = get_tree_structure(tree, home) #org. and save tree data and get new terms
            final_wordlist = wordlist + new_searchterms
            
            os.chdir(home)
            return final_wordlist, tree, tree_data
        
        except: 
            os.chdir(home)
            print("No tree available")
            return wordlist, {}, {}
    
    else: 
        os.chdir(home)
        return wordlist, {}, {}

