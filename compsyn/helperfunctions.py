#!/usr/bin/env python
# coding: utf-8

from __future__ import annotations
import datetime
import gzip
import hashlib
import io 
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import random
import rapidjson
import requests
import time
import urllib.request
from PIL import Image
from bs4 import BeautifulSoup
from google.cloud import vision_v1p2beta1 as vision
from google.protobuf.json_format import MessageToDict
from google_images_download import google_images_download 
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from mpl_toolkits.mplot3d import Axes3D
from nltk.corpus import wordnet as wn
from numba import jit
from random_word import RandomWords
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from sklearn.manifold import TSNE
from textblob import Word
from textblob.wordnet import Synset


def get_webdriver(browser: str, executable_path: Optional[str]) -> webdriver:
    """
        Flexible webdriver getter and configuration
        
        In:
        browser: Browser name, will be taken as the webdriver attribute to be instantiated, i.e. webdriver.Browser
        executable_path: Path to browser executable, optional. Default behaviour is to assume the executable will be in $PATH

        Out:
        webdriver: instantiated webdriver, can be used as a context manager.
    """

    try:
        WebDriver = getattr(webdriver, browser)
    except AttributeError as e:
        raise Exception(f"no webdriver attribute called {browser}, try setting browser to 'Chrome' or 'Firefox'") from e

    # optionally configure webdriver executable path, selenium will look in $PATH by default
    if executable is None:
        return WebDriver()
    else:
        return WebDriver(executable_path=executable_path)


def settings(application_cred_name: str, driver_browser: str, driver_executable_path: str) -> None:
    #This client for the Google API needs to be set for the VISION classification
    #but it is not necessary for the selenium scaper for image downloading 
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=application_cred_name
    client = vision.ImageAnnotatorClient() # authentification via environment variable

    #See here for scraper details: 
    #https://towardsdatascience.com/image-scraping-with-python-a96feda8af2d
    wd = get_webdriver(browser=driver_browser, executable_path=driver_executable_path)
    wd.quit()



def fuzzy_sleep(min_time: int) -> None:
    """
        Fuzz wait times between [min_time, min_time*2]
    """
    time.sleep(min_time + min_time * random.random())


def fetch_image_urls(
    query: str, 
    number_of_links_to_fetch: int, 
    wd: webdriver, 
    thumb_css: str = "img.Q4LuWd", 
    img_css: str = "img.n3VNCb", 
    load_page_css: str = ".mye4qd", 
    sleep_between_interactions: float = 0.4
) -> List[str]:
    
    """
        Scrape all image urls from Google for search term 'query'. The script continues to load new 
        Google search pages as needed until number_of_links_to_fetch is reached.
        query: term to search in Google
        number_of_links_to_fetch: number of links to download from Google for query
        wd: path to the webdriver for selenium (Chrome or Firefox)
        thumb_css, img_css, load_page_css: css tags to identify IMG urls 
        sleep_between_interactions: sleep behavior to avoid red flags with Google. 
            Fuzzy sleep randomly varies sleep intervals to emulate human users. 
    """
        
    def scroll_to_end(wd):
        wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        fuzzy_sleep(sleep_between_interactions)    
    
    # build the google query
    search_url = "https://www.google.com/search?safe=off&site=&tbm=isch&source=hp&q={q}&oq={q}&gs_l=img"

    # load the page
    wd.get(search_url.format(q=query))

    image_urls = set()
    image_count = 0
    results_start = 0

    while image_count < number_of_links_to_fetch:
        
        scroll_to_end(wd)
        thumbnail_results = wd.find_elements_by_css_selector(thumb_css) # get all image thumbnail results
        if len(thumbnail_results) == 0:
            print(f"WARNING: found no thumbnails using the selector {thumb_css}")
        number_results = len(thumbnail_results)
        
        print(f"Found: {number_results} search results. Extracting links from {results_start}:{number_results}")
        
        for img in thumbnail_results[results_start:number_results]:
            # try to click every thumbnail such that we can get the real image behind it
            try:
                img.click()
                fuzzy_sleep(sleep_between_interactions)
            except Exception:
                continue

            # extract image urls    
            actual_images = wd.find_elements_by_css_selector(img_css)
            if len(actual_images) == 0:
                print(f"WARNING: found no images using the selector {img_css}")
            for actual_image in actual_images:
                if actual_image.get_attribute('src') and 'http' in actual_image.get_attribute('src'):
                    image_urls.add(actual_image.get_attribute('src'))
                    image_count += 1
                    if image_count >= number_of_links_to_fetch:
                        print(f"Found: {image_count} image links, done!")
                        return image_urls

        else:
            print(f"Found: {image_count} image links, looking for more ...")
            load_more_button = wd.find_element_by_css_selector(load_page_css)
            if load_more_button:
                fuzzy_sleep(sleep_between_interactions)
                wd.execute_script(f"document.querySelector('{load_page_css}').click();")
            else:
                print(
                    f"WARNING: {image_count}/{number_of_links_to_fetch} images gathered, but no 'load_more_button' found with the selector '{load_page_css}', returning what we have so far"
                )
                return image_urls

        # move the result startpoint further down
        results_start = len(thumbnail_results)


def save_image(folder_path:str,url:str) -> None:
    
    """
        Try to download the image correspond to the url scraped from the function, fetch_image_urls. 
        folder_path: file location for saving images 
        url: image url to download image from 
    """
        
    try:
        image_content = requests.get(url).content

    except Exception as e:
        print(f"ERROR - Could not download url")

    try:
        image_file = io.BytesIO(image_content)
        image = Image.open(image_file).convert('RGB')
        file_path = os.path.join(folder_path,hashlib.sha1(image_content).hexdigest()[:10] + '.jpg')
        
        with open(file_path, 'wb') as f:
            image.save(f, "JPEG", quality=85)
        #print(f"SUCCESS - saved {url} - as {file_path}")
        
    except Exception as e:
        print(f"ERROR - Could not save url")
        pass



def search_and_download(
    search_term:str,
    driver_browser: str,
    home: str, 
    driver_executable_path: str,
    target_path: str = './downloads',
    number_images: int = 5,
    sleep_time: float = 0.4
) -> List[str]:
    
    """
       Scrape and save images from Google using selenium to automate Google search. Save the raw images 
       collected into the folder, './downloads'. number_images determines the number of images to 
       collect for each search term.    
       
       search_term: term to use in Google query 
       driver_path: path to the webdriver for selenium (Chrome or Firefox)
       home: path to home directory of notebook
       target_path: file location to save images 
       number_images: number of images to download for each query
       sleep_time: general rate of sleep activity (lower values raise red flags for Google)
    """
        
    target_folder = os.path.join(target_path, search_term)

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    with get_webdriver(browser=driver_browser, driver_executable_path=driver_executablepath) as wd:
        urls = fetch_image_urls(search_term, number_images, wd=wd, sleep_between_interactions=sleep_time)
    
    for url in urls:
        save_image(target_folder,url)
    
    wd.quit()
    os.chdir(home)
    
    return urls


def run_google_vision(img_urls_dict: Dict[str, List[str]]) -> Dict[str, Dict[str, Any]]:
    
    """
       Use the Google vision API to return a set of classification labels for each image collected from 
       Google using the search_and_download function. Each label assigned by Google vision is associated 
       with a score indicating Google's confidence in the fit fo the label for the image.
       
       img_urls_dict: dictionary containing image_urls
    """
        
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


def write_to_json(to_save: Dict[str, Any], filename: str) -> None:
    """ add and write dictionary to existing json file"""
    with open(filename, 'a') as to_write_to:
        json.dump(to_save, to_write_to, indent=4)


def write_img_classifications_to_file(home: str, search_terms: List[str], img_classified_dict: Dict[str, Any]) -> None:

    """
       Store Google vision's classifications for images in a json file, which can then be retrieved for 
       the purposes of filtering and also statistical analyses.  
       
       home: home directory of notebook
       search_terms: terms used for querying Google
       img_classified_dict: dictionary of image URLs and classifications from Google Vision
    """
        
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


def get_branching_factor(search_terms: List[str]) -> Dict[str, int]:

    """
    Query WordNet to retrieve the branching factor for each search term. The default setting is to 
    retrieve the branching factor associated with each term's primary definition in WordNet. If a term 
    is polysemous and has multiple meanings, then the specific synset (i.e. WordNet definition) for this 
    term will need to be manually retrieved. Branching factor is the number of subsets a term indexes 
    plus 1, such that terms with no branches have a branching factor of 1. 
    """
        
    branching_dict={}
    
    for term in search_terms:  
        
        try: 
            wordsyn = wn.synsets(term)[0]
            branches = set([i for i in wordsyn.closure(lambda s:s.hyponyms())])
            branching_factor = len(branches)+1 #1 for the node itself 
            branching_dict[term] = branching_factor
        except: 
            branching_dict[term] = 0
        
    return branching_dict  


def expandTree(search_terms: List[str]) -> Dict[str, Any]:
 
    """
    Query WordNet to retrieve the full taxonomic tree associated with each search term. Hypernyms are 
    the supersets for a search term (i.e. the more general classes its a part of); Hyponyms are the 
    subsets for a search term (i.e. its branches in the tree); meronyms indicate part-whole relations 
    and associations, via synecdoche and metonymy.  
    """
    
    treeneighbors = {}
    
    for word in search_terms:
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


def get_tree_structure(tree: Dict[str, Any], home: str) -> Tuple[Dict[str, Any], List[str]]:
    
    """
    Query WordNet to retrieve various properties of each search term.
    ref_term indicates the term used to retrieve a new term from WordNet. 
    new_term indicates the new term retrieved from WordNet as part of another term's taxonomic tree. 
    role indicates the part of speech of the new term.
    synset indicates the meaning (sense) of the new term according to WordNet.
    Branch_fact indicates the branching factor of the new term in WordNet.
    Num_senses indicates how polysemous the new term is in WordNet, based on how many meanings it is 
    linked to in WordNet. 
    
    Tree: dictionary of search terms and their corresponding parents and children in WordNet's taxonomy
    """
        
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



def get_wordnet_tree_data(search_terms: List[str], home: str, get_trees: bool = True) -> Tuple[List[str], Dict[str, Any], Dict[str, Any]]: 
    
    """
    Use get_tree_structure to collect new terms from the taxonomic trees associated with each search 
    term provided as input. 
    
    home: home directory of notebook
    If get_trees = False, then return original search_term list
    """
        
    if get_trees: 
        try:
            tree = expandTree(search_terms) #get tree for wordlist
            tree_data, new_searchterms = get_tree_structure(tree, home) 
            final_wordlist = search_terms + new_searchterms
            
            os.chdir(home)
            return final_wordlist, tree, tree_data
        
        except: 
            os.chdir(home)
            print("No tree available")
            return wordlist, dict(), dict()
    
    else: 
        os.chdir(home)
        return wordlist, dict(), dict()

