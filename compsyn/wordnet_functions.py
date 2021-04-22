# Need to check imports carefully
import os
import random

import pandas as pd
from nltk.corpus import wordnet as wn

from .logger import get_logger


def get_branching_factor(search_terms):

    """
    Query WordNet to retrieve the branching factor for each search term. The default setting is to 
    retrieve the branching factor associated with each term's primary definition in WordNet. If a term 
    is polysemous and has multiple meanings, then the specific synset (i.e. WordNet definition) for this 
    term will need to be manually retrieved. Branching factor is the number of subsets a term indexes 
    plus 1, such that terms with no branches have a branching factor of 1. 
    """

    branching_dict = {}

    for term in search_terms:

        try:
            wordsyn = wn.synsets(term)[0]
            branches = set([i for i in wordsyn.closure(lambda s: s.hyponyms())])
            branching_factor = len(branches) + 1  # 1 for the node itself
            branching_dict[term] = branching_factor
        except:
            branching_dict[term] = 0

    return branching_dict


def expandTree(search_terms):

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

            if hyponyms:
                hyponyms = [f.name() for f in hyponyms]
            if hypernyms:
                hypernyms = [f.name() for f in hypernyms]

            all_hyponyms.extend(hyponyms)

            all_hypernyms.extend(hypernyms)

        neighbors = {
            "hyponyms": all_hyponyms,
            "hypernyms": all_hypernyms,
            "substanceMeronyms": [],
            "partMeronyms": [],
        }

        treeneighbors[word] = neighbors

    return treeneighbors


def get_tree_structure(tree, home):

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

    tree_data = pd.DataFrame(
        columns=["ref_term", "new_term", "role", "synset", "Branch_fact", "Num_senses"]
    )

    for term in tree.keys():
        hyponyms = tree[term]["hyponyms"]
        hypernyms = tree[term]["hypernyms"]
        substanceMeronyms = tree[term]["substanceMeronyms"]
        partMeronyms = tree[term]["partMeronyms"]

        for hypo in hyponyms:
            hypo = wn.synset(hypo)
            new_term = [t.name() for t in hypo.lemmas()][0]
            term_branch = get_branching_factor([new_term])
            term_branch = term_branch[new_term]
            row = {
                "ref_term": term,
                "new_term": new_term,
                "role": "hyponym",
                "synset": hypo,
                "Branch_fact": term_branch,
                "Num_senses": len(wn.synsets(new_term)),
            }

            tree_data = tree_data.append(row, ignore_index=True)

        for hyper in hypernyms:
            hyper = wn.synset(hyper)
            new_term = [t.name() for t in hyper.lemmas()][0]
            term_branch = get_branching_factor([new_term])
            term_branch = term_branch[new_term]
            row = {
                "ref_term": term,
                "new_term": new_term,
                "role": "hypernym",
                "synset": hyper,
                "Branch_fact": term_branch,
                "Num_senses": len(wn.synsets(new_term)),
            }
            tree_data = tree_data.append(row, ignore_index=True)

        for subst in substanceMeronyms:
            subst = wn.synset(subst)
            new_term = [t.name() for t in subst.lemmas()][0]
            term_branch = get_branching_factor([new_term])
            term_branch = term_branch[new_term]
            row = {
                "ref_term": term,
                "new_term": new_term,
                "role": "substmeronym",
                "synset": subst,
                "Branch_fact": term_branch,
                "Num_senses": len(wn.synsets(new_term)),
            }
            tree_data = tree_data.append(row, ignore_index=True)

        for part in partMeronyms:
            part = wn.synset(part)
            new_term = [t.name() for t in part.lemmas()][0]
            term_branch = get_branching_factor([new_term])
            term_branch = term_branch[new_term]
            row = {
                "ref_term": term,
                "new_term": new_term,
                "role": "partMeronyms",
                "synset": part,
                "Branch_fact": term_branch,
                "Num_senses": len(wn.synsets(new_term)),
            }
            tree_data = tree_data.append(row, ignore_index=True)

    new_searchterms = list(set(tree_data["new_term"].values))
    new_searchterms = [t.replace("_", " ") for t in new_searchterms]

    tree_data_path = "tree_data/"
    try:
        os.mkdir(tree_data_path)
    except:
        pass

    os.chdir(tree_data_path)

    for term in tree.keys():
        tree_data_term = tree_data[tree_data["ref_term"] == term]
        tree_data_term.to_json(r"tree_data_" + term + ".json")

    return tree_data, new_searchterms


def get_wordnet_tree_data(search_terms, home, get_trees=True):

    """
    Use get_tree_structure to collect new terms from the taxonomic trees associated with each search 
    term provided as input. 
    
    home: home directory of notebook
    If get_trees = False, then return original search_term list
    """

    log = get_logger("get_wordnet_tree_data")

    if get_trees:
        try:
            tree = expandTree(search_terms)  # get tree for wordlist
            tree_data, new_searchterms = get_tree_structure(tree, home)
            final_wordlist = search_terms + new_searchterms

            os.chdir(home)
            return final_wordlist, tree, tree_data

        except Exception as exc:
            os.chdir(home)
            log.error(f"No tree available due to error {exc}")
            return wordlist, {}, {}

    else:
        os.chdir(home)
        return wordlist, {}, {}
