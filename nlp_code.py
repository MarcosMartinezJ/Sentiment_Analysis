"""
NLP CODE (2022) - Marcos Martinez Jimenez
-------------------------------------

Module with several utilities for NLP (namely opinion mining) analysis.
"""


#from __future__ import annotations

### Module import ###

import numpy as np
import pandas as pd
import nltk
import json
import time
import re

from nltk.corpus import wordnet as wn
from nltk.parse.corenlp import CoreNLPDependencyParser

# Utilities for function annotation
from typing import Dict, Union, List
from typing import Callable, Tuple

# Functions for displaying the progress of loops
from tqdm import tqdm
from IPython.display import clear_output

### - ###


### Internal variables ###

# ANSI codes for displaying text in bold red (RED)
#                                   bold magenta (MAG)
#                               and going back to normal (REG)
RED = "\033[01m\033[31m"
MAG = "\033[03m\033[35m"
REG = "\033[0m"

# List of POS tags allowed as opinions (adjs/advbs of the aspect term)
# (for basic opinion identification)
aspect_classes = ['JJ','JJR','JJS','RB','RBR','RBS']

# List of dependencies allowed as opinions with the aspect term as head
# (for dependency parsing opinion identification)
aspect_relations_1 = ['amod','advmod']

# List of dependencies allowed as as opinions with the aspect term as dependent
aspect_relations_2 = ['nsubj']

# List of dependencies allowed as modifications (modifiers of aspect opinions)
aspect_subrelations = ['amod','advmod']

### - ###


### 1. Loading datasets ###

def parse_batch(filename: str) -> List[Dict]:
    """
    Load all entries of a json file simultaneously (batch).

    Parameters
    ----------
    filename:
        Path of the json file

    Returns
    -------
    json_file:
        List with all entries of the file
    """
    with open(filename, encoding='utf-8') as f:
        json_file = json.load(f)
    f.close()
    return json_file


def parse_lines(filename: str) -> Dict:
    """
    Generator for loading entries of a json file one at a time.

    Parameters
    ----------
    filename:
        Path of the json file

    Yields
    -------
    json_line:
        Each entry of the file
    """
    with open(filename, encoding='utf-8') as f:
        # Walrus operator for checking whether next line exists
        # and saving it to file simultaneously
        while (line := f.readline()):
            # Json files can be problematic if
            # 1) they're empty strings
            if line == "":
                pass
            # 2) they end with \n (in which case remove it)
            if line[-1]=='\n':
                line = line[:-1]
            # 3) they end in , (in which case remove it)
            if line[-1]==',':
                line = line[:-1]
            try:
                json_line = json.loads(line)
            except:
                continue
            yield json_line
    f.close()
    return

### - ###


### 2. Identification of aspect terms ###

def matchRE(
    review: str,
    aspects: pd.DataFrame,
    show_context: bool = False
) -> Union[Dict[str,int], None]:
    """
    Identification of all occurrences of aspect terms in a text.

    Parameters
    ----------
    review:
        Text where the aspect terms will be searched
    aspects:
        Pandas dataframe with "aspect" and "term" columns. Terms include the
        words that will be identified and aspects are the corresponding area
        of opinion (e.g. food, landscape, ...)
    show_context:
        Boolean flag that controls the output (see Returns)

    Returns
    -------
    aspect_counts <= show_context::False
        Dictionary with the number of terms identified for each aspect
    None          <= show_context::True
        None; Prints the context (+/-10 characters around each term) instead
    """
    aspect_counts = {}
    for i in range(len(aspects)):
        aspect = aspects.loc[i,'aspect']
        # To show context we need the start and end of each match
        # re.findall (used otherwise) doesn't return the re.match object with
        # the span() method, so we have to manually find one term at a time and
        # remove the text already covered
        if show_context:
            # copy of the text that will be trimmed as matches are found
            subtext = review
            while (match := re.search(aspects.loc[i,'term'], subtext)):
                st,end = match.span()
                subtext = subtext[end:]
                                       # Adds fixed number of spaces between
                                       # the start of each part and the next
                                       # so it acts like a tab
                print(f"Aspect: {aspect}{' '*(20-len(aspect))}" +
                      f"Term: {match.group()}{' '*(15-len(match.group()))}" +
                      f"Context:{review[st-10:end+10]}")

        # If not showing context we can just use re.findall()
        else:
            matches = re.findall(aspects.loc[i,'term'], review)
            for match in matches:
                # Dictionary value actualization fails if the key hasn't been
                # initialized so this initializes in case it hasn't been yet
                try:
                    aspect_counts[aspect] += 1
                except:
                    aspect_counts[aspect] = 1
    # Return control
    if show_context:
        return
    else:
        return aspect_counts


def match_token(
    review: str,
    aspects: pd.DataFrame,
    tol: float = 0,
    out: str = "count"
) -> Union[Dict[str,int],
           Dict[str,List[Tuple[int,int]]],
           None]:
    """
    Identification of occurrences of aspect terms using text divided in tokens,
    and a threshold of letter mismatches.

    Parameters
    ----------
    review:
        Text where the aspect terms will be searched
    aspects:
        Pandas dataframe with "aspect" and "term" columns. Terms include the
        words that will be identified and aspects are the corresponding area
        of opinion (e.g. food, landscape, ...)
    tol:
        Tolerance of term matching. % of letter mismatches allowed to consider
        that a token matches an aspect term
    out:
        Keywords that control the output (see Returns)

    Returns
    -------
    aspect_counts     <= out::'count'
        Dictionary with the number of terms identified for each aspect
    None              <= out::'context'
        None; Prints the context (+/-3 tokens around each term) instead
    aspect_positions  <= out::'pos'
        Dictionary with List[(phr_pos,tok_pos)] for each identified aspect term
            phr_pos: phrase number of the term
            tok_pos: token number of the term
    """
    # Controls initialization of the output
    if out == "count":
        aspect_counts = {}
    elif out == "pos":
        aspect_positions = {}
    elif out == "context":
        pass

    # Tokenization of text
    sentences = nltk.sent_tokenize(review)
    sentences = [nltk.word_tokenize(s) for s in sentences]

    # Obtains max lengths of text's tokens and of aspect's terms
    # and max of those two (required for token<->term matching code)
    max_length_rev = max([max([len(w) for w in s]) for s in sentences])
    max_length_asp = max([len(aspect) for aspect in aspects['term']])
    max_length = max(max_length_rev, max_length_asp)

    # Numpy array with aspect terms (one term for column, one letter of
    #                                each term for row)
    # (the array is actually in 3D so it broadcasts adequately with the tokens)
    mat_aspects = np.array([list(aspect)+["."]*(max_length - len(aspect))
                            for aspect in aspects['term']])
    s1,s2 = mat_aspects.shape
    mat_aspects = mat_aspects.reshape((1,s1,s2))

    # Array with length of each aspect (will be needed to obtain the maximum
    #                                   length of each comparison to normalize
    #                                   the number of mismatches)
    asp_lengths = np.array([len(aspect) for aspect in aspects['term']])
    asp_lengths.reshape(1,s1)

    for sidx, sentence in enumerate(sentences):
        # Numpy array with sentence tokens with shape that allows simultaneous
        # comparisons of all tokens to all aspect terms
        mat_sentence = np.array([list(w)+["*"]*(max_length - len(w))
                                 for w in sentence])
        s1,s2 = mat_sentence.shape
        mat_sentence = mat_sentence.reshape((s1,1,s2))

        # Array with length of each token (to obtain maximum)
        wor_lengths = np.array([len(w) for w in sentence])
        wor_lengths = wor_lengths.reshape((s1,1))

        # Simultaneous comparison through broadcasting and count of the
        # number of matches
        match_scores = np.sum(mat_sentence == mat_aspects, axis=2)

        # Meshgrid with token and term lengths
        La,Ls = np.meshgrid(asp_lengths,wor_lengths)
        L = La

        # Each position in the meshgrid corresponds to one token-term
        # comparison so the maximum length is stored
        # (that is the length that should be used for the normalization)
        L[Ls>La] = Ls[Ls>La]
        match_scores = match_scores/L

        # We obtain the terms that best match each token of the phrase
        matches,scores = np.argmax(match_scores, axis=1),np.max(match_scores, axis=1)
        for i in range(len(matches)):
            # If the mismatches are low enough it's added to the output
            # (either counts, appending to the list of terms or showing
            # the context)
            if scores[i]>=1-tol:
                aspect = aspects.loc[matches[i],'aspect']
                if out == "context":
                    print(f"Aspect: {aspect}{' '*(20-len(aspect))}" +
                          f"Term: {aspects.loc[matches[i],'term']}{' '*(15-len(aspects.loc[matches[i],'term']))}" +
                          f"Context:{' '.join(sentence[max(0,i-3):i])}" +
                          # RED + "..." + REG makes the inner text bold red
                          f"{RED} {sentence[i]} {REG}{' '.join(sentence[i+1:i+3])}")
                elif out == "count":
                    try:
                        aspect_counts[aspect] += 1
                    except:
                        aspect_counts[aspect] = 1
                elif out == "pos":
                    try:
                        aspect_positions[aspect].append((sidx, i))
                    except:
                        aspect_positions[aspect] = [(sidx,i)]

    if out == "context":
        return
    elif out == "count":
        return aspect_counts
    elif out == "pos":
        return aspect_positions


def deambiguate_terms(aspects:pd.DataFrame) -> Dict[str,List["nltk.Synset"]]:
    """
    Deambiguate a list of terms to WordNet synsets.

    Function that interactively asks the user to select the correct synset
    for each aspect term.

    Parameters
    ----------
    aspects:
        Pandas dataframe with "aspect" and "term" columns. Terms include the
        words that will be identified and aspects are the corresponding area
        of opinion (e.g. food, landscape, ...)

    Returns
    -------
    final_synsets:
        Dictionary with the original aspects as keys and the lists of term
        synsets associated with the aspects as values
    """
    # Initialization of the dict output (alternative to the try/except)
    term_synsets = {aspect: [] for aspect in aspects['aspect']}
    for i in range(len(aspects)):
        # clear_output and sleep allow the cell's output to
        # update after each answer
        clear_output()
        time.sleep(0.5)
        aspect = aspects.loc[i,'aspect']
        term = aspects.loc[i,'term']
        # Prints tabbed the current Aspect | Term
        print(f"Aspect: {aspect.upper()}{' '*(20-len(aspect))}" +
              f"Term: {term.upper()}\n{'-'*30}\n")
        synsets = wn.synsets(term)
        for j,s in enumerate(synsets):
            print(f"{j}: {s.definition()}")
            print("\t", s.examples())
        if len(synsets)>0:
            choice = int(input("Enter correct synset [0..n]: "))
            term_synsets[aspect].append(synsets[choice])

    # Now get the final synsets by reconstructing the dictionary
    # without adding synsets that are already present in this aspect's
    final_synsets = {aspect: [] for aspect in term_synsets.keys()}
    for aspect in term_synsets.keys():
     for s in term_synsets[aspect]:
         if not s in final_synsets[aspect]:
             final_synsets[aspect].append(s)
    return final_synsets

def gather_terms(
    synsets: Dict[str,List["nltk.Synset"]]
) -> Dict[str,List[str]]:
    """
    Obtain aspect terms from WordNet synsets.

    Parameters
    ----------
    synsets:
        Dictionary with aspects as keys and lists of term
        synsets associated with the aspects as values

    Returns
    -------
    terms:
        Dictionary with aspects as keys and lists of string
        terms associated with the aspects' synsets as values
    """
    terms = {asp:[] for asp in synsets.keys()}
    aspects = list(synsets.keys())
    # tqdm module offers a progress bar in a for loop
    for i in tqdm(range(len(aspects)),desc="Aspect progress"):
        aspect = aspects[i]
        for synset in synsets[aspect]:
            # Extract from a synset:
                # All hyponyms (and from those, their lemmas)
                # All lemmas and their derived forms
            # And append them to the list if they aren't already
            # included
            for hypo_synset in synset.hyponyms():
                for lemma in hypo_synset.lemmas():
                    for form in lemma.derivationally_related_forms():
                        if not (term := form.name()) in terms[aspect]:
                            term = term.replace("_"," ")
                            terms[aspect].append(term)
                    if not (term := lemma.name()) in terms[aspect]:
                        term = term.replace("_"," ")
                        terms[aspect].append(term)
            for lemma in synset.lemmas():
                for form in lemma.derivationally_related_forms():
                    if not (term := form.name()) in terms[aspect]:
                        term = term.replace("_"," ")
                        terms[aspect].append(term)
                if not (term := lemma.name()) in terms[aspect]:
                    term = term.replace("_"," ")
                    terms[aspect].append(term)
    return terms


def basic_parse_opinion(
    review: str,
    asp_pos: Dict[str,List[Tuple[int,int]]],
    op_lexicon: ...,
    context_range: int = 1,
    show_context:bool = False
) -> Union[Dict[str,List[Tuple[str,str,float]]],
           None]:
    """
    Identifies opinions related to previously found terms by identifying
    adjectives or adverbs around the term's context (no dependency parsing).

    Parameters
    ----------
    review:
        Text to be analyzed
    asp_pos:
        Dictionary with List[(phr_pos,tok_pos)] for each aspect term
        (output of match functions)
            phr_pos: phrase number of the term
            tok_pos: token number of the term
    op_lexicon:
        Object with .positive() and .negative() methods that evaluate to lists
        with positive (polarity=+1) and negative (polarity=-1) opinion words
        respectively
    context_range:
        Number of tokens around the aspect term used in finding opinion
        words
    show_context:
        Boolean flag controlling the output (see Returns)

    Returns
    -------
    aspect_opinions <= show_context::False
        Dictionary with aspects as keys and lists of tuples of opinions as
        values; List[(term, op_word, polarity)]
            term: aspect term
            op_word: opinion word (adjective or adverb)
            polarity: +/-1
    None            <= show_context::True
        None; Prints the context (+/-context_range tokens around each
        term) instead
    """
    aspect_opinions = {}

    # Sentence tokenization and POS tagging
    # POS tags are flattened to list of lists like sentences
    sentences = nltk.sent_tokenize(review)
    sentences = [nltk.word_tokenize(s) for s in sentences]
    pos_tags = [[t[1] for t in nltk.pos_tag(s)] for s in sentences]

    for aspect in asp_pos.keys():
        for term in asp_pos[aspect]:
            # for word in context...
            # try statement to check if there's a +/-i times
            # word
            for i in [j for j in range(-context_range,context_range+1) if j!=0]:
                try:
                    this_word = sentences[term[0]][term[1]+i]
                    this_pos = pos_tags[term[0]][term[1]+i]
                except:
                    continue
                # If the pos tag of the word is in the propper
                # class (adj/adv; aspect_classes defined at
                # the beginning)
                if this_pos in aspect_classes:
                    if this_word in op_lexicon.positive():
                        polar = 1
                    elif this_word in op_lexicon.negative():
                        polar = -1
                    else:
                        continue
                    # tupple of term, op_word, polarity to
                    # append
                    value = [sentences[term[0]][term[1]],
                             sentences[term[0]][term[1]+i],
                             polar]
                    if show_context:
                        # The aspect term is printed in red
                        # with RED + "..." + REG and the
                        # opinion word in magenta

                        # To use the colors in the correct
                        # order we consider the case where
                        # each goes first
                        if i<0:
                            print(f"Aspect: {aspect}{' '*(20-len(aspect))}" +
                                  f"Term: {sentences[term[0]][term[1]]}{' '*(15-len(sentences[term[0]][term[1]]))}\n" +
                                  f"Context: {' '.join(sentences[term[0]][:term[1]+i])}" +
                                  f"{MAG} {sentences[term[0]][term[1]+i]} {REG}"+
                                  f"{' '.join(sentences[term[0]][term[1]+i+1:term[1]])}"+
                                  f"{RED} {sentences[term[0]][term[1]]} {REG}"+
                                  f"{' '.join(sentences[term[0]][term[1]+1:])}\n\n")
                        if i>0:
                            print(f"Aspect: {aspect}{' '*(20-len(aspect))}" +
                                  f"Term: {sentences[term[0]][term[1]]}{' '*(15-len(sentences[term[0]][term[1]]))}\n" +
                                  f"Context: {' '.join(sentences[term[0]][:term[1]])}" +
                                  f"{RED} {sentences[term[0]][term[1]]} {REG}"+
                                  f"{' '.join(sentences[term[0]][term[1]+1:term[1]+i])}" +
                                  f"{MAG} {sentences[term[0]][term[1]+i]} {REG}"+
                                  f"{' '.join(sentences[term[0]][term[1]+i+1:])}\n\n")
                    try:
                        aspect_opinions[aspect].append(value)
                    except:
                        aspect_opinions[aspect] = [value]

    if show_context:
        return
    return aspect_opinions


def advanced_parse_opinion(
    review: str,
    asp_pos: Dict[str,List[Tuple[int,int]]],
    op_lexicon: ...,
    modifiers: pd.DataFrame,
    show_context: bool = False,
    corenlp_port: int = 9000
) -> Union[Dict[str,List[Tuple[str,str,str,str,float]]],
           None]:
    """
    Identifies opinions related to previously found terms by identifying
    adjectives or adverbs dependent to the term. Also finds modifiers of
    the opinion words and negations.

    IMPORTANT: uses the corenlp server for dependency parsing

    Parameters
    ----------
    review:
        Text to be analyzed
    asp_pos:
        Dictionary with List[(phr_pos,tok_pos)] for each aspect term
        (output of match functions)
            phr_pos: phrase number of the term
            tok_pos: token number of the term
    op_lexicon:
        Object with .positive() and .negative() methods that evaluate to lists
        with positive (polarity=+1) and negative (polarity=-1) opinion words
        respectively
    modifiers:
        Pandas dataframe with "term" and "polarity" columns. Terms include the
        words that will be identified and polarity are the modifier's polarity
        values (considered multiplicative)
    show_context:
        Boolean flag controlling the output (see Returns)
    corenlp_port:
        Port number for the CoreNLP server (default is 9000 but that one was
        in use in my computer so had to allow the option to change it)

    Returns
    -------
    aspect_opinions <= show_context::False
        Dictionary with aspects as keys and lists of tuples of opinions as
        values; List[(term, op_word, modifier, "Neg=...", polarity)]
            term: aspect term
            op_word: opinion word (adjective or adverb)
            "Neg=...": where ... stands for "True" or "False"
            polarity: +/-1
    None            <= show_context::True
        None; Prints the context instead (the whole phrase so modifiers
        and negations can be understood in context)
    """
    aspect_opinions = {}

    # Dependency parser connection to the specified port
    dep_parser = CoreNLPDependencyParser(url=f'http://localhost:{corenlp_port}')

    sentences = nltk.sent_tokenize(review)

    for aspect in asp_pos.keys():
        for term in asp_pos[aspect]:
            # For each aspect,term pair get the sentence and
            # term and parse the sentence
            this_sentence = sentences[term[0]]
            this_term = nltk.word_tokenize(this_sentence)[term[1]]
            dependencies, = dep_parser.raw_parse(this_sentence)

            # flag to keep track of sentence negation
            is_negated = False
            for head,relation,dependent in dependencies.triples():
                # For each relation
                #   * Check if the term is a head to a no/not dependence
                #   * Check if the term is a head in an adj/adv dependence
                #   * Check if the term is a dependent in an adj/adv ...

                # Keep track of whether an opinion word was found through
                # any of the above mechanisms
                this_opinion = None

                if head[0].lower() == this_term and dependent[0].lower() in ['not','no']:
                    is_negated=True
                # aspect_relations_1 contains accepted relations with term
                # as a head (defined at the start)
                if head[0].lower() == this_term and relation in aspect_relations_1:
                    this_opinion = dependent[0].lower()
                # aspect_relations_1 contains accepted relations with term
                # as a dependent (defined at the start)
                elif dependent[0].lower() == this_term and relation in aspect_relations_2:
                    this_opinion = head[0].lower()
                if this_opinion:
                    # If opinion word was found stablish polarity
                    if this_opinion in op_lexicon.positive():
                        polar = 1
                    elif this_opinion in op_lexicon.negative():
                        polar = -1
                    # If it doesn't appear in the lexicon drop it
                    else:
                        continue

                    # bool flag that keeps track of whether a
                    # modifier was found
                    open = True

                    # Once an opinion word is found check if it
                    # has modifiers by finding negation dependences
                    # or it as a head in adv/adj dependences
                    for subhead, subrelation, subdependent in dependencies.triples():
                        if subhead[0].lower() == this_opinion and subdependent[0].lower() in ['not','no']:
                            is_negated=True
                        # aspect_subrelations holds relation types
                        # accepted for modifiers
                        if subhead[0].lower() == this_opinion and subrelation in aspect_subrelations:
                            this_modifier = subdependent[0].lower()
                            if len(mod_info := modifiers.loc[modifiers['term'] == this_modifier]) > 0:
                                # modifiers are considered
                                # multiplicative
                                polar *= mod_info['polarity'].values[0]
                                # negation inverts polarity
                                polar = -1*polar if is_negated else polar
                                value = [this_term, this_opinion, this_modifier, f"Neg={is_negated}", polar]
                                if show_context:
                                    print(f"Aspect: {aspect}{' '*(20-len(aspect))}" +
                                          f"Term: {this_term}{' '*(15-len(this_term))}" +
                                          f"Modif: {this_modifier}{' '*(15-len(this_modifier))}\n" +
                                          f"Value: {value}\n" +
                                          f"Context: {this_sentence}\n\n")
                                try:
                                    aspect_opinions[aspect].append(value)
                                except:
                                    aspect_opinions[aspect] = [value]
                                open = False
                    # If no modifier was found append/show a value
                    # without modifier
                    if open:
                        polar = -1*polar if is_negated else polar
                        value = [this_term, this_opinion, "", f"Neg={is_negated}", polar]
                        if show_context:
                            print(f"Aspect: {aspect}{' '*(20-len(aspect))}" +
                                  f"Term: {this_term}{' '*(15-len(this_term))}\n" +
                                  f"Value: {value}\n" +
                                  f"Context: {this_sentence}\n\n")
                        try:
                            aspect_opinions[aspect].append(value)
                        except:
                            aspect_opinions[aspect] = [value]
    if show_context:
        return
    return aspect_opinions


def display_opinions(
    aspect_opinions: Dict[str, List[Tuple[str,str,str,str,float]]]
) -> None:
    """
    Display identified opinion tuples.

    Parameters
    ----------
    aspect_opinions:
        Dictionary with aspects as keys and lists of tuples of opinions as
        values; List[(term, op_word, modifier, "Neg=...", polarity)]
            term: aspect term
            op_word: opinion word (adjective or adverb)
            "Neg=...": where ... stands for "True" or "False"
            polarity: +/-1

    Returns
    -------
    None:
        Shows each aspect and their opinion tuples in the following format:

        Aspect: ...
            Term: ... (polarity): opinion_word, modifier, Neg=...
            ...
        ...
    """
    for aspect in aspect_opinions.keys():
        print(f"Aspect: {aspect}")
        for term,opinion,modif,neg,polar in aspect_opinions[aspect]:
            print(f"\tTerm: {term} ({polar}): {opinion}, {modif}, {neg}")


def summarize(
    aspect_opinions: Dict[str, List[Tuple[str,str,str,str,float]]]
) -> pd.DataFrame:
    """
    Calculates several summary measures of aspect opinions.

    Parameters
    -----------
    aspect_opinions:
        Dictionary with aspects as keys and lists of tuples of opinions as
        values; List[(term, op_word, modifier, "Neg=...", polarity)]
            term: aspect term
            op_word: opinion word (adjective or adverb)
            "Neg=...": where ... stands for "True" or "False"
            polarity: +/-1

    Returns
    -------
    summary:
        Pandas dataframe with the following columns:
            Aspect: each of the aspects
            Positive count: total number of positive opinions
            Negative count: total number of negative opinions
            Total polarity: sum of all opinion's polarities
                            (of a given aspect)
            Mean polarity: mean of opinion's polarities
                            (of a given aspect)
            Polarity variance: variance of opinion's polarities
                               (of a given aspect)
    """
    summary = []
    for aspect in aspect_opinions.keys():
        # polarity list as the last value in opinion tuples
        polarity = np.array([op[-1] for op in aspect_opinions[aspect]])
        value = {'Aspect':aspect,
                 'Positive count': np.sum(polarity>0),
                 'Negative count': np.sum(polarity<0),
                 'Total polarity': np.sum(polarity),
                 'Mean polarity': np.mean(polarity),
                 'Polarity variance': np.var(polarity)}
        summary.append(value)

    # Turn list of dictionaries into DataFrame
    summary = pd.DataFrame(summary)
    summary = summary.sort_values(by='Total polarity', ascending=False)
    summary.reset_index(drop=True, inplace=True)
    return summary


def main():
    pass


if __name__ == '__main__':
    main()
