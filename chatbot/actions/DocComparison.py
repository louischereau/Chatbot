import nltk
#nltk.download('wordnet')
from nltk.corpus import wordnet as wn
import string
import wikipediaapi
import textacy

class DocComparer:

    def __init__(self):
        self.wiki = wikipediaapi.Wikipedia(language='en', extract_format=wikipediaapi.ExtractFormat.WIKI)
    
    def _convert_tag(self, tag):
        """Convert the tag given by nltk.pos_tag to the tag used by wordnet.synsets"""
        tag_dict = {'N': 'n', 'J': 'a', 'R': 'r', 'V': 'v'}
        try:
            return tag_dict[tag[0]]
        except KeyError:
            return None

    def _doc_to_synsets(self, doc : string):
        """
        Returns a list of synsets in document.
        Tokenizes and tags the words in the document doc.
        Then finds the first synset for each word/tag combination.
        If a synset is not found for that combination it is skipped.
        Args:
        doc: string to be converted
        Returns:
        list of synsets
        Example:
        doc_to_synsets('Fish are nvqjp friends.')
        Out: [Synset('fish.n.01'), Synset('be.v.01'), Synset('friend.n.01')]
        """
        doc_tokenized = nltk.word_tokenize(doc)
        doc_tokenized_tagged = nltk.pos_tag(doc_tokenized)
        tags = [x[1] for x in doc_tokenized_tagged]
        new_tags =[self._convert_tag(tag) for tag in tags]
        new_doc_tokenized_tagged = list(zip(doc_tokenized, new_tags))
        synsets = [wn.synsets(x[0], x[1])[0] for x in new_doc_tokenized_tagged if len(wn.synsets(x[0], x[1])) > 0]
        return synsets

    def _similarity_score(self, s1, s2):
        """
        Calculate the normalized similarity score of s1 onto s2
        For each synset in s1, finds the synset in s2 with the largest similarity value.
        Sum of all of the largest similarity values and normalize this value by dividing it by the
        number of largest similarity values found.
        Args:
        s1, s2: list of synsets from doc_to_synsets
        Returns:
        normalized similarity score of s1 onto s2
        Example:
        synsets1 = doc_to_synsets('I like cats')
        synsets2 = doc_to_synsets('I like dogs')
        similarity_score(synsets1, synsets2)
        Out: 0.73333333333333339
        """
        scores = list()
        for x in s1:
            scores.append([x.path_similarity(y) for y in s2 if x.path_similarity(y) != None])
        no_empty_list_scores = [x for x in scores if x !=[]]
        best_scores = [max(x) for x in no_empty_list_scores]
        if len(best_scores) == 0:
            return 0
        normalized_score = sum(best_scores)/len(best_scores)
        return normalized_score

    def document_path_similarity(self, doc1 : string, doc2 : string):
        """Finds the symmetrical similarity between doc1 and doc2"""

        #if (doc1 in doc2) or (doc2 in doc1):
        #    return 1
        
        synsets1 = self._doc_to_synsets(doc1)
        synsets2 = self._doc_to_synsets(doc2)
        return (self._similarity_score(synsets1, synsets2) + self._similarity_score(synsets2, synsets1)) / 2

    def levenshtein_similarity(self, doc1 : string, doc2 : string):
        return textacy.similarity.edits.levenshtein(doc1, doc2)