import wikipediaapi
from operator import itemgetter
from nltk.probability import FreqDist
from nltk.util import ngrams
import yake
from collections import Counter
import string
from nltk.tokenize import word_tokenize, sent_tokenize
import re
from bisect import bisect_left
from collections import Counter
from nltk.stem.snowball import SnowballStemmer



class KeywordExtractor:

    def __init__(self):
        self.wiki = wikipediaapi.Wikipedia(language='en', extract_format=wikipediaapi.ExtractFormat.WIKI)

    
    def _take_closest(self, myList : list, myNumber : int):
        """
        Assumes myList is sorted. Returns closest value to myNumber.

        If two numbers are equally close, return the smallest number.
        """
        pos = bisect_left(myList, myNumber)
        if pos == 0:
            return myList[0]
        if pos == len(myList):
            return myList[-1]
        before = myList[pos - 1]
        after = myList[pos]
        if after - myNumber < myNumber - before:
            return after
        else:
            return before


    def _find_end_of_sentences(self, text : string):
        indices_object = re.finditer('\.', text)
        return [index.start() for index in indices_object]

    def _decompose_paragraph(self, paragraph:string):
        
        indeces = [self._take_closest(self._find_end_of_sentences(paragraph), i * 500) for i in range(1, round(len(paragraph)/500))]

        sub_text = []

        for i in range(len(indeces)+1):
            if i == 0:
                sub_text.append(paragraph[:indeces[i]+1])
            elif i == len(indeces):
                sub_text.append(paragraph[indeces[i-1]+1:])
            else:
                sub_text.append(paragraph[indeces[i-1]+1:indeces[i]+1])

        return sub_text

    def _flatten(self, A:list):
        rt = []
        for i in A:
            if isinstance(i,list): rt.extend(self._flatten(i))
            else: rt.append(i)
        return rt

    def _get_paragraphs(self, topic : string):
        document = wikipediaapi.Wikipedia(language='en', extract_format=wikipediaapi.ExtractFormat.WIKI).page(topic).text
        document = [x for x in document.split("\n") if len(word_tokenize(x)) >= 10]
        paragraphs = [self._decompose_paragraph(x) if len(x)/500 >= 1.5 else x for x in document]
        paragraphs = self._flatten(paragraphs)
        paragraphs = [paragraph for paragraph in paragraphs if len(sent_tokenize(paragraph)) > 1]
        return paragraphs
    
        
    def _most_frequent_words(self, text : list):
        freq = FreqDist(text)
        return sorted(freq.items(), key=itemgetter(1), reverse=True)[:20]

    def _get_n_grams(self, text : string, n : int):
        result = []
        n_grams = ngrams(text.split(), n)
        for grams in n_grams :
            result.append(grams)
        return result

    def _is_duplicate(self, keyword : string, keywords : list):

        if [self.wiki.page(x).text[:50] for x in keywords].count(self.wiki.page(keyword).text[:50]) >= 1:
            return True
        
        return False


    def _remove_duplicates(self, keywords : list):
        for keyword in keywords:
            if self._is_duplicate(keyword, keywords):
                keywords.remove(keyword)
        return keywords

    def _get_answer_keywords(self, answer : string):
        language = "en"
        max_ngram_size = 3
        deduplication_thresold = 0.9
        deduplication_algo = 'seqm'
        windowSize = 1
        numOfKeywords = 20
        custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresold, dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords, features=None)
        keywords = custom_kw_extractor.extract_keywords(answer)
        return [x[0].lower() for x in keywords if self.wiki.page(x[0]).exists()]

    def _topic_relevance_score(self, word, keywords : list):
        score = 0
        for y in keywords:
            if y != word:
                score += [x.lower() for x in word_tokenize(self.wiki.page(word).text)].count(y)
        return score

    def _get_stopwords(self):
        file = open("./actions/stopwords.txt", "rb")
        stopwords = []
        for word in file:
            stopwords.append(SnowballStemmer("english").stem(re.sub('\n', '', word.decode("utf-8"))))
        return stopwords

    def _remove_stopwords(self, document : string):
        stopwords = self._get_stopwords()
        words = [word.lower() for word in word_tokenize(document) if SnowballStemmer("english").stem(word.lower()) not in stopwords and word.isalpha() is True]
        return " ".join(words)

    def get_keywords_from_previous_answer(self, previous_answer: string, topics: dict):
    
        keywords = self._get_answer_keywords(previous_answer)

        new_topics_from_previous_answer = {}
            
        for keyword in keywords:
            if list(topics.keys()).count(keyword) == 0:
                score = self._topic_relevance_score(keyword, list(topics.keys()))
                if score > 0 and score < 10:
                    new_topics_from_previous_answer[keyword] = {"topic_weight": 1, "contexts": self._get_paragraphs(keyword)}
        
        return new_topics_from_previous_answer

    def get_doc_keywords(self, text:string):

        bigrams = Counter(self._get_n_grams(self._remove_stopwords(text), 2)).most_common(10)
        trigrams = Counter(self._get_n_grams(self._remove_stopwords(text), 2)).most_common(10)
        keywords = [] + [x[0][0] + " " + x[0][1] for x in bigrams] + [x[0][0] + " " + x[0][1] + x[0][1] for x in trigrams]
        keywords.append(self._most_frequent_words(word_tokenize(self._remove_stopwords(text)))[0][0])
        keywords = keywords + [x[0] for x in self._most_frequent_words([x for y in trigrams for x in y[0]])[:3]]
        keywords = keywords + [x[0] for x in self._most_frequent_words([x for y in bigrams for x in y[0]])[:3]]
        keywords = [x for x in keywords if self.wiki.page(x).exists() and self.wiki.page(x).text[:100].find("may refer to") == -1]
        keywords = self._remove_duplicates(keywords)

        topic_relevance_scores = {}
            
        for kw in keywords:
            topic_relevance_scores[kw] = self._topic_relevance_score(kw, keywords)  
        
        keywords = [x for x in topic_relevance_scores if topic_relevance_scores[x] >= 10]

        topics = {}
                
        for topic in keywords:
            topics[topic] = {"topic_weight": 1, "contexts": self._get_paragraphs(topic)}

        return topics