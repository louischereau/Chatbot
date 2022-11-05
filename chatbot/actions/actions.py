# This files contains your custom actions which can be used to run
# custom Python code.

# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

from typing import Any, Text, Dict, List

from sklearn.metrics import f1_score

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet
from rasa_sdk.events import AllSlotsReset
from rasa_sdk.events import EventType
from rasa_sdk.events import Restarted
from rasa_sdk import FormValidationAction
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.types import DomainDict

import boto3
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')
#nltk.download('omw-1.4')
from actions.DocComparison import DocComparer
from actions.KeywordExtractor import KeywordExtractor
from actions.t5 import T5Model

import os
import random
import string
from tabulate import tabulate
from nltk.tokenize import sent_tokenize, word_tokenize
from datetime import datetime 



BUCKET_NAME = 'individualprojectfiles'
ASSESSMENT_BUCKET_NAME = 'individualprojectreports'
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        )

s3 = session.resource('s3')


doccomparer = DocComparer()
keywordextractor = KeywordExtractor()
questiongenerator = T5Model()


question_types = ["t5_question", "template_question"]


template_questions = {
                        "individualproject": ["Why did you choose this project?", "What did you learn during this project?", "What are the major challenges you encountered during this project?", "If you had to start over what would you do differently?", "In your opinion was the outcome of your project a success? Explain why or why not.", "Did you enjoy doing this project? Explain why or why not.", "What tools or technologies, if any, did you use in your project?", "How could this project be used to help people?", "Are there any other potential explanations (not mentioned in your thesis) for the results you obtained? If so what are they?"], 
                        "groupproject": ["What were your contributions in this group project?", "Who did your work with for this group project?", "What did you learn during this group project?", "What are the major challenges you encountered during this group project?", "If you had to start over what would you do differently?", "In your opinion was the outcome of your group project a success? Explain why or why not.", "Did you enjoy doing this group project? Explain why or why not.", "What tools or technologies, if any, did you use in your group project?", "How could this project be used to help people?", "Are there any other potential explanations (not mentioned in your thesis) for the results you obtained? If so what are they?"],
                        "essay": ["What did you learn during the writing of this essay?", "What was your major source of information to write this essay?", "What reference managament tool, if any, did you use?", "Did you enjoy writing this essay? Explain why or why not.", "If you had to write this essay again what would you add and/or remove and why?", "If you had to rewrite your conclusion, what would be your changes?"], 
                        "review":["What did you learn during this review?", "What was your major source of information to conduct this review?", "What reference managament tool, if any, did you use?", "Did you enjoy doing this review? Explain why or why not.", "If you had to conduct this review again what would you add and/or remove and why?"], 
                        "report": ["What did you learn during this assignment?", "What are the major challenges you encountered during this assignment?", "If you had to start over what would you do differently?", "In your opinion was the outcome of this assignment a success? Explain why or why not.", "Did you enjoy doing this assignment? Explain why or why not.", "What tools or technologies, if any, did you use in this assignment?"]
                     }



def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string, re

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_f1(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()
    
    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    
    common_tokens = set(pred_tokens) & set(truth_tokens)
    
    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0
    
    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)
    
    return 2 * (prec * rec) / (prec + rec)



def determine_current_score(previous_answer : string, previous_expected_answer : string, current_score : float):

    if len(previous_expected_answer) == 0 or previous_expected_answer == " ":
        return current_score
    
    if len(word_tokenize(previous_answer)) > 5:
        previous_expected_answer_sentences = sent_tokenize(previous_expected_answer)
        score = 1 if round(max([doccomparer.document_path_similarity(normalize_text(previous_answer), normalize_text(x)) for x in  previous_expected_answer_sentences]),2) > 0.4 else 0
    else:
        score = 1 if round(compute_f1(normalize_text(previous_expected_answer), normalize_text(previous_answer)), 2) >= 0.5 else 0

    current_score = (current_score + score) 

    return current_score


def generate_t5_question(previous_answer:string, previous_expected_answer:string, current_score: float, topics:list, number_of_t5_questions_asked : float):
    
    current_score = determine_current_score(previous_answer, previous_expected_answer, current_score)

    new_topics = keywordextractor.get_keywords_from_previous_answer(previous_answer, topics)
        
    if (current_score / number_of_t5_questions_asked) > 0.4 and len(new_topics) > 2:
        question, expected_answer, topics = questiongenerator.get_next_question(new_topics, "hard", based_on_previous_answer = True)
    elif (current_score / number_of_t5_questions_asked) > 0.4 and ((len(new_topics) > 0) and (len(new_topics) < 2)):
        question, expected_answer, topics = questiongenerator.get_next_question(new_topics, "easy", based_on_previous_answer = True)
    elif (current_score / number_of_t5_questions_asked) > 0.4 and len(new_topics) == 0:
        question, expected_answer, topics = questiongenerator.get_next_question(topics, "hard", based_on_previous_answer = False)
    else:
        question, expected_answer, topics = questiongenerator.get_next_question(topics, "easy", based_on_previous_answer = False)

    return question, expected_answer, topics, current_score



    

class ActionHelloWorld(Action):

    def name(self) -> Text:
        return "action_hello_world"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        dispatcher.utter_message(text="Hello. I am your examiner, Padric. First, upload a text file of your work. Then send me the name of your file.")

        return []


class ActionSetup(Action):

     def name(self) -> Text:
         return "action_setup"

     def _get_file(self, filename : string):
        s3.Bucket(BUCKET_NAME).download_file(filename, "./" + filename)
        file = open(filename, "rb")
        lines = []
        for line in file:
            lines.append(line.decode("utf-8").strip())
        return " ".join(lines)


     def run(self, dispatcher: CollectingDispatcher,
             tracker: Tracker,
             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        start_time = datetime.now()

        filename = tracker.get_slot('filename')

        if ".txt" not in filename:
            filename = filename + ".txt"

        my_bucket = s3.Bucket('individualprojectfiles')

        project_files = []

        for my_bucket_object in my_bucket.objects.all():
            project_files.append(my_bucket_object.key)

        if filename in project_files:
            
            text = self._get_file(filename)

            assignment_type = filename.split("_")[1][:-4]

            topics = keywordextractor.get_doc_keywords(text)

            print('Time elapsed (hh:mm:ss.ms) {}'.format(datetime.now() - start_time))

            return [SlotSet("file_uploaded", True), SlotSet("filename", filename), 
            SlotSet("topics", topics), SlotSet("current_score", 0.0), 
            SlotSet("template_questions", template_questions), 
            SlotSet("assignment_type", assignment_type), 
            SlotSet("number_of_t5_questions_asked", 0.0)]
        
        else:
            
            dispatcher.utter_message(response="utter_file_not_found", filename = filename)

            print('Time elapsed (hh:mm:ss.ms) {}'.format(datetime.now() - start_time))
            
            return [SlotSet("file_uploaded", False)]


class AskForAnswer1Action(Action):
    def name(self) -> Text:
        return "action_ask_answer1"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:

        start_time = datetime.now()

        assignment_type = tracker.get_slot("assignment_type")
        
        questions = tracker.get_slot("template_questions")
        
        question = random.choice(questions[assignment_type])

        questions[assignment_type].remove(question)
        
        dispatcher.utter_message(text=question)

        print('Time elapsed (hh:mm:ss.ms) {}'.format(datetime.now() - start_time))

        return [SlotSet("template_questions", questions), SlotSet("question1", (question, ""))]

class AskForAnswer2Action(Action):
    def name(self) -> Text:
        return "action_ask_answer2"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:

        start_time = datetime.now()

        previous_answer = (tracker.latest_message)["text"]
        previous_expected_answer = tracker.get_slot("question1")[1]
        current_score = tracker.get_slot("current_score")
        topics = tracker.get_slot("topics")
        assignment_type = tracker.get_slot("assignment_type")
        questions = tracker.get_slot("template_questions")
        number_of_t5_questions_asked = tracker.get_slot("number_of_t5_questions_asked")
        
        question_type = random.choices(question_types, weights=[1.5, 1], k=1)[0]

        if question_type == "t5_question":
            number_of_t5_questions_asked += 1
            question, expected_answer, topics, current_score = generate_t5_question(previous_answer, previous_expected_answer, current_score, topics, number_of_t5_questions_asked)
            if "?" not in question:
                number_of_t5_questions_asked -= 1
                question_type = "template_question"
        

        if question_type == "template_question": 
            
            question = random.choice(questions[assignment_type])
            expected_answer = ""
            questions[assignment_type].remove(question)
            
        
        dispatcher.utter_message(text=question)

        print('Time elapsed (hh:mm:ss.ms) {}'.format(datetime.now() - start_time))


        return [SlotSet("template_questions", questions), 
        SlotSet("topics", topics), 
        SlotSet("question2", (question, expected_answer)), 
        SlotSet("current_score", current_score), 
        SlotSet("number_of_t5_questions_asked", number_of_t5_questions_asked)]



class AskForAnswer3Action(Action):
    def name(self) -> Text:
        return "action_ask_answer3"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:

        start_time = datetime.now()
        
        previous_answer = (tracker.latest_message)["text"]
        previous_expected_answer = tracker.get_slot("question2")[1]
        current_score = tracker.get_slot("current_score")
        topics = tracker.get_slot("topics")
        assignment_type = tracker.get_slot("assignment_type")
        questions = tracker.get_slot("template_questions")
        number_of_t5_questions_asked = tracker.get_slot("number_of_t5_questions_asked")
        
        question_type = random.choices(question_types, weights=[1.5, 1], k=1)[0]

        if question_type == "t5_question":
            number_of_t5_questions_asked += 1
            question, expected_answer, topics, current_score = generate_t5_question(previous_answer, previous_expected_answer, current_score, topics, number_of_t5_questions_asked)
            if "?" not in question:
                number_of_t5_questions_asked -= 1
                question_type = "template_question"
        

        if question_type == "template_question": 
            
            question = random.choice(questions[assignment_type])
            expected_answer = ""
            questions[assignment_type].remove(question)

        dispatcher.utter_message(text=question)

        print('Time elapsed (hh:mm:ss.ms) {}'.format(datetime.now() - start_time))


        return [SlotSet("template_questions", questions), 
        SlotSet("topics", topics), 
        SlotSet("question3", (question, expected_answer)), 
        SlotSet("current_score", current_score),
        SlotSet("number_of_t5_questions_asked", number_of_t5_questions_asked)]

class AskForAnswer4Action(Action):
    def name(self) -> Text:
        return "action_ask_answer4"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:

        start_time = datetime.now()
        
        previous_answer = (tracker.latest_message)["text"]
        previous_expected_answer = tracker.get_slot("question3")[1]
        current_score = tracker.get_slot("current_score")
        topics = tracker.get_slot("topics")
        questions = tracker.get_slot("template_questions")
        assignment_type = tracker.get_slot("assignment_type")
        number_of_t5_questions_asked = tracker.get_slot("number_of_t5_questions_asked")

        question_type = random.choices(question_types, weights=[1.5, 1], k=1)[0]

        if question_type == "t5_question":
            number_of_t5_questions_asked += 1
            question, expected_answer, topics, current_score = generate_t5_question(previous_answer, previous_expected_answer, current_score, topics, number_of_t5_questions_asked)
            if "?" not in question:
                number_of_t5_questions_asked -= 1
                question_type = "template_question"
        

        if question_type == "template_question": 
           
            question = random.choice(questions[assignment_type])
            expected_answer = ""
            questions[assignment_type].remove(question)


        dispatcher.utter_message(text=question)

        print('Time elapsed (hh:mm:ss.ms) {}'.format(datetime.now() - start_time))


        return [SlotSet("template_questions", questions), 
        SlotSet("topics", topics), 
        SlotSet("question4", (question, expected_answer)), 
        SlotSet("current_score", current_score),
        SlotSet("number_of_t5_questions_asked", number_of_t5_questions_asked)]


class AskForAnswer5Action(Action):
    def name(self) -> Text:
        return "action_ask_answer5"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:

        start_time = datetime.now()
        
        previous_answer = (tracker.latest_message)["text"]
        previous_expected_answer = tracker.get_slot("question4")[1]
        current_score = tracker.get_slot("current_score")
        topics = tracker.get_slot("topics")
        questions = tracker.get_slot("template_questions")
        assignment_type = tracker.get_slot("assignment_type")
        number_of_t5_questions_asked = tracker.get_slot("number_of_t5_questions_asked")


        question_type = random.choices(question_types, weights=[1.5, 1], k=1)[0]

        if question_type == "t5_question":
            number_of_t5_questions_asked += 1
            question, expected_answer, topics, current_score = generate_t5_question(previous_answer, previous_expected_answer, current_score, topics, number_of_t5_questions_asked)
            if "?" not in question:
                number_of_t5_questions_asked -= 1
                question_type = "template_question"
        

        if question_type == "template_question":
            
            question = random.choice(questions[assignment_type])
            expected_answer = ""
            questions[assignment_type].remove(question)

        
        dispatcher.utter_message(text=question)

        print('Time elapsed (hh:mm:ss.ms) {}'.format(datetime.now() - start_time))

        
        return [SlotSet("template_questions", questions), 
        SlotSet("topics", topics), 
        SlotSet("question5", (question, expected_answer)), 
        SlotSet("current_score", current_score),
        SlotSet("number_of_t5_questions_asked", number_of_t5_questions_asked)]


class AskForAnswer6Action(Action):
    def name(self) -> Text:
        return "action_ask_answer6"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:

        start_time = datetime.now()
        
        previous_answer = (tracker.latest_message)["text"]
        previous_expected_answer = tracker.get_slot("question5")[1]
        current_score = tracker.get_slot("current_score")
        topics = tracker.get_slot("topics")
        questions = tracker.get_slot("template_questions")
        assignment_type = tracker.get_slot("assignment_type")
        number_of_t5_questions_asked = tracker.get_slot("number_of_t5_questions_asked")

        question_type = random.choices(question_types, weights=[1.5, 1], k=1)[0]

        if question_type == "t5_question":
            number_of_t5_questions_asked += 1
            question, expected_answer, topics, current_score = generate_t5_question(previous_answer, previous_expected_answer, current_score, topics, number_of_t5_questions_asked)
            if "?" not in question:
                number_of_t5_questions_asked -= 1
                question_type = "template_question"
        

        if question_type == "template_question": 
           
            question = random.choice(questions[assignment_type])
            expected_answer = ""
            questions[assignment_type].remove(question)

        
        dispatcher.utter_message(text=question)

        print('Time elapsed (hh:mm:ss.ms) {}'.format(datetime.now() - start_time))

        
        return [SlotSet("template_questions", questions), 
        SlotSet("topics", topics), 
        SlotSet("question6", (question, expected_answer)), 
        SlotSet("current_score", current_score),
        SlotSet("number_of_t5_questions_asked", number_of_t5_questions_asked)]


class AskForAnswer7Action(Action):
    def name(self) -> Text:
        return "action_ask_answer7"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:

        start_time = datetime.now()
        
        previous_answer = (tracker.latest_message)["text"]
        previous_expected_answer = tracker.get_slot("question6")[1]
        current_score = tracker.get_slot("current_score")
        topics = tracker.get_slot("topics")
        questions = tracker.get_slot("template_questions")
        assignment_type = tracker.get_slot("assignment_type")
        number_of_t5_questions_asked = tracker.get_slot("number_of_t5_questions_asked")


        question_type = random.choices(question_types, weights=[1.5, 1], k=1)[0]

        if question_type == "t5_question":
            number_of_t5_questions_asked += 1
            question, expected_answer, topics, current_score = generate_t5_question(previous_answer, previous_expected_answer, current_score, topics, number_of_t5_questions_asked)
            if "?" not in question:
                number_of_t5_questions_asked -= 1
                question_type = "template_question"
        

        if question_type == "template_question": 
            
            question = random.choice(questions[assignment_type])
            expected_answer = ""
            questions[assignment_type].remove(question)
     
        
        dispatcher.utter_message(text=question)

        print('Time elapsed (hh:mm:ss.ms) {}'.format(datetime.now() - start_time))


        return [SlotSet("template_questions", questions), 
        SlotSet("topics", topics), 
        SlotSet("question7", (question, expected_answer)), 
        SlotSet("current_score", current_score),
        SlotSet("number_of_t5_questions_asked", number_of_t5_questions_asked)]


class AskForAnswer8Action(Action):
    def name(self) -> Text:
        return "action_ask_answer8"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:

        start_time = datetime.now()
        
        previous_answer = (tracker.latest_message)["text"]
        previous_expected_answer = tracker.get_slot("question7")[1]
        current_score = tracker.get_slot("current_score")
        topics = tracker.get_slot("topics")
        questions = tracker.get_slot("template_questions")
        assignment_type = tracker.get_slot("assignment_type")
        number_of_t5_questions_asked = tracker.get_slot("number_of_t5_questions_asked")

        question_type = random.choices(question_types, weights=[1.5, 1], k=1)[0]

        if question_type == "t5_question":
            number_of_t5_questions_asked += 1
            question, expected_answer, topics, current_score = generate_t5_question(previous_answer, previous_expected_answer, current_score, topics, number_of_t5_questions_asked)
            if "?" not in question:
                number_of_t5_questions_asked -= 1
                question_type = "template_question"
        

        if question_type == "template_question":
            question = random.choice(questions[assignment_type])
            expected_answer = ""
            questions[assignment_type].remove(question)

        
        dispatcher.utter_message(text=question)

        print('Time elapsed (hh:mm:ss.ms) {}'.format(datetime.now() - start_time))


        return [SlotSet("template_questions", questions), 
        SlotSet("topics", topics), 
        SlotSet("question8", (question, expected_answer)), 
        SlotSet("current_score", current_score),
        SlotSet("number_of_t5_questions_asked", number_of_t5_questions_asked)]


class AskForAnswer9Action(Action):
    def name(self) -> Text:
        return "action_ask_answer9"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:

        start_time = datetime.now()
        
        previous_answer = (tracker.latest_message)["text"]
        previous_expected_answer = tracker.get_slot("question8")[1]
        current_score = tracker.get_slot("current_score")
        topics = tracker.get_slot("topics")
        questions = tracker.get_slot("template_questions")
        assignment_type = tracker.get_slot("assignment_type")
        number_of_t5_questions_asked = tracker.get_slot("number_of_t5_questions_asked")
       

        question_type = random.choices(question_types, weights=[1.5, 1], k=1)[0]

        if question_type == "t5_question":
            number_of_t5_questions_asked += 1
            question, expected_answer, topics, current_score = generate_t5_question(previous_answer, previous_expected_answer, current_score, topics, number_of_t5_questions_asked)
            if "?" not in question:
                number_of_t5_questions_asked -= 1
                question_type = "template_question"
        

        if question_type == "template_question": 
            
            question = random.choice(questions[assignment_type])
            expected_answer = ""
            questions[assignment_type].remove(question)
        
        dispatcher.utter_message(text=question)

        print('Time elapsed (hh:mm:ss.ms) {}'.format(datetime.now() - start_time))


        return [SlotSet("template_questions", questions), 
        SlotSet("topics", topics), 
        SlotSet("question9", (question, expected_answer)), 
        SlotSet("current_score", current_score),
        SlotSet("number_of_t5_questions_asked", number_of_t5_questions_asked)]



class AskForAnswer10Action(Action):
    def name(self) -> Text:
        return "action_ask_answer10"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:

        start_time = datetime.now()
        
        previous_answer = (tracker.latest_message)["text"]
        previous_expected_answer = tracker.get_slot("question9")[1]
        current_score = tracker.get_slot("current_score")
        topics = tracker.get_slot("topics")
        questions = tracker.get_slot("template_questions")
        assignment_type = tracker.get_slot("assignment_type")
        number_of_t5_questions_asked = tracker.get_slot("number_of_t5_questions_asked")

        question_type = random.choices(question_types, weights=[1.5, 1], k=1)[0]

        if question_type == "t5_question":
            number_of_t5_questions_asked += 1
            question, expected_answer, topics, current_score = generate_t5_question(previous_answer, previous_expected_answer, current_score, topics, number_of_t5_questions_asked)
            if "?" not in question:
                number_of_t5_questions_asked -= 1
                question_type = "template_question"
        

        if question_type == "template_question":  
            question = random.choice(questions[assignment_type])
            expected_answer = ""
            questions[assignment_type].remove(question)
            
        
        dispatcher.utter_message(text=question)

        print('Time elapsed (hh:mm:ss.ms) {}'.format(datetime.now() - start_time))


        return [SlotSet("template_questions", questions), 
        SlotSet("topics", topics), 
        SlotSet("question10", (question, expected_answer)), 
        SlotSet("current_score", current_score),
        SlotSet("number_of_t5_questions_asked", number_of_t5_questions_asked)]



class SubmitReportAction(Action):
    
    def name(self) -> Text:
        return "action_submit_report"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:

        filename = tracker.get_slot('filename')
        
        report = open(filename.split(".")[0] + "_assessment.txt", "w")

        table_score = []

        for i in range(1, 11):
            
            question, expected_answer = tracker.get_slot("question" + str(i))
            answer_provided = tracker.get_slot("answer" + str(i))

            if len(expected_answer) > 0 and expected_answer != " ":
                if len(word_tokenize(answer_provided)) > 5:
                    score = 1 if round(max([doccomparer.document_path_similarity(normalize_text(x), normalize_text(answer_provided)) for x in sent_tokenize(expected_answer)]), 2) > 0.4 else 0
                else:
                    score = 1 if round(compute_f1(normalize_text(expected_answer), normalize_text(answer_provided)), 2) >= 0.5 else 0

                table_score.append(["Question " + str(i), score])

            report.write("Question " + str(i) + ": " + question + "\n\n" + "Expected Answer: " + expected_answer + "\n\n" + "Provided Answer: " + answer_provided)
            report.write("\n\n==========================================================================\n\n")

        report.write("\n\nScore:\n")

        report.write(tabulate(table_score, headers = ["QUESTION#", "Score"], tablefmt="psql"))

        report.close()

        s3.meta.client.upload_file(filename.split(".")[0] + "_assessment.txt", ASSESSMENT_BUCKET_NAME, filename.split(".")[0] + "_assessment.txt")

        os.remove(filename)
        os.remove(filename.split(".")[0] + "_assessment.txt")

        dispatcher.utter_message(response="utter_finished")

        return [AllSlotsReset()]
