
import pytorch_lightning as pl

from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer
)

import string
import random
import nltk


class T5Model(pl.LightningModule):
  
  def __init__(self):
    super().__init__()
    self.easy_model_qg = T5ForConditionalGeneration.from_pretrained("./actions/t5_squad_qg", return_dict=True)
    self.hard_model_qg = T5ForConditionalGeneration.from_pretrained("./actions/t5_hotpot_qg", return_dict=True)
    self.easy_model_qa = T5ForConditionalGeneration.from_pretrained("./actions/t5_squad_qa", return_dict=True)
    self.hard_model_qa = T5ForConditionalGeneration.from_pretrained("./actions/t5_hotpot_qa", return_dict=True)
    self.easy_tokenizer_qg = T5Tokenizer.from_pretrained("./actions/t5_tokenizer_squad_qg")
    self.hard_tokenizer_qg = T5Tokenizer.from_pretrained("./actions/t5_tokenizer_hotpot_qg")
    self.easy_tokenizer_qa = T5Tokenizer.from_pretrained("./actions/t5_tokenizer_squad_qa")
    self.hard_tokenizer_qa = T5Tokenizer.from_pretrained("./actions/t5_tokenizer_hotpot_qa")

  def _generate_question(self, contexts, difficulty):

    if difficulty == "hard":
      context1, context2 = contexts
      input = "context1: %s context2: %s </s>" % (context1, context2)
      source_max_len = 528
      target_max_len = 100
      model = self.hard_model_qg
      tokenizer = self.hard_tokenizer_qg
    else:
      context = contexts[0]
      input = "context: %s </s>" % (context)
      source_max_len = 387
      target_max_len = 34
      model = self.easy_model_qg
      tokenizer = self.easy_tokenizer_qg

    source_encoding=tokenizer(
        input,
        max_length = source_max_len,
        padding="max_length",
        truncation="only_first",
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt"
      )

    generated_ids = model.generate(
        input_ids=source_encoding["input_ids"],
        attention_mask=source_encoding["attention_mask"],
        num_beams=1,  # greedy search
        max_length=target_max_len,
        repetition_penalty=2.5,
        early_stopping=True,
        use_cache=True)

    preds = [tokenizer.decode(generated_id, skip_special_tokens=True, clean_up_tokenization_spaces=True) for generated_id in generated_ids]
      
    return "".join(preds)



  def _generate_answer(self, question, contexts, difficulty):

    
    if difficulty == "hard":
      context1, context2 = contexts
      input = "question: %s context1: %s context2: %s </s>" % (question, context1, context2)
      source_max_len = 628
      target_max_len = 42
      model = self.hard_model_qa
      tokenizer = self.hard_tokenizer_qa
    else:
      context = contexts[0]
      input = "question: %s context: %s </s>" % (question, context)
      source_max_len = 421
      target_max_len = 22
      model = self.easy_model_qa
      tokenizer = self.easy_tokenizer_qa

    source_encoding = tokenizer(
          input,
          max_length = source_max_len,
          padding="max_length",
          truncation="only_second",
          return_attention_mask=True,
          add_special_tokens=True,
          return_tensors = "pt"
      )
    
    generated_ids = model.generate(
          input_ids=source_encoding["input_ids"],
          attention_mask=source_encoding["attention_mask"],
          num_beams=1,
          max_length=target_max_len,
          repetition_penalty=2.5,
          length_penalty=1.0,
          early_stopping=True,
          use_cache=True
      )
    
    preds = [
              tokenizer.decode(generated_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
              for generated_id in generated_ids 
      ]
    
    return "".join(preds)

  
  def _update_topics(self, topics : list, topic: string, contexts: list):

    for context in contexts:
      if context in topics[topic]["contexts"]:
        topics[topic]["contexts"].remove(context)

    if len(topics[topic]["contexts"]) == 0:
      topics.pop(topic)
        
    topics[topic]["topic_weight"] += 1

    return topics


  def get_next_question(self, topics : list, difficulty : string, based_on_previous_answer = False):

      if len(topics) == 0:
          return "", "", topics

      topic = random.choices(list(topics.keys()), weights = [1/x for x in [y["topic_weight"] for y in list(topics.values())]], k = 1)[0]
      
      if difficulty == "hard" and len(topics[topic]["contexts"]) >= 2:
          contexts = random.sample(topics[topic]["contexts"], 2)
          question = self._generate_question(contexts, "hard")
          answer = self._generate_answer(question, contexts, "hard")
      else:
          contexts = random.sample(topics[topic]["contexts"], 1)
          question = self._generate_question(contexts, "easy")
          answer = self._generate_answer(question, contexts, "easy")

      topics = self._update_topics(topics, topic, contexts)

      if based_on_previous_answer: 
        question = "You just mentionned " + str(topic) + ". " + str(question)

      return question, answer, topics
  


