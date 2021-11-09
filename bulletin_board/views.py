from django.shortcuts import render
# from .models import Post
from django.shortcuts import redirect
import datetime
import pandas as pd
import numpy as np
import re
import fasttext
import fasttext.util
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, scan
from elasticsearch_dsl import Search
from elasticsearch_dsl import Q
from ast import literal_eval
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer


# Create your views here.

def home(request):
	return render(request, 'bulletin_board/home.html')

def results(request):
	search_q = request.GET.get('search')
	search_query = []
	search_keyword = {
		'keyword': search_q
	}
	search_query.append(search_keyword)

	fasttext.util.download_model('tr', if_exists='ignore')

	try:
		ft = fasttext.load_model('cc.tr.300.bin')
	except:
		print("Couldn't load the model")

	# DOCUMENT SEARCH
	documents = document_search(search_q, ft)

	print("\n")
	print("DOCUMENT SEARCH RESULT")
	print(documents)

	# QUESTION & ANSWER
	answer, score = question_answer(search_q, documents)
	answer = answer.capitalize()
	answer_query = []
	answer_d = {
		'answer_key': answer,
		'score': score
	}
	answer_query.append(answer_d)

	print("\n")
	
	display_data = []
	for index, row in documents.iterrows():
		document_dict = {}
		document_dict['document_id'] = row['document_id']
		document_dict['topic_id'] = row['topic_id']
		document_dict['document_name'] = row['document_name']
		document_dict['topic'] = row['topic']
		document_dict['content'] = row['content']
		document_dict['score'] = row['score']
		display_data.append(document_dict)

	context = {
		'posts': display_data,
		'search_q': search_query,
		'answer_key': answer_query
	}

	return render(request, 'bulletin_board/results.html', context)

def question_answer(input_query, document_search_result):
	model = AutoModelForQuestionAnswering.from_pretrained("kuzgunlar/electra-turkish-qa")
	tokenizer = AutoTokenizer.from_pretrained("kuzgunlar/electra-turkish-qa")
	qa=pipeline('question-answering', model=model, tokenizer=tokenizer)

	qa_content = ' '.join(row['topic_content_result'] for index, row in document_search_result[:5].iterrows())
	qa_result = qa(question=input_query, context=qa_content)
	return qa_result['answer'], qa_result['score']
	
def semantic_search(input_query):
	script_query = {
            "script_score": {
                "query": {
                    "match_all": {}
                },
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'topic_content_vector') + 1.0",
                    "params": {
                        "query_vector": input_query
                    }
                }
            }
        }

	client = Elasticsearch(hosts=["localhost"], http_auth=('elastic', 'changeme'))

	response = client.search(
		index='akbank_data',
		body={
			"query": script_query
		},
		size=999
	)

	all_hits = response['hits']['hits']

	document_id_list = []
	topic_id_list = []
	document_name_list = []
	topic = []
	content = []
	topic_content_result = []
	search_score = []
	for i in range(len(all_hits)):
		score = all_hits[i]['_score']
		
		document_id = all_hits[i]['_source']
		document_id = document_id['document_id']
		
		topic_id = all_hits[i]['_source']
		topic_id = topic_id['topic_id']
		
		document_name = all_hits[i]['_source']
		document_name = document_name['document_name']
		
		content_text = all_hits[i]['_source']
		content_text = content_text['content']
		
		topic_text = all_hits[i]['_source']
		topic_text = topic_text['topic']
		
		topic_content = all_hits[i]['_source']
		topic_content = topic_content['topic_content']
		
		document_id_list.append(document_id)
		topic_id_list.append(topic_id)
		document_name_list.append(document_name)
		topic.append(topic_text)
		content.append(content_text)
		topic_content_result.append(topic_content)
		search_score.append(score)

	document_search_df = pd.DataFrame()
	document_search_df['document_id'] = document_id_list
	document_search_df['topic_id'] = topic_id_list
	document_search_df['document_name'] = document_name_list
	document_search_df['topic'] = topic
	document_search_df['content'] = content
	document_search_df['topic_content_result'] = topic_content_result
	document_search_df['score'] = search_score

	return document_search_df

def document_search(keyword, ft):
	input_query_vector = ft.get_sentence_vector(keyword)
	document_search_result = semantic_search(input_query_vector)
	return document_search_result