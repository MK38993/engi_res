import os
import re
import pandas as pd
import numpy as np

import datetime as dt

import flask
from flask import Flask, request

import pickle

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from textblob import TextBlob
import string

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


with open('topic_matrix','rb') as f:
	topic_matrix1=pickle.load(f)
f.close()

with open('feature_names','rb') as f:
	xtffn=pickle.load(f)
f.close()
	
with open('RandomForestRegressor','rb') as f:
	RFR=pickle.load(f)
f.close()



def listify(to_listify, uni=False):
	if uni:
		listed=re.findall("'(.+?)'",to_listify)
		for i in range(len(listed)):
			listed[i]=unicodify(listed[i])
		return(listed)
	else:
		return(re.findall("'(.+?)'",to_listify))


def remove_noise(tweet_tokens, stop_words = stopwords.words('english')):
	cleaned_tokens = []

	for token, tag in pos_tag(tweet_tokens):
		token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|''(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
		token = re.sub("(@[A-Za-z0-9_]+)","", token)
		token = re.sub("[0-9]+","number_", token)
		token = re.sub("[,.'-:;!]"," ", token)
		token = re.sub("&amp"," ", token)
		token = re.sub("  "," ", re.sub("  "," ", token))

		if tag.startswith("NN"):
			pos = 'n'
		elif tag.startswith('VB'):
			pos = 'v'
		elif tag.startswith('JJ') or tag.startswith('NNP'):
			pos = 'del'
		else:
			pos = 'a'

		lemmatizer = WordNetLemmatizer()
		if pos!='del':
			token = lemmatizer.lemmatize(token, pos)
		if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words and pos!='adj':
			cleaned_tokens.append(token.lower())
	return cleaned_tokens


def soft_cluster(tokens):
	cluster_score={}
	for i in range(64):
		cluster_score[f'lda_topic{i}']=0
	for token in listify(tokens):
		if token in xtffn:
			tok_num=xtffn.index(token)
			for topic in range(len(topic_matrix1)):
				cluster_score[f'lda_topic{topic}']+=topic_matrix1[topic][tok_num]
	return(cluster_score)


app = Flask(__name__)  # create instance of Flask class

@app.route('/')  # the site to route to, index/main in this case
def landingpage() -> str:
	return(flask.render_template('landingpage.html'))


@app.route('/predict')  # the site to route to, index/main in this case
def predictpage() -> str:
	arg_dict=request.args
	
	back_to_landing=False
	for i in arg_dict:
		print(i)
		if arg_dict[i]=='' and i!='description':
		 	back_to_landing=True
		 	print('blank')
	
	if int(arg_dict['beds']) not in list(range(51)):
		print('beds')
		back_to_landing=True
	if int(arg_dict['baths']) not in range(51):
		print('baths')
		back_to_landing=True
	if int(arg_dict['sqft']) not in range(300,10001):
		print('sqft')
		back_to_landing=True
	if int(arg_dict['stories']) not in range(1,81):
		print('stories')
		back_to_landing=True
	if int(arg_dict['rooms']) not in range(1,31):
		print('rooms')
		back_to_landing=True
	if int(arg_dict['year_built']) not in range(1833,int(dt.datetime.now().date().strftime("%Y"))+1):
		print('year')
		back_to_landing=True
	
	if back_to_landing:
		return(flask.render_template('landingpage.html')+"<p1>Please enter valid values.</p1>")
	
	
	arg_frame=pd.DataFrame(arg_dict,index=['0'])

	arg_frame.rename(columns={'year_built':'building_age'},inplace=True)
	arg_frame['building_age']=[int(dt.datetime.now().date().strftime("%Y"))-int(arg_frame['building_age'].iloc[0])]
	
	arg_frame['pol']=[TextBlob(arg_frame.description.iloc[0]).sentiment[0]]
	arg_frame['sub']=[TextBlob(arg_frame.description.iloc[0]).sentiment[0]]
	
	arg_frame['property_type']=[str(arg_frame['property_type'].iloc[0]).replace('_',' ')]
	
	if len(str(arg_frame.description.iloc[0]))==0:
		arg_frame['description']=['blankdescriptionplaceholder']
	else:
		arg_frame['description']=[str(arg_frame['description'].iloc[0]).lower()]

	description=soft_cluster(arg_frame['description'].iloc[0])
	description=pd.DataFrame(description,index=[0])
	
	for column in description:
		arg_frame[column]=list(description[column])
	
	
	#Create dummy columns
	ptype_dict={}
	ptype_dict["Commercial"]=0
	ptype_dict["Condo"]=0
	ptype_dict["Multi-Family Home"]=0
	ptype_dict["Other"]=0
	ptype_dict["Single-Family Home"]=0
	
	dummies=pd.DataFrame(ptype_dict,index=[0])
	for column in dummies:
		arg_frame[column]=dummies[column]
	
	borough_dict={}
	borough_dict["Bronx"]=0
	borough_dict["Brooklyn"]=0
	borough_dict["Manhattan"]=0
	borough_dict["Queens"]=0
	borough_dict["Staten Island"]=0
	
	dummies=pd.DataFrame(borough_dict,index=[0])
	for column in dummies:
		arg_frame[column]=dummies[column]

	
	#curse you, "smart" quotes		
	arg_frame[arg_frame['property_type'].iloc[0].replace('”','')]=[1]
	arg_frame[arg_frame['borough'].iloc[0].replace('”','')]=[1]
	
	#We don't need these columns now that we have dummies
	arg_frame.drop(['description','property_type','borough'],axis=1,inplace=True)
	arg_frame.fillna('0',inplace=True)
	
	#Run the model.
	prediction=RFR.predict(arg_frame)[0]
	
	#Head to the prediction page.
	return(flask.render_template('predictpage.html', x_input=request.args, prediction=int(prediction*100)/100))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0') # Local

