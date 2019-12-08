#Load Libraries
from flask import Flask,escape, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


app = Flask(__name__)
#model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/test_predict',methods=['GET', 'POST'])
def test_predict():
	summery = request.form['summery'] 
	skill = request.form['skill']
	experiance = request.form['experiance']
	
	train_set_summery=[]
	train_set_skill=[]
	train_set_experiance=[]
	
	#CV_compareing
	ddfff=pd.read_csv('cv_details_processed.csv')

	ddfff.loc[ddfff['Top_Skills']=='','Top_Skills'] = 'No Skill'
	ddfff['Top_Skills'].fillna("No Skill", inplace = True) 
	
	train_set_summery.append(summery)
	train_set_skill.append(skill)
	train_set_experiance.append(experiance)

	for i in range(len(ddfff['Top_Skills'])):
		train_set_skill.append(ddfff['Top_Skills'][i])
	
	for i in range(len(ddfff['Summary'])):
		train_set_summery.append(ddfff['Summary'][i])
	
	for i in range(len(ddfff['Experience'])):
		train_set_experiance.append(ddfff['Experience'][i])


	tfidf_vectorizer = TfidfVectorizer()
	tfidf_matrix_summery = tfidf_vectorizer.fit_transform(train_set_summery)
	tfidf_matrix_skill = tfidf_vectorizer.fit_transform(train_set_skill)
	tfidf_matrix_experiance = tfidf_vectorizer.fit_transform(train_set_experiance)
	
	sval_summery=cosine_similarity(tfidf_matrix_summery[0:1], tfidf_matrix_summery)
	lst = sval_summery[0][1:]
	ddfff['cos_sim_summery']=lst
	
	
	sval_skill=cosine_similarity(tfidf_matrix_skill[0:1], tfidf_matrix_skill)
	lst = sval_skill[0][1:]
	ddfff['cos_sim_skill']=lst
	
	
	sval_experiance=cosine_similarity(tfidf_matrix_experiance[0:1], tfidf_matrix_experiance)
	lst = sval_experiance[0][1:]
	ddfff['cos_sim_experiance']=lst
	
	ddfff.drop(['Unnamed: 0','Top_Skills','Summary','Experience'],axis=1,inplace=True)
	ddfff.sort_values(['cos_sim_skill','cos_sim_summery', 'cos_sim_experiance'], axis = 0, ascending = False,inplace = True, na_position ='last') 
	
	#ddfff=ddfff.replace(0,np.nan).dropna(subset=['cos_sim_skill'])
	k=list(enumerate(ddfff['Name'][0:11]))
	
	Prof_list = []
	for i in range(len(k)):
		var_ = k[i][1]
		Prof_list.append(var_)
		
	return render_template("result.html", user=Prof_list)

if __name__ == "__main__":
    app.run(debug=True)