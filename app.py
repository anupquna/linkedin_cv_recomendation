from flask import Flask,escape, request, jsonify, render_template
import numpy as np
import pandas as pd
# libraries for making count matrix and similarity matrix
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
	#int_features = [x for x in request.form.values()]
	#final_features = [np.array(int_features)]
	skill = request.form['abcd']
	train_set=[]
	#CV_compareing
	ddfff=pd.read_csv('cv_details.csv')
	ddfff['Job_Skill']=skill
	ddfff.loc[ddfff['Top_Skills']=='','Top_Skills'] = 'No Skill'
	ddfff['Top_Skills'].fillna("No Skill", inplace = True) 
	train_set.append(skill)

	for i in range(len(ddfff['Top_Skills'])):
		train_set.append(ddfff['Top_Skills'][i])


	tfidf_vectorizer = TfidfVectorizer()
	tfidf_matrix_train = tfidf_vectorizer.fit_transform(train_set)
	
	sval=cosine_similarity(tfidf_matrix_train[0:1], tfidf_matrix_train)
	lst = sval[0][1:]
	ddfff['cos_sim_val']=lst
	
	ddfff.sort_values("cos_sim_val", axis = 0, ascending = False,inplace = True, na_position ='last') 
				 
	k=list(enumerate(ddfff['Name'][0:11]))
	
	l = []
	for i in range(len(k)):
		a = k[i][1]
		l.append(a)
		
	return render_template("result.html", user=l)

if __name__ == "__main__":
    app.run(debug=True)