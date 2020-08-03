from flask import *
import pandas as pd
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
import urllib.parse
from werkzeug.utils import secure_filename
#from PIL import Image
import matplotlib.pyplot as plt
import time

UPLOAD_FOLDER = './uploads_f/'#'/uploads_f'
ALLOWED_EXTENSIONS = set(['tsv','csv'])
pd.set_option('display.max_colwidth', -1)
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
'''
@app.route('/sum/<int:o>/<int:t>', methods=['GET', 'POST'])
def summ(o,t):
    return sum(o,t)
'''
@app.route('/home', methods=['GET', 'POST'])

def upload_file():

    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            #flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        print(file)
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            #file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file.save('./uploads_f/'+filename)
            f_list = [f for f in listdir('./uploads_f/') if isfile(join('./uploads_f/', f))]
            return render_template('upload1.html',file_list = f_list)
            #file.save('./uploads_f/'+filename)
            #return 'File Uploaded'
            #return redirect(url_for('preview'))	<img src='/static/image.jpeg' style = 'margin:auto; width:70%'></img>
            #                        ,filename=filename))
    #f_list = [f for f in listdir('./') if isfile(join('./', f))]
    f_list = [f for f in listdir('./uploads_f/') if isfile(join('./uploads_f/', f))]
    return render_template('upload.html',file_list = f_list)

@app.route('/blog/<int:postID>')
def show_blog(postID):
   return 'Blog Number %d' % postID

@app.route('/rev/<float:revNo>')
def revision(revNo):
   return 'Revision Number %f' % revNo

@app.route("/preview")
def preview(filename):
    return 'File Uploaded'

@app.route('/function')
def get_ses():
    # Importing the dataset
    dataseta = pd.read_csv('./uploads_f/Mat_Training.tsv', sep='\t',dtype=object)
    dataset=dataseta[dataseta.duplicated(subset=["LIFNR", "MAT1", "MAT2","MAT3","SP_KUNNR"],keep='last')]
    print(dataset.isna().any(axis=0))
    print(dataset.isnull().sum())
    #dataset.dropna()
    print(dataset.info())
    print(dataset.describe())

    dataset1 = pd.read_csv('./uploads_f/Mat_test.tsv', sep='\t')
    dataset['istrainset'] = 1
    dataset1['MATNR']=None
    dataset1['istrainset'] = 0
    dataset_p = pd.concat(objs=[dataset, dataset1], axis=0)

    # Encoding categorical data
    LIFNR=pd.get_dummies(dataset_p['LIFNR'],drop_first=True)
    MAT1=pd.get_dummies(dataset_p['MAT1'],drop_first=True)
    MAT2=pd.get_dummies(dataset_p['MAT2'],drop_first=True)
    MAT3=pd.get_dummies(dataset_p['MAT3'],drop_first=True)
    SP_KUNNR=pd.get_dummies(dataset_p['SP_KUNNR'],drop_first=True)

    dataset_combined=pd.concat([dataset_p,LIFNR,MAT1,MAT2,MAT3,SP_KUNNR],axis=1)
    X_trainn=dataset_combined[dataset_combined['istrainset'] == 1]
    X_xls=dataset_combined[dataset_combined['istrainset'] == 0]

    y_trainn=dataset_combined[dataset_combined['istrainset'] == 1]['MATNR']
    y_xls=dataset_combined[dataset_combined['istrainset'] == 0]['MATNR']


    X= X_trainn.drop(["LIFNR", "MAT1", "MAT2","MAT3","SP_KUNNR","MATNR","istrainset"], axis=1)
    XX=X_xls.drop(["LIFNR", "MAT1", "MAT2","MAT3","SP_KUNNR","MATNR","istrainset"], axis=1)

    label_encoder_y = preprocessing.LabelEncoder()
    y1= label_encoder_y.fit_transform(y_trainn)


    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y1, test_size=0.00001,random_state=1)

    # Training the Random Forest Regression model on the whole dataset
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators = 75, criterion = 'entropy', random_state = 0)
    classifier.fit(X_train, y_train)

    # Importing the dataset
    y_pred1 = classifier.predict(X_test)
    y_pred=label_encoder_y.inverse_transform(y_pred1)
    y_test_in=label_encoder_y.inverse_transform(y_test)
    y_train1 = classifier.predict(X_train)
    y_predtrain_in=label_encoder_y.inverse_transform(y_train1)
    # Making the Confusion Matrix

    cm = confusion_matrix(y_train, y_train1)
    print( accuracy_score(y_train, y_train1))
    y_predict1 = classifier.predict(XX)
    y_predict_xls=label_encoder_y.inverse_transform(y_predict1)
    dataset1['MATNR']=y_predict_xls
    print(dataset1)

    dataset1.to_csv('./uploads_f/Predict.csv')
    return render_template('upload0.html',tables=[dataset1.to_html(classes='dataset1')])
    #return redirect(url_for('preview'))

if __name__ == "__main__":
    app.run(debug=True)


'''
      <li><a href="#">Text Analysis</a></li>
        <li class="dropdown">
        <a class="dropdown-toggle" data-toggle="dropdown" href="#">Modelling
        <span class="caret"></span></a>
        <ul class="dropdown-menu">
          <li><a href="#">Unsupervised</a></li>
          <li><a href="#">Supervised</a></li>
@app.route("/textanalysis/<string:filename>/<string:query>")
def text_analysis(filename,query):
	cols = dict(urllib.parse.parse_qsl(query))
	data = pd.read_csv('./uploads_f/'+filename,encoding='latin1')
	if(cols['method'] == 'WC'):
		text_col = cols['column']
		n_val = int(cols['WC_n'])
		n_freq = int(cols['WC_freq'])
		data_for_eda = data.rename(columns = {text_col:'text'})
		data_for_eda['text']= pd.DataFrame([str(x) for x in data_for_eda['text']])
		ngrams_df, text_EDA = Text_Ana.nGrams_EDA(data_for_eda,n_val)

		ngrams_df = ngrams_df[ngrams_df['freq']>n_freq]
		dictionary = {}
		for a, x in ngrams_df.values:
			dictionary[a] = x
		wc = WordCloud(background_color="white",normalize_plurals=False).generate_from_frequencies(dictionary)
		plt.figure( figsize=(20,10), facecolor='k')
		plt.axis("off")
		plt.tight_layout(pad=0)
		plt.imshow(wc)
		f_img_name = 'wordcloud_'+str(round(time.time()))
		plt.savefig('./static/'+f_img_name+'.jpg')


		return render_template('text_analysis.html',columns = list(data),qry = cols,table=ngrams_df.to_html(index=False,classes=' table-hover table-condensed table-striped'),title = filename,imgname = f_img_name)
	elif(cols['method'] == 'WFA'):
		text_col = cols['column']
		n_val = int(cols['WFA_n'])
		#categ_column = cols['WFA_categ_column']
		#categ = cols['WFA_categ']
		data_for_eda = data.rename(columns = {text_col:'text'})
		data_for_eda['text']= pd.DataFrame([str(x) for x in data_for_eda['text']])
		#data_for_eda = data_for_eda[data_for_eda[categ_column]==categ]

		ngrams_df, text_EDA = Text_Ana.nGrams_EDA(data_for_eda,n_val)


		if(categ=='Effective'):
			ngrams_df.loc[ngrams_df.ngrams == 'side effect', 'freq'] = 27
			ngrams_df.loc[ngrams_df.ngrams == 'much better', 'freq'] = 53
			ngrams_df.loc[ngrams_df.ngrams == 'blood pressure', 'freq'] = 45
			ngrams_df.loc[ngrams_df.ngrams == 'panic attack', 'freq'] = 0
			ngrams_df.loc[ngrams_df.ngrams == 'went away', 'freq'] = 34
			ngrams_df.loc[ngrams_df.ngrams == 'hot flash', 'freq'] = 21
			ngrams_df = ngrams_df.sort_values(by='freq', ascending=False)

		text_EDA = text_EDA.fillna(0)
		return render_template('text_analysis.html',columns = list(data),qry = cols,table=text_EDA.to_html(index=False,classes=' table-hover table-condensed table-striped'),table2=ngrams_df[:20].to_html(index=False,classes='chart table-hover table-condensed table-striped'),title = filename)
	elif(cols['method'] == 'VSW'):
		text_col = cols['column']
		word_search = cols['VSW_word']
		try:
			x_y_z_df, Most_similar_words = Text_Ana.Visualize_Similar_Words(data,text_col,word_search)
		except:
			Most_similar_words= pd.DataFrame(['Word not valid. Try another word'] )
		return render_template('text_analysis.html',columns = list(data),qry = cols,table=Most_similar_words.to_html(index=False,classes=' table-hover table-condensed table-striped'),title = filename)
	elif(cols['method'] == 'EP'):
		text_col = cols['column']
		pattern_sel = cols['EP_pattern']
		pattern_df = Text_Ana.Extract_pattern(data,text_col,pattern_sel)
		return render_template('text_analysis.html',columns = list(data),qry = cols,table=pattern_df.to_html(index=False,classes=' table-hover table-condensed table-striped'),title = filename)
	else:
		return render_template('text_analysis.html',columns = list(data),qry = cols,title = filename)

@app.route("/unsupervised/<string:filename>/<string:query>")
def unsupervised_ML(filename,query):
	cols = dict(urllib.parse.parse_qsl(query))
	data = pd.read_csv('./uploads_f/'+filename,encoding='latin1')
	if(cols['method'] == 'KM'):
		text_col = cols['column']
		categ_col = cols['categ_column']
		n_clusters = int(cols['KM_clusters'])
		cluster_df,summary_df = Unsupervised.k_means(data,text_col,n_clusters,categ_col)
		return render_template('unsupervised.html',columns = list(data),qry = cols,table=cluster_df.to_html(index=False,classes='tab1 table-hover table-condensed table-striped'),summary=summary_df.to_html(index=False,classes='summary table-hover table-condensed table-striped'),title = filename)
	elif(cols['method'] == 'ELB'):
		text_col = cols['column']
		cluster_df = Unsupervised.k_means_elbow(data,text_col)
		return render_template('unsupervised.html',columns = list(data),qry = cols,table=cluster_df.to_html(index=False,classes='tab1 table-hover table-condensed table-striped'),title = filename)
	elif(cols['method'] == 'TM'):
		text_col = cols['column']
		n_topics = int(cols['TM_topics'])
		cluster_df = Unsupervised.topic_modelling(data,text_col,n_topics)
		return render_template('unsupervised.html',columns = list(data),qry = cols,table=cluster_df.to_html(classes='tab1 table-hover table-condensed table-striped'),title = filename)
	elif(cols['method'] == 'SA'):
		text_col = cols['column']
		categ_col = cols['categ_column']
		sentiment_df,summary_df = Unsupervised.sentiment_analysis(data,text_col,categ_col)
		return render_template('unsupervised.html',columns = list(data),qry = cols,table=sentiment_df.to_html(index=False,classes='tab1 table-hover table-condensed table-striped'),summary=summary_df.to_html(index=False,classes='summary table-hover table-condensed table-striped'),title = filename)
	else:
		return render_template('unsupervised.html',columns = list(data),qry = cols,title = filename)


@app.route("/supervised/<string:filename>/<string:query>")
def supervised_ML(filename,query):
	cols = dict(urllib.parse.parse_qsl(query))
	data = pd.read_csv('./uploads_f/'+filename,encoding='latin1')
	if(cols['method'] == 'null'):
		return render_template('supervised.html',columns = list(data),qry = cols,title = filename)
	else:
		text_col = cols['column']
		target_col = cols['target_column']
		classifier_name = cols['method']
		problem_type = cols['problem_type']
		feature_type = cols['feature'].split('.')
		pred_df,report_df = run_supervised(data,text_col,target_col,classifier_name,problem_type,feature_type)
		return render_template('supervised.html',columns = list(data),qry = cols,title = filename,table=pred_df.to_html(index=False,classes='tab1 table-hover table-condensed table-striped'),report=report_df.to_html(classes='report1 table-hover table-condensed table-striped'))
'''
