from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from pandas.plotting import scatter_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import pickle

df=pd.read_csv('./cleanedResume.csv')
def modelTrain(df): 
    # labelling
    new_df=df.copy()
    var_mod = ['Category']
    le = LabelEncoder()
    for i in var_mod:
        df[i] = le.fit_transform(df[i])
    requiredText = df['Resume'].values
    requiredTarget = df['Category'].values
    # print(requiredTarget)
    # print(requiredText)
    word_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        stop_words='english')
    word_vectorizer.fit(requiredText)
    WordFeatures = word_vectorizer.transform(requiredText)
    # print(WordFeatures)
    print ("Feature completed .....")
    X_train,X_test,y_train,y_test = train_test_split(WordFeatures,requiredTarget,random_state=1, test_size=0.2,shuffle=True, stratify=requiredTarget)
    clf = KNeighborsClassifier(n_neighbors=22)
    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)
    # save the model using pickle
    print('Accuracy of KNeighbors Classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
    print('Accuracy of KNeighbors Classifier on test set:     {:.2f}'.format(clf.score(X_test, y_test)))
    print(prediction)
    print("\n Classification report for classifier %s:\n%s\n" % (clf, metrics.classification_report(y_test, prediction)))
    
    pickle.dump(clf, open('model.pkl','wb'))
    # dump the tfidf vectorizer
    pickle.dump(word_vectorizer, open('tfidf.pkl','wb'))

def predictResults(requiredText):
    print(requiredText)
    labels = ['Java Developer','Testing','DevOps Engineer','Javascript Developer', 'Web Designing', 'HR','Hadoop', 'Blockchain','ETL Developer', 'Operations Manager','Data Science','Sales','Mechanical Engineer', 'Arts', 'Database', 'Electrical Engineering','Health and fitness','PMO','Business Analyst', 'DotNet Developer', 'Python Developer','Network Security Engineer','SAP Developer', 'Civil Engineer','Advocate']
    # for cat in df['Category']:
    #     if cat not in labels:
    #         labels.append(cat)
    # # print(labels)
    # load the model from disk
    loaded_model = pickle.load(open('model.pkl', 'rb'))
    loaded_word_vectorizer = pickle.load(open('tfidf.pkl', 'rb'))
    WordFeatures = loaded_word_vectorizer.transform(requiredText)
    results=loaded_model.predict_proba(WordFeatures)
    # print(results)
    # for i in range(len(labels)):
    #     print(labels[i],": ",results[0][i]*100)
    # add each result to a dictionary
    result_dict = {}
    for i in range(len(labels)):
        result_dict[labels[i]] = results[0][i]*100
    # sort the dictionary
    sorted_dict = sorted(result_dict.items(), key=lambda x: x[1], reverse=True)
    print(sorted_dict)
    # return the json
    return sorted_dict
    



modelTrain(df)
