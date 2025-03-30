

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd
import seaborn as sns
import nltk
from nltk.corpus import stopwords
import string
from wordcloud import WordCloud
import re
def cleanResume(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
    resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText) 
    resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
    return resumeText

def visualiseData(df):
    # show the job description
    plt.figure(figsize=(10,5))
    plt.xticks(rotation=90)
    ax=sns.countplot(x="Category", data=df)
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))
    plt.grid()
    plt.savefig('./output/output1.png')
    # plt.show()

    # show the category distribution

    targetCounts = df['Category'].value_counts()
    targetLabels  = df['Category'].unique()
    # Make square figures and axes
    plt.figure(1, figsize=(10,5))
    the_grid = GridSpec(3, 3)


    cmap = plt.get_cmap('coolwarm')
    plt.subplot(the_grid[0, 1], aspect=1, title='CATEGORY DISTRIBUTION')

    source_pie = plt.pie(targetCounts, labels=targetLabels, autopct='%1.1f%%', shadow=True)
    plt.show()  
    oneSetOfStopWords = set(stopwords.words('english')+['``',"''"])
    totalWords =[]
    Sentences = df['Resume'].values
    cleanedSentences = ""
    for records in Sentences:
        cleanedText = cleanResume(records)
        cleanedSentences += cleanedText
        requiredWords = nltk.word_tokenize(cleanedText)
        for word in requiredWords:
            if word not in oneSetOfStopWords and word not in string.punctuation:
                totalWords.append(word)
        
    wordfreqdist = nltk.FreqDist(totalWords)
    mostcommon = wordfreqdist.most_common(50)
    print(mostcommon)
    wc = WordCloud(background_color='white').generate(cleanedSentences)
    plt.figure(figsize=(16,16))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    # plt.show()
    plt.savefig('./output/output2.png')




# df = pd.read_csv('./cleanedResume.csv')
# visualiseData(df)




