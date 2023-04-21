from django.shortcuts import render
from django.middleware.csrf import get_token
from django.core.files.storage import FileSystemStorage


import pickle
import pandas as pd
cv = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


def home(request):
    return render(request, "home.html")


def predict(request):
    import re
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer

    # function for preprocess the dataset.
    def preprocess(dataset):

        corpus = []
        for i in range(0, dataset.shape[0]):
            review = re.sub('[^a-zA-Z]', ' ', dataset['v2'][i])
            review = review.lower()
            review = review.split()
            ps = PorterStemmer()
            all_stopwords = stopwords.words('english')
            all_stopwords.remove('not')
            review = [ps.stem(word)
                      for word in review if not word in set(all_stopwords)]
            review = ' '.join(review)
            corpus.append(review)

        return corpus

    # get input from website
    context = {}
    if request.method == 'POST':
        # for csrf error
        csrf_token = get_token(request)
        # get files from the website
        uploaded_file = request.FILES['file']
        # print(uploaded_file.name)
        # print(uploaded_file.size)
        fs = FileSystemStorage()
        name = fs.save(uploaded_file.name, uploaded_file)
        context['url'] = fs.url(name)

    # get dataset from the uploaded data
    # file name should be unique
    dataset = pd.read_csv(
        r'E:\sem6\ML\django\spam-detection-ass8\spamModel\media\data.csv', encoding="ISO-8859-1")
    # preprocess the input
    transformed_sms = preprocess(dataset)

    # preformed column vectorizer and transformed sparse array into dense array
    vector_input = cv.transform(transformed_sms).toarray()
    # predict the input data with the model come from pickle file
    result = model.predict(vector_input)
    # append the result column with the given dataset
    dataset['output'] = result
    # finally save the output file to given location
    dataset.to_csv(
        r'E:\sem6\ML\django\spam-detection-ass8\spamModel\media\outputdata.csv')
    # saved the output file location to context variable and passed it's link to frontend
    context['output_url'] = '/media/outputdata.csv'
    # send home.html file for rendering with the context data.
    return render(request, "home.html", context)
