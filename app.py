# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 11:47:10 2020

@author: BerniceYeow
"""


import pandas as pd


import malaya


import re


import streamlit as st

english_stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

from stop_words import get_stop_words

malay_stop_words = get_stop_words('Indonesian')

def main():
    st.set_option('deprecation.showfileUploaderEncoding', False)


            
    @st.cache(suppress_st_warning=True)
    def load_data(uploaded_file):
        

        df = pd.read_csv(uploaded_file)
                
 
        return df
    



        
    uploaded_file = st.file_uploader('Upload CSV file to begin', type='csv')

    #if upload then show left bar
    if uploaded_file is not None:
        df = load_data(uploaded_file)






        st.sidebar.subheader("Text column to analyse")
        st_ms = st.sidebar.selectbox("Select Text Columns To Analyse", (df.columns.tolist()))
        import nltk


        import top2vec
        from top2vec import Top2Vec
        
        #INITIALIZE AN EMPTY DATAFRAME, CONVERT THE TEXT INTO STRING AND APPEND INTO THE NEW COLUMN
        d1 = pd.DataFrame()
        d1['text'] = ""
        d1['text'] = df[st_ms]
        d1['text'] = d1['text'].astype(str)
        
        for x in range(len(d1)):
                    d1.text.iloc[x] = d1.text.iloc[x].lower() #to lower case
                    d1.text.iloc[x] = re.sub(r"@\S+","", d1.text.iloc[x]) #remove mentions
                    d1.text.iloc[x] = re.sub(r"http\S+","", d1.text.iloc[x]) #remove hyperlinks
                    d1.text.iloc[x] = ''.join([word for word in d1.text.iloc[x] if not word.isdigit()]) #remove numbers
                    d1.text.iloc[x] =  nltk.word_tokenize(d1.text.iloc[x]) #tokenising
                    d1.text.iloc[x] = [i for i in d1.text.iloc[x] if not i in english_stop_words] #remove stop words
                    d1.text.iloc[x] = [i for i in d1.text.iloc[x] if not i in malay_stop_words]
                    d1.text.iloc[x] = [i for i in d1.text.iloc[x] if len(i) > 2] #too short potong
                    print('Completed line : ',x)
                
        

        #INITIALIZE THE TOP2VEC MODEL AND FIT THE TEXT
        #model.build_vocab(df_list, update=False)
        model = Top2Vec(documents=d1['text'], speed="learn", workers=10)
        
        topic_sizes, topic_nums = model.get_topic_sizes()
        for topic in topic_nums:
            st.pyplot(model.generate_topic_wordcloud(topic))
            # Display the generated image:

    




if __name__ == '__main__':
    main()