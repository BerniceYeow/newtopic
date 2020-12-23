# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 11:47:10 2020

@author: BerniceYeow
"""


import pandas as pd


import malaya





import streamlit as st


from PIL import Image

preprocessing = malaya.preprocessing.preprocessing()



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
        

        df_list = list(df)
 

        import top2vec
        from top2vec import Top2Vec
        
        #INITIALIZE AN EMPTY DATAFRAME, CONVERT THE TEXT INTO STRING AND APPEND INTO THE NEW COLUMN
        d1 = pd.DataFrame()
        d1['text'] = ""
        d1['text'] = df[st_ms]
        d1['text'] = d1['text'].astype(str)
        d1['text'] = d1['text'].apply(preprocessing.process)
        
        

        #INITIALIZE THE TOP2VEC MODEL AND FIT THE TEXT
        #model.build_vocab(df_list, update=False)
        model = Top2Vec(documents=d1['text'], speed="learn", workers=10)
        
        topic_sizes, topic_nums = model.get_topic_sizes()
        for topic in topic_nums:
            st.pyplot(model.generate_topic_wordcloud(topic))
            # Display the generated image:

    




if __name__ == '__main__':
    main()