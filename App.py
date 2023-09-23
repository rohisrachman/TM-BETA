import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import gensim
from gensim import corpora, models
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from gensim.models import CoherenceModel
from collections import Counter

st.set_page_config(page_title="TM-BETA", 
                   page_icon=":robot_face:",
                   layout="wide",
                   initial_sidebar_state="expanded"
                   )

def p_title(title):
    st.markdown(f'<h3 style="text-align: left; color:#F63366; font-size:28px;">{title}</h3>', unsafe_allow_html=True)

#########
#SIDEBAR
########

nav = st.sidebar.radio('',['Go to homepage', 'Topic Modelling', 'About'])
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')
st.sidebar.write('')

if nav == 'Go to homepage':
    st.markdown("<h1 style='text-align: center; color: white; font-size:28px;'>Helloo Welcome to TM-BETA!</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; font-size:56px;'<p>&#129302;</p></h3>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: grey; font-size:20px;'>Implementation of Topic Modelling</h3>", unsafe_allow_html=True)

    st.markdown("<h3 style='text-align: left; color:#F63366; font-size:18px;'><b>What is this App about?<b></h3>", unsafe_allow_html=True)
    st.write("this app is build for my satisfaction")
    st.markdown("<h3 style='text-align: left; color:#F63366; font-size:18px;'><b>Who is this App for?<b></h3>", unsafe_allow_html=True)
    st.write("Anyone can use this App completely for free! If you like it :heart:, show your support by sharing :+1: ")

if nav == 'Topic Modelling':
    st.markdown("<h1 style='text-align: center; color: white; font-size:28px;'>Topic Modelling</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: grey; font-size:20px;'>Drop your text, set up, analyze, and voilaa..</h3>", unsafe_allow_html=True)

    def preprocess_text(text, user_stopwords=[]):
        text = re.sub(r'[^\w\s]', '', text)
        factory = StopWordRemoverFactory()
        stopword_sastrawi = factory.get_stop_words()
        tokens = text.split()
        tokens = [word for word in tokens if word not in stopword_sastrawi]
        text = ' '.join(tokens)
        stop_words = gensim.parsing.preprocessing.STOPWORDS.union(set(user_stopwords))
        paragraphs = text.split('\n')
        preprosesing_text = [[token for token in gensim.utils.simple_preprocess(paragraph) if token not in stop_words] for paragraph in paragraphs]
        return preprosesing_text

    def perform_topic_modeling(transcript_text, num_topics, num_words, user_stopwords=[]):
        preprosesing_text = preprocess_text(transcript_text, user_stopwords)
        dictionary = corpora.Dictionary(preprosesing_text)
        corpus = [dictionary.doc2bow(text) for text in preprosesing_text]
        lsa_model = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
        topics = []
        for idx, topic in lsa_model.print_topics(num_topics=num_topics, num_words=num_words):
            topic_words = [word.split('*')[1].replace('"', '').strip() for word in topic.split('+')]
            topics.append((f"Topic {idx}", topic_words))
        return topics

    def generate_wordcloud(preprocessed_text):
        flat_words = [word for sublist in preprocessed_text for word in sublist]
        word_freq = Counter(flat_words)
        wc = WordCloud(background_color="white", width=800, height=400)
        wc.generate_from_frequencies(word_freq)
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt)


    col1, col2 = st.columns([3, 1])  # Penambahan kolom ketiga untuk elemen dari sidebar

    with col1:
        text_input = st.text_area('paste your text', height=500)
        st.write(f'You wrote {len(text_input)} characters.')

    with col2:  # Elemen sidebar sekarang ada di kolom ketiga
        st.title("Settings")
        num_clusters = st.slider('Number of Clusters', 1, 10, 3)
        num_words = st.slider('Number of Words per Topic', 1, 10, 3)
        user_stopwords = st.text_area('Stopword Tambahan', height=200)
        user_stopwords_list = [word.strip() for word in user_stopwords.split(",")]

        # Membuat tombol lebih mencolok
        start_analysis = st.button("Analyze Text")

    col3 = st.columns([1])
    with col3[0]:
        st.info("Initial Word Cloud")
        if text_input:
            preprocessed_text = preprocess_text(text_input, user_stopwords_list)
            generate_wordcloud(preprocessed_text)

    # Text Analysis Section
    if start_analysis and text_input:
        col4, col5 = st.columns([1, 1])  # Membuat kolom untuk visualisasi dan hasil pemodelan
        with col4:
            st.info("Topics in Text")
            topics = perform_topic_modeling(text_input, num_clusters, num_words, user_stopwords_list)
            for topic in topics:
                st.success(f"{topic[0]}: {', '.join(topic[1])}")
        with col5:
            st.info("Word Cloud from Topics")
            if topics:
                topic_words = [word for topic in topics for word in topic[1]]
                generate_wordcloud([topic_words])

if nav == 'About':
    st.markdown("<h1 style='text-align: center; color: white; font-size:28px;'>this is about</h1>", unsafe_allow_html=True)