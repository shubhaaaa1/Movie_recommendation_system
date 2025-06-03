import streamlit as st
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Page Config ---
st.set_page_config(page_title="Movie Recommender", page_icon="🎥", layout="centered")

# --- Title ---
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🎬 Movie Recommendation System</h1>", unsafe_allow_html=True)
st.write("Welcome! Find movies similar to your favorite one.")

# --- Sidebar Instructions ---
st.sidebar.header("📌 How to use")
st.sidebar.markdown(
"""
1. Type or select a movie from the dropdown.  
2. Get top 30 recommended movies!  
3. Enjoy your watchlist! 🍿
"""
)

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("C:/Users/shubh/Downloads/movies.csv")
    for feature in ['genres', 'keywords', 'tagline', 'cast', 'director']:
        df[feature] = df[feature].fillna('')
    df['combined_features'] = df['genres'] + ' ' + df['keywords'] + ' ' + df['tagline'] + ' ' + df['cast'] + ' ' + df['director']
    return df

movies_data = load_data()

@st.cache_resource
def compute_similarity(data):
    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(data['combined_features'])
    similarity = cosine_similarity(feature_vectors)
    return similarity

similarity = compute_similarity(movies_data)

# --- User Input ---
movie_list = movies_data['title'].tolist()
movie_name = st.selectbox("🎞️ Select or type a movie you like:", movie_list)

if st.button("🎯 Recommend"):
    find_close_match = difflib.get_close_matches(movie_name, movie_list)

    if find_close_match:
        close_match = find_close_match[0]
        index_of_the_movie = movies_data[movies_data.title == close_match].index[0]

        similarity_score = list(enumerate(similarity[index_of_the_movie]))
        sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

        st.success(f"Top movies similar to **{close_match}**:")
        for i, movie in enumerate(sorted_similar_movies[1:31], start=1):
            title_from_index = movies_data.iloc[movie[0]]['title']
            st.markdown(f"**{i}.** {title_from_index}")
    else:
        st.error("❌ No close match found. Please try a different movie.")
