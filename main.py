import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title=" AI Movie Recommender", page_icon="🎥", layout="centered")

# Add CSS and HTML for background and styles (all text white)
page_bg_img = """
<style>
body {
    background: url('https://wallpapers.com/images/file/netflix-background-gs7hjuwvv2g0e9fj.jpg') no-repeat center center fixed;
    background-size: cover;
    color: #ffffff !important;
    font-weight: 600;
    text-shadow: 1.5px 1.5px 4px rgba(0, 0, 0, 0.8);
}

.stApp {
    background-color: rgba(0, 0, 0, 0.35) !important;
    padding: 2rem;
    border-radius: 12px;
    box-shadow: 0 0 20px rgba(0, 0, 0, 0.7);
    color: #ffffff !important;
    font-weight: 600;
}

h1, h2, h3, h4 {
    color: #ffffff !important;
    text-shadow: 2px 2px 6px rgba(0, 0, 0, 0.85);
    font-weight: 700;
    animation: fadeInText 1s ease forwards;
}

.stApp * {
    color: #ffffff !important;
}

button {
    background-color: #ffd700 !important;
    color: #000000 !important;
    font-weight: 700;
    border-radius: 8px;
    padding: 12px 28px;
    border: none;
    cursor: pointer;
    transition: background-color 0.3s ease, transform 0.2s ease;
    box-shadow: 1px 1px 7px rgba(0,0,0,0.6);
}

button:hover {
    background-color: #ffbf00 !important;
    transform: scale(1.05);
}

[role="radiogroup"] {
    color: #ffffff !important;
    font-weight: 700 !important;
    text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.8);
}

[role="radiogroup"] label {
    color: #ffffff !important;
    font-weight: 700 !important;
    cursor: pointer;
    user-select: none;
    text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.8);
}

[role="radiogroup"] input[type="radio"] {
    accent-color: #ffd700;
    cursor: pointer;
}

.stSlider label {
    color: #ffffff !important;
    font-weight: 700 !important;
    text-shadow: 1.5px 1.5px 5px rgba(0, 0, 0, 0.9);
}

footer {
    color: #ffd700;
    font-weight: 700;
    text-align: center;
    margin-top: 3rem;
    font-style: italic;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.8);
}

@keyframes fadeInText {
    from {
        opacity: 0;
        transform: translateY(-10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('C:/Users/D/OneDrive/Desktop/ai recommendation system/tmdb_5000_credits.csv')
    df['cast'] = df['cast'].fillna('')
    df['crew'] = df['crew'].fillna('')
    df['combined'] = df['cast'] + ' ' + df['crew']
    return df

df = load_data()

@st.cache_data
def compute_sim():
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined'])
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

cosine_sim = compute_sim()
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

genre_keywords = {
    "Comedy": ["Will Ferrell", "Kevin Hart", "Seth Rogen"],
    "Thriller": ["David Fincher", "Jake Gyllenhaal", "Rosamund Pike"],
    "Sci-Fi": ["Ridley Scott", "Keanu Reeves", "James Cameron"],
    "Romance": ["Emma Stone", "Ryan Gosling", "Julia Roberts"],
    "Fantasy": ["Peter Jackson", "Orlando Bloom", "Ian McKellen"],
    "Action": ["Tom Cruise", "Dwayne Johnson", "Michael Bay"],
    "Drama": ["Meryl Streep", "Daniel Day-Lewis", "Kate Winslet"],
    "Animation": ["Pixar", "Disney", "DreamWorks"]
}

def recommend_by_keywords(keywords, filter_keyword, num_recs):
    results = df.copy()
    if filter_keyword:
        results = results[
            results['cast'].str.lower().str.contains(filter_keyword) |
            results['crew'].str.lower().str.contains(filter_keyword)
        ]
    if keywords:
        results = results[
            results['cast'].apply(lambda x: any(k.lower() in x.lower() for k in keywords)) |
            results['crew'].apply(lambda x: any(k.lower() in x.lower() for k in keywords))
        ]

    if results.empty:
        return None

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(results['combined'])
    cosine_sim_local = cosine_similarity(tfidf_matrix, tfidf_matrix)

    recs = []
    for i in range(min(num_recs, len(results))):
        recs.append({
            'title': results.iloc[i]['title'],
            'cast': results.iloc[i]['cast'][:150],
            'crew': results.iloc[i]['crew'][:150],
            'similarity': cosine_sim_local[i].mean()
        })

    return pd.DataFrame(recs)

# SESSION STATE INIT
if "step" not in st.session_state:
    st.session_state.step = 0
if "selected_genre" not in st.session_state:
    st.session_state.selected_genre = None
if "story_preference" not in st.session_state:
    st.session_state.story_preference = None
if "keyword" not in st.session_state:
    st.session_state.keyword = ""
if "num_recs" not in st.session_state:
    st.session_state.num_recs = 5

# PAGE FLOW
def page_welcome():
    st.title(" WATCH YOUR FAVOURITE MOVIE")
    st.markdown("We’re here to help you find the perfect movie — based on **what you love** and **how you feel**.")
    st.markdown("Click the button below to start your personalized movie journey!")
    if st.button(" Start Now"):
        st.session_state.step = 1
        st.rerun()

def page_genre_selection():
    st.header("Step 1: What type of movie are you in the mood for?")
    genre = st.radio(
        "Choose one genre that excites you the most:",
        options=list(genre_keywords.keys())
    )
    if st.button("Next"):
        st.session_state.selected_genre = genre
        st.session_state.step = 2
        st.rerun()

def page_story_preference():
    st.header("Step 2: What kind of story do you prefer?")
    story = st.radio(
        "Pick the vibe you want:",
        options=[
            "Light & Funny",
            "Thrilling & Suspenseful",
            "Romantic & Heartwarming",
            "Mind-bending & Sci-fi",
            "Epic & Fantasy",
            "Fast-paced & Action-packed",
            "Deep & Emotional",
            "Family & Animation"
        ]
    )
    if st.button("Next"):
        st.session_state.story_preference = story
        st.session_state.step = 3
        st.rerun()

def page_keyword_input():
    st.header("Step 3: Any favorite actors, directors, or keywords? (Optional)")
    kw = st.text_input("Type names or words to filter movies (leave empty to skip):")
    if st.button("Next"):
        st.session_state.keyword = kw.lower()
        st.session_state.step = 4
        st.rerun()

def page_num_recommendations():
    st.header("Step 4: How many recommendations would you like?")
    num = st.slider("Select number of movies:", min_value=3, max_value=10, value=5)
    if st.button("Get Recommendations"):
        st.session_state.num_recs = num
        st.session_state.step = 5
        st.rerun()

def page_recommendations():
    st.header(" Your Personalized Movie Recommendations")

    selected_genre = st.session_state.selected_genre
    story = st.session_state.story_preference
    keyword = st.session_state.keyword
    num_recs = st.session_state.num_recs

    story_map = {
        "Light & Funny": ["Comedy", "Funny", "Humor"],
        "Thrilling & Suspenseful": ["Thriller", "Suspense", "Mystery"],
        "Romantic & Heartwarming": ["Romance", "Love", "Relationship"],
        "Mind-bending & Sci-fi": ["Sci-Fi", "Science Fiction", "Futuristic"],
        "Epic & Fantasy": ["Fantasy", "Magic", "Epic"],
        "Fast-paced & Action-packed": ["Action", "Adventure", "Fight"],
        "Deep & Emotional": ["Drama", "Emotional", "Serious"],
        "Family & Animation": ["Animation", "Family", "Cartoon"]
    }

    keywords = genre_keywords.get(selected_genre, []) + story_map.get(story, [])

    recommendations = recommend_by_keywords(keywords, keyword, num_recs)

    if recommendations is None or recommendations.empty:
        st.warning("Sorry, we couldn't find any movies matching your preferences. Try adjusting your inputs!")
    else:
        st.success(f"Here are {len(recommendations)} movies just for you 🎉")
        for _, row in recommendations.iterrows():
            with st.expander(f"🎬 {row['title']} (Similarity: {row['similarity']:.2f})"):
                st.markdown(f"**👥 Cast:** `{row['cast']}`")
                st.markdown(f"**🎬 Crew:** `{row['crew']}`")

    if st.button("Start Over"):
        st.session_state.step = 0
        st.rerun()

page_functions = {
    0: page_welcome,
    1: page_genre_selection,
    2: page_story_preference,
    3: page_keyword_input,
    4: page_num_recommendations,
    5: page_recommendations,
}

page_functions[st.session_state.step]()

# Footer with movie quote
st.markdown("""
<footer>
  "Movies touch our hearts and awaken our vision." – Martin Scorsese 
</footer>
""", unsafe_allow_html=True)
