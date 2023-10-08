import streamlit as st
from streamlit_cookies_manager import EncryptedCookieManager
import os
import warnings
import csv
import random
import uuid
import pickle
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import requests

st.set_page_config(page_title="NBDT Recommendation Engine Arena")

cookies = EncryptedCookieManager(
    prefix="LUL/streamlit-cookies-manager/",
    password=os.environ.get("COOKIES_PASSWORD", "uDnda87,kGFdi&jh.kjsk/jk4DF369*^jhGks"),
)
warnings.filterwarnings("ignore")


user_id = cookies.get('user_id')  # Attempt to retrieve the user ID cookie

if user_id is None:
    user_id = str(uuid.uuid4())  # Generate a random user ID
    cookies['user_id'] = user_id  # Set the cookie


def download_file(url, local_filename):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename

# URLs of your remote files
article_list_url = "https://huggingface.co/spaces/atrytone/ArenaTester/resolve/main/article_list.pkl"


# Local paths where the files will be downloaded
article_list_path = "article_list_2.pkl"


# Download the files
download_file(article_list_url, article_list_path)


# Now load the files from the local paths
with open(article_list_path, "rb") as articles:
    article_list = tuple(pickle.load(articles))
    
INDEXES = ["miread_large", "miread_contrastive", "scibert_contrastive"]
MODELS = [
    "biodatlab/MIReAD-Neuro-Large",
    "biodatlab/MIReAD-Neuro-Contrastive",
    "biodatlab/SciBERT-Neuro-Contrastive",
]
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
faiss_embedders = [HuggingFaceEmbeddings(
    model_name=name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs) for name in MODELS]

vecdbs = [FAISS.load_local(index_name, faiss_embedder)
          for index_name, faiss_embedder in zip(INDEXES, faiss_embedders)]


def get_matchup():
    choices = INDEXES
    left, right = random.sample(choices, 2)
    return left, right


def get_comp(prompt):
    left, right = get_matchup()
    left_output = inference(prompt, left)
    right_output = inference(prompt, right)
    return left_output, right_output

def get_article():
    return random.choice(article_list)



def send_result(l_output, r_output, prompt, pick):
    with open('results.csv', 'a') as res_file:
        writer = csv.writer(res_file)
        row = [user_id, l_output, r_output, prompt, pick]
        writer.writerow(row)
    new_prompt = get_article()
    return new_prompt


def get_matches(query, db_name="miread_contrastive"):
    """
    Wrapper to call the similarity search on the required index
    """
    matches = vecdbs[INDEXES.index(
        db_name)].similarity_search_with_score(query, k=30)
    return matches


def inference(query, model="miread_contrastive"):
    """
    This function processes information retrieved by the get_matches() function
    Returns - Streamlit output for the authors, abstracts, and journals tabular output
    """
    matches = get_matches(query, model)
    auth_counts = {}
    n_table = []
    scores = [round(match[1].item(), 3) for match in matches]
    min_score = min(scores)
    max_score = max(scores)

    def normaliser(x): return round(1 - (x-min_score)/max_score, 3)

    i = 1
    for match in matches:
        doc = match[0]
        score = round(normaliser(round(match[1].item(), 3)), 3)
        title = doc.metadata['title']
        author = doc.metadata['authors'][0].title()
        date = doc.metadata.get('date', 'None')
        link = doc.metadata.get('link', 'None')

        # For authors
        record = [score,
                  author,
                  title,
                  link,
                  date]
        if auth_counts.get(author, 0) < 2:
            n_table.append([i, ]+record)
            i += 1
            if auth_counts.get(author, 0) == 0:
                auth_counts[author] = 1
            else:
                auth_counts[author] += 1

    return n_table[:10]


# Style the title and description with Markdown and HTML
st.markdown(
    "<h1 style='text-align: center; color: #0066cc;'>NBDT Recommendation Engine Arena</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='font-size: 18px; text-align: center; color: #333;'>"
    "NBDT Recommendation Engine for Editors is a tool for neuroscience authors/abstracts/journals recommendation built for NBDT journal editors. "
    "It aims to help an editor find similar reviewers, abstracts, and journals for a given submitted abstract. "
    "To get a recommendation, paste a `title[SEP]abstract` or `abstract` in the text box below and click the appropriate 'Get Comparison' button. "
    "Then, explore the suggested lists of authors, abstracts, and journals. "
    "The data in our current demo includes authors associated with the NBDT Journal, and we regularly update it for the latest publications."
    "</p>",
    unsafe_allow_html=True
)


article = get_article()
prompt = st.text_area("Enter Abstract", article, height=200)
action_btn = st.button("Get Comparison")


# Create a layout with two columns for Model A and Model B results
col1, col2 = st.columns(2)

if action_btn:
    l_output, r_output = get_comp(prompt)
    
    # Display Model A results in the first column
    with col1:
        st.write("Model A Results:")
        st.dataframe(l_output, width=800)
    
    # Display Model B results in the second column
    with col2:
        st.write("Model B Results:")
        st.dataframe(r_output, width=800)
    # Align "Model A is better" and "Model B is better" buttons horizontally
    st.write("")  # Add some space
    st.markdown("### Choose the Better Model:")
    with st.beta_container():  # Create a container for alignment
      with st.beta_columns(2):  # Create two columns for buttons
          with st.beta_container():
             l_btn = st.button("Model A is better")
          with st.beta_container():
             r_btn = st.button("Model B is better")



