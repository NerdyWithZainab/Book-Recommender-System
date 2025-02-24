import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document  # Import the Document class
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
import os
import requests
import gradio as gr

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
books = pd.read_csv("books_with_emotions.csv")
def validate_url(url):
    """Ensure the URL is valid and properly formatted."""
    if isinstance(url, str) and url.startswith("http"):
        return url + "&fife=w800"
    return "cover-not-found.jpg"

books["large_thumbnail"] = books["thumbnail"].fillna("").apply(validate_url)
books["large_thumbnail"] = books["large_thumbnail"].apply(
    lambda x: x + "&fife=w800" if isinstance(x, str) and x.startswith("http") else "cover-not-found.jpg"
)

# Create a directory to store images
IMAGE_DIR = "book_covers"
os.makedirs(IMAGE_DIR, exist_ok=True)

def download_image(url, isbn):
    """Download and save the book cover image locally."""
    if not isinstance(url, str) or not url.startswith("http"):
        return "cover-not-found.jpg"
    
    image_path = os.path.join(IMAGE_DIR, f"{isbn}.jpg")
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            with open(image_path, "wb") as f:
                f.write(response.content)
            return image_path
    except requests.RequestException:
        pass
    return "cover-not-found.jpg"

# Process and download images
books["image_path"] = books.apply(lambda row: download_image(row["thumbnail"], row["isbn13"]), axis=1)

# Load tagged descriptions
with open("tagged_descriptions.txt", "r", encoding="utf-8") as file:
    raw_text = file.read()

# Split and process documents
raw_documents = [raw_text]  # Convert to list format expected by text_splitter
raw_documents = [Document(page_content=text) for text in raw_documents]
text_splitter = CharacterTextSplitter(separator="\n",chunk_size=0,chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
db_books = Chroma.from_documents(documents, OpenAIEmbeddings(api_key="OPENAI_KEY"))

def retreive_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:
    recs = db_books.similarity_search(query, initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(final_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category][:final_top_k]
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs.sort_values(by="joy",ascending=False,inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise",ascending=False,inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger",ascending=False,inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear",ascending=False,inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness",ascending=False,inplace=True)

    return book_recs

def recommend_books(query: str, category: str, tone: str):
    recommendations = retreive_semantic_recommendations(query, category, tone)
    
    image_paths = recommendations["image_path"].tolist()
    titles = recommendations["title"].tolist()
    authors = recommendations["authors"].tolist()
    
    # Format captions
    captions = [f"{title} by {author}" for title, author in zip(titles, authors)]
    
    # Ensure 8 recommendations are returned
    while len(image_paths) < 8:
        image_paths.append("cover-not-found.jpg")
        captions.append("No book found")

    return image_paths[:8], captions[:8]

categories = ["All"] + sorted(books["simple_categories"].dropna().astype(str).unique())
tones =["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic book recommender")

    with gr.Row():
        user_query = gr.Textbox(label="Enter a book description:", placeholder="e.g., A story about forgiveness")
        category_dropdown = gr.Dropdown(choices=categories, label="Select a category", value="All")
        tones_dropdown = gr.Dropdown(choices=tones, label="Select a tone", value="All")
        submit_button = gr.Button(value="Get recommendations")

    gr.Markdown("## Recommendations")
    
    with gr.Row():
        image_outputs = [gr.Image(label=f"Book {i+1}") for i in range(8)]
    
    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tones_dropdown],
        outputs=image_outputs
    )

if __name__ == "__main__":
    dashboard.launch()
