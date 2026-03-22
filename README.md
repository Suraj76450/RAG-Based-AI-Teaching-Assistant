## 1. Project Title

RAG-Based AI Teaching Assistant

## 2. Short Description

A RAG-based AI Teaching Assistant that processes video content using Whisper, converts it into structured text chunks, generates embeddings, and retrieves relevant context to answer user queries using an LLM. Designed for efficient learning, semantic search, and accurate context-aware responses.

## 3. Tech Stack

* Python
* Pandas & NumPy
* Scikit-learn (Cosine Similarity)
* Whisper (Speech-to-Text)
* Ollama / Local LLM API
* Embedding Model: `mxbai-embed-large`
* Joblib (for saving embeddings/data)

## 4. Feature Highlights

* 🎥 Converts video lectures into text with timestamps
* ✂️ Smart chunking of content with metadata
* 🔍 Semantic search using embeddings
* 📊 Efficient storage using Pandas & Joblib
* 🧠 Retrieval-Augmented Generation (RAG) pipeline
* ⚡ Fast similarity search using cosine similarity
* 🤖 Generates contextual answers using LLM

## 5. Sceenshots
output looks like. - ![Alt text](https://github.com/Suraj76450/RAG-Based-AI-Teaching-Assistant/blob/main/Output.png)

# How to use this RAG Based AI Teaching assistant on your own data
## Step 1 - Collect your videos
Move all your video files to the videos folder

## Step 2 - Convert to mp3
Convert all the video files to mp3 by ruunning video_to_mp3

## Step 3 - Convert mp3 to json 
Convert all the mp3 files to json by ruunning mp3_to_json

## Step 4 - Convert the json files to Vectors
Use the file read_chunks.py to convert the json files to a dataframe with Embeddings and save it as a joblib pickle

## Step 5 - Prompt generation and feeding to LLM


Read the joblib file and load it into the memory. Then create a relevant prompt as per the user query and feed it to the LLM




