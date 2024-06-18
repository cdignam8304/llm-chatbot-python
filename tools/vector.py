#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 19:41:39 2024

@author: christopherdignam
"""

from neo4j import GraphDatabase
import streamlit as st
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from llm import llm  # embeddings
from langchain.chains import RetrievalQA
# from langchain_community.embeddings import OpenAIEmbeddings  # deprecated
from langchain_openai import OpenAIEmbeddings
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


openai_key = st.secrets["OPENAI_API_KEY"]
embedding_model = OpenAIEmbeddings(openai_api_key=openai_key)



# Neo4j connection details
uri = st.secrets["NEO4J_URI"]
user = st.secrets["NEO4J_USERNAME"]
password = st.secrets["NEO4J_PASSWORD"]
# Debug...
# print("For debug purposes...")
# print("uri: %s, user: %s, password: %s" % (uri, user, password))


# Initialize the Neo4j driver
driver = GraphDatabase.driver(uri, auth=(user, password))


# Function to retrieve embeddings from Neo4j
def fetch_embeddings(tx):
    query = "MATCH (n) RETURN n.name AS name, n.nameEmbedding AS embedding"
    result = tx.run(query)
    return [(record["name"], record["embedding"]) for record in result]


# Fetch embeddings from Neo4j
with driver.session() as session:
    embeddings = session.read_transaction(fetch_embeddings)

# Debugging: Log the embeddings retrieved
# logger.info(f"Retrieved embeddings: {embeddings}")


# Ensure embeddings are lists of floats, not tuples
names = []
vectors = []
for name, embedding in embeddings:
    if isinstance(embedding, list) and all(isinstance(i, float) for i in embedding):
        names.append(name)
        vectors.append(embedding)
    else:
        logger.error(f"Invalid embedding format for node {name}: {embedding}")


embeddings = OpenAIEmbeddings()
text_embeddings = embeddings.embed_documents(names)
text_embedding_pairs = list(zip(names, text_embeddings))
neo4jvector = Neo4jVector.from_embeddings(
    text_embedding_pairs,
    embeddings,
#     retrieval_query="""
# RETURN
#     node.name AS text,
#     score,
#     {
#         name: node.name
#     } AS metadata
#         """
    )


# Use the vector_store for further operations
# print(neo4jvector)

# https://graphacademy.neo4j.com/courses/llm-chatbot-python/3-tools/1-vector-tool/

# Creating a Retriever
# In Langchain applications, Retrievers are classes that are designed to retrieve documents from a Store.
# Vector Retrievers are a specific type of retriever that are designed to retrieve documents from a 
# Vector Store based on similarity.
retriever = neo4jvector.as_retriever()

# Retrieval QA Chain
# The RetrievalQA chain will be responsible for creating an embedding from the user’s input, calling the Retriever to identify similar documents, and passing them to an LLM to generate a response.
# Call the static .from_llm() method on the RetrievalQA to create a new chain, passing the following parameters:
# 1. The LLM that used to process the chain
# 2. A Stuff chain is a relatively straightforward chain that stuffs, or inserts, documents into a prompt and passes that prompt to an LLM.
# 3. The Chain should use the Neo4jVectorRetriever created in the previous step.

# Comment out while using a dummy kg_qa function (see below)
kg_qa = RetrievalQA.from_chain_type(
    llm,                  # (1)
    chain_type="stuff",   # (2)
    retriever=retriever,  # (3)
)

# Test function used for debugging
# def kg_qa(query):
#     # Simulate a tool function
#     return f"Mocked response for query: {query}"
