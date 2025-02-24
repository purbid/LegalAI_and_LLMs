import os

import numpy as np
import openai
import chromadb

from tqdm import tqdm

from dotenv import load_dotenv
from chromadb.config import Settings
from chromadb.utils import embedding_functions

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class ChromaDB():

    def __init__(self, collection_name = "supreme_court_rag",
                description = "Legal judgements from the Supreme Court with rhetorical roles",
                embedding_model = "text-embedding-3-small"):
                # embedding_model = "text-embedding-ada-002"):

        self.client = chromadb.Client(Settings(
            # chroma_db_impl="duckdb+parquet",   ### raises error with newer version.
            persist_directory="./chroma_db"
        ))

        self.delete_collection_if_exists("supreme_court_rag")

        self.collection = self.client.create_collection(
            name=collection_name,
            metadata={"description": description}
        )

        self.openai_embedder = embedding_functions.OpenAIEmbeddingFunction(
            api_key=openai.api_key,
            model_name=embedding_model
        )

    def delete_collection_if_exists(self, collection_name):
        try:
            self.client.delete_collection(collection_name)
            print(f"Collection '{collection_name}' deleted successfully.")
        except Exception as e:
            print(f"Collection '{collection_name}' may not exist or deletion failed: {e}")

    def get_embedding(self, text):
        # print(text)
        # emb =  self.openai_embedder(text)
        return self.openai_embedder([text])

    def populate(self, docs):


        total_docs = len(docs)

        with tqdm(total=total_docs, desc="Processing documents") as pbar:
            for doc_num, curr_doc in enumerate(docs):
                for doc_line, line in enumerate(curr_doc):
                    text, role, line_id, file_name = line
                    # print(f" the length of the sentence is {len(text.split(' '))}" )

                    embedding = self.get_embedding(text)

                    self.collection.add(
                        documents=[text],
                        embeddings=embedding,
                        metadatas=[{
                            "rhetorical_role": role,
                            "document_id": f"doc_{doc_num}",
                            "line_number": doc_line
                        }],
                        ids=[line_id]
                    )

                pbar.update(1)
