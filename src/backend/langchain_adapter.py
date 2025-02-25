import ast
from typing import List
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

from prompt_utils import rhetorical_role_identify_prompt, generation_prompt_rag



class LegalRAG:

    def __init__(self, openai_api_key = '',
                 vector_store_path = './chroma_db',
                 collection_name = 'supreme_court_rag',
                 model_name = 'text-davinci-003'):


        self.embedding_function = OpenAIEmbeddings(
            model = 'text-embedding-3-small',
            openai_api_key = openai_api_key
        )

        self.chroma_obj = Chroma(
            collection_name = collection_name,
            embedding_function = self.embedding_function,
            persist_directory=vector_store_path
        )


        self.llm = OpenAI(
            openai_api_key = openai_api_key,
            model_name = model_name
        )

        self.role_selector_chain = LLMChain(
            llm = self.llm,
            prompt = rhetorical_role_identify_prompt
        )

        self.final_qa_chain = LLMChain(
            llm = self.llm,
            prompt = generation_prompt_rag
        )

    def retrieve_weighted_lines(self, query: str, search_only=["ARG"], top_k=5):

        query_embedding = self.chroma_obj.get_embedding(query)

        ### we first get 20 results, then narrow it down to top k
        ### I will replace this with a better weighted matching algo later.

        large_results = self.chroma_obj.collection.query(query_embeddings=[query_embedding], n_results=top_k*5)

        combined = []
        for i, doc_text in enumerate(large_results["documents"][0]):
            meta = large_results["metadatas"][0][i]
            dist = large_results["distances"][0][i]

            boost = 1.0
            if meta["rhetorical_role"] in search_only:
                boost = 1.25  ## 1.25 x importance, if the sentence had the rhet role to look out for
            adjusted_dist = dist * (1 / boost)

            combined.append({
                "text": doc_text,
                "metadata": meta,
                "original_distance": dist,
                "adjusted_distance": adjusted_dist
            })

        return sorted(combined, key=lambda x: x["adjusted_distance"])[:top_k]


    def answer_query(self, query: str):

        # 1. We get the roles from an LLM query
        roles_list_str = self.role_selector_chain.run({"query": query})
        try:
            roles_list = ast.literal_eval(roles_list_str)
            #verify we have a list of strs
            if not isinstance(roles_list, list):
                roles_list = []
        except Exception as e:
            roles_list = []

        # 2. Fetch the best matched sentences from vectorDB
        context_for_llm_qa = self.retrieve_weighted_lines(query, roles_list)



        ### incomplete, need to add the context builder method.
        llm_response = self.final_qa_chain.run({"context": context_for_llm_qa,
                                                "question": context_for_llm_qa})
        print(llm_response)