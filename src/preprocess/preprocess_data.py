import os

import sys
sys.path.append('/Users/purbidbambroo/PycharmProjects/LLMs/LegalAI_and_LLMs')

from src.backend.chromadb import ChromaDB



def read_files(docs_path = "data/UK-train-set"):

    doc_chunks_for_ingestion = []
    doc_num = 1

    for file_path in os.listdir(docs_path):

        with open(os.path.join(docs_path, file_path), "r") as f:
            lines = f.readlines()

        curr_doc = []
        for doc_line, line in enumerate(lines):
            line_contents = line.strip().split("\t")
            if len(line_contents) != 2:  ### check why this might happen
                continue
            text, role = line_contents
            #### 15_1 is line 1 on doc 15. Good identifier.
            line_id = f"doc{doc_num}_line{doc_line}"

            curr_doc.append([text, role, line_id, file_path])
        doc_chunks_for_ingestion.append(curr_doc)
        doc_num += 1
    return doc_chunks_for_ingestion

docs_folder = "data/UK-train-set"
doc_chunks_for_ingestion = read_files()



### init an object with default names and models
chroma_obj = ChromaDB()
chroma_obj.populate(doc_chunks_for_ingestion)




# for file_path in os.listdir(docs_folder):
#     with open(os.path.join(docs_folder, file_path), "r") as f:
#         lines = f.readlines()
#     for doc_line, line in enumerate(lines):
#         parts = line.strip().split("\t")
#         if len(parts) != 2:  ### check why this happened
#             continue
#         text, role = parts
#         print(text, role)
#         embedding = get_embedding(text)
#
#         #### 15_1 is line 1 on doc 15. Good identifier.
#         line_id = f"doc{doc_num}_line{doc_line}"
#
#         collection.add(
#             documents=[text],
#             embeddings=[embedding],
#             metadatas=[{
#                 "rhetorical_role": role,
#                 "document_id": f"doc_{doc_num}",
#                 "line_number": doc_line
#             }],
#             ids=[line_id]
#         )
#     doc_num += 1
#
# client.persist()
