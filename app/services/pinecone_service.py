#import pinecone
from pinecone import Pinecone, PodSpec
from app.services.openai_service import get_embedding
import os

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
#pinecone.init(api_key=PINECONE_API_KEY, environment='gcp-starter')
pc = Pinecone(api_key=PINECONE_API_KEY)
EMBEDDING_DIMENSION = 1536

def embed_chunks_and_upload_to_pinecone(chunks, index_name):
    matches = [x for x in pc.list_indexes() if index_name in x["name"]]
    if len(matches)>0:
        print("\nIndex already exists. Deleting index ...{}".format(index_name))
        pc.delete_index(name=index_name)
    
    print("\nCreating a new index: ", index_name)
    
    pc.create_index(
        name=index_name,
        dimension=EMBEDDING_DIMENSION,
        metric="cosine",
        spec=PodSpec(
            environment="gcp-starter"
        )
    )

    index = pc.Index(index_name)

    # Embedding each chunk and preparing for upload
    print("\nEmbedding chunks using OpenAI ...")
    embeddings_with_ids = []
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        embeddings_with_ids.append((str(i), embedding, chunk))

    print("\nUploading chunks to Pinecone ...")
    upserts = [(id, vec, {"chunk_text": text}) for id, vec, text in embeddings_with_ids]
    index.upsert(vectors=upserts)

    print(f"\nUploaded {len(chunks)} chunks to Pinecone index\n'{index_name}'.")


def get_most_similar_chunks_for_query(query, index_name):
    print("\nEmbedding query using OpenAI ...")
    question_embedding = get_embedding(query)

    print("\nQuerying Pinecone index ...")
    index = pc.Index(index_name)
    query_results = index.query(vector=[question_embedding], top_k=3, include_metadata=True) # problem
    context_chunks = [x['metadata']['chunk_text'] for x in query_results['matches']]

    return context_chunks   


def delete_index(index_name):
  matches = [x for x in pc.list_indexes() if index_name in x["name"]]
  if len(matches)>0:
    print("\nDeleting index ...{}".format(index_name))
    pc.delete_index(name=index_name)
    print(f"Index {index_name} deleted successfully")
  else:
     print("\nNo index to delete!")