import os
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain.embeddings.openai import OpenAIEmbeddings

url = os.environ["NEO4J_URL"]
username = os.environ["NEO4J_USERNAME"]
password = os.environ["NEO4J_PASSWORD"]

vector_index = Neo4jVector.from_existing_graph(
    OpenAIEmbeddings(),
    url=url,
    username=username,
    password=password,
    index_name="blb",
    node_label="Embeddable",
    text_node_properties=["name", "oem", "severity", "status"],
    embedding_node_property="embedding",
)


question = "list yuri's events"
response = vector_index.similarity_search(question, k=1)
print(response[0].page_content)
