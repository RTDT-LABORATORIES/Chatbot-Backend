import os
import json
from langchain_community.graphs import Neo4jGraph
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import GraphCypherQAChain, RetrievalQA
from langchain.agents import AgentType, initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
from langchain.chains.openai_functions.openapi import get_openapi_chain
from langchain.prompts import (
    PromptTemplate,
)
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import MessagesPlaceholder

# url = "neo4j://localhost:7687"
# password = "password"
NEO4J_URL = os.environ["NEO4J_URL"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]

graph = Neo4jGraph(url=NEO4J_URL, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)
graph.refresh_schema()

chat_history = {}

memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="input",
    output_key="output",
    return_messages=True,
)


class Chatbot:
    _session = None

    def __init__(self, session):
        self._session = session

    def _get_blb_chain(self):
        BLB_API_KEY = os.environ["BLB_API_KEY"]
        headers = {
            "Authorization": f"Token {BLB_API_KEY}",
            "Content-Type": "application/json",
        }
        chain = get_openapi_chain(
            spec=os.environ["BLB_API_OPENAPI_SPEC_URL"],
            headers=headers
            # "https://www.klarna.com/us/shopping/public/openai/v0/api-docs/"
        )

        return chain

    def _get_cypher_chain(self):
        CYPHER_GENERATION_TEMPLATE = """Task:Generate Cypher statement to query a graph database.
            Instructions:
            Use only the provided relationship types and properties in the schema.
            Do not use any other relationship types or properties that are not provided.
            When search for users, events and wind turbines by their names, use text similarity search because they might be incomplete and contain typos.
            Schema:
            {schema}
            Note: Do not include any explanations or apologies in your responses.
            Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
            Do not include any text except the generated Cypher statement.

            The question is:
            {question}"""
        CYPHER_GENERATION_PROMPT = PromptTemplate(
            input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
        )

        cypher_chain = GraphCypherQAChain.from_llm(
            cypher_llm=ChatOpenAI(temperature=0, model_name="gpt-4-1106-preview"),
            qa_llm=ChatOpenAI(temperature=0, model_name="gpt-4-1106-preview"),
            cypher_prompt=CYPHER_GENERATION_PROMPT,
            graph=graph,
            verbose=True,
            validate_cypher=True,
            return_intermediate_steps=True,
            return_direct=True,
        )

        return cypher_chain

    def _get_vector_chain(self):
        vector_index = Neo4jVector.from_existing_index(
            OpenAIEmbeddings(),
            url=NEO4J_URL,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            database="neo4j",
            index_name="blb",
            text_node_property="embedding",
            # retrieval_query=retrieval_query,
        )
        chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(),
            chain_type="stuff",
            retriever=vector_index.as_retriever(k=1),
        )

        return chain

    def run(self, prompt):
        blb_chain = self._get_blb_chain()
        cypher_chain = self._get_cypher_chain()
        vector_chain = self._get_vector_chain()

        tools = [
            Tool(
                name="BLB",
                func=blb_chain,
                description="""
                    Useful for posting actions, assigning events to users and changing events statuses.
                    This tool requires that you have in hand the IDs of objects that you can fetch with the Graph tool
                """,
            ),
            # Tool(
            #     name="Vector",
            #     func=vector_chain.run,
            #     description="""
            #         Useful for searching for users, events and wind turbines by using text filters such as their names
            #     """,
            # ),
            Tool(
                name="Graph",
                func=cypher_chain.run,
                description="""
                    Useful ONLY when you need to answer questions about wind turbines, events and users. This tool can only READ information.
                    Also useful for any sort of aggregation like counting the number of things and traversing. Return fields other than IDs when asked for them.
                    Use full question as input.
                """,
            ),
        ]

        agent = initialize_agent(
            tools,
            ChatOpenAI(temperature=0, model_name="gpt-4-1106-preview"),
            agent=AgentType.OPENAI_FUNCTIONS,
            memory=memory,
            verbose=True,
            return_intermediate_steps=True,
            agent_kwargs={
                "extra_prompt_messages": [
                    MessagesPlaceholder(variable_name="chat_history")
                ],
            },
        )

        response = agent(prompt)

        #       if (chatHistory[sessionId]) {
        #   chatHistory[sessionId].addMessage(new HumanMessage(input));
        #   chatHistory[sessionId].addMessage(new AIMessage(result.output));
        # } else {
        #   chatHistory[sessionId] = new ChatMessageHistory([
        #     new HumanMessage(input),
        #     new AIMessage(result.output),
        #   ]);
        # }

        return response
