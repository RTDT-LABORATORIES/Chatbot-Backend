import os
import json
from langchain.graphs import Neo4jGraph
from langchain.chat_models import ChatOpenAI
from langchain.chains import GraphCypherQAChain, APIChain, RetrievalQA
from langchain.agents import AgentType, initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
from langchain.chains.openai_functions.openapi import get_openapi_chain
from langchain_community.utilities.openapi import OpenAPISpec
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
)
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain.embeddings.openai import OpenAIEmbeddings


# url = "neo4j://localhost:7687"
# password = "password"
NEO4J_URL = os.environ["NEO4J_URL"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]

graph = Neo4jGraph(url=NEO4J_URL, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)
graph.refresh_schema()

chat_history = {}

memory = ConversationBufferMemory(
    memory_key="chat_history", input_key="input", output_key="output"
)


class Chatbot:
    _session = None

    def __init__(self, session):
        self._session = session

    # def _get_blb_chain(self):
    #     with open("blb_openapi_spec.json") as f:
    #         blb_openapi_spec = json.load(f)

    #     blb_openapi_spec["securityDefinitions"] = {}
    #     blb_openapi_spec["security"] = []

    #     BLB_API_KEY = os.environ["BLB_API_KEY"]
    #     headers = {"Authorization": f"Token {BLB_API_KEY}"}

    #     API_URL_PROMPT_TEMPLATE = """You are given the below API Documentation:
    #     {api_docs}
    #     Using this documentation, generate the full API url to call for answering the user question.
    #     You should build the API url in order to get a response that is as short as possible, while still getting the necessary information to answer the question. Pay attention to deliberately exclude any unnecessary pieces of data in the API call.

    #     Background:

    #     Id for user "Yuri Jean Fabris" is 1
    #     Id for user "Yuri Jean Fabris 2" is 6
    #     Id for user "Yuri Jean Fabris 3" is 7

    #     Id for event "Fatigue increased" is 1

    #     Id for wind turbine "P04" from farm "San Gottardo" is 1
    #     Id for wind turbine "P05" from farm "San Gottardo" is 2
    #     Id for wind turbine "P09" from farm "Ptoon" is 3

    #     Question:{question}
    #     API url:"""

    #     API_URL_PROMPT = PromptTemplate(
    #         input_variables=[
    #             "api_docs",
    #             "question",
    #         ],
    #         template=API_URL_PROMPT_TEMPLATE,
    #     )

    #     blb_chain = APIChain.from_llm_and_api_docs(
    #         llm=ChatOpenAI(temperature=0, model_name="gpt-4-1106-preview"),
    #         api_docs=json.dumps(blb_openapi_spec),
    #         headers=headers,
    #         api_url_prompt=API_URL_PROMPT,
    #         verbose=True,
    #         limit_to_domains=[
    #             # "*"
    #             f"http://{blb_openapi_spec['host']}",
    #             # f"https://{blb_openapi_spec['host']}",
    #         ],
    #     )

    #     return blb_chain

    def _get_blb_chain(self):
        BLB_API_KEY = os.environ["BLB_API_KEY"]
        headers = {"Authorization": f"Token {BLB_API_KEY}"}
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
            # Tool(
            #     name="Tasks",
            #     func=vector_qa.run,
            #     description="""Useful when you need to answer questions about descriptions of tasks.
            #     Not useful for counting the number of tasks.
            #     Use full question as input.
            #     """,
            # ),
            Tool(
                name="BLB",
                func=blb_chain,
                description="""
                    Useful for posting actions, assigning events to users and changing events statuses.
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
                    Also useful for any sort of aggregation like counting the number of things and traversing.
                    Use full question as input.
                """,
            ),
        ]

        # prompt = ChatPromptTemplate(
        #     messages=[
        #         SystemMessagePromptTemplate.from_template(
        #             """
        #             You have the following information about users, wind turbines and events. Use it to extract IDs to pass to the BLB tool.

        #                                                   Users:

        #             [
        #               { "name": "Yuri Jean Fabris", "id": 1 },
        #               { "name": "Yuri Jean Fabris 2", "id": 6 },
        #               { "name": "Yuri Jean Fabris 3", "id": 7 },
        #             ]

        #             Events:

        #             [
        #               { "name": "Fatigue increased", "id": 1 },
        #             ]

        #             Wind turbines:

        #             [
        #               { "name": "P04", "farm": "San Gottardo", "id": 1 },
        #               { "name": "P05", "farm": "San Gottardo", "id": 2 },
        #               { "name": "P09", "farm": "Ptoon", "id": 3 },
        #             ]
        #         """
        #         ),
        #         MessagesPlaceholder(variable_name="chat_history"),
        #         HumanMessagePromptTemplate.from_template("{question}"),
        #     ]
        # )

        agent = initialize_agent(
            tools,
            ChatOpenAI(temperature=0, model_name="gpt-4-1106-preview"),
            agent=AgentType.OPENAI_FUNCTIONS,
            memory=memory,
            verbose=True,
            return_intermediate_steps=True,
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
