import os
import json
from langchain.graphs import Neo4jGraph
from langchain.chat_models import ChatOpenAI
from langchain.chains import GraphCypherQAChain
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain_community.agent_toolkits.openapi.spec import reduce_openapi_spec
from langchain.requests import RequestsWrapper
from langchain_community.tools.json.tool import JsonSpec
from langchain_community.agent_toolkits import OpenAPIToolkit

from langchain.chains import APIChain

# url = "neo4j://localhost:7687"
# password = "password"
url = os.environ["NEO4J_URL"]
username = os.environ["NEO4J_USERNAME"]
password = os.environ["NEO4J_PASSWORD"]

graph = Neo4jGraph(url=url, username=username, password=password)
graph.refresh_schema()


class Chatbot:
    _session = None

    def __init__(self, session):
        self._session = session

    def _get_blb_chain(self):
        with open("blb_openapi_spec.json") as f:
            blb_openapi_spec = json.load(f)

        # blb_openapi_spec = JsonSpec(dict_=blb_raw_openapi_spec, max_value_length=4000)
        # blb_openapi_spec = reduce_openapi_spec(blb_raw_openapi_spec)

        BLB_API_KEY = os.environ["BLB_API_KEY"]
        headers = {"Authorization": f"Bearer {BLB_API_KEY}"}

        blb_chain = APIChain.from_llm_and_api_docs(
            llm=ChatOpenAI(temperature=0, model_name="gpt-4-1106-preview"),
            api_docs=json.dumps(blb_openapi_spec),
            headers=headers,
            verbose=True,
            limit_to_domains=["https://listen-api.listennotes.com/"],
        )

        return blb_chain

    def _get_cypher_chain(self):
        cypher_chain = GraphCypherQAChain.from_llm(
            cypher_llm=ChatOpenAI(temperature=0, model_name="gpt-4-1106-preview"),
            qa_llm=ChatOpenAI(temperature=0, model_name="gpt-4-1106-preview"),
            graph=graph,
            verbose=True,
            validate_cypher=True,
            return_intermediate_steps=True,
        )

        return cypher_chain

    def run(self, prompt):
        blb_chain = self._get_blb_chain()
        cypher_chain = self._get_cypher_chain()

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
                func=blb_chain.run,
                description="""
                    Useful for posting/creating/registering actions, assigning events to users and changing events statuses.
                """,
            ),
            Tool(
                name="Graph",
                func=cypher_chain.run,
                description="""
                    Useful ONLY when you need to answer questions about wind turbines, events and users.
                    Also useful for any sort of aggregation like counting the number of things and traversing.
                    Use full question as input.
                """,
            ),
        ]

        agent = initialize_agent(
            tools,
            ChatOpenAI(temperature=0, model_name="gpt-4-1106-preview"),
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=True,
            return_intermediate_steps=True,
        )

        response = agent(prompt)

        print(response)

        return response
