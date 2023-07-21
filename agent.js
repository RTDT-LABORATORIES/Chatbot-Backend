const { Calculator } = require("langchain/tools/calculator");
const { ChainTool, Tool } = require("langchain/tools");
const { ChatOpenAI } = require("langchain/chat_models/openai");
const axios = require("axios");
const { initializeAgentExecutorWithOptions } = require("langchain/agents");
const { BufferMemory, ChatMessageHistory } = require("langchain/memory");
const { PineconeClient } = require("@pinecone-database/pinecone");
const { PineconeStore } = require("langchain/vectorstores/pinecone");
const { OpenAIEmbeddings } = require("langchain/embeddings/openai");
const { VectorDBQAChain } = require("langchain/chains");
const { AIMessage, HumanMessage } = require("langchain/schema");

class WindTurbineSearch extends Tool {
  name = "wind-turbine-search";

  /** @ignore */
  async _call() {
    const response = await axios.get("https://rtdt-blb.herokuapp.com/wtg/", {
      headers: {
        Authorization: `Token ${process.env.BLB_TOKEN}`,
      },
    });
    const wtgs = response.data.map((wtg) => ({
      name: wtg.name,
      status: wtg.status,
      location: wtg.location,
      farm: {
        name: wtg.farm_data.name,
        operator: wtg.farm_data.operator,
        owner: wtg.farm_data.owner,
      },
      oem: {
        name: wtg.oem_model_data.oem,
        arrangement: wtg.oem_model_data.arrangement,
        control: wtg.oem_model_data.control,
        cut_in_ws: wtg.oem_model_data.cut_in_ws,
        cut_out_ws: wtg.oem_model_data.cut_out_ws,
        deployment: wtg.oem_model_data.deployment,
        drive: wtg.oem_model_data.drive,
        hub_height_m: wtg.oem_model_data.hub_height_m,
        model: wtg.oem_model_data.model,
        rated_power_kw: wtg.oem_model_data.rated_power_kw,
        num_blades: wtg.oem_model_data.num_blades,
        rated_ws: wtg.oem_model_data.rated_ws,
        rotor_diameter_m: wtg.oem_model_data.rotor_diameter_m,
      },
      project: {
        name: wtg.project_data[0].model,
      },
      events: wtg.recorded_events.map((event) => ({
        severity: event.category.name,
        name: event.text,
        status: event.status,
        assigned_to: event.assigned_to,
      })),
    }));

    try {
      return JSON.stringify(wtgs);
    } catch (error) {
      return "I don't know how to do that.";
    }
  }

  description = `Useful for fetching information about existing wind turbines. These wind turbines are owned by him. Wind turbines can also be referred to as "wtgs" or "turbines", so with these words the user probably wants to you list their wind turbines. This data is always up to date.`;
}

class EventSearch extends Tool {
  name = "event-search";

  /** @ignore */
  async _call() {
    const response = await axios.get("https://rtdt-blb.herokuapp.com/wtg/", {
      headers: {
        Authorization: `Token ${process.env.BLB_TOKEN}`,
      },
    });

    const events = response.data.reduce(
      (acc, wtg) => [
        ...acc,
        ...wtg.recorded_events.map((event) => ({
          severity: event.category.name,
          name: event.text,
          status: event.status,
          assigned_to: event.assigned_to,
        })),
      ],
      []
    );

    try {
      return JSON.stringify(events);
    } catch (error) {
      return "I don't know how to do that.";
    }
  }

  description = `Useful for fetching information about wind turbine events or diagnostics. If the user asks for events, you should use this tool. This data is always up to date.`;
}

class EventAssign extends Tool {
  name = "event-assign";

  /** @ignore */
  async _call() {
    try {
      return "Event assigned";
    } catch (error) {
      return "I don't know how to do that.";
    }
  }

  description = `Useful for assigning wind turbine events to users. Input should be in the form eventName,assignee. This tool should NOT be used for fetching information about events.`;
}

const getPineconeVectorStoreTool = async (model) => {
  const client = new PineconeClient();
  await client.init({
    apiKey: process.env.PINECONE_API_KEY,
    environment: process.env.PINECONE_ENVIRONMENT,
  });
  const pineconeIndex = client.Index(process.env.PINECONE_INDEX_NAME);

  const vectorStore = await PineconeStore.fromExistingIndex(
    new OpenAIEmbeddings(),
    { pineconeIndex }
  );
  const chain = VectorDBQAChain.fromLLM(model, vectorStore, {
    // returnSourceDocuments: true,
  });
  const qaTool = new ChainTool({
    name: "qa-oem-wind-turbines",
    description:
      "Useful for answering general questions and questions related to wind turbines operation and maintenance. Always use this tool for answering in case others aren't fit.",
    chain: chain,
  });

  return qaTool;
};

const chatHistory = {};

module.exports = async ({ input, sessionId }) => {
  const model = new ChatOpenAI({ modelName: "gpt-3.5-turbo", temperature: 0 });
  const tools = [
    new WindTurbineSearch(),
    new EventSearch(),
    new EventAssign(),
    new Calculator(),
    await getPineconeVectorStoreTool(model),
  ];
  const executor = await initializeAgentExecutorWithOptions(tools, model, {
    agentType: "openai-functions",
    // verbose: true,
    returnIntermediateSteps: true,
    memory: new BufferMemory({
      chatHistory: chatHistory[sessionId],
      returnMessages: true,
      memoryKey: "chat_history",
      inputKey: "input",
      outputKey: "output",
    }),
  });

  const result = await executor.call({ input });

  if (chatHistory[sessionId]) {
    chatHistory[sessionId].addMessage(new HumanMessage(input));
    chatHistory[sessionId].addMessage(new AIMessage(result.output));
  } else {
    chatHistory[sessionId] = new ChatMessageHistory([
      new HumanMessage(input),
      new AIMessage(result.output),
    ]);
  }

  return result;
};
