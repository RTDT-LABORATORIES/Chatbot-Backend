const express = require("express");
const cors = require("cors");
const agent = require("./agent.js");

require("dotenv").config();

const app = express();

app.use(cors());
app.use(express.json());

app.post("/prompt", async function (req, res) {
  console.log(req.body);

  const result = await agent({
    input: req.body.query,
    sessionId: req.body.sessionId,
  });

  console.log(result);

  res.json(result);
});

app.listen(3000);
