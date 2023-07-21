const express = require("express");
const cors = require("cors");

require("dotenv").config();

const agent = require("./agent.js");
const app = express();
const port = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

app.post("/prompt", async function (req, res, next) {
  try {
    const result = await agent({
      input: req.body.query,
      sessionId: req.body.sessionId,
    });

    res.json(result);
  } catch (e) {
    next(e);
  }
});

app.listen(port, () => {
  console.log(`Server listening on port ${port}`);
});
