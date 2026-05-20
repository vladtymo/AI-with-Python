import axios from "axios";
import dotenv from "dotenv";
import Groq from "groq-sdk";

dotenv.config();

const groq = new Groq({
  apiKey: process.env.GROQ_API_KEY,
});

const MCP_SERVER = "http://localhost:3000";

async function getRepositoryInfo(owner, repo) {

  const response = await axios.get(
    `${MCP_SERVER}/repo/${owner}/${repo}`
  );

  return response.data;
}

async function analyzeRepository(owner, repo) {

  // STEP 1: Get repository data from MCP server
  const repository = await getRepositoryInfo(owner, repo);

  console.log("Repository data loaded.");

  // STEP 2: Send repository data to Groq LLM
  const completion = await groq.chat.completions.create({

    model: "llama-3.3-70b-versatile",

    messages: [
      {
        role: "system",
        content: `
You are a GitHub repository analysis agent.

Analyze repositories and provide:
- short project summary
- technology stack
- project purpose
- possible use cases
- beginner friendliness
- architecture observations
        `,
      },

      {
        role: "user",
        content: `
Repository data:

Name: ${repository.name}
Description: ${repository.description}
Language: ${repository.language}
Stars: ${repository.stars}
Forks: ${repository.forks}
URL: ${repository.url}
        `,
      },
    ],

    temperature: 0.3,
    max_tokens: 500,
  });

  return completion.choices[0].message.content;
}

async function main() {

  try {

    const result = await analyzeRepository(
      "microsoft",
      "vscode"
    );

    console.log("\n=== AI ANALYSIS ===\n");

    console.log(result);

  } catch (error) {

    console.error("\nError:");

    if (error.status === 429) {
      console.log("Rate limit exceeded.");
    } else {
      console.log(error.message);
    }
  }
}

main();