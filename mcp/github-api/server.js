import express from "express";
import axios from "axios";
import dotenv from "dotenv";

dotenv.config();

const app = express();

const github = axios.create({
  baseURL: "https://api.github.com",
  headers: {
    Authorization: `Bearer ${process.env.GITHUB_TOKEN}`,
    Accept: "application/vnd.github+json",
  },
});

app.get("/repo/:owner/:repo", async (req, res) => {
  try {
    const { owner, repo } = req.params;

    // Get repository info
    const repoResponse = await github.get(`/repos/${owner}/${repo}`);

    // Get README
    const readmeResponse = await github.get(
      `/repos/${owner}/${repo}/readme`
    );

    const result = {
      name: repoResponse.data.name,
      description: repoResponse.data.description,
      stars: repoResponse.data.stargazers_count,
      forks: repoResponse.data.forks_count,
      language: repoResponse.data.language,
      url: repoResponse.data.html_url,
      readmeBase64: readmeResponse.data.content,
    };

    res.json(result);

  } catch (error) {
    console.error(error.message);

    res.status(500).json({
      error: error.message,
    });
  }
});

app.listen(3000, () => {
  console.log("MCP GitHub server running on port 3000");
});