import express from "express";
import cors from "cors";
import multer from "multer";
import OpenAI from "openai";
import dotenv from "dotenv";

dotenv.config();

const app = express();

const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 8 * 1024 * 1024,
  },
});

app.use(cors());
app.use(express.json());

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

app.get("/", (req, res) => {
  res.json({
    status: "ok",
    service: "nutrition-ai-server",
  });
});

app.post("/analyze-meal", upload.single("image"), async (req, res) => {
  try {
    if (!process.env.OPENAI_API_KEY) {
      return res.status(500).json({
        error: "OPENAI_API_KEY is missing",
      });
    }

    if (!req.file) {
      return res.status(400).json({
        error: "No image uploaded",
      });
    }

    const base64Image = req.file.buffer.toString("base64");

    let mimeType = req.file.mimetype;

    if (!mimeType || mimeType === "application/octet-stream") {
      const fileName = req.file.originalname?.toLowerCase() || "";

      if (fileName.endsWith(".png")) {
        mimeType = "image/png";
      } else if (fileName.endsWith(".webp")) {
        mimeType = "image/webp";
      } else {
        mimeType = "image/jpeg";
      }
    }

    const response = await client.responses.create({
      model: "gpt-5.4-mini",
      input: [
        {
          role: "user",
          content: [
            {
              type: "input_text",
              text:
                "Analyze the meal in this image. Return ONLY valid JSON, no markdown. " +
                "Estimate calories, protein grams, fat grams and carbs grams. " +
                "The response language should be Hebrew. " +
                "Use realistic nutrition estimation. " +
                "If quantities are unclear, estimate reasonably and lower confidence. " +
                "Use this exact JSON structure: " +
                '{"meal_name":"string","calories":0,"protein":0,"fat":0,"carbs":0,"confidence":"low|medium|high","notes":"string"}',
            },
            {
              type: "input_image",
              image_url: `data:${mimeType};base64,${base64Image}`,
            },
          ],
        },
      ],
    });

    const text = response.output_text ?? "";

    let parsed;

    try {
      parsed = JSON.parse(text);
    } catch (error) {
      return res.status(500).json({
        error: "AI returned invalid JSON",
        raw: text,
      });
    }

    return res.json(parsed);
  } catch (error) {
    console.error(error);

    return res.status(500).json({
      error: "Failed to analyze meal",
      details: error.message,
    });
  }
});

const port = process.env.PORT || 3000;

app.listen(port, () => {
  console.log(`Nutrition AI server is running on port ${port}`);
});