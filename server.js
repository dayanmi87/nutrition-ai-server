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

function detectImageMimeType(buffer, originalName = "") {
  if (!buffer || buffer.length < 12) {
    return "image/jpeg";
  }

  // JPEG: FF D8 FF
  if (buffer[0] === 0xff && buffer[1] === 0xd8 && buffer[2] === 0xff) {
    return "image/jpeg";
  }

  // PNG: 89 50 4E 47
  if (
    buffer[0] === 0x89 &&
    buffer[1] === 0x50 &&
    buffer[2] === 0x4e &&
    buffer[3] === 0x47
  ) {
    return "image/png";
  }

  // WEBP: RIFF....WEBP
  const riff = buffer.toString("ascii", 0, 4);
  const webp = buffer.toString("ascii", 8, 12);
  if (riff === "RIFF" && webp === "WEBP") {
    return "image/webp";
  }

  const fileName = originalName.toLowerCase();

  if (fileName.endsWith(".png")) return "image/png";
  if (fileName.endsWith(".webp")) return "image/webp";
  if (fileName.endsWith(".jpg") || fileName.endsWith(".jpeg")) {
    return "image/jpeg";
  }

  return "image/jpeg";
}

app.get("/", (req, res) => {
  res.json({
    status: "ok",
    service: "nutrition-ai-server",
    version: "mime-fix-2",
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
    const mimeType = detectImageMimeType(
      req.file.buffer,
      req.file.originalname
    );

    console.log("Uploaded file:", {
      originalname: req.file.originalname,
      multerMimeType: req.file.mimetype,
      detectedMimeType: mimeType,
      size: req.file.size,
    });

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