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
app.use(express.json({ limit: "2mb" }));

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

function detectImageMimeType(buffer, originalName = "") {
  if (!buffer || buffer.length < 12) return "image/jpeg";

  if (buffer[0] === 0xff && buffer[1] === 0xd8 && buffer[2] === 0xff) {
    return "image/jpeg";
  }

  if (
    buffer[0] === 0x89 &&
    buffer[1] === 0x50 &&
    buffer[2] === 0x4e &&
    buffer[3] === 0x47
  ) {
    return "image/png";
  }

  const riff = buffer.toString("ascii", 0, 4);
  const webp = buffer.toString("ascii", 8, 12);
  if (riff === "RIFF" && webp === "WEBP") return "image/webp";

  const fileName = originalName.toLowerCase();
  if (fileName.endsWith(".png")) return "image/png";
  if (fileName.endsWith(".webp")) return "image/webp";
  if (fileName.endsWith(".jpg") || fileName.endsWith(".jpeg")) return "image/jpeg";

  return "image/jpeg";
}

function toNumber(value) {
  if (typeof value === "number" && Number.isFinite(value)) return Math.round(value);

  if (typeof value === "string") {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return Math.round(parsed);
  }

  return 0;
}

function normalizeQuantity(value) {
  if (typeof value === "number" && Number.isFinite(value)) return value;

  if (typeof value === "string") {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }

  return 1;
}

function normalizeAiResult(raw) {
  const items = Array.isArray(raw.items)
    ? raw.items.map((item) => ({
        name: String(item.name || "רכיב לא מזוהה"),
        quantity: normalizeQuantity(item.quantity),
        unit: String(item.unit || "מנה"),
        calories: toNumber(item.calories),
        protein: toNumber(item.protein),
        fat: toNumber(item.fat),
        carbs: toNumber(item.carbs),
        notes: String(item.notes || ""),
      }))
    : [];

  const totalsFromItems = items.reduce(
    (sum, item) => {
      sum.calories += item.calories;
      sum.protein += item.protein;
      sum.fat += item.fat;
      sum.carbs += item.carbs;
      return sum;
    },
    { calories: 0, protein: 0, fat: 0, carbs: 0 }
  );

  return {
    meal_name: String(raw.meal_name || "ארוחה מנותחת"),
    calories: toNumber(raw.calories) || totalsFromItems.calories,
    protein: toNumber(raw.protein) || totalsFromItems.protein,
    fat: toNumber(raw.fat) || totalsFromItems.fat,
    carbs: toNumber(raw.carbs) || totalsFromItems.carbs,
    confidence: ["low", "medium", "high"].includes(raw.confidence)
      ? raw.confidence
      : "medium",
    notes: String(raw.notes || "הערכה תזונתית. מומלץ לאמת כמויות."),
    items,
  };
}

async function parseJsonFromAi(response) {
  const text = response.output_text ?? "";

  try {
    return JSON.parse(text);
  } catch (error) {
    const match = text.match(/\{[\s\S]*\}/);
    if (match) return JSON.parse(match[0]);
    throw new Error(`AI returned invalid JSON: ${text}`);
  }
}

function requireApiKey(res) {
  if (!process.env.OPENAI_API_KEY) {
    res.status(500).json({ error: "OPENAI_API_KEY is missing" });
    return false;
  }
  return true;
}

app.get("/", (req, res) => {
  res.json({
    status: "ok",
    service: "nutrition-ai-server",
    version: "text-composition-v1",
    endpoints: ["/analyze-meal", "/analyze-text-meal"],
  });
});

app.post("/analyze-meal", upload.single("image"), async (req, res) => {
  try {
    if (!requireApiKey(res)) return;

    if (!req.file) {
      return res.status(400).json({ error: "No image uploaded" });
    }

    const base64Image = req.file.buffer.toString("base64");
    const mimeType = detectImageMimeType(req.file.buffer, req.file.originalname);

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
                "The response language should be Hebrew. " +
                "Break the meal into separate visible food items. " +
                "For each item estimate quantity, unit, calories, protein grams, fat grams, carbs grams. " +
                "Use realistic nutrition estimation. If quantities are unclear, estimate reasonably and lower confidence. " +
                "The total meal values must equal approximately the sum of item values. " +
                "Use this exact JSON structure: " +
                '{"meal_name":"string","calories":0,"protein":0,"fat":0,"carbs":0,"confidence":"low|medium|high","notes":"string","items":[{"name":"string","quantity":1,"unit":"string","calories":0,"protein":0,"fat":0,"carbs":0,"notes":"string"}]}',
            },
            {
              type: "input_image",
              image_url: `data:${mimeType};base64,${base64Image}`,
            },
          ],
        },
      ],
    });

    const parsed = await parseJsonFromAi(response);
    return res.json(normalizeAiResult(parsed));
  } catch (error) {
    console.error(error);
    return res.status(500).json({
      error: "Failed to analyze meal",
      details: error.message,
    });
  }
});

app.post("/analyze-text-meal", async (req, res) => {
  try {
    if (!requireApiKey(res)) return;

    const mealName = String(req.body?.meal_name || "ארוחה ידנית");
    const notes = String(req.body?.notes || "");
    const items = Array.isArray(req.body?.items) ? req.body.items : [];

    if (items.length === 0) {
      return res.status(400).json({ error: "No food items provided" });
    }

    const simplifiedItems = items.map((item) => ({
      name: String(item.name || ""),
      quantity: normalizeQuantity(item.quantity),
      unit: String(item.unit || "מנה"),
      calories: toNumber(item.calories),
      protein: toNumber(item.protein),
      fat: toNumber(item.fat),
      carbs: toNumber(item.carbs),
      notes: String(item.notes || ""),
    }));

    const response = await client.responses.create({
      model: "gpt-5.4-mini",
      input: [
        {
          role: "user",
          content: [
            {
              type: "input_text",
              text:
                "You are a nutrition analysis engine. Return ONLY valid JSON, no markdown. " +
                "The response language should be Hebrew. " +
                "The user entered a manual/fixed meal with food item names, quantities and units. " +
                "Analyze and correct the meal composition and nutritional values. " +
                "If an item has missing or zero nutrition values, estimate them. " +
                "If the user wrote a broad item, split it only when clearly useful. " +
                "Preserve the user's intended foods and quantities unless clearly wrong. " +
                "Use realistic Israeli/common nutrition values. " +
                "The total meal values must equal approximately the sum of item values. " +
                "Meal name: " +
                mealName +
                "\nNotes: " +
                notes +
                "\nItems JSON: " +
                JSON.stringify(simplifiedItems) +
                "\nUse this exact JSON structure: " +
                '{"meal_name":"string","calories":0,"protein":0,"fat":0,"carbs":0,"confidence":"low|medium|high","notes":"string","items":[{"name":"string","quantity":1,"unit":"string","calories":0,"protein":0,"fat":0,"carbs":0,"notes":"string"}]}',
            },
          ],
        },
      ],
    });

    const parsed = await parseJsonFromAi(response);
    return res.json(normalizeAiResult(parsed));
  } catch (error) {
    console.error(error);
    return res.status(500).json({
      error: "Failed to analyze text meal",
      details: error.message,
    });
  }
});

const port = process.env.PORT || 3000;

app.listen(port, () => {
  console.log(`Nutrition AI server is running on port ${port}`);
});
