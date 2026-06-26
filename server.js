import express from "express";
import cors from "cors";
import multer from "multer";
import OpenAI from "openai";
import dotenv from "dotenv";

dotenv.config();

const app = express();

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 12 * 1024 * 1024 },
});

app.use(cors());
app.use(express.json({ limit: "4mb" }));

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

function requireApiKey(res) {
  if (!process.env.OPENAI_API_KEY) {
    res.status(500).json({ error: "OPENAI_API_KEY is missing on the server" });
    return false;
  }
  return true;
}

function detectImageMimeType(buffer, originalName = "") {
  if (buffer?.length >= 4) {
    if (buffer[0] === 0xff && buffer[1] === 0xd8 && buffer[2] === 0xff) return "image/jpeg";
    if (buffer[0] === 0x89 && buffer[1] === 0x50 && buffer[2] === 0x4e && buffer[3] === 0x47) return "image/png";
    if (buffer[0] === 0x52 && buffer[1] === 0x49 && buffer[2] === 0x46 && buffer[3] === 0x46) return "image/webp";
  }

  const lower = String(originalName || "").toLowerCase();
  if (lower.endsWith(".png")) return "image/png";
  if (lower.endsWith(".webp")) return "image/webp";
  return "image/jpeg";
}

function extractTextFromOpenAiResponse(response) {
  if (typeof response.output_text === "string" && response.output_text.trim()) {
    return response.output_text.trim();
  }

  const parts = [];
  for (const item of response.output || []) {
    for (const content of item.content || []) {
      if (typeof content.text === "string") parts.push(content.text);
    }
  }

  return parts.join("\n").trim();
}

function extractJson(text) {
  const raw = String(text || "").trim();

  try {
    return JSON.parse(raw);
  } catch (_) {}

  const fenced = raw.match(/```(?:json)?\s*([\s\S]*?)```/i);
  if (fenced) {
    try {
      return JSON.parse(fenced[1].trim());
    } catch (_) {}
  }

  const first = raw.indexOf("{");
  const last = raw.lastIndexOf("}");
  if (first >= 0 && last > first) {
    return JSON.parse(raw.slice(first, last + 1));
  }

  throw new Error("OpenAI did not return valid JSON");
}

async function parseJsonFromAi(response) {
  const text = extractTextFromOpenAiResponse(response);
  return extractJson(text);
}

function toNumber(value) {
  const number = Number(String(value ?? "").replace(",", ".").replace(/[^\d.-]/g, ""));
  return Number.isFinite(number) ? number : 0;
}

function normalizeQuantity(value) {
  const n = toNumber(value);
  if (n <= 0) return 1;
  return n;
}

function round1(value) {
  return Math.round(toNumber(value) * 10) / 10;
}

function normalizeItem(item) {
  return {
    name: String(item?.name || "רכיב מזון").trim(),
    quantity: normalizeQuantity(item?.quantity),
    unit: String(item?.unit || "מנה").trim(),
    calories: round1(item?.calories),
    protein: round1(item?.protein),
    fat: round1(item?.fat),
    carbs: round1(item?.carbs),
    notes: String(item?.notes || "").trim(),
    source: "chatgpt_direct_analysis",
    confidence: String(item?.confidence || "").trim() || undefined,
  };
}

function normalizeMealResult(parsed, fallbackName = "ארוחה") {
  const items = Array.isArray(parsed?.items) ? parsed.items.map(normalizeItem) : [];

  const totals = items.reduce(
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
    meal_name: String(parsed?.meal_name || fallbackName).trim(),
    calories: Math.round(totals.calories),
    protein: round1(totals.protein),
    fat: round1(totals.fat),
    carbs: round1(totals.carbs),
    confidence: ["low", "medium", "high"].includes(String(parsed?.confidence)) ? parsed.confidence : "medium",
    notes: String(parsed?.notes || "הערכים חושבו ישירות על ידי ChatGPT לפי הרכיבים והכמויות שזוהו.").trim(),
    source: "chatgpt_direct_analysis",
    items,
  };
}

function systemPrompt() {
  return `
אתה מנוע תזונה קליני בתוך אפליקציה לניטור תזונה.
כל ניתוח חייב להתבצע ישירות על ידך, ChatGPT, ללא שימוש במאגר פנימי ישן וללא ניחוש שרירותי.
המטרה: להחזיר הערכה שימושית של קלוריות, חלבון, שומן ופחמימות.

כללים מחייבים:
1. החזר JSON בלבד. בלי Markdown, בלי הסברים מחוץ ל-JSON.
2. נתח כל רכיב בנפרד.
3. הערך כמות לכל רכיב ביחידה שהמשתמש נתן או ביחידה שימושית בישראל: גרם, מ"ל, כף, כפית, פרוסה, יחידה, סקופ, קערה, מנה.
4. אם מדובר במוצר ממותג/חטיף מוכר, השתמש בערכים תזונתיים מוכרים למוצר ולגודל מנה מקובל.
5. אם מדובר בתמונה, זהה את כל מה שנראה: חלבון, פחמימה, ירקות, רטבים, שמנים, אגוזים, גבינות, לחם, שתייה ותוספות.
6. אם אינך בטוח, תן אומדן שמרני ומציאותי וציין את ההנחה ב-notes.
7. totals חייבים להיות סכום items.
8. אל תחזיר רכיב עם אפס ערכים אלא אם באמת אין קלוריות משמעותיות.
9. התשובה בעברית.
10. שים לב:
   - סקופ אבקת חלבון רגיל: לרוב 30 גרם, כ-120 קלוריות, כ-24 גרם חלבון.
   - ביצה רגילה: כ-70 קלוריות, כ-6 גרם חלבון.
   - כף שמן: כ-120 קלוריות, כ-14 גרם שומן.
   - פרוסת לחם רגילה: כ-70-90 קלוריות.
   - עוגיית אוראו אחת: כ-50-55 קלוריות, בעיקר פחמימות ושומן.
   - במבה 25 גרם: כ-130-140 קלוריות.

מבנה JSON חובה:
{
  "meal_name": "string",
  "calories": 0,
  "protein": 0,
  "fat": 0,
  "carbs": 0,
  "confidence": "low|medium|high",
  "notes": "string",
  "items": [
    {
      "name": "string",
      "quantity": 1,
      "unit": "string",
      "calories": 0,
      "protein": 0,
      "fat": 0,
      "carbs": 0,
      "confidence": "low|medium|high",
      "notes": "string"
    }
  ]
}
`.trim();
}

async function analyzeImageWithChatGpt({ base64Image, mimeType }) {
  const response = await client.responses.create({
    model: process.env.OPENAI_MODEL || "gpt-5.4-mini",
    input: [
      {
        role: "system",
        content: [{ type: "input_text", text: systemPrompt() }],
      },
      {
        role: "user",
        content: [
          {
            type: "input_text",
            text:
              "נתח את התמונה המצורפת כארוחה מלאה. " +
              "חובה לזהות רכיבים, להעריך כמות לכל רכיב, ולחשב קלוריות/חלבון/שומן/פחמימות לכל רכיב ולכל הארוחה. " +
              "הנתונים צריכים להגיע מהניתוח שלך על התמונה בלבד. החזר JSON בלבד.",
          },
          {
            type: "input_image",
            image_url: `data:${mimeType};base64,${base64Image}`,
          },
        ],
      },
    ],
  });

  return normalizeMealResult(await parseJsonFromAi(response), "ארוחה מצולמת");
}

function buildTextMealPrompt(mealName, items) {
  const lines = items
    .map((item, index) => {
      const name = String(item?.name || "").trim();
      const quantity = normalizeQuantity(item?.quantity);
      const unit = String(item?.unit || "מנה").trim();
      return `${index + 1}. ${name} | ${quantity} | ${unit}`;
    })
    .join("\n");

  return `
נתח את הארוחה הבאה:
שם הארוחה: ${mealName}

רכיבים:
${lines}

עבור כל רכיב:
- חשב קלוריות, חלבון, שומן ופחמימות.
- השתמש בכמות וביחידה שניתנו.
- אם היחידה היא "יחידה", "מנה", "כף", "כפית", "פרוסה" או "סקופ" — המר אותה לכמות תזונתית מקובלת.
- אם זה מוצר מוכר או חטיף מוכר, השתמש בערך התזונתי המקובל למוצר הזה.
- החזר JSON בלבד לפי הסכמה.
`.trim();
}

async function analyzeTextMealWithChatGpt({ mealName, items }) {
  const response = await client.responses.create({
    model: process.env.OPENAI_MODEL || "gpt-5.4-mini",
    input: [
      {
        role: "system",
        content: [{ type: "input_text", text: systemPrompt() }],
      },
      {
        role: "user",
        content: [{ type: "input_text", text: buildTextMealPrompt(mealName, items) }],
      },
    ],
  });

  return normalizeMealResult(await parseJsonFromAi(response), mealName);
}

app.get("/", (req, res) => {
  res.json({
    status: "ok",
    service: "nutrition-ai-server",
    version: "metric-meal-v9-chatgpt-first-from-scratch",
    model: process.env.OPENAI_MODEL || "gpt-5.4-mini",
    rule: "Every image and every text/manual item is sent to ChatGPT for direct nutrition analysis. No internal food DB overrides the result.",
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

    const result = await analyzeImageWithChatGpt({ base64Image, mimeType });
    return res.json(result);
  } catch (error) {
    console.error("analyze-meal failed:", error);
    return res.status(500).json({
      error: "Failed to analyze meal image with ChatGPT",
      details: error.message,
    });
  }
});

app.post("/analyze-text-meal", async (req, res) => {
  try {
    if (!requireApiKey(res)) return;

    const mealName = String(req.body?.meal_name || "ארוחה ידנית");
    const items = Array.isArray(req.body?.items) ? req.body.items : [];

    if (items.length === 0) {
      return res.status(400).json({ error: "No food items provided" });
    }

    const result = await analyzeTextMealWithChatGpt({ mealName, items });
    return res.json(result);
  } catch (error) {
    console.error("analyze-text-meal failed:", error);
    return res.status(500).json({
      error: "Failed to analyze text meal with ChatGPT",
      details: error.message,
    });
  }
});

const port = process.env.PORT || 3000;

app.listen(port, () => {
  console.log(`Nutrition AI server v9 is running on port ${port}`);
});
