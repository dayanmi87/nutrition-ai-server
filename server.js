import express from "express";
import cors from "cors";
import multer from "multer";
import OpenAI from "openai";
import dotenv from "dotenv";
import crypto from "crypto";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

dotenv.config();

const app = express();
const upload = multer({ storage: multer.memoryStorage(), limits: { fileSize: 12 * 1024 * 1024 } });
app.use(cors());
app.use(express.json({ limit: "4mb" }));

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const OPEN_FOOD_FACTS_USER_AGENT =
  process.env.OPEN_FOOD_FACTS_USER_AGENT ||
  "nutrition-app/1.0 (contact: nutrition-app-user)";
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const CACHE_DIR = path.join(__dirname, ".nutrition-cache");
const CACHE_FILE = path.join(CACHE_DIR, "analysis-cache.json");
const CACHE_ENABLED = String(process.env.NUTRITION_CACHE_ENABLED || "true").toLowerCase() !== "false";

function loadCache() {
  if (!CACHE_ENABLED) return {};
  try {
    if (!fs.existsSync(CACHE_DIR)) fs.mkdirSync(CACHE_DIR, { recursive: true });
    if (!fs.existsSync(CACHE_FILE)) return {};
    return JSON.parse(fs.readFileSync(CACHE_FILE, "utf8"));
  } catch (error) {
    console.error("Failed loading nutrition cache:", error.message);
    return {};
  }
}
let analysisCache = loadCache();
function saveCache() {
  if (!CACHE_ENABLED) return;
  try {
    if (!fs.existsSync(CACHE_DIR)) fs.mkdirSync(CACHE_DIR, { recursive: true });
    fs.writeFileSync(CACHE_FILE, JSON.stringify(analysisCache, null, 2), "utf8");
  } catch (error) { console.error("Failed saving nutrition cache:", error.message); }
}
function hashValue(value) { return crypto.createHash("sha256").update(value).digest("hex"); }
function stableStringify(value) {
  if (Array.isArray(value)) return `[${value.map(stableStringify).join(",")}]`;
  if (value && typeof value === "object") return `{${Object.keys(value).sort().map(k => `${JSON.stringify(k)}:${stableStringify(value[k])}`).join(",")}}`;
  return JSON.stringify(value);
}
function requireApiKey(res) {
  if (!process.env.OPENAI_API_KEY) { res.status(500).json({ error: "OPENAI_API_KEY is missing on the server" }); return false; }
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
  if (typeof response.output_text === "string" && response.output_text.trim()) return response.output_text.trim();
  const parts = [];
  for (const item of response.output || []) for (const content of item.content || []) if (typeof content.text === "string") parts.push(content.text);
  return parts.join("\n").trim();
}
function extractJson(text) {
  const raw = String(text || "").trim();
  try { return JSON.parse(raw); } catch (_) {}
  const fenced = raw.match(/```(?:json)?\s*([\s\S]*?)```/i);
  if (fenced) { try { return JSON.parse(fenced[1].trim()); } catch (_) {} }
  const first = raw.indexOf("{"); const last = raw.lastIndexOf("}");
  if (first >= 0 && last > first) return JSON.parse(raw.slice(first, last + 1));
  throw new Error("OpenAI did not return valid JSON");
}
async function parseJsonFromAi(response) { return extractJson(extractTextFromOpenAiResponse(response)); }
function toNumber(value) { const n = Number(String(value ?? "").replace(",", ".").replace(/[^\d.-]/g, "")); return Number.isFinite(n) ? n : 0; }
function normalizeQuantity(value) { const n = toNumber(value); return n <= 0 ? 1 : n; }
function round1(value) { return Math.round(toNumber(value) * 10) / 10; }
function normalizeName(value) { return String(value || "").toLowerCase().replace(/[׳']/g, "'").replace(/״/g, '"').replace(/[^\p{L}\p{N}\s."'%-]/gu, " ").replace(/\s+/g, " ").trim(); }
function normalizeInputItems(items) { return (Array.isArray(items) ? items : []).map(item => ({ name: normalizeName(item?.name), quantity: normalizeQuantity(item?.quantity), unit: normalizeName(item?.unit || "מנה") })); }
function normalizeItem(item) {
  return { name: String(item?.name || "רכיב מזון").trim(), quantity: normalizeQuantity(item?.quantity), unit: String(item?.unit || "מנה").trim(), calories: round1(item?.calories), protein: round1(item?.protein), fat: round1(item?.fat), carbs: round1(item?.carbs), notes: String(item?.notes || "").trim(), source: "chatgpt_direct_analysis_cached_stable", confidence: String(item?.confidence || "").trim() || undefined };
}
function normalizeMealResult(parsed, fallbackName = "ארוחה") {
  const items = Array.isArray(parsed?.items) ? parsed.items.map(normalizeItem) : [];
  const totals = items.reduce((s, i) => ({ calories: s.calories + i.calories, protein: s.protein + i.protein, fat: s.fat + i.fat, carbs: s.carbs + i.carbs }), { calories: 0, protein: 0, fat: 0, carbs: 0 });
  return { meal_name: String(parsed?.meal_name || fallbackName).trim(), calories: Math.round(totals.calories), protein: round1(totals.protein), fat: round1(totals.fat), carbs: round1(totals.carbs), confidence: ["low","medium","high"].includes(String(parsed?.confidence)) ? parsed.confidence : "medium", notes: String(parsed?.notes || "הערכים חושבו ישירות על ידי ChatGPT לפי הרכיבים והכמויות שזוהו.").trim(), source: "chatgpt_direct_analysis_cached_stable", items };
}
function withCacheMeta(result, cacheKey, cacheHit) {
  return { ...result, cache_key: cacheKey, cache_hit: cacheHit, notes: `${result.notes || ""}${cacheHit ? " | תוצאה זהה נשלפה מהמטמון כדי למנוע שינוי בין ניתוחים חוזרים." : " | תוצאה חדשה נשמרה במטמון כדי שבניתוח חוזר לאותם נתונים הערכים יישארו קבועים."}`.trim() };
}
function systemPrompt() { return `אתה מנוע תזונה קליני בתוך אפליקציה לניטור תזונה.
המטרה: להחזיר הערכה יציבה, עקבית ושימושית של קלוריות, חלבון, שומן ופחמימות.
אותו קלט חייב להחזיר אותה תוצאה ככל האפשר.
כללים מחייבים:
1. החזר JSON בלבד. בלי Markdown.
2. נתח כל רכיב בנפרד.
3. אל תשנה הערכה סתם בין ניתוחים. בחר ערך נקודתי אחד וסביר, לא טווח.
4. אם מדובר במוצר/מנה מוכרים, השתמש בערכים תזונתיים מקובלים למוצר ולגודל מנה מקובל.
5. אם מדובר בתמונה, זהה את כל הרכיבים הנראים והערך כמות לכל רכיב.
6. אם אינך בטוח, תן אומדן שמרני ומציאותי וציין את ההנחה ב-notes.
7. totals חייבים להיות סכום items בלבד.
8. התשובה בעברית.
כללי בסיס: סקופ אבקת חלבון 30 גרם כ-120 קלוריות וכ-24 גרם חלבון; ביצה רגילה כ-70 קלוריות וכ-6 גרם חלבון; כף שמן כ-120 קלוריות וכ-14 גרם שומן; עוגיית אוראו אחת כ-50-55 קלוריות; במבה 25 גרם כ-130-140 קלוריות.
מבנה JSON חובה:
{"meal_name":"string","calories":0,"protein":0,"fat":0,"carbs":0,"confidence":"low|medium|high","notes":"string","items":[{"name":"string","quantity":1,"unit":"string","calories":0,"protein":0,"fat":0,"carbs":0,"confidence":"low|medium|high","notes":"string"}]}`; }
async function createResponse(input) {
  const payload = { model: process.env.OPENAI_MODEL || "gpt-5.4-mini", input };
  if (process.env.OPENAI_TEMPERATURE !== undefined) payload.temperature = Number(process.env.OPENAI_TEMPERATURE);
  return client.responses.create(payload);
}
async function analyzeImageWithChatGpt({ base64Image, mimeType }) {
  const cacheKey = `image:${hashValue(base64Image)}`;
  if (analysisCache[cacheKey]) return withCacheMeta(analysisCache[cacheKey], cacheKey, true);
  const response = await createResponse([{ role: "system", content: [{ type: "input_text", text: systemPrompt() }] }, { role: "user", content: [{ type: "input_text", text: "נתח את התמונה כארוחה מלאה. זהה רכיבים, הערך כמות לכל רכיב, חשב קלוריות/חלבון/שומן/פחמימות לכל רכיב ולכל הארוחה. בחר הערכה נקודתית אחת ואל תחזיר טווח. החזר JSON בלבד." }, { type: "input_image", image_url: `data:${mimeType};base64,${base64Image}` }] }]);
  const result = normalizeMealResult(await parseJsonFromAi(response), "ארוחה מצולמת");
  analysisCache[cacheKey] = result; saveCache(); return withCacheMeta(result, cacheKey, false);
}
function buildTextMealPrompt(mealName, items) {
  const lines = normalizeInputItems(items).map((item, i) => `${i + 1}. ${item.name} | ${item.quantity} | ${item.unit}`).join("\n");
  return `נתח את הארוחה הבאה:\nשם הארוחה: ${mealName}\n\nרכיבים:\n${lines}\n\nעבור כל רכיב חשב קלוריות, חלבון, שומן ופחמימות. השתמש בכמות וביחידה שניתנו. אם היחידה היא יחידה/מנה/כף/כפית/פרוסה/סקופ המר לכמות מקובלת. אם זה מוצר מוכר השתמש בערך המקובל. בחר הערכה נקודתית אחת ואל תחזיר טווח. החזר JSON בלבד.`;
}
async function analyzeTextMealWithChatGpt({ mealName, items }) {
  const normalizedItems = normalizeInputItems(items);
  const cacheKey = `text:${hashValue(stableStringify({ mealName: normalizeName(mealName), items: normalizedItems }))}`;
  if (analysisCache[cacheKey]) return withCacheMeta(analysisCache[cacheKey], cacheKey, true);
  const response = await createResponse([{ role: "system", content: [{ type: "input_text", text: systemPrompt() }] }, { role: "user", content: [{ type: "input_text", text: buildTextMealPrompt(mealName, items) }] }]);
  const result = normalizeMealResult(await parseJsonFromAi(response), mealName);
  analysisCache[cacheKey] = result; saveCache(); return withCacheMeta(result, cacheKey, false);
}

function buildMealImagePrompt({ mealName, description, items }) {
  const itemText = Array.isArray(items)
    ? items
        .map((item) => `${item.name || ""} ${item.quantity || ""} ${item.unit || ""}`.trim())
        .filter(Boolean)
        .join(", ")
    : "";

  return `
Create a realistic, appetizing, high-quality photo of the exact meal described below.
The image should look like a real smartphone food photo for a nutrition tracking app.
Show the actual foods described, not a random healthy plate.
No text, no labels, no logos, no hands, no people.
Use a clean plate or bowl, natural light, top-down or 45-degree angle.
Meal name: ${mealName || "meal"}
Meal description: ${description || ""}
Meal items: ${itemText || ""}
`.trim();
}

app.post("/generate-meal-image", async (req, res) => {
  try {
    if (!requireApiKey(res)) return;

    const mealName = String(req.body?.meal_name || "ארוחה");
    const description = String(req.body?.description || "");
    const items = Array.isArray(req.body?.items) ? req.body.items : [];
    const prompt = buildMealImagePrompt({ mealName, description, items });

    const image = await client.images.generate({
      model: process.env.OPENAI_IMAGE_MODEL || "gpt-image-1",
      prompt,
      n: 1,
      size: "1024x1024",
      quality: process.env.OPENAI_IMAGE_QUALITY || "low",
      output_format: "jpeg",
    });

    const first = image?.data?.[0] || {};
    return res.json({
      status: "ok",
      image_base64: first.b64_json || "",
      image_url: first.url || "",
      source: "openai_image_generation",
    });
  } catch (error) {
    console.error("generate-meal-image failed:", error);
    return res.status(500).json({
      error: "Failed to generate meal image",
      details: error.message,
    });
  }
});



function numberFrom(value) {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  const parsed = Number(String(value ?? "").replace(",", ".").replace(/[^\d.-]/g, ""));
  return Number.isFinite(parsed) ? parsed : 0;
}

function parseQuantityGrams(value) {
  const text = String(value || "").toLowerCase().replace(",", ".");
  const gramMatch = text.match(/([\d.]+)\s*(g|גרם|grams?)/);
  if (gramMatch) return Number(gramMatch[1]);
  const kgMatch = text.match(/([\d.]+)\s*(kg|ק\"ג|קג|kilograms?)/);
  if (kgMatch) return Number(kgMatch[1]) * 1000;
  const mlMatch = text.match(/([\d.]+)\s*(ml|מ\"ל|מל)/);
  if (mlMatch) return Number(mlMatch[1]);
  const lMatch = text.match(/([\d.]+)\s*(l|liter|litre|ליטר)/);
  if (lMatch) return Number(lMatch[1]) * 1000;
  return 0;
}

function firstPositive(obj, keys) {
  for (const key of keys) {
    const value = numberFrom(obj?.[key]);
    if (value > 0) return value;
  }
  return 0;
}

function kcalFromEnergy(value) {
  const n = numberFrom(value);
  if (n <= 0) return 0;
  return n > 1000 ? n / 4.184 : n;
}

function kcalPer100(nutriments) {
  return firstPositive(nutriments, ["energy-kcal_100g", "energy-kcal"]) || kcalFromEnergy(nutriments?.["energy_100g"]) || kcalFromEnergy(nutriments?.energy);
}

function macroPer100(nutriments, macro) {
  return firstPositive(nutriments, [`${macro}_100g`, macro]);
}

function kcalServing(nutriments) {
  return firstPositive(nutriments, ["energy-kcal_serving"]) || kcalFromEnergy(nutriments?.["energy_serving"]);
}

function macroServing(nutriments, macro) {
  return firstPositive(nutriments, [`${macro}_serving`]);
}

function roundMacro(value) {
  return Math.round(numberFrom(value) * 10) / 10;
}

function roundKcal(value) {
  return Math.round(numberFrom(value));
}

function barcodeServingGrams(product) {
  const servingQuantity = numberFrom(product.serving_quantity);
  if (servingQuantity > 0) return servingQuantity;
  const servingSize = parseQuantityGrams(product.serving_size);
  if (servingSize > 0) return servingSize;
  const packageQuantity = parseQuantityGrams(product.quantity);
  if (packageQuantity > 0 && packageQuantity <= 250) return packageQuantity;
  return 100;
}

function barcodeNutritionFromProduct(product) {
  const n = product.nutriments || {};
  const servingGrams = barcodeServingGrams(product);
  const factor = servingGrams / 100;

  const per100 = {
    calories: kcalPer100(n),
    protein: macroPer100(n, "proteins"),
    fat: macroPer100(n, "fat"),
    carbs: macroPer100(n, "carbohydrates"),
  };

  const perServing = {
    calories: kcalServing(n),
    protein: macroServing(n, "proteins"),
    fat: macroServing(n, "fat"),
    carbs: macroServing(n, "carbohydrates"),
  };

  const hasFullServing = perServing.calories > 0 && perServing.protein > 0 && perServing.fat > 0 && perServing.carbs > 0;

  const values = hasFullServing
    ? perServing
    : {
        calories: per100.calories * factor,
        protein: per100.protein * factor,
        fat: per100.fat * factor,
        carbs: per100.carbs * factor,
      };

  return {
    servingGrams,
    per100,
    values: {
      calories: roundKcal(values.calories),
      protein: roundMacro(values.protein),
      fat: roundMacro(values.fat),
      carbs: roundMacro(values.carbs),
    },
    calculationSource: hasFullServing ? "serving" : "per_100g_scaled",
  };
}

async function openFoodFactsByBarcode(barcode) {
  const url = `https://world.openfoodfacts.org/api/v2/product/${encodeURIComponent(barcode)}.json?fields=product_name,product_name_he,brands,quantity,serving_size,serving_quantity,nutriments,image_url,image_front_url,selected_images`;
  const response = await fetch(url, { headers: { Accept: "application/json", "User-Agent": OPEN_FOOD_FACTS_USER_AGENT } });
  if (!response.ok) return null;

  const data = await response.json();
  if (!data || data.status !== 1 || !data.product) return null;

  const product = data.product;
  const nutrition = barcodeNutritionFromProduct(product);
  const hasUsefulValues = nutrition.values.calories > 0 && (nutrition.values.protein > 0 || nutrition.values.fat > 0 || nutrition.values.carbs > 0);
  if (!hasUsefulValues) return null;

  const nameParts = [product.product_name_he, product.product_name, product.brands].filter(Boolean);
  const name = nameParts.length > 0 ? nameParts.join(" - ") : `מוצר ברקוד ${barcode}`;
  const imageUrl = product.image_front_url || product.image_url || product.selected_images?.front?.display?.he || product.selected_images?.front?.display?.en || "";

  const sourceText = nutrition.calculationSource === "serving"
    ? "ערכי serving מתוך Open Food Facts"
    : `ערכי 100 גרם × ${Math.round(nutrition.servingGrams)} גרם מתוך Open Food Facts`;

  return {
    meal_name: name,
    calories: nutrition.values.calories,
    protein: nutrition.values.protein,
    fat: nutrition.values.fat,
    carbs: nutrition.values.carbs,
    confidence: "high",
    notes: `${sourceText}. ל-100 גרם: ${roundKcal(nutrition.per100.calories)} קל׳, ${roundMacro(nutrition.per100.protein)} חלבון, ${roundMacro(nutrition.per100.fat)} שומן, ${roundMacro(nutrition.per100.carbs)} פחמימות. ברקוד: ${barcode}.`,
    image_url: imageUrl,
    items: [
      {
        name,
        quantity: Math.round(nutrition.servingGrams),
        unit: "גרם",
        calories: nutrition.values.calories,
        protein: nutrition.values.protein,
        fat: nutrition.values.fat,
        carbs: nutrition.values.carbs,
        confidence: "high",
        notes: sourceText,
      },
    ],
  };
}

async function analyzeBarcodeWithChatGptFallback(barcode) {
  const response = await createResponse([
    { role: "system", content: [{ type: "input_text", text: systemPrompt() }] },
    { role: "user", content: [{ type: "input_text", text: `לא נמצא מוצר עם ערכים תקינים ב-Open Food Facts לפי הברקוד ${barcode}. אל תנחש מותג ספציפי. אם אינך יודע לזהות את המוצר מהברקוד בלבד, החזר פריט כללי עם confidence low והערה שהברקוד לא נמצא במאגר.` }] },
  ]);
  const result = normalizeMealResult(await parseJsonFromAi(response), `מוצר ברקוד ${barcode}`);
  return { ...result, notes: `${result.notes || ""} | הברקוד לא נמצא במאגר מוצרים עם ערכים תקינים.` };
}

async function analyzeNutritionLabelWithChatGpt({ base64Image, mimeType }) {
  const response = await createResponse([
    {
      role: "system",
      content: [
        {
          type: "input_text",
          text:
            systemPrompt() +
            "\n\nאתה מנתח צילום של תווית תזונתית מאריזת מזון. המשימה היא OCR מדויק לטבלת הערכים, לא הערכה כללית. העתק את המספרים מהתווית כפי שמופיעים. אם קיימת עמודת 'למנה' או 'per serving' — השתמש בה כברירת מחדל. אם קיימת רק עמודת 'ל-100 גרם' — החזר פריט בכמות 100 גרם. אם קיימים גם 100 גרם וגם מנה — כתוב בהערות את ערכי 100 גרם ואת גודל המנה. אין לנחש ערכים אם המספרים לא קריאים; במקרה כזה confidence low והערה מפורשת. החזר JSON בלבד.",
        },
      ],
    },
    {
      role: "user",
      content: [
        {
          type: "input_text",
          text:
            "קרא את טבלת הערכים התזונתיים מהתמונה. צור מוצר אחד בלבד עם הערכים המדויקים מהתווית: קלוריות, חלבון, שומן ופחמימות. השתמש בשם המוצר אם מופיע בתמונה; אחרת קרא לו 'מוצר מתווית תזונתית'.",
        },
        { type: "input_image", image_url: `data:${mimeType};base64,${base64Image}` },
      ],
    },
  ]);
  return normalizeMealResult(await parseJsonFromAi(response), "מוצר מתווית תזונתית");
}

app.post("/analyze-barcode", async (req, res) => {
  try {
    const barcode = String(req.body?.barcode || "").trim();
    if (!barcode) return res.status(400).json({ error: "Barcode is missing" });
    const offResult = await openFoodFactsByBarcode(barcode);
    if (offResult) return res.json(offResult);
    if (!requireApiKey(res)) return;
    const fallback = await analyzeBarcodeWithChatGptFallback(barcode);
    return res.json(fallback);
  } catch (error) {
    console.error("analyze-barcode failed:", error);
    return res.status(500).json({ error: "Failed to analyze barcode", details: error.message });
  }
});

app.post("/analyze-nutrition-label", upload.single("image"), async (req, res) => {
  try {
    if (!requireApiKey(res)) return;
    if (!req.file) return res.status(400).json({ error: "No image uploaded" });
    const result = await analyzeNutritionLabelWithChatGpt({ base64Image: req.file.buffer.toString("base64"), mimeType: detectImageMimeType(req.file.buffer, req.file.originalname) });
    return res.json(result);
  } catch (error) {
    console.error("analyze-nutrition-label failed:", error);
    return res.status(500).json({ error: "Failed to analyze nutrition label", details: error.message });
  }
});

app.get("/", (req, res) => res.json({ status: "ok", service: "nutrition-ai-server", version: "metric-meal-v16-barcode-label-accuracy-image-fix", model: process.env.OPENAI_MODEL || "gpt-5.4-mini", cache_enabled: CACHE_ENABLED, cached_analyses: Object.keys(analysisCache).length, rule: "ChatGPT analyzes every image/text meal. Identical input is cached and reused so repeated analysis of the same meal does not change values.", endpoints: ["/analyze-meal", "/analyze-text-meal", "/analyze-barcode", "/analyze-nutrition-label", "/generate-meal-image", "/clear-cache"] }));
app.post("/clear-cache", (req, res) => { analysisCache = {}; saveCache(); res.json({ status: "ok", cleared: true }); });
app.post("/analyze-meal", upload.single("image"), async (req, res) => { try { if (!requireApiKey(res)) return; if (!req.file) return res.status(400).json({ error: "No image uploaded" }); const result = await analyzeImageWithChatGpt({ base64Image: req.file.buffer.toString("base64"), mimeType: detectImageMimeType(req.file.buffer, req.file.originalname) }); return res.json(result); } catch (error) { console.error("analyze-meal failed:", error); return res.status(500).json({ error: "Failed to analyze meal image with ChatGPT", details: error.message }); } });
app.post("/analyze-text-meal", async (req, res) => { try { if (!requireApiKey(res)) return; const mealName = String(req.body?.meal_name || "ארוחה ידנית"); const items = Array.isArray(req.body?.items) ? req.body.items : []; if (items.length === 0) return res.status(400).json({ error: "No food items provided" }); const result = await analyzeTextMealWithChatGpt({ mealName, items }); return res.json(result); } catch (error) { console.error("analyze-text-meal failed:", error); return res.status(500).json({ error: "Failed to analyze text meal with ChatGPT", details: error.message }); } });
const port = process.env.PORT || 3000;
app.listen(port, () => console.log(`Nutrition AI server v16 is running on port ${port}`));
