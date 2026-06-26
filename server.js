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
app.get("/", (req, res) => res.json({ status: "ok", service: "nutrition-ai-server", version: "metric-meal-v10-stable-chatgpt-cache", model: process.env.OPENAI_MODEL || "gpt-5.4-mini", cache_enabled: CACHE_ENABLED, cached_analyses: Object.keys(analysisCache).length, rule: "ChatGPT analyzes every image/text meal. Identical input is cached and reused so repeated analysis of the same meal does not change values.", endpoints: ["/analyze-meal", "/analyze-text-meal", "/clear-cache"] }));
app.post("/clear-cache", (req, res) => { analysisCache = {}; saveCache(); res.json({ status: "ok", cleared: true }); });
app.post("/analyze-meal", upload.single("image"), async (req, res) => { try { if (!requireApiKey(res)) return; if (!req.file) return res.status(400).json({ error: "No image uploaded" }); const result = await analyzeImageWithChatGpt({ base64Image: req.file.buffer.toString("base64"), mimeType: detectImageMimeType(req.file.buffer, req.file.originalname) }); return res.json(result); } catch (error) { console.error("analyze-meal failed:", error); return res.status(500).json({ error: "Failed to analyze meal image with ChatGPT", details: error.message }); } });
app.post("/analyze-text-meal", async (req, res) => { try { if (!requireApiKey(res)) return; const mealName = String(req.body?.meal_name || "ארוחה ידנית"); const items = Array.isArray(req.body?.items) ? req.body.items : []; if (items.length === 0) return res.status(400).json({ error: "No food items provided" }); const result = await analyzeTextMealWithChatGpt({ mealName, items }); return res.json(result); } catch (error) { console.error("analyze-text-meal failed:", error); return res.status(500).json({ error: "Failed to analyze text meal with ChatGPT", details: error.message }); } });
const port = process.env.PORT || 3000;
app.listen(port, () => console.log(`Nutrition AI server v10 is running on port ${port}`));
