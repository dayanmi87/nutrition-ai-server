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
app.use(express.json({ limit: "3mb" }));

const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const FDC_API_KEY = process.env.FDC_API_KEY || "";
const OPEN_FOOD_FACTS_USER_AGENT =
  process.env.OPEN_FOOD_FACTS_USER_AGENT ||
  "nutrition-app/1.0 (contact: nutrition-app-user)";

function requireApiKey(res) {
  if (!process.env.OPENAI_API_KEY) {
    res.status(500).json({ error: "OPENAI_API_KEY is missing" });
    return false;
  }

  return true;
}

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
    const parsed = Number(value.replace(",", "."));
    if (Number.isFinite(parsed)) return Math.round(parsed);
  }

  return 0;
}

function normalizeQuantity(value) {
  if (typeof value === "number" && Number.isFinite(value)) return value;

  if (typeof value === "string") {
    const parsed = Number(value.replace(",", "."));
    if (Number.isFinite(parsed)) return parsed;
  }

  return 1;
}

function normalizeText(value) {
  return String(value || "")
    .toLowerCase()
    .replace(/[׳']/g, "'")
    .replace(/״/g, '"')
    .replace(/[^\p{L}\p{N}\s."'%-]/gu, " ")
    .replace(/\s+/g, " ")
    .trim();
}

const FOOD_DB = [
  {
    keys: ["שיבולת שועל", "קוואקר", "oats", "oatmeal", "rolled oats"],
    english: "rolled oats",
    per100g: { calories: 389, protein: 16.9, carbs: 66.3, fat: 6.9 },
    unitGrams: { "כף": 8, "כפות": 8, "גרם": 1, "גרמים": 1, "מנה": 40, "cup": 80, "כוס": 80 },
  },
  {
    keys: ["זרעי צ׳יה", "זרעי צ'יה", "צ׳יה", "צ'יה", "chia", "chia seeds"],
    english: "chia seeds",
    per100g: { calories: 486, protein: 16.5, carbs: 42.1, fat: 30.7 },
    unitGrams: { "כפית": 4, "כפיות": 4, "כף": 12, "כפות": 12, "גרם": 1, "גרמים": 1 },
  },
  {
    keys: ["תערובת אגוזים", "אגוזים", "nuts", "mixed nuts"],
    english: "mixed nuts",
    per100g: { calories: 610, protein: 17, carbs: 22, fat: 55 },
    unitGrams: { "כף": 9, "כפות": 9, "חופן": 25, "גרם": 1, "גרמים": 1, "מנה": 30 },
  },
  {
    keys: ["אבקת חלבון", "חלבון", "protein powder", "whey", "whey protein"],
    english: "whey protein powder",
    per100g: { calories: 400, protein: 80, carbs: 6.7, fat: 5 },
    unitGrams: { "סקופ": 30, "מנה": 30, "כף": 10, "כפות": 10, "כפית": 3.3, "כפיות": 3.3, "גרם": 1, "גרמים": 1 },
  },
  {
    keys: ["בננה", "banana"],
    english: "banana raw",
    per100g: { calories: 89, protein: 1.1, carbs: 22.8, fat: 0.3 },
    unitGrams: { "יחידה": 120, "בננה": 120, "גרם": 1, "גרמים": 1, "medium": 120 },
  },
  {
    keys: ["ביצה", "ביצים", "egg", "eggs"],
    english: "egg whole raw",
    perUnit: { calories: 72, protein: 6.3, carbs: 0.4, fat: 5 },
    unitGrams: { "יחידה": 1, "ביצה": 1, "ביצים": 1, "מנה": 1 },
  },
  {
    keys: ["אורז מבושל", "אורז", "rice"],
    english: "rice white cooked",
    per100g: { calories: 130, protein: 2.7, carbs: 28, fat: 0.3 },
    unitGrams: { "גרם": 1, "גרמים": 1, "כף": 15, "כפות": 15, "כוס": 160, "מנה": 180 },
  },
  {
    keys: ["חלב", "milk"],
    english: "milk",
    per100g: { calories: 60, protein: 3.2, carbs: 4.8, fat: 3.3 },
    unitGrams: { "מ״ל": 1.03, "מל": 1.03, "כוס": 240, "גרם": 1, "גרמים": 1 },
  },
  {
    keys: ["יוגורט יווני", "יוגורט", "greek yogurt", "yogurt"],
    english: "greek yogurt plain",
    per100g: { calories: 97, protein: 9, carbs: 3.6, fat: 5 },
    unitGrams: { "גרם": 1, "גרמים": 1, "גביע": 150, "כף": 15, "כפות": 15 },
  },
  {
    keys: ["טונה", "tuna"],
    english: "tuna canned",
    per100g: { calories: 132, protein: 29, carbs: 0, fat: 1 },
    unitGrams: { "גרם": 1, "גרמים": 1, "קופסה": 112, "מנה": 112 },
  },
  {
    keys: ["פיתה", "פיתה לבנה", "pita"],
    english: "pita bread",
    per100g: { calories: 275, protein: 9, carbs: 56, fat: 1.2 },
    unitGrams: { "יחידה": 100, "חצי": 50, "גרם": 1, "גרמים": 1 },
  },
  {
    keys: ["חומוס", "hummus"],
    english: "hummus",
    per100g: { calories: 250, protein: 8, carbs: 20, fat: 16 },
    unitGrams: { "כף": 15, "כפות": 15, "מנה": 100, "גרם": 1, "גרמים": 1 },
  },
  {
    keys: ["טחינה", "tahini"],
    english: "tahini",
    per100g: { calories: 595, protein: 17, carbs: 21, fat: 53 },
    unitGrams: { "כף": 15, "כפות": 15, "כפית": 5, "מנה": 30, "גרם": 1, "גרמים": 1 },
  },
  {
    keys: ["פלאפל", "falafel"],
    english: "falafel",
    perUnit: { calories: 60, protein: 2.5, carbs: 6, fat: 3.5 },
    unitGrams: { "כדור": 1, "כדורים": 1, "יחידה": 1, "מנה": 5 },
  },
  {
    keys: ["קוטג", "קוטג׳", "קוטג'", "cottage"],
    english: "cottage cheese",
    per100g: { calories: 95, protein: 11, carbs: 3, fat: 5 },
    unitGrams: { "גרם": 1, "גרמים": 1, "גביע": 250, "כף": 15, "כפות": 15 },
  },
  {
    keys: ["גבינה צהובה", "yellow cheese", "cheese slice"],
    english: "cheese",
    per100g: { calories: 330, protein: 25, carbs: 2, fat: 25 },
    unitGrams: { "פרוסה": 22, "פרוסות": 22, "גרם": 1, "גרמים": 1 },
  },
  {
    keys: ["חזה עוף", "chicken breast"],
    english: "chicken breast cooked",
    per100g: { calories: 165, protein: 31, carbs: 0, fat: 3.6 },
    unitGrams: { "גרם": 1, "גרמים": 1, "מנה": 150, "יחידה": 150 },
  },
  {
    keys: ["סלט ישראלי", "סלט ירקות", "israeli salad", "vegetable salad"],
    english: "vegetable salad",
    per100g: { calories: 35, protein: 1, carbs: 6, fat: 0.5 },
    unitGrams: { "גרם": 1, "גרמים": 1, "קערה": 250, "מנה": 200, "כף": 15, "כפות": 15 },
  },
  {
    keys: ["קרקרים", "קרקר", "cracker", "crackers"],
    english: "crackers",
    per100g: { calories: 430, protein: 9, carbs: 72, fat: 12 },
    unitGrams: { "יחידה": 8, "קרקר": 8, "קרקרים": 8, "גרם": 1, "גרמים": 1 },
  },

  {
    keys: ["אוראו", "עוגיית אוראו", "oreo", "oreo cookie", "oreo cookies"],
    english: "oreo cookies",
    perUnit: { calories: 53, protein: 0.7, carbs: 8.3, fat: 2.3 },
    unitGrams: { "עוגייה": 1, "עוגיה": 1, "יחידה": 1, "מנה": 3 },
  },
  {
    keys: ["ביסקוויט", "biscuit", "cookie", "cookies"],
    english: "cookies",
    per100g: { calories: 480, protein: 6, carbs: 70, fat: 20 },
    unitGrams: { "עוגייה": 11, "עוגיה": 11, "יחידה": 11, "גרם": 1, "גרמים": 1 },
  },
];

function findFoodRecord(name) {
  const normalized = normalizeText(name);
  return FOOD_DB.find((record) =>
    record.keys.some((key) => normalized.includes(normalizeText(key)))
  );
}

function gramsFromItem(item, record = null) {
  const quantity = normalizeQuantity(item.quantity) || 1;
  const unit = String(item.unit || "מנה").trim();

  if (String(unit).match(/גרם|גרמים|gram|grams|g\b/i)) return quantity;
  if (String(unit).match(/קג|ק״ג|kg|קילו/i)) return quantity * 1000;

  const unitTable = record?.unitGrams || {};
  const matchedUnit = Object.keys(unitTable).find(
    (key) => normalizeText(key) === normalizeText(unit)
  );

  if (matchedUnit) return quantity * unitTable[matchedUnit];

  const text = `${item.name || ""} ${item.notes || ""} ${item.unit || ""}`;
  const gramsMatch = text.match(/(\d+(?:[.,]\d+)?)\s*(גרם|גרמים|g|gram|grams)/i);
  if (gramsMatch) return Number(gramsMatch[1].replace(",", "."));

  return null;
}

function calcKnownFoodItem(item) {
  const record = findFoodRecord(item.name);
  if (!record) return null;

  const quantity = normalizeQuantity(item.quantity) || 1;
  const unit = String(item.unit || "מנה").trim();

  if (record.perUnit) {
    return {
      name: String(item.name || "רכיב מזון"),
      quantity,
      unit,
      calories: Math.round(record.perUnit.calories * quantity),
      protein: Math.round(record.perUnit.protein * quantity),
      fat: Math.round(record.perUnit.fat * quantity),
      carbs: Math.round(record.perUnit.carbs * quantity),
      notes: "חושב לפי מאגר פנימי",
      source: "internal_db",
      confidence: "high",
    };
  }

  const grams = gramsFromItem(item, record) ?? quantity;
  const multiplier = grams / 100;

  return {
    name: String(item.name || "רכיב מזון"),
    quantity,
    unit,
    calories: Math.round(record.per100g.calories * multiplier),
    protein: Math.round(record.per100g.protein * multiplier),
    fat: Math.round(record.per100g.fat * multiplier),
    carbs: Math.round(record.per100g.carbs * multiplier),
    notes: `חושב לפי מאגר פנימי: כ-${Math.round(grams)} גרם`,
    source: "internal_db",
    confidence: "high",
  };
}

function getNutrient(food, nutrientNameCandidates) {
  const nutrients = Array.isArray(food.foodNutrients) ? food.foodNutrients : [];

  for (const candidate of nutrientNameCandidates) {
    const found = nutrients.find((n) => {
      const name = normalizeText(n.nutrientName || n.name || "");
      return name.includes(normalizeText(candidate));
    });

    if (found && Number.isFinite(Number(found.value))) {
      return Number(found.value);
    }
  }

  return 0;
}

async function translateFoodNameToEnglish(name) {
  const record = findFoodRecord(name);
  if (record?.english) return record.english;

  try {
    const response = await client.responses.create({
      model: "gpt-5.4-mini",
      input: [
        {
          role: "user",
          content: [
            {
              type: "input_text",
              text:
                "Translate this food name to a concise English USDA FoodData Central search query. " +
                "Return only the English query, no explanation: " +
                String(name || ""),
            },
          ],
        },
      ],
    });

    return String(response.output_text || name || "").trim().replace(/^["']|["']$/g, "");
  } catch {
    return String(name || "");
  }
}


async function calcOpenFoodFactsItem(item) {
  const originalName = String(item.name || "").trim();
  if (!originalName) return null;

  const query = await translateFoodNameToEnglish(originalName);
  const url = new URL("https://world.openfoodfacts.org/cgi/search.pl");
  url.searchParams.set("search_terms", query);
  url.searchParams.set("search_simple", "1");
  url.searchParams.set("action", "process");
  url.searchParams.set("json", "1");
  url.searchParams.set("page_size", "5");
  url.searchParams.set(
    "fields",
    "product_name,brands,quantity,serving_size,nutriments"
  );

  const response = await fetch(url.toString(), {
    headers: {
      Accept: "application/json",
      "User-Agent": OPEN_FOOD_FACTS_USER_AGENT,
    },
  });

  if (!response.ok) return null;

  const data = await response.json();
  const products = Array.isArray(data.products) ? data.products : [];
  if (products.length === 0) return null;

  const product = products.find((p) => p?.nutriments) || products[0];
  const n = product.nutriments || {};

  const caloriesPer100 = Number(n["energy-kcal_100g"] ?? n["energy-kcal"] ?? 0);
  const proteinPer100 = Number(n["proteins_100g"] ?? 0);
  const fatPer100 = Number(n["fat_100g"] ?? 0);
  const carbsPer100 = Number(n["carbohydrates_100g"] ?? 0);

  if (!caloriesPer100 && !proteinPer100 && !fatPer100 && !carbsPer100) return null;

  let grams = gramsFromItem(item, findFoodRecord(item.name));

  if (!grams && String(item.unit || "").match(/יחידה|עוגייה|עוגיה|cookie/i)) {
    const serving = String(product.serving_size || "");
    const match = serving.match(/(\d+(?:[.,]\d+)?)\s*g/i);
    if (match) grams = Number(match[1].replace(",", "."));
  }

  if (!grams && normalizeText(item.name).includes("oreo")) grams = 11;
  if (!grams && normalizeText(item.name).includes("אוראו")) grams = 11;
  if (!grams) grams = normalizeQuantity(item.quantity);

  const multiplier = grams / 100;

  return {
    name: originalName || product.product_name || "רכיב מזון",
    quantity: normalizeQuantity(item.quantity),
    unit: String(item.unit || "מנה"),
    calories: Math.round(caloriesPer100 * multiplier),
    protein: Math.round(proteinPer100 * multiplier),
    fat: Math.round(fatPer100 * multiplier),
    carbs: Math.round(carbsPer100 * multiplier),
    notes: `Open Food Facts: ${product.product_name || query}; כ-${Math.round(grams)} גרם`,
    source: "open_food_facts",
    confidence: "high",
  };
}

async function calcUsdaFoodItem(item) {
  if (!FDC_API_KEY) return null;

  const query = await translateFoodNameToEnglish(item.name);
  if (!query) return null;

  const url = new URL("https://api.nal.usda.gov/fdc/v1/foods/search");
  url.searchParams.set("api_key", FDC_API_KEY);
  url.searchParams.set("query", query);
  url.searchParams.set("pageSize", "5");
  url.searchParams.set("dataType", "Foundation,SR Legacy,Survey (FNDDS),Branded");

  const response = await fetch(url.toString(), {
    headers: { Accept: "application/json" },
  });

  if (!response.ok) return null;

  const data = await response.json();
  const foods = Array.isArray(data.foods) ? data.foods : [];
  if (foods.length === 0) return null;

  const food =
    foods.find((f) => String(f.dataType || "").toLowerCase().includes("foundation")) ||
    foods.find((f) => String(f.dataType || "").toLowerCase().includes("sr legacy")) ||
    foods[0];

  const caloriesPer100 = getNutrient(food, ["energy"]);
  const proteinPer100 = getNutrient(food, ["protein"]);
  const fatPer100 = getNutrient(food, ["total lipid", "total fat", "fat"]);
  const carbsPer100 = getNutrient(food, ["carbohydrate"]);

  if (!caloriesPer100 && !proteinPer100 && !fatPer100 && !carbsPer100) return null;

  const grams = gramsFromItem(item, findFoodRecord(item.name)) ?? normalizeQuantity(item.quantity);
  const multiplier = grams / 100;

  return {
    name: String(item.name || food.description || "רכיב מזון"),
    quantity: normalizeQuantity(item.quantity),
    unit: String(item.unit || "גרם"),
    calories: Math.round(caloriesPer100 * multiplier),
    protein: Math.round(proteinPer100 * multiplier),
    fat: Math.round(fatPer100 * multiplier),
    carbs: Math.round(carbsPer100 * multiplier),
    notes: `USDA: ${food.description || query}; כ-${Math.round(grams)} גרם`,
    source: "usda",
    confidence: "high",
  };
}

function sanityCheckItem(item) {
  const name = normalizeText(item.name);

  if ((name.includes("protein") || name.includes("חלבון")) && String(item.unit).includes("סקופ")) {
    if (item.protein < 18) {
      item.protein = 24;
      item.calories = Math.max(item.calories, 120);
      item.notes = `${item.notes || ""} | תוקן בבדיקת סבירות לסקופ חלבון`.trim();
      item.source = `${item.source || "unknown"}+sanity_check`;
      item.confidence = "high";
    }
  }

  if (item.calories < 0) item.calories = 0;
  if (item.protein < 0) item.protein = 0;
  if (item.fat < 0) item.fat = 0;
  if (item.carbs < 0) item.carbs = 0;

  return item;
}

async function verifyAndCalculateItems(items) {
  const verified = [];
  const needsAi = [];

  for (const item of items) {
    const simplified = {
      name: String(item.name || ""),
      quantity: normalizeQuantity(item.quantity),
      unit: String(item.unit || "מנה"),
      calories: toNumber(item.calories),
      protein: toNumber(item.protein),
      fat: toNumber(item.fat),
      carbs: toNumber(item.carbs),
      notes: String(item.notes || ""),
    };

    const internal = calcKnownFoodItem(simplified);
    if (internal) {
      verified.push(sanityCheckItem(internal));
      continue;
    }

    const openFoodFacts = await calcOpenFoodFactsItem(simplified);
    if (openFoodFacts) {
      verified.push(sanityCheckItem(openFoodFacts));
      continue;
    }

    const usda = await calcUsdaFoodItem(simplified);
    if (usda) {
      verified.push(sanityCheckItem(usda));
      continue;
    }

    needsAi.push(simplified);
  }

  if (needsAi.length > 0) {
    const response = await client.responses.create({
      model: "gpt-5.4-mini",
      input: [
        {
          role: "user",
          content: [
            {
              type: "input_text",
              text:
                "You are a strict nutrition calculation engine. Return ONLY valid JSON, no markdown. " +
                "Estimate ONLY these unresolved food items. Keep quantities and units exactly as given. " +
                "Do not underestimate protein powder. Use realistic nutrition values. " +
                "Items JSON: " +
                JSON.stringify(needsAi) +
                "\nUse this exact JSON structure: " +
                '{"meal_name":"string","calories":0,"protein":0,"fat":0,"carbs":0,"confidence":"low|medium|high","notes":"string","items":[{"name":"string","quantity":1,"unit":"string","calories":0,"protein":0,"fat":0,"carbs":0,"notes":"string"}]}',
            },
          ],
        },
      ],
    });

    const parsed = await parseJsonFromAi(response);
    const aiResult = normalizeAiResult(parsed);
    const aiItems = aiResult.items.map((item) =>
      sanityCheckItem({
        ...item,
        source: "ai_estimate",
        confidence: item.confidence || "medium",
      })
    );

    verified.push(...aiItems);
  }

  return verified;
}

function normalizeAiResult(raw) {
  const items = Array.isArray(raw.items)
    ? raw.items.map((item) => ({
        name: String(item.name || "רכיב לא מזוהה"),
        quantity:
          typeof item.quantity === "number"
            ? item.quantity
            : Number(String(item.quantity || "1").replace(",", ".")) || 1,
        unit: String(item.unit || "מנה"),
        calories: toNumber(item.calories),
        protein: toNumber(item.protein),
        fat: toNumber(item.fat),
        carbs: toNumber(item.carbs),
        notes: String(item.notes || ""),
        source: String(item.source || "ai_estimate"),
        confidence: String(item.confidence || raw.confidence || "medium"),
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

function buildMealResultFromItems(mealName, items, notes = "") {
  const totals = items.reduce(
    (sum, item) => {
      sum.calories += toNumber(item.calories);
      sum.protein += toNumber(item.protein);
      sum.fat += toNumber(item.fat);
      sum.carbs += toNumber(item.carbs);
      return sum;
    },
    { calories: 0, protein: 0, fat: 0, carbs: 0 }
  );

  const highCount = items.filter((i) => i.confidence === "high").length;
  const confidence =
    items.length === 0
      ? "low"
      : highCount === items.length
      ? "high"
      : highCount >= Math.ceil(items.length / 2)
      ? "medium"
      : "low";

  return {
    meal_name: mealName || "ארוחה",
    calories: totals.calories,
    protein: totals.protein,
    fat: totals.fat,
    carbs: totals.carbs,
    confidence,
    notes:
      notes ||
      "הנתונים אומתו מול מאגר פנימי, USDA או AI לפי זמינות. מומלץ לדייק כמויות לפי משקל.",
    items,
  };
}

async function parseJsonFromAi(response) {
  const text = response.output_text ?? "";

  try {
    return JSON.parse(text);
  } catch {
    const match = text.match(/\{[\s\S]*\}/);

    if (match) {
      return JSON.parse(match[0]);
    }

    throw new Error(`AI returned invalid JSON: ${text}`);
  }
}

app.get("/", (req, res) => {
  res.json({
    status: "ok",
    service: "nutrition-ai-server",
    version: "metric-meal-v5-off-usda",
    usda_enabled: Boolean(FDC_API_KEY),
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
                "אתה מנוע ניתוח תזונתי כמו בשיחת ChatGPT רגילה. נתח את התמונה בזהירות. החזר JSON תקין בלבד, בלי Markdown. " +
                "זהה את כל רכיבי המנה הנראים בתמונה, כולל תוספות, רטבים, לחם, שמן, אגוזים, גבינות וחלבונים. " +
                "הדגש החשוב ביותר: הערך כמות לכל רכיב ביחידות שימושיות בישראל: גרם, כפות, כפיות, פרוסות, יחידות, סקופ, קערה או מנה. " +
                "אם אינך בטוח בכמות, תן אומדן שמרני אך מציאותי וכתוב בהערות מה הנחת. " +
                "לאחר זיהוי הרכיבים, הערך קלוריות, חלבון, שומן ופחמימות לכל רכיב. " +
                "שים לב במיוחד לאבקת חלבון: סקופ רגיל הוא לרוב כ-30 גרם עם כ-24 גרם חלבון וכ-120 קלוריות. " +
                "התשובה בעברית. הערכים הכוללים חייבים להיות סכום הרכיבים. " +
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
    const normalized = normalizeAiResult(parsed);
    const verifiedItems = await verifyAndCalculateItems(normalized.items);

    return res.json(
      buildMealResultFromItems(
        normalized.meal_name,
        verifiedItems,
        "תמונה נותחה עם AI; הערכים אומתו מול מאגר פנימי/Open Food Facts/USDA/AI לפי זמינות."
      )
    );
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
    const items = Array.isArray(req.body?.items) ? req.body.items : [];

    if (items.length === 0) {
      return res.status(400).json({ error: "No food items provided" });
    }

    const verifiedItems = await verifyAndCalculateItems(items);

    return res.json(
      buildMealResultFromItems(
        mealName,
        verifiedItems,
        "מאכלים מוכרים חושבו לפי מאגר פנימי; מוצרים ארוזים מול Open Food Facts; השאר מול USDA או AI."
      )
    );
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
