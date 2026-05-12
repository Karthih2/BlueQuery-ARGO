from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, text
import os, json, httpx
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="BlueQuery API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_ui():
    return FileResponse("static/index.html")

RDS_ENDPOINT = os.getenv("DB_HOST")
USERNAME      = os.getenv("DB_USER")
PASSWORD      = os.getenv("DB_PASS")
DB_NAME       = os.getenv("DB_NAME")

engine = create_engine(
    f"postgresql://{USERNAME}:{PASSWORD}@{RDS_ENDPOINT}/{DB_NAME}",
    pool_pre_ping=True, pool_size=5, max_overflow=10
)

ANTHROPIC_KEY = os.getenv("ANTHROPIC_KEY", "")
GEMINI_KEY    = os.getenv("GEMINI_KEY", "")
GROK_KEY      = os.getenv("GROK_KEY", "")

SCHEMA_DESC = """
PostgreSQL table: argo_profiles
Columns:
  data_center  TEXT        -- 'incois', 'coriolis', 'csiro'
  float_id     TEXT        -- e.g. '1902669'
  cycle_number INTEGER
  timestamp    TIMESTAMP
  latitude     DOUBLE PRECISION   -- -68.55 to 25.13
  longitude    DOUBLE PRECISION   -- 33.13 to 119.57
  depth        DOUBLE PRECISION   -- 0 to 4041.7 meters
  temperature  DOUBLE PRECISION   -- -2.13 to 32.77 Celsius
  salinity     DOUBLE PRECISION   -- 0 to 42 PSU
Total rows: ~5,633,032
"""

def save_chat(user_message, sql, answer, llm_used):
    from datetime import datetime
    entry = {"time": datetime.now().isoformat(), "user": user_message, "sql": sql, "answer": answer, "llm": llm_used}
    try:
        os.makedirs("logs", exist_ok=True)
        with open("logs/chat_history.json", "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"Chat log error: {e}")

async def call_llm(prompt: str, max_tokens: int = 400):
    # 1. ANTHROPIC
    if ANTHROPIC_KEY:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={"x-api-key": ANTHROPIC_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                    json={"model": "claude-3-5-haiku-20241022", "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]}
                )
                if r.status_code == 200:
                    return r.json()["content"][0]["text"].strip(), "Claude"
                print(f"Anthropic {r.status_code}: {r.text[:200]}")
        except Exception as e:
            print(f"Anthropic failed: {e}")

    # 2. GEMINI
    if GEMINI_KEY:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_KEY}",
                    json={"contents": [{"parts": [{"text": prompt}]}]}
                )
                if r.status_code == 200:
                    return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip(), "Gemini"
                print(f"Gemini {r.status_code}: {r.text[:200]}")
        except Exception as e:
            print(f"Gemini failed: {e}")

    # 3. GROQ
    if GROK_KEY:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {GROK_KEY}", "Content-Type": "application/json"},
                    json={"model": "llama-3.3-70b-versatile", "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]}
                )
                if r.status_code == 200:
                    return r.json()["choices"][0]["message"]["content"].strip(), "Groq"
                print(f"Groq {r.status_code}: {r.text[:200]}")
        except Exception as e:
            print(f"Groq failed: {e}")

    raise HTTPException(500, "All LLMs failed — check your API keys in .env")


@app.get("/floats")
def get_floats():
    q = text("""SELECT DISTINCT ON (float_id) float_id, data_center,
               ROUND(latitude::numeric,4) AS latitude, ROUND(longitude::numeric,4) AS longitude, timestamp
        FROM argo_profiles ORDER BY float_id, timestamp DESC LIMIT 717""")
    with engine.connect() as conn:
        return [dict(r) for r in conn.execute(q).mappings()]

@app.get("/floats/summary")
def get_floats_summary():
    q = text("SELECT DISTINCT float_id, data_center FROM argo_profiles ORDER BY data_center, float_id LIMIT 800")
    with engine.connect() as conn:
        return [dict(r) for r in conn.execute(q).mappings()]

@app.get("/profile")
def get_profile(float_id: str, cycle: int = 1):
    q = text("""SELECT ROUND(depth::numeric,1) AS depth, ROUND(temperature::numeric,3) AS temperature,
               ROUND(salinity::numeric,3) AS salinity FROM argo_profiles
        WHERE float_id=:fid AND cycle_number=:cycle ORDER BY depth ASC""")
    with engine.connect() as conn:
        return [dict(r) for r in conn.execute(q, {"fid": float_id, "cycle": cycle}).mappings()]

@app.get("/cycles")
def get_cycles(float_id: str):
    q = text("SELECT DISTINCT cycle_number FROM argo_profiles WHERE float_id=:fid ORDER BY cycle_number")
    with engine.connect() as conn:
        return [r["cycle_number"] for r in conn.execute(q, {"fid": float_id}).mappings()]

@app.get("/region")
def get_region(lat_min: float, lat_max: float, lon_min: float, lon_max: float, year: int = None):
    params = {"lat_min": lat_min, "lat_max": lat_max, "lon_min": lon_min, "lon_max": lon_max}
    year_clause = ""
    if year:
        year_clause = "AND EXTRACT(YEAR FROM timestamp) = :year"
        params["year"] = year
    q = text(f"""SELECT ROUND(AVG(temperature)::numeric,2) AS avg_temp, ROUND(AVG(salinity)::numeric,2) AS avg_sal,
               ROUND(MIN(temperature)::numeric,2) AS min_temp, ROUND(MAX(temperature)::numeric,2) AS max_temp,
               ROUND(STDDEV(temperature)::numeric,3) AS std_temp, COUNT(DISTINCT float_id) AS float_count, COUNT(*) AS total_obs
        FROM argo_profiles WHERE latitude BETWEEN :lat_min AND :lat_max AND longitude BETWEEN :lon_min AND :lon_max {year_clause}""")
    with engine.connect() as conn:
        return dict(conn.execute(q, params).mappings().first())

@app.get("/overview")
def get_overview(data_center: str = None, depth_max: float = 10):
    cond = "AND data_center = :dc" if data_center else ""
    params: dict = {"depth_max": depth_max}
    if data_center:
        params["dc"] = data_center
    q = text(f"""SELECT data_center, ROUND(AVG(temperature)::numeric,2) AS avg_temp,
               ROUND(AVG(salinity)::numeric,2) AS avg_sal, ROUND(MIN(temperature)::numeric,2) AS min_temp,
               ROUND(MAX(temperature)::numeric,2) AS max_temp, COUNT(DISTINCT float_id) AS float_count, COUNT(*) AS total_obs
        FROM argo_profiles WHERE depth <= :depth_max {cond} GROUP BY data_center ORDER BY data_center""")
    with engine.connect() as conn:
        return [dict(r) for r in conn.execute(q, params).mappings()]

@app.get("/center-profiles")
def get_center_profiles(data_center: str, depth_max: float = 500):
    q = text("""SELECT ROUND(depth::numeric/50)*50 AS depth_band,
               ROUND(AVG(temperature)::numeric,2) AS avg_temp, ROUND(AVG(salinity)::numeric,2) AS avg_sal,
               COUNT(DISTINCT float_id) AS float_count
        FROM argo_profiles WHERE data_center=:dc AND depth<=:dmax
        GROUP BY ROUND(depth::numeric/50)*50 ORDER BY depth_band""")
    with engine.connect() as conn:
        return [dict(r) for r in conn.execute(q, {"dc": data_center, "dmax": depth_max}).mappings()]

@app.get("/timeseries")
def get_timeseries(lat_min: float=6, lat_max: float=22, lon_min: float=80, lon_max: float=100):
    q = text("""SELECT TO_CHAR(DATE_TRUNC('month',timestamp),'YYYY-MM') AS month,
               ROUND(AVG(temperature)::numeric,2) AS avg_temp, ROUND(AVG(salinity)::numeric,2) AS avg_sal, COUNT(*) AS obs
        FROM argo_profiles WHERE latitude BETWEEN :lat_min AND :lat_max AND longitude BETWEEN :lon_min AND :lon_max AND depth<10
        GROUP BY DATE_TRUNC('month',timestamp) ORDER BY DATE_TRUNC('month',timestamp)""")
    with engine.connect() as conn:
        return [dict(r) for r in conn.execute(q, {"lat_min":lat_min,"lat_max":lat_max,"lon_min":lon_min,"lon_max":lon_max}).mappings()]

@app.get("/rows")
def get_rows(float_id: str=None, center: str=None, limit: int=50, offset: int=0):
    conditions, params = [], {"limit": limit, "offset": offset}
    if float_id:
        conditions.append("float_id=:float_id"); params["float_id"] = float_id
    if center:
        conditions.append("data_center=:center"); params["center"] = center
    where = ("WHERE "+" AND ".join(conditions)) if conditions else ""
    q = text(f"""SELECT data_center, float_id, cycle_number, timestamp,
               ROUND(latitude::numeric,4) AS latitude, ROUND(longitude::numeric,4) AS longitude,
               ROUND(depth::numeric,1) AS depth, ROUND(temperature::numeric,3) AS temperature,
               ROUND(salinity::numeric,3) AS salinity
        FROM argo_profiles {where} ORDER BY timestamp DESC LIMIT :limit OFFSET :offset""")
    with engine.connect() as conn:
        return [dict(r) for r in conn.execute(q, params).mappings()]

@app.get("/shap")
def get_shap(float_id: str="1902669", cycle: int=1):
    q = text("""SELECT depth, latitude, longitude, salinity, temperature FROM argo_profiles
        WHERE float_id=:fid AND cycle_number=:cycle ORDER BY depth ASC LIMIT 500""")
    with engine.connect() as conn:
        rows = [dict(r) for r in conn.execute(q, {"fid": float_id, "cycle": cycle}).mappings()]
    if len(rows) < 5:
        return {"error": "Not enough data", "shap_values": []}
    import numpy as np
    X = np.array([[r["depth"],r["latitude"],r["longitude"],r["salinity"]] for r in rows])
    y = np.array([r["temperature"] for r in rows])
    X_mean = X.mean(axis=0); X_std = X.std(axis=0)+1e-8
    X_norm = (X-X_mean)/X_std; y_mean = y.mean()
    try:
        coeffs,_,_,_ = np.linalg.lstsq(np.column_stack([X_norm, np.ones(len(X_norm))]), y, rcond=None)
    except:
        coeffs = np.zeros(5)
    features = ["Depth","Latitude","Longitude","Salinity"]
    shap_vals = []
    for i, feat in enumerate(features):
        contrib = float(coeffs[i]*X_norm[:,i].mean())
        shap_vals.append({"feature":feat,"value":round(contrib,4),"mean_feature_val":round(float(X_mean[i]),3),"direction":"positive" if contrib>0 else "negative"})
    shap_vals.sort(key=lambda x: abs(x["value"]), reverse=True)
    return {"float_id":float_id,"cycle":cycle,"target":"temperature","shap_values":shap_vals,
            "model_type":"Linear Regression (SHAP approx)","n_samples":len(rows),"baseline_temp":round(float(y_mean),3)}

def _compute_ohc(depths, temps):
    rho, cp = 1025, 3985
    ohc = 0.0
    for i in range(1, len(depths)):
        dz = depths[i]-depths[i-1]
        tavg = (temps[i]+temps[i-1])/2
        ohc += rho*cp*tavg*dz
    return round(ohc/1e6, 2)

def _layer_ohc(depths, temps, d_min, d_max):
    pairs = [(d,t) for d,t in zip(depths,temps) if d_min<=d<=d_max]
    if len(pairs)<2: return 0.0
    return _compute_ohc([p[0] for p in pairs], [p[1] for p in pairs])

@app.get("/ohc")
def get_ohc(float_id: str="1902669", cycle: int=1, depth_max: float=700):
    q = text("""SELECT ROUND(depth::numeric,1) AS depth, ROUND(temperature::numeric,3) AS temperature,
               ROUND(salinity::numeric,3) AS salinity
        FROM argo_profiles WHERE float_id=:fid AND cycle_number=:cycle AND depth<=:dmax ORDER BY depth ASC""")
    with engine.connect() as conn:
        rows = [dict(r) for r in conn.execute(q, {"fid":float_id,"cycle":cycle,"dmax":depth_max}).mappings()]
    if len(rows)<2:
        return {"error":"Not enough data"}
    depths = [float(r["depth"]) for r in rows]
    temps  = [float(r["temperature"]) for r in rows]
    sals   = [float(r["salinity"]) for r in rows]
    ohc_total = _compute_ohc(depths, temps)
    tc_depth, max_grad = None, 0
    for i in range(1, len(depths)):
        dz = depths[i]-depths[i-1]
        if dz==0: continue
        grad = abs((temps[i]-temps[i-1])/dz)
        if grad>max_grad:
            max_grad = grad; tc_depth = depths[i]
    return {
        "float_id": float_id, "cycle": cycle, "depth_max": depth_max, "n_levels": len(rows),
        "ohc_total":   ohc_total,
        "ohc_surface": _layer_ohc(depths,temps,0,100),
        "ohc_mixed":   _layer_ohc(depths,temps,100,300),
        "ohc_deep":    _layer_ohc(depths,temps,300,depth_max),
        "avg_temp": round(sum(temps)/len(temps),2),
        "avg_sal":  round(sum(sals)/len(sals),2),
        "thermocline_depth": round(tc_depth,1) if tc_depth else None,
        "unit": "MJ/m²",
        "profile": [{"depth":d,"temperature":t,"salinity":s} for d,t,s in zip(depths,temps,sals)]
    }

@app.get("/query")
async def ai_query(q: str):
    sql_prompt = f"""You are a PostgreSQL expert for ocean data.
{SCHEMA_DESC}
User question: {q}

Rules:
- Write ONE valid SQL SELECT query only. No markdown, no backticks, no explanation. Raw SQL only.
- Always add LIMIT 100 unless the user asks for aggregates (AVG, COUNT, SUM, MIN, MAX).
- CRITICAL: All columns are DOUBLE PRECISION. PostgreSQL cannot ROUND(double precision, n) directly.
  Always cast to numeric first: ROUND(AVG(temperature)::numeric, 2) not ROUND(AVG(temperature), 2).
  Apply this to every ROUND() call without exception.
- For salinity queries, always cast: ROUND(AVG(salinity)::numeric, 2)
- For temperature queries, always cast: ROUND(AVG(temperature)::numeric, 2)
- For depth queries, always cast: ROUND(depth::numeric, 1)
"""
    sql_raw, llm_used = await call_llm(sql_prompt, max_tokens=400)
    sql = sql_raw.replace("```sql","").replace("```","").strip()
    # Safety net: auto-fix any ROUND(col, n) without ::numeric cast
    import re
    def fix_round(m):
        inner = m.group(1)
        if "::numeric" not in inner:
            inner = inner + "::numeric"
        return f"ROUND({inner}, {m.group(2)})"
    sql = re.sub(r"ROUND\(([^,]+),\s*(\d+)\)", fix_round, sql)

    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            rows = [dict(r) for r in result.mappings()]
            row_count = len(rows)
            data_preview = str(rows[:5])
    except Exception as e:
        return {"sql":sql,"rows":[],"row_count":0,"answer":f"SQL error: {str(e)}","llm":llm_used,"error":True,"chart_type":None}

    # Chart intent detection
    ql = q.lower()
    chart_type = None
    if any(w in ql for w in ["chart","plot","graph","visualize","show me","trend","over time","monthly","time series"]):
        cols = list(rows[0].keys()) if rows else []
        has_time = any(c for c in cols if any(k in c.lower() for k in ["month","date","year","time"]))
        has_num  = any(c for c in cols if any(k in c.lower() for k in ["temp","sal","depth","count","avg","sum","obs"]))
        chart_type = "line" if has_time else ("bar" if has_num and len(rows)>=2 else None)

    explain_prompt = f"""You are BlueQuery, an ocean data AI assistant.
User asked: "{q}"
SQL returned {row_count} rows. First 5 rows: {data_preview}
Write 2-3 sentences of scientific explanation. Be specific with numbers. No bullets, no markdown."""
    explanation, _ = await call_llm(explain_prompt, max_tokens=300)
    save_chat(q, sql, explanation, llm_used)
    return {"sql":sql,"rows":rows[:100],"row_count":row_count,"answer":explanation,
            "llm":llm_used,"columns":list(rows[0].keys()) if rows else [],"chart_type":chart_type}

@app.get("/llm-status")
async def llm_status():
    results = {}
    checks = [
        (ANTHROPIC_KEY, "anthropic",
         "https://api.anthropic.com/v1/messages",
         {"x-api-key": ANTHROPIC_KEY, "anthropic-version":"2023-06-01","content-type":"application/json"},
         {"model":"claude-3-5-haiku-20241022","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}),
        (GEMINI_KEY, "gemini",
         f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_KEY}",
         {}, {"contents":[{"parts":[{"text":"hi"}]}]}),
        (GROK_KEY, "grok",
         "https://api.groq.com/openai/v1/chat/completions",
         {"Authorization":f"Bearer {GROK_KEY}","Content-Type":"application/json"},
         {"model":"llama-3.3-70b-versatile","max_tokens":10,"messages":[{"role":"user","content":"hi"}]})
    ]
    for key, name, url, headers, body in checks:
        if not key:
            results[name] = "⚠️ No key in .env"
            continue
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.post(url, headers=headers, json=body)
                results[name] = "✅ Working" if r.status_code==200 else f"❌ {r.status_code}: {r.text[:80]}"
        except Exception as e:
            results[name] = f"❌ {str(e)[:80]}"
    return results

@app.get("/health")
def health():
    try:
        with engine.connect() as conn:
            count = conn.execute(text("SELECT COUNT(*) FROM argo_profiles")).scalar()
        return {"status":"ok","db_rows":count}
    except Exception as e:
        return {"status":"error","detail":str(e)}