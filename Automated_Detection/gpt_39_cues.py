import os, base64, time, pandas as pd, requests
from pathlib import Path
from tqdm import tqdm

# Paths & config
csv_labels_path  = Path("data/rico_filtered_labels.csv")
image_dir        = Path("images")
excel_file_path  = Path("outputs/GPT_features_39_RICO.xlsx")
sheet_name       = "Sheet1"

# Optional controls
SINGLE_IMAGE_PATH = None      # e.g., Path("images/12345.jpg") to process one image only
FIRST_N           = None      # e.g., 500 to limit processing from CSV (None = all)
CHECKPOINT_EVERY  = 20        # write Excel every K rows
MODEL             = os.getenv("OPENAI_MODEL", "gpt-4o")

# API key (required)
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise SystemExit("OPENAI_API_KEY is not set. Export it in your environment.")

# column layout: screen, 1…39, true_label
cols = ["screen"] + [str(i) for i in range(1, 40)] + ["true_label"]

# Prompts
system_prompt = """
You are an expert mobile-UI auditor.

**Task**
• Look at the image you are given.
• Decide, for each of cues c1-c39 (defined below), whether the cue is present (1) or absent (0).
• After thinking, respond with **exactly one line**: 39 comma-separated 0/1 integers, no spaces, no other text.

**Contract**
– Think step-by-step *silently*; do not reveal any reasoning or explanation.
– Output nothing except the single 39-integer line.
– If a cue is only partially visible, count as present.
– If you are not certain, default to 0.
– Keep the order c1…c39 exactly as given.

Cue definitions
c1  text such as “Remove Ads”, “Ad-Free”, “Upgrade to remove ads”
c2  ad-free badge/icon with an “×” overlay
c3  full-screen overlay blocks taps
c4  overlay dims or blurs background
c5  overlay shows video controls
c6  overlay shows countdown / “Skip in …”
c7  overlay has **no** close button
c8  overlay close button is tiny
c9  tiny close icon top-right
c10 tiny close icon in any other corner
c11 text “Ad”, “Advertisement”, or “Sponsored”
c12 store logo / “Install” / price badge / star strip
c13 banner ad ≤ 25 % of screen height
c14 banner blue-triangle ad logo
c15 banner tiny “×” inside
c16 tiny “×” left edge of banner
c17 tiny “×” right edge of banner
c18 ≥ 2 choices with different **background** colours
c19 ≥ 2 choices with different **text** colours
c20 choices differ only by border / outline
c21 preferred option filled/coloured vs grey/outline
c22 preferred ≥ 25 % larger than alternative
c23  ≥ 2 choices where the positive option helps the app (payments, data, tracking, rating, etc.)
c24 decline wording negative/shaming (“No thanks”, “I like ads”)
c25 preferred tagged “Recommended” / “Best Value”
c26 checkbox visible
c27 toggle switch visible
c28 any checkbox/toggle is **checked by default**
c29 checked mentions notifications
c30 checked mentions privacy/data
c31 checked mentions marketing/emails
c32 email **and** password fields present
c33 OAuth buttons (Google / Facebook / Apple)
c34 register / sign-up button
c35 generic login / sign-in button
c36 banner with **product-image strip**
c37 small “AdChoices” text label
c38 full-screen ad overlay with no border
c39 overlay covers Android navigation bar
"""

prompt_39 = """
STEP 1 – Inspect the screenshot
Carefully inspect the attached screenshot for all 39 cues.

STEP 2 – Decide 0/1 for each cue
Remember: 1 = present, 0 = absent, 0 if unsure.

STEP 3 – Internal check
Silently review all 39 decisions once more to catch mistakes.

STEP 4 – Output
Respond with **one line only**:
c1,c2,…,c39   (39 comma-separated integers, no spaces, no labels)
"""

# Helpers
def encode_image(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("utf-8")

def check_internet() -> bool:
    try:
        requests.get("https://www.google.com", timeout=5)
        return True
    except Exception:
        return False

def call_gpt(b64: str) -> str:
    payload = {
        "model": MODEL,
        "temperature": 0.3,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": [
                {"type": "text",      "text": prompt_39},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/jpeg;base64,{b64}",
                    "detail": "high"
                }}
            ]}
        ]
    }
    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
        json=payload, timeout=120
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()

def parse_vector(resp: str):
    for ln in resp.splitlines():
        if ln.count(",") >= 38:
            vec = [p.strip() for p in ln.split(",")]
            return (vec + [""] * 39)[:39]
    return [""] * 39

# Load / init results file
if excel_file_path.exists():
    df = pd.read_excel(excel_file_path, sheet_name=sheet_name)
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols]
else:
    excel_file_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(columns=cols)
done = set(df["screen"].astype(str))

# Build job list
jobs = []

if SINGLE_IMAGE_PATH:
    sid = Path(SINGLE_IMAGE_PATH).stem
    if sid not in done:
        jobs.append((sid, str(SINGLE_IMAGE_PATH), "external"))
else:
    labels_df = pd.read_csv(csv_labels_path)
    for _, row in labels_df.iterrows():
        sid        = str(row["screen"]).strip()
        true_label = str(row.get("true_label", ""))
        img_path   = image_dir / f"{sid}.jpg"
        if img_path.is_file() and sid not in done:
            jobs.append((sid, str(img_path), true_label))
            if FIRST_N is not None and len(jobs) >= FIRST_N:
                break

print("Will process", len(jobs), "screens.")

# GPT loop
count = 0
for sid, path, lbl in tqdm(jobs, desc=MODEL):
    b64 = encode_image(path)
    vector = [""] * 39

    for attempt in range(3):
        if not check_internet():
            time.sleep(30)
            continue
        try:
            resp = call_gpt(b64)
            vector = parse_vector(resp)
            break
        except Exception as e:
            if attempt < 2:
                time.sleep(30)
            else:
                print(f"{sid} failed:", e)

    df.loc[len(df)] = [sid] + vector + [lbl]
    done.add(sid)
    count += 1

    if count % CHECKPOINT_EVERY == 0:
        excel_file_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(excel_file_path, sheet_name=sheet_name, index=False)
        print("Checkpoint:", len(df), "rows")

# Final save
df["screen"] = pd.to_numeric(df["screen"], errors="ignore")
excel_file_path.parent.mkdir(parents=True, exist_ok=True)
df.to_excel(excel_file_path, sheet_name=sheet_name, index=False)
print("Finished –", len(df), "rows saved to", excel_file_path)
