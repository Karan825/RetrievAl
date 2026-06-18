import json
from pathlib import Path
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
from docx import Document
import numpy as np
import sys
import os

# Resolve paths relative to this script directory
VRD_DIR = Path(__file__).resolve().parent

# ==========================================
# 1. INITIALIZE MODELS
# ==========================================
def load_model():
    # Automatically download models if they don't exist
    try:
        sys.path.append(str(VRD_DIR))
        from download_model import ensure_models_exist
        ensure_models_exist()
    except Exception as e:
        print(f"Warning: Could not check/download models: {e}")

    print("Loading SLM (Qwen2.5 1.5B)...")
    model_path = VRD_DIR / "qwen2.5-1.5b-instruct-q4_k_m.gguf"
    llm = Llama(
        model_path=str(model_path),
        n_ctx=8192,
        n_threads=4,
        verbose=False
    )
    print("Loading Embedding Model (bge-small)...")
    embedder_path = VRD_DIR / "local_bge_model"
    embedder = SentenceTransformer(str(embedder_path))
    return llm, embedder


def safe_llm_call(llm, prompt, max_tokens=1500, stop="<|im_end|>", temperature=0.1, min_tokens=16):
    """Call llama_cpp Llama safely, retrying with smaller max_tokens if the requested tokens exceed the context window."""
    attempt_max = max_tokens
    for i in range(5):
        try:
            return llm(prompt, max_tokens=attempt_max, stop=stop, temperature=temperature)
        except ValueError as e:
            msg = str(e)
            if "Requested tokens" in msg and "exceed context window" in msg:
                attempt_max = max(min_tokens, attempt_max // 2)
                print(f"LLM call failed due to context window; reducing max_tokens to {attempt_max} and retrying...")
                continue
            raise

    short_prompt = prompt[:1500]
    print("LLM call still failing — retrying with truncated prompt and minimal tokens.")
    return llm(short_prompt, max_tokens=min_tokens, stop=stop, temperature=temperature)


# ==========================================
# 2. PARSE THE JD (100% Generalized)
# ==========================================
def extract_single_rule(llm, item_text):
    prompt = f"""<|im_start|>system
You are a technical recruiter. Your task is to analyze a disqualifying background/trait/negative pattern and extract structured verification fields in JSON.
Output JSON with exactly these keys:
1. "description": The input text.
2. "trigger_skills": A list of specific skill/technology names (lowercase, e.g. ["computer vision", "yolo", "speech recognition", "tts", "asr", "robotics"]) that would trigger this disqualifier. Set to [] if skills are not relevant.
3. "exception_skills": A list of specific skill/technology names (lowercase, e.g. ["nlp", "retrieval", "search", "rag", "embeddings", "llm", "transformers"]) that override/nullify the trigger. If a candidate has any of these exception skills, the disqualifier MUST NOT be triggered. Set to [] if no exception applies.
4. "company_keywords": A list of lowercase substrings (e.g. ["consulting", "tcs", "wipro", "infosys", "accenture", "cognizant", "capgemini", "services", "agency"]) to check in candidate company names. Set to [] if not checking consulting/agency background.

Example 1:
Input: "People who have only worked at consulting firms (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, etc.)"
Output: {{
  "description": "People who have only worked at consulting firms (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini, etc.)",
  "trigger_skills": [],
  "exception_skills": [],
  "company_keywords": ["consulting", "tcs", "wipro", "infosys", "accenture", "cognizant", "capgemini", "services", "agency"]
}}

Example 2:
Input: "People whose primary expertise is computer vision, speech, or robotics without significant NLP/IR exposure"
Output: {{
  "description": "People whose primary expertise is computer vision, speech, or robotics without significant NLP/IR exposure",
  "trigger_skills": ["computer vision", "yolo", "speech recognition", "tts", "asr", "robotics", "cv", "image classification", "object detection", "speech to text", "text to speech"],
  "exception_skills": ["nlp", "retrieval", "search", "rag", "embeddings", "llm", "transformers", "natural language processing", "information retrieval"],
  "company_keywords": []
}}

Output ONLY valid JSON. No explanation or markdown.<|im_end|>
<|im_start|>user
Input: "{item_text}"<|im_end|>
<|im_start|>assistant
"""
    response = safe_llm_call(llm, prompt, max_tokens=600, stop="<|im_end|>", temperature=0.1)
    try:
        raw_text = response['choices'][0]['text'].strip()
        cleaned_text = raw_text.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned_text)
    except Exception as e:
        print(f"Failed to parse single rule for: {item_text}", e)
        return {
            "description": item_text,
            "trigger_skills": [],
            "exception_skills": [],
            "company_keywords": []
        }


def parse_jd(llm, jd_text):
    print("\n[Step 1] Extracting Implicit and Explicit Intents from JD...")
    prompt = f"""<|im_start|>system
You are an expert recruiter analyzing a complex Job Description.
Extract the hiring intents into a strict JSON format with exactly these keys:
1. "company": The name of the company hiring for this role (e.g., 'Redrob').
2. "job_title": The exact role being hired for (e.g., "Senior AI Engineer", "Marketing Manager", "Data Scientist").
3. "must_have_hard_skills": Core technical or functional requirements as full phrases.
4. "execution_and_impact_signals": Signals of real-world delivery, scale, or impact.
5. "culture_and_behavior": Preferred working styles and mindset.
6. "abstract_disqualifiers": Backgrounds, traits, or firm types they explicitly reject.
7. "negative_tools_or_patterns": Tools, methodologies, or approaches they view as negative.
8. "metadata_constraints": A dictionary with keys "min_yoe", "max_yoe", "preferred_company_type", "preferred_locations".
9. "domain_keywords": 5-8 short keywords (1-3 words each) defining the correct professional domain for this role.
   Example: ["embeddings", "retrieval", "NLP", "semantic search", "ranking", "vector database"].
   These must reflect what a GOOD candidate should know — not what is disqualifying.
10. "must_have_skills_short": A flat list of up to 10 short skill/technology names extracted from must_have_hard_skills.
    These should be concise identifiers (e.g., ["FAISS", "Pinecone", "sentence-transformers", "Python", "NDCG"]).
11. "seniority_target": One word indicating the seniority level of the role.
    Must be one of: "junior", "mid", "senior", "staff", "lead", "principal", "director".
12. "title_family_keywords": A list of 3-5 lowercase single-word nouns or roots (e.g., ["engineer", "developer", "scientist", "programmer", "specialist"]) that define acceptable job title variations for this role.
13. "unacceptable_title_keywords": A list of 5-10 lowercase single-word roots (e.g., ["civil", "mechanical", "chemical", "hr", "recruiter", "writer", "sales", "operations", "designer", "finance", "accounting"]) that are NOT acceptable for this role.
Output ONLY valid JSON. No explanation or markdown.<|im_end|>
<|im_start|>user
Job Description:
{jd_text}<|im_end|>
<|im_start|>assistant
"""

    response = safe_llm_call(llm, prompt, max_tokens=1500, stop="<|im_end|>", temperature=0.1)

    try:
        raw_text = response['choices'][0]['text'].strip()
        cleaned_text = raw_text.replace("```json", "").replace("```", "").strip()
        result = json.loads(cleaned_text)

        # Ensure required keys always exist
        if "company" not in result:
            result["company"] = "Redrob"
        if "job_title" not in result:
            result["job_title"] = "Professional"
        if "domain_keywords" not in result:
            result["domain_keywords"] = []
        if "must_have_skills_short" not in result:
            result["must_have_skills_short"] = [
                phrase.split()[0] for phrase in result.get("must_have_hard_skills", [])[:8]
            ]
        if "seniority_target" not in result:
            result["seniority_target"] = "senior"
        if "title_family_keywords" not in result:
            result["title_family_keywords"] = []
        if "unacceptable_title_keywords" not in result:
            result["unacceptable_title_keywords"] = []

        # Now extract the disqualifier rules for each item
        disq_items = result.get("abstract_disqualifiers", []) + result.get("negative_tools_or_patterns", [])
        rules = []
        for item in disq_items:
            if item and item.strip():
                print(f"Extracting structured rules for: {item.strip()}")
                rule = extract_single_rule(llm, item.strip())
                rules.append(rule)
        result["disqualifier_rules"] = rules

        return result
    except Exception as e:
        print("Failed to parse JSON, check SLM output.", e)
        return {
            "company": "Redrob",
            "job_title": "Professional",
            "must_have_hard_skills": [],
            "execution_and_impact_signals": [],
            "culture_and_behavior": [],
            "abstract_disqualifiers": [],
            "negative_tools_or_patterns": [],
            "metadata_constraints": {},
            "domain_keywords": [],
            "must_have_skills_short": [],
            "seniority_target": "senior",
            "title_family_keywords": [],
            "unacceptable_title_keywords": [],
            "disqualifier_rules": [],
        }


# ==========================================
# 3. EXPAND DISQUALIFIERS (Dynamic Context & Few-Shot)
# ==========================================
def expand_disqualifiers(llm, disqualifiers, job_title):
    print(f"\n[Step 2] Expanding Abstract Disqualifiers for a {job_title}...")
    if not disqualifiers:
        return ""
    expanded_text = ""

    for disq in disqualifiers:
        prompt = f"""<|im_start|>system
You are a technical recruiter hiring a {job_title}. 
Your task is to generate BAD resume bullet points that indicate a candidate has a specific NEGATIVE trait. 
Here are examples of how you must answer regardless of industry:
Trait: "Consulting-only background" -> "Spent entire career at an agency", "Only worked on client-facing consulting projects"
Trait: "Over-reliant on basic tools" -> "Only used drag-and-drop tools", "No fundamental understanding of the underlying systems"
Trait: "Job hoppers" -> "Switched companies every 8 months", "Unjustified promotions after 1 year"

Now, generate 5 concrete, short examples of this NEGATIVE trait specifically for a {job_title}: "{disq}". 
Just list the examples separated by commas. No explanations.<|im_end|>
<|im_start|>user
Trait: "{disq}"<|im_end|>
<|im_start|>assistant
"""
        response = safe_llm_call(llm, prompt, max_tokens=200, stop="<|im_end|>", temperature=0.1)
        expanded_text += response['choices'][0]['text'].strip() + ", "

    return expanded_text.strip(", ")


# ==========================================
# 4. GENERATE MULTI-VECTORS
# ==========================================
def create_embeddings(embedder, jd_dict, expanded_neg_text):
    print("\n[Step 3] Generating Intent Embeddings...")

    core_text = ". ".join(
        jd_dict.get("must_have_hard_skills", []) + jd_dict.get("execution_and_impact_signals", [])
    )
    v_core = embedder.encode([core_text], normalize_embeddings=True)

    culture_text = ". ".join(jd_dict.get("culture_and_behavior", []))
    v_culture = embedder.encode([culture_text], normalize_embeddings=True)

    negative_frameworks = ". ".join(jd_dict.get("negative_tools_or_patterns", []))
    # Combine abstract disqualifiers, expanded examples, and negative tool patterns
    full_negative = (
        ". ".join(jd_dict.get("abstract_disqualifiers", []))
        + " " + expanded_neg_text
        + " " + negative_frameworks
    ).strip()
    v_neg = embedder.encode([full_negative], normalize_embeddings=True)

    # [NEW] Skills vector: embed the short skill identifiers for fast candidate skill matching
    skills_short = jd_dict.get("must_have_skills_short", [])
    domain_keywords = jd_dict.get("domain_keywords", [])
    combined_skills = skills_short + domain_keywords
    if combined_skills:
        skills_text = " ".join(combined_skills)
        v_skills = embedder.encode([skills_text], normalize_embeddings=True)
    else:
        v_skills = v_core  # fallback: use core vector if no skills extracted
        print("[Warning] No skills/domain keywords found — using v_core as v_skills fallback.")

    return v_core, v_culture, v_neg, v_skills

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print('Loading JD...')

    def read_docx(path):
        requested = Path(path)
        if not requested.is_file():
            repo_root = Path(__file__).resolve().parent.parent
            requested = (repo_root / path).resolve()
        if not requested.is_file():
            raise FileNotFoundError(f"Job description not found at '{path}' or '{requested}'")

        print(f"Reading JD from: {requested}")
        doc = Document(requested)
        text = [para.text for para in doc.paragraphs]
        return "\n".join(text)

    raw_jd = read_docx("./job_description.docx")
    print(raw_jd[:67])

    print('Loading Model...')
    llm, embedder = load_model()

    print('Parsing JD...')
    parsed_jd = parse_jd(llm, raw_jd)
    print(json.dumps(parsed_jd, indent=2))

    expanded_negatives = expand_disqualifiers(
        llm, parsed_jd.get("abstract_disqualifiers", []), parsed_jd.get("job_title", "Professional")
    )
    print(f"\nExpanded Negative Examples: {expanded_negatives}")

    v_core, v_culture, v_neg, v_skills = create_embeddings(embedder, parsed_jd, expanded_negatives)

    print("\n SUCCESS! JD is fully parsed into Intent Vectors.")
    print(f"  domain_keywords     : {parsed_jd.get('domain_keywords', [])}")
    print(f"  must_have_skills_short: {parsed_jd.get('must_have_skills_short', [])}")
    print(f"  seniority_target    : {parsed_jd.get('seniority_target', 'N/A')}")

    # 1. Save all vectors to disk (including new v_skills)
    # Use VRD_DIR so this always saves to vrd/ regardless of the calling CWD
    embed_save_path = VRD_DIR / "jd_embeddings.npz"
    np.savez(
        str(embed_save_path),
        v_core=v_core,
        v_culture=v_culture,
        v_neg=v_neg,
        v_skills=v_skills,
    )
    print(f"Saved embeddings to {embed_save_path} (v_core, v_culture, v_neg, v_skills)")

    # 2. Save the parsed metadata (YOE constraints, job title, new fields, etc.)
    meta_save_path = VRD_DIR / "jd_metadata.json"
    with open(str(meta_save_path), "w") as f:
        json.dump(parsed_jd, f, indent=2)

    print(f"Saved rules to {meta_save_path}")
    print(f"Core Vector Shape   : {v_core.shape}")
    print(f"Culture Vector Shape: {v_culture.shape}")
    print(f"Negative Vector Shape: {v_neg.shape}")
    print(f"Skills Vector Shape : {v_skills.shape}")