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
def parse_jd(llm, jd_text):
    print("\n[Step 1] Extracting Implicit and Explicit Intents from JD...")
    prompt = f"""<|im_start|>system
You are an expert recruiter analyzing a complex Job Description. 
Extract the hiring intents into a strict JSON format with exactly these keys:
1. "job_title": The exact role being hired for (e.g., "Senior AI Engineer", "Marketing Manager", "Data Scientist").
2. "must_have_hard_skills": Core technical or functional requirements.
3. "execution_and_impact_signals": Signals of real-world delivery, scale, or impact.
4. "culture_and_behavior": Preferred working styles and mindset.
5. "abstract_disqualifiers": Backgrounds, traits, or firm types they explicitly reject.
6. "negative_tools_or_patterns": Tools, methodologies, or approaches they view as negative.
7. "metadata_constraints": A dictionary with keys "min_yoe", "max_yoe", "preferred_company_type", "preferred_locations".
Output ONLY valid JSON.<|im_end|>
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

        # Ensure job_title always exists
        if "job_title" not in result:
            result["job_title"] = "Professional"

        return result
    except Exception as e:
        print("Failed to parse JSON, check SLM output.", e)
        return {
            "job_title": "Professional", "must_have_hard_skills": [], "execution_and_impact_signals": [],
            "culture_and_behavior": [], "abstract_disqualifiers": [], "negative_tools_or_patterns": [],
            "metadata_constraints": {}
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

    core_text = ". ".join(jd_dict.get("must_have_hard_skills", []) + jd_dict.get("execution_and_impact_signals", []))
    v_core = embedder.encode([core_text], normalize_embeddings=True)

    culture_text = ". ".join(jd_dict.get("culture_and_behavior", []))
    v_culture = embedder.encode([culture_text], normalize_embeddings=True)

    negative_frameworks = ". ".join(jd_dict.get("negative_tools_or_patterns", []))
    full_negative = expanded_neg_text + " " + negative_frameworks
    v_neg = embedder.encode([full_negative], normalize_embeddings=True)

    return v_core, v_culture, v_neg

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

    expanded_negatives = expand_disqualifiers(llm, parsed_jd.get("abstract_disqualifiers", []), parsed_jd.get("job_title", "Professional"))
    print(f"\nExpanded Negative Examples: {expanded_negatives}")

    v_core, v_culture, v_neg = create_embeddings(embedder, parsed_jd, expanded_negatives)

    print("\n SUCCESS! JD is fully parsed into Intent Vectors.")

    # 1. Save the vectors to disk
    np.savez(
        "./jd_embeddings.npz",
        v_core=v_core,
        v_culture=v_culture,
        v_neg=v_neg
    )
    print("Saved embeddings to jd_embeddings.npz")

    # 2. Save the parsed metadata (YOE constraints, job title, etc.)
    with open("./jd_metadata.json", "w") as f:
        json.dump(parsed_jd, f, indent=2)

    print("Saved rules to jd_metadata.json")
    print(f"Core Vector Shape: {v_core.shape}")
    print(f"Culture Vector Shape: {v_culture.shape}")
    print(f"Negative Vector Shape: {v_neg.shape}")