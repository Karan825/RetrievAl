import json
from pathlib import Path
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
from docx import Document


# ==========================================
# 1. INITIALIZE MODELS
# ==========================================
def load_model():
    print("Loading SLM (Qwen2.5 1.5B)...")
    llm = Llama(
        model_path="./qwen2.5-1.5b-instruct-q4_k_m.gguf",
        n_ctx=8192,
        n_threads=4,
        verbose=False
    )
    print("Loading Embedding Model (bge-small)...")
    embedder = SentenceTransformer("./local_bge_model")
    return llm, embedder


def safe_llm_call(llm, prompt, max_tokens=1500, stop="<|im_end|>", temperature=0.1, min_tokens=16):
    """Call llama_cpp Llama safely, retrying with smaller max_tokens if the requested tokens exceed the context window.
    This helps when a long prompt + requested max_tokens would overflow n_ctx.
    """
    attempt_max = max_tokens
    for i in range(5):
        try:
            return llm(prompt, max_tokens=attempt_max, stop=stop, temperature=temperature)
        except ValueError as e:
            msg = str(e)
            if "Requested tokens" in msg and "exceed context window" in msg:
                # Reduce the requested generation length and retry
                attempt_max = max(min_tokens, attempt_max // 2)
                print(f"LLM call failed due to context window; reducing max_tokens to {attempt_max} and retrying...")
                continue
            raise
    # As a last resort, truncate the prompt and request a small generation
    short_prompt = prompt[:1500]
    print("LLM call still failing — retrying with truncated prompt and minimal tokens.")
    return llm(short_prompt, max_tokens=min_tokens, stop=stop, temperature=temperature)


# ==========================================
# 2. PARSE THE JD (SLM)
# ==========================================
def parse_jd(llm, jd_text):
    print("\n[Step 1] Extracting Intents and Metadata from JD...")
    prompt = f"""<|im_start|>system
You are an expert AI recruiter. Read the job description and extract data into a strict JSON format with exactly three keys:
1. "positive_experiences": A list of specific things the candidate must have built or done.
2. "abstract_disqualifiers": A list of backgrounds, traits, or firm types the company explicitly does NOT want.
3. "hard_constraints": A dictionary with keys like "min_yoe", "max_yoe", "preferred_locations".
Output ONLY valid JSON.<|im_end|>
<|im_start|>user
Job Description:
{jd_text}<|im_end|>
<|im_start|>assistant
"""
    # 🔴 FIX APPLIED HERE: We pass max_tokens=1500 so it doesn't get capped at 300!
    response = safe_llm_call(llm, prompt, max_tokens=1500, stop="<|im_end|>", temperature=0.1)

    try:
        result = json.loads(response['choices'][0]['text'].strip())
        return result
    except Exception as e:
        print("Failed to parse JSON, check SLM output.", e)
        print("Raw output:", response['choices'][0]['text'])
        return {"positive_experiences": [], "abstract_disqualifiers": [], "hard_constraints": {}}


# ==========================================
# 3. EXPAND DISQUALIFIERS (WORLD KNOWLEDGE)
# ==========================================
def expand_disqualifiers(llm, disqualifiers):
    print("\n[Step 2] Expanding Abstract Disqualifiers...")
    if not disqualifiers:
        return ""
    expanded_text = ""

    for disq in disqualifiers:
        prompt = f"""<|im_start|>system
You are a career expert. The user will give you an abstract negative trait a company does NOT want. 
List 5 to 10 concrete examples, company names, or resume phrases that represent this trait. 
Do not explain, just list the examples separated by commas.<|im_end|>
<|im_start|>user
Abstract Trait: "{disq}"<|im_end|>
<|im_start|>assistant
"""
        # Kept at 200 because we only need a short comma-separated list here
        response = safe_llm_call(llm, prompt, max_tokens=200, stop="<|im_end|>", temperature=0.1)
        expanded_text += response['choices'][0]['text'].strip() + ", "

    return expanded_text.strip(", ")


# ==========================================
# 4. GENERATE MULTI-VECTORS
# ==========================================
def create_embeddings(embedder, jd_dict, expanded_neg_text):
    print("\n[Step 3] Generating Positive and Negative Embeddings...")

    # Combine positive experiences into a single descriptive string
    positive_text = ". ".join(jd_dict.get("positive_experiences", []))
    print(f" -> Embedding Positive Text: '{positive_text[:75]}...'")

    # Create the embeddings and normalize them (important for cosine similarity)
    v_pos = embedder.encode([positive_text], normalize_embeddings=True)

    print(f" -> Embedding Negative Text: '{expanded_neg_text[:75]}...'")
    v_neg = embedder.encode([expanded_neg_text], normalize_embeddings=True)

    return v_pos, v_neg


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


    raw_jd = read_docx("Dataset/job_description.docx")
    print(raw_jd[:67])

    print('Loading Model...')
    # Load Models
    llm, embedder = load_model()

    print('Parsing JD...')
    # Parse JD
    parsed_jd = parse_jd(llm, raw_jd)
    print(json.dumps(parsed_jd, indent=2))

    expanded_negatives = expand_disqualifiers(llm, parsed_jd.get("abstract_disqualifiers", []))
    print(f"\nExpanded Negative Examples: {expanded_negatives}")

    v_pos, v_neg = create_embeddings(embedder, parsed_jd, expanded_negatives)
    print("\n SUCCESS! JD is fully parsed and embedded.")
    print(f"Positive Vector Shape: {v_pos.shape}")
    print(f"Negative Vector Shape: {v_neg.shape}")
