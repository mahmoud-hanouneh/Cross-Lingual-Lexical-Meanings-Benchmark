import requests
import json
import random
import time
import os
from dotenv import load_dotenv

# --- 1. Configuration ---

# Load API Key from a .env file for security
load_dotenv()
API_KEY = os.getenv('MY_API_KEY')

# Define the languages you want in your benchmark
TARGET_LANGUAGES = [("DE", "German"), ("FR", "French")]
SOURCE_LANGUAGE = "EN"

# Define your seed words with their Part-of-Speech (NOUN, VERB, ADJ)
SEED_WORDS_WITH_POS = [
    ("house", "NOUN"), ("water", "NOUN"), ("sun", "NOUN"), ("tree", "NOUN"),
    ("eat", "VERB"), ("walk", "VERB"), ("see", "VERB"),
    ("big", "ADJ"), ("small", "ADJ"), ("happy", "ADJ"), ("sad", "ADJ")
]

# API endpoints
GET_SYNSER_IDS_URL = "https://babelnet.io/v9/getSynsetIds"
GET_SYNSER_URL = "https://babelnet.io/v9/getSynset"
GET_EDGES_URL = "https://babelnet.io/v9/getOutgoingEdges" # <-- NEW endpoint for relations

def get_primary_synset_id(word, pos, lang):
    """Finds the most relevant synset ID for a word."""
    try:
        params = {
            "lemma": word,
            "searchLang": lang,
            "pos": pos,
            "source": "WN",
            "key": API_KEY
        }
        response = requests.get(GET_SYNSER_IDS_URL, params=params)
        response.raise_for_status()
        synset_ids = response.json()
        if synset_ids:
            return synset_ids[0]['id'] # Return the ID of the top result
    except requests.exceptions.RequestException as e:
        print(f"    -> API Error finding synset for '{word}': {e}")
    return None

def get_word_from_synset(synset_id, lang):
    """Gets the primary word for a given synset ID in a target language."""
    try:
        time.sleep(0.5) # Be respectful of the API rate limit
        params = {
            "id": synset_id,
            "targetLang": [lang],
            "key": API_KEY
        }
        response = requests.get(GET_SYNSER_URL, params=params)
        response.raise_for_status()
        data = response.json()
        senses = data.get('senses', [])
        for sense in senses:
            if sense['properties']['language'] == lang:
                return sense['properties']['fullLemma'].replace("_", " ")
    except requests.exceptions.RequestException as e:
        print(f"    -> API Error getting word for synset '{synset_id}': {e}")
    return None

def get_distractors(synset_id, target_lang, num_distractors=3):
    """
    Finds semantically related but incorrect words to use as distractors.
    """
    distractor_words = set()
    relation_groups_to_try = ["HYPERNYM", "MERONYM_PART", "MERONYM_MEMBER"]
    
    for relation in relation_groups_to_try:
        if len(distractor_words) >= num_distractors:
            break
        try:
            time.sleep(0.5)
            params = {
                "id": synset_id,
                "relationGroup": relation,
                "key": API_KEY
            }
            response = requests.get(GET_EDGES_URL, params=params)
            response.raise_for_status()
            edges = response.json()

            for edge in edges:
                related_synset_id = edge.get('target')
                if related_synset_id:
                    distractor = get_word_from_synset(related_synset_id, target_lang)
                    if distractor:
                        distractor_words.add(distractor)
                        if len(distractor_words) >= num_distractors:
                            return list(distractor_words)

        except requests.exceptions.RequestException as e:
            print(f"    -> API Error getting distractors for '{synset_id}': {e}")
            continue
            
    return list(distractor_words)


# --- 2. Main Generation Loop ---
benchmark_data = []
task_counter = 0

for word_to_translate, part_of_speech in SEED_WORDS_WITH_POS:
    for lang_code, lang_name in TARGET_LANGUAGES:
        print(f"\nProcessing '{word_to_translate}' ({part_of_speech}) -> {lang_name}...")

        # --- Step A: Get the main concept and the correct answer ---
        main_synset_id = get_primary_synset_id(word_to_translate, part_of_speech, SOURCE_LANGUAGE)
        if not main_synset_id:
            print(f"  -> Could not find main concept for '{word_to_translate}'. Skipping.")
            continue
            
        correct_answer = get_word_from_synset(main_synset_id, lang_code)
        if not correct_answer:
            print(f"  -> Could not find translation for '{word_to_translate}'. Skipping.")
            continue
        
        print(f"  -> Found translation: '{correct_answer}'")
        
        # --- Step B: Get high-quality, semantically related distractors ---
        print("  -> Generating semantic distractors...")
        distractors = get_distractors(main_synset_id, lang_code)
        distractors = set(distractors)
        distractors.discard(correct_answer)
        # Fallback strategy: If we can't find enough semantic distractors, use random ones.
        if len(distractors) < 3:
            print("  -> Not enough semantic distractors found, using random fallback...")
            distractor_pool = [item for item in SEED_WORDS_WITH_POS if item[0] != word_to_translate]
            words_for_distractors = random.sample(distractor_pool, 3)
            for distractor_word_en, distractor_pos in words_for_distractors:
                distractor_synset_id = get_primary_synset_id(distractor_word_en, distractor_pos, SOURCE_LANGUAGE)
                if distractor_synset_id:
                    distractor = get_word_from_synset(distractor_synset_id, lang_code)
                    if distractor and distractor != correct_answer:
                        distractors.add(distractor) if isinstance(distractors, set) else distractors.append(distractor)


        if len(distractors) < 3:
            print(f"  -> Could not generate enough distractors. Skipping.")
            continue

        # --- Step C: Assemble the final JSON object ---
        choices = list(distractors)[:3] # Ensure exactly 3 distractors
        choices.append(correct_answer)
        random.shuffle(choices)

        task_counter += 1
        data_point = {
            "task_id": f"CSI-{SOURCE_LANGUAGE}-{lang_code}-{task_counter:03d}",
            "task_type": "csi_custom_task",
            "source_word": word_to_translate,
            "source_lang": SOURCE_LANGUAGE,
            "target_lang": lang_code,
            "question": f"Which word has the same meaning as '{word_to_translate}' in {lang_name}?",
            "choices": choices,
            "answer": correct_answer
        }
        benchmark_data.append(data_point)

# --- 3. Save the benchmark to a file ---
output_file = "../benchmark/csi_benchmark_advanced.jsonl"
with open(output_file, "w", encoding="utf-8") as f:
    for item in benchmark_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"\n--- DONE ---")
print(f"Generated {len(benchmark_data)} examples. Saved to {output_file}")