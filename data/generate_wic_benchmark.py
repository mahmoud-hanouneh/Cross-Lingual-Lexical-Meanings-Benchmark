import requests
import json
import random
import time
import os
from dotenv import load_dotenv

# --- 1. Configuration ---

load_dotenv()
API_KEY = os.getenv('MY_API_KEY')

# Define the languages to use for creating pairs
LANG_1 = "EN"
LANG_2 = "DE" 

# A seed list of ambiguous English words
AMBIGUOUS_WORDS = ["bank", "crane", "match", "right", "ring", "bat", "bar", "bass", "firm", "star"]

# API endpoints
GET_SYNSER_IDS_URL = "https://babelnet.io/v9/getSynsetIds"
GET_SYNSER_URL = "https://babelnet.io/v9/getSynset"

def get_synset_details(synset_id):
    """
    Retrieves the full details for a given synset ID, including glosses (definitions).
    Returns a dictionary of language -> list of glosses.
    """
    try:
        time.sleep(0.5) # To be respectful of the API rate limit/ for now testing with their API because my BabelNet local copy seems to be incomplete :(
        params = { "id": synset_id, "key": API_KEY }
        response = requests.get(GET_SYNSER_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        glosses_by_lang = {}
        for gloss_data in data.get('glosses', []):
            lang = gloss_data['language']
            gloss_text = gloss_data['gloss']
            # --- Enhancement: Basic quality filter for glosses ---
            if len(gloss_text.split()) > 3: # Only accept "sentences" with more than 3 words
                if lang not in glosses_by_lang:
                    glosses_by_lang[lang] = []
                glosses_by_lang[lang].append(gloss_text)
        
        return glosses_by_lang
    except requests.exceptions.RequestException as e:
        print(f"    -> API Error for synset ID '{synset_id}': {e}")
        return {}


# --- 2. Main Generation Loop ---
benchmark_data = []
task_counter = 0

for word in AMBIGUOUS_WORDS:
    print(f"\nProcessing ambiguous word: '{word}'...")
    try:
        # --- Find distinct meanings (synsets) for the word ---
        params_ids = { "lemma": word, "searchLang": "EN", "source": "WN", "key": API_KEY }
        response_ids = requests.get(GET_SYNSER_IDS_URL, params=params_ids)
        response_ids.raise_for_status()
        synset_ids_data = response_ids.json()

        if len(synset_ids_data) < 2:
            print(f"  -> Could not find at least two distinct meanings for '{word}'. Skipping.")
            continue

        # --- Enhancement: Smarter search for two valid meanings with sentences ---
        meanings_with_sentences = []
        for synset_info in synset_ids_data[:5]: # Search top 5 meanings
            details = get_synset_details(synset_info['id'])
            if LANG_1 in details and details[LANG_1]:
                meanings_with_sentences.append(details)
            if len(meanings_with_sentences) == 2:
                break # Found two good meanings, we can stop searching

        if len(meanings_with_sentences) < 2:
            print(f"  -> Failed to find two meanings with usable sentences. Skipping.")
            continue

        details_1, details_2 = meanings_with_sentences

        # --- Generate a "False" example (meanings are different) ---
        sentence1 = random.choice(details_1[LANG_1])
        sentence2 = random.choice(details_2[LANG_2 if LANG_2 in details_2 else LANG_1]) # Fallback to LANG_1
        
        task_counter += 1
        benchmark_data.append({
            "task_id": f"WiC-{LANG_1}-{LANG_2}-{task_counter:03d}",
            "task_type": "word_in_context", "word": word,
            "lang1": LANG_1, "sentence1": sentence1,
            "lang2": LANG_2, "sentence2": sentence2,
            "question": f"Does the word '{word}' have the same meaning in both sentences? Answer with 'true' or 'false'.",
            "label": False
        })
        print("  -> Generated a 'False' example.")

        # --- Generate a "True" example (meanings are the same) ---
        # --- Enhancement: Check for at least two UNIQUE sentences ---
        if LANG_1 in details_1 and len(set(details_1[LANG_1])) > 1:
            # Use random.sample to pick two unique sentences from the list
            sentence1, sentence2 = random.sample(details_1[LANG_1], 2)

            task_counter += 1
            benchmark_data.append({
                "task_id": f"WiC-{LANG_1}-{LANG_1}-{task_counter:03d}",
                "task_type": "word_in_context", "word": word,
                "lang1": LANG_1, "sentence1": sentence1,
                "lang2": LANG_1, "sentence2": sentence2,
                "question": f"Does the word '{word}' have the same meaning in both sentences? Answer with 'true' or 'false'.",
                "label": True
            })
            print("  -> Generated a 'True' example.")

    except requests.exceptions.RequestException as e:
        print(f"An API error occurred for word '{word}': {e}")
    except Exception as e:
        print(f"A general error occurred for word '{word}': {e}")


# --- 3. Save the benchmark to a file ---
output_file = "wic_benchmark.jsonl"
with open(output_file, "w", encoding="utf-8") as f:
    for item in benchmark_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"\n--- DONE ---")
print(f"Generated {len(benchmark_data)} examples. Saved to {output_file}")