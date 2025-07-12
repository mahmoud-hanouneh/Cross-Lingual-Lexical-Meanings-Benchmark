# This script is to work with the BabelNet instance via RPC/Docker. I used BabelNet 5.0 for this project
# It generates separate benchmark files for high, medium, and low-resource languages.

import babelnet as bn
import json
import random
import os

# Imports for the pybabelnet library's specific types
from babelnet.language import Language
from babelnet.pos import POS
from babelnet.data.source import BabelSenseSource
from babelnet.data.relation import BabelPointer
from babelnet.resources import BabelSynsetID


from seed_words import SEED_WORDS_WITH_POS
from language_config import LANGUAGE_CONFIG

#  Configuration
SOURCE_LANGUAGE_ENUM = Language.EN
SOURCE_LANGUAGE_STR = "EN"
OUTPUT_DIR = "./data" 


# Helper functions 

def get_primary_synset(word, pos, lang_enum):
    # This func finds the most relevant BabelSynset object for a word using the local library
    try:
        # Call library function directly 
        synsets = bn.get_synsets(word, from_langs=[lang_enum], poses=[POS[pos]], sources=[BabelSenseSource.WN])
        if synsets:
            return synsets[0].id # Returning the ID of the synset object
    except Exception as e:
        print(f"    -> Error finding synset for '{word}': {e}")
    return None

def get_word_from_synset(synset, target_lang_code):
    # This function gets the primary word for a given synset object in a target language (via its ID)

    w_f_syn = bn.get_synset(BabelSynsetID(str(synset)))
    if not w_f_syn:
        return None
    # We can now directly access the senses from the synset object
    for sense in w_f_syn.senses():
        if sense.language.name.upper() == target_lang_code.upper():
            return sense.full_lemma.replace("_", " ")
    return None

# The logic here focuses on generating polysomous words for the distractors to make the benchmark more efficient and challenging 
def get_distractors(main_synset, target_lang_code, num_distractors=3):
    # Finds semantically related but incorrect words to use as distractors.
    if not main_synset:
        return []
    distractor_words = set()

    # Use BabelPointer Enums for relation types
    relation_types_to_try = [BabelPointer.ANY_HYPERNYM, BabelPointer.ANY_MERONYM]
    
    for rel_type in relation_types_to_try:
        if len(distractor_words) >= num_distractors:
            break

        related_edges = main_synset.outgoing_edges(rel_type)


        for edge in related_edges:
            # The edge gives us the ID, so we need to fetch the full synset object
            # related_synset = bn.get_synset(edge.target)
            related_synset = bn.get_synset(BabelSynsetID(str(edge.target)))
            if related_synset:
                distractor = get_word_from_synset(related_synset.id, target_lang_code)
                if distractor:
                    distractor_words.add(distractor)
                    if len(distractor_words) >= num_distractors:
                        return list(distractor_words)
    return list(distractor_words)


# Main Generation Loop

# benchmark_data = [] 

# Updated to multilingually extend the benchmark and generate data in tiers 

os.makedirs(OUTPUT_DIR, exist_ok=True)

for tier_name, languages_in_tier in LANGUAGE_CONFIG.items():
    print(f"\n== Starting Generating for Tier: {tier_name.upper()} ==\n")
    
    tier_specific_data = []
    
    for lang_enum, lang_details in languages_in_tier.items():
        lang_code = lang_details['code']
        lang_name = lang_details['name']
        print(f"--- Processing Language: {lang_name} ({lang_code}) ---")
        
        for word_to_translate, part_of_speech in SEED_WORDS_WITH_POS:
            #EDIT: Added the detailed print statement back in ---
            print(f"  -> Processing '{word_to_translate}'...")

            main_synset_id = get_primary_synset(word_to_translate, part_of_speech, SOURCE_LANGUAGE_ENUM)
            if not main_synset_id:
                # print(f"  -> Could not find main concept for '{word_to_translate}'. Skipping.")
                continue
                
            correct_answer = get_word_from_synset(main_synset_id, lang_code)
            if not correct_answer:
                # print(f"  -> Could not find translation for '{word_to_translate}'. Skipping.")
                continue
            
            # print(f"  -> Found translation: '{correct_answer}'")

            main_synset_obj = bn.get_synset(BabelSynsetID(str(main_synset_id)))
            distractors = set(get_distractors(main_synset_obj, lang_code, num_distractors=5)) # Get a few extra
            # Remove the correct answer if it's there - to prevent duplicatoins
            distractors.discard(correct_answer) 


             # Fallback strategy in case there are NOT enough semantic distractors
            if len(distractors) < 3:
                distractor_pool = [item for item in SEED_WORDS_WITH_POS if item[0] != word_to_translate]
                if len(distractor_pool) >= 3:
                    words_for_distractors = random.sample(distractor_pool, 3)
                    for distractor_word_en, distractor_pos in words_for_distractors:
                        distractor_synset_id = get_primary_synset(distractor_word_en, distractor_pos, SOURCE_LANGUAGE_ENUM)
                        if distractor_synset_id:
                            distractor = get_word_from_synset(distractor_synset_id, lang_code)
                            if distractor and distractor != correct_answer:
                                distractors.add(distractor)

            if len(distractors) < 3:
                print(f"  -> Could not generate enough unique distractors. Skipping.")
                continue

            # Assemble the final JSON object
            choices = random.sample(list(distractors), 3)
            choices.append(correct_answer)
            random.shuffle(choices)

            task_id = f"MSI-{SOURCE_LANGUAGE_STR}-{lang_code}-{len(tier_specific_data) + 1:04d}"

            data_point = {
                "task_id": task_id,
                "task_type": "msi_custom_task",
                "source_word": word_to_translate,
                "source_lang": SOURCE_LANGUAGE_STR,
                "target_lang": lang_code,
                "question": f"Which word has the same meaning as '{word_to_translate}' in {lang_name}?",
                "choices": choices,
                "answer": correct_answer
            }
            tier_specific_data.append(data_point)

    if tier_specific_data:
        output_file_path = os.path.join(OUTPUT_DIR, f"msi_benchmark_{tier_name}.jsonl")
        with open(output_file_path, "w", encoding="utf-8") as f:
            for item in tier_specific_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"\n--- TIER {tier_name.upper()} DONE ---")
        print(f"Generated {len(tier_specific_data)} total examples. Saved to {output_file_path}")
    else:
        print(f"\n--- TIER {tier_name.upper()} DONE ---")
        print(f"No examples were generated for this tier.")

print("\n===== ALL TIERS PROCESSED! =====\n")