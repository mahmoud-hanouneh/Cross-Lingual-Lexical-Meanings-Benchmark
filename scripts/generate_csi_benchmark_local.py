# This script is to work with the BabelNet instance via RPC/Docker. I used BabelNet 5.0 for this project
import babelnet as bn
import json
import random

from babelnet.language import Language
from babelnet.pos import POS
from babelnet.data.source import BabelSenseSource
from babelnet.data.relation import BabelPointer
from babelnet.resources import BabelSynsetID
from seed_words import SEED_WORDS_WITH_POS
# Configuration - No need for API key
# The babelnet_conf.yml file handles the connection.

TARGET_LANGUAGES = [("DE", "German"), ("FR", "French")]
SOURCE_LANGUAGE_ENUM = Language.EN 
SOURCE_LANGUAGE_STR = "EN"  


# Will be extended later 
# SEED_WORDS_WITH_POS = [
#     ("house", "NOUN"), ("water", "NOUN"), ("sun", "NOUN"), ("tree", "NOUN"),
#     ("eat", "VERB"), ("walk", "VERB"), ("see", "VERB"),
#     ("big", "ADJ"), ("small", "ADJ"), ("happy", "ADJ"), ("sad", "ADJ")
# ]
# SEED_WORDS_WITH_POS = [
#     ("house", "NOUN"), ("water", "NOUN"), ("sun", "NOUN"), ("tree", "NOUN"),
#     ("eat", "VERB"), ("walk", "VERB"), ("see", "VERB"),
#     ("big", "ADJ"), ("small", "ADJ"), ("happy", "ADJ"), ("sad", "ADJ")
# ]


def get_primary_synset(word, pos, lang_enum):
    # This func finds the most relevant BabelSynset object for a word using the local library
    try:
        # Call library function directly 
        synsets = bn.get_synsets(word, from_langs=[lang_enum], poses=[POS[pos]], sources=[BabelSenseSource.WN])
        if synsets:
            return synsets[0].id  # Returning the ID of the synset object
    except Exception as e:
        print(f"    -> Error finding synset for '{word}': {e}")
    return None

def get_word_from_synset(synset, target_lang_code):
    # This function gets the primary word for a given synset object in a target language

    w_f_syn = bn.get_synset(BabelSynsetID(str(synset)))
    if not w_f_syn:
        return None
    # We can now directly access the senses from the synset object
    for sense in w_f_syn.senses():
        if sense.language.name == target_lang_code:
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


#  Main Generation Loop 
benchmark_data = []
task_counter = 0

for word_to_translate, part_of_speech in SEED_WORDS_WITH_POS:
    for lang_code, lang_name in TARGET_LANGUAGES:
        print(f"\nProcessing '{word_to_translate}' ({part_of_speech}) -> {lang_name}...")

        # Step A: Get the main concept and the correct answer 
        main_synset = get_primary_synset(word_to_translate, part_of_speech, SOURCE_LANGUAGE_ENUM) #EN
        if not main_synset:
            print(f"  -> Could not find main concept for '{word_to_translate}'. Skipping.")
            continue
            
        correct_answer = get_word_from_synset(main_synset, lang_code)
        if not correct_answer:
            print(f"  -> Could not find translation for '{word_to_translate}'. Skipping.")
            continue
        
        print(f"  -> Found translation: '{correct_answer}'")
        
        # Step B: Get high-quality, semantically related distractors
        print("  -> Generating semantic distractors...")
        # distractors = set(get_distractors(main_synset, lang_code, num_distractors=5)) # Get a few extra
        distractors = set(get_distractors(bn.get_synset(BabelSynsetID(str(main_synset))), lang_code, num_distractors=5)) # Get a few extra

        distractors.discard(correct_answer) # Remove the correct answer if it's there - to prevent duplicatoins
        
        # Fallback strategy in case there are NOT enough semantic distractors
        if len(distractors) < 3:
            print("  -> Not enough semantic distractors found, using random fallback...")
            distractor_pool = [item for item in SEED_WORDS_WITH_POS if item[0] != word_to_translate]
            if len(distractor_pool) >= 3:
                words_for_distractors = random.sample(distractor_pool, 3)
                for distractor_word_en, distractor_pos in words_for_distractors:
                    distractor_synset = get_primary_synset(distractor_word_en, distractor_pos, SOURCE_LANGUAGE_ENUM)
                    if distractor_synset:
                        distractor = get_word_from_synset(distractor_synset, lang_code)
                        if distractor and distractor != correct_answer:
                            distractors.add(distractor)

        if len(distractors) < 3:
            print(f"  -> Could not generate enough unique distractors. Skipping.")
            continue

        # Step C: Assemble the final JSON object
        choices = random.sample(list(distractors), 3)
        choices.append(correct_answer)
        random.shuffle(choices)

        task_counter += 1
        data_point = {
            "task_id": f"CSI-{SOURCE_LANGUAGE_STR}-{lang_code}-{task_counter:03d}",
            "task_type": "csi_custom_task",
            "source_word": word_to_translate,
            "source_lang": SOURCE_LANGUAGE_STR,
            "target_lang": lang_code,
            "question": f"Which word has the same meaning as '{word_to_translate}' in {lang_name}?",
            "choices": choices,
            "answer": correct_answer
        }
        benchmark_data.append(data_point)

# Save the benchmark to a file ---
output_file = "csi_benchmark_local.jsonl"
with open(output_file, "w", encoding="utf-8") as f:
    for item in benchmark_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"\n--- DONE ---")
print(f"Generated {len(benchmark_data)} examples. Saved to {output_file}")