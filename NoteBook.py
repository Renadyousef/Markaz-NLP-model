#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import torch
device = 0 if torch.cuda.is_available() else -1  # 0 = first GPU, -1 = CPU for speed
from arabert.preprocess import ArabertPreprocessor #preprocess text
from dotenv import load_dotenv
import re
from nltk.tokenize import sent_tokenize
import random
from transformers import pipeline
# from camel_tools.tokenizers.sent import simple_sentence_tokenize



load_dotenv()  # Loads HF_TOKEN automatically


# In[2]:


#directly thro hugging face
from transformers import pipeline

# POS tagging
pos = pipeline(
    "token-classification",
    model="CAMeL-Lab/bert-base-arabic-camelbert-da-pos-msa",
    framework="pt",
    trust_remote_code=True  # VERY IMPORTANT
)

# NER
ner = pipeline(
    "ner",
    model="CAMeL-Lab/bert-base-arabic-camelbert-msa-ner",
    framework="pt",
    trust_remote_code=True  # VERY IMPORTANT
)


# In[3]:


# genrating other MQS options
load_dotenv()  
from transformers import pipeline
from dotenv import load_dotenv
import os
import openai

# Load .env file
load_dotenv()

# Get the key
api_key = os.getenv("OPENAI_API_KEY")

# Debug check is it connecting?
if api_key:
    print("✅ OpenAI API key loaded successfully")
else:
    print("❌ Failed to load OpenAI API key. Check your .env file")

# Assign to OpenAI
openai.api_key = api_key


# In[4]:


import re

def arabic_sentence_split(text, max_words=150):
    # Split by sentence-ending punctuation first
    sentences = re.split(r'(?<=[\.\؟\!])\s+', text.strip())
    result = []
    for s in sentences:
        words = s.split()
        if len(words) <= max_words:
            result.append(s)
        else:
            # Chunk long sentences safely
            for i in range(0, len(words), max_words):
                result.append(" ".join(words[i:i+max_words]))
    return result


# In[5]:


def generate_question_text(entity, pos, word, sentence=None):
    """Generate natural Arabic question from entity + POS with more coverage and variety."""
    
    # Person entity
    if entity == "PER":
        if pos in ["noun", "noun_prop"]:
            return f"من هو {word}؟"
        elif pos == "adj":
            return f"أي شخص وُصف بـ {word}؟"
        else:
            return f"إلى أي شخص تشير كلمة '{word}'؟"
    
    # Location 
    if entity == "LOC":
        return f"أين تقع {word}؟"
    
    # Organization
    if entity == "ORG":
        if pos == "noun":
            return f"ما هي المنظمة المسماة {word}؟"
        elif pos == "adj":
            return f"أي منظمة وُصفت بأنها {word}؟"
        else:
            return f"اذكر المنظمة المرتبطة بكلمة '{word}'؟"
    
    #Date / Time 
    if entity in ["DATE", "TIME"]:
        return f"متى حدث ذلك؟ (الإشارة إلى {word})"
    
    #Number / Quantity
    if entity in ["NUM", "QUANTITY", "PERCENT"]:
        return f"ما هي القيمة العددية المذكورة: ____ (الإجابة {word})؟"
    
    # Miscellaneous / Product / Event
    if entity in ["EVENT", "WORK_OF_ART", "MISC"]:
        return f"إلى أي شيء يشير {word}؟"
    
    # Default Cloze (fill-in-the-blank)
    if sentence:
        return f"أكمل الفراغ: {sentence.replace(word, '____')}"
    
    #Fallback in case
    return f"صف الكلمة: {word}"


# In[6]:


import random

def generate_TF_question(sentence, entity_word, entity_type, ner_tags):
    """
    Generate a True/False question using GPT-3.5 to make the False statement more natural.

    sentence: the original sentence
    entity_word: the correct entity in the sentence
    entity_type: NER type, e.g., "LOC", "PER", "ORG"
    ner_tags: output from ner() for the text
    """
    # True statement
    true_statement = f"{sentence} (صح أم خطأ؟)"

    # Prepare prompt for GPT-3.5
    prompt = f"اجعل الجملة التالية خاطئة مع الحفاظ على المعنى العام والموضوع: {sentence}"

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "أنت مساعد لإنشاء أسئلة صح أو خطأ باللغة العربية."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.7
        )

        # Extract generated text
        false_sentence = response['choices'][0]['message']['content'].strip()
        false_statement = false_sentence + " (صح أم خطأ؟)"

    except Exception as e:
        print("Error generating false statement:", e)
        # fallback: swap entity as before
        other_entities = [tag['word'] for tag in ner_tags 
                          if tag['entity'] == entity_type and tag['word'] != entity_word]
        distractor = random.choice(other_entities) if other_entities else entity_word + "X"
        false_statement = sentence.replace(entity_word, distractor) + " (صح أم خطأ؟)"

    # Randomly choose True or False
    if random.choice([True, False]):
        return {"type": "TF", "statement": true_statement, "answer": True}
    else:
        return {"type": "TF", "statement": false_statement, "answer": False}


# In[7]:


difficulty_settings = {
    "easy": {
        "num_questions": 5,
        "tf_ratio": 0.7,       
        "mcq_distractor_type": "simple"  
    },
    "medium": {
        "num_questions": 10,
        "tf_ratio": 0.5,       
        "mcq_distractor_type": "medium"  
    },
    "hard": {
        "num_questions": 15,
        "tf_ratio": 0.3,       
        "mcq_distractor_type": "challenging"  
    }
}


# In[8]:


import random
import openai

def make_mq_options(entity_label, correct_entity, sentence, num_distractors=2):
    distractors = []

    prompt = (
        f"أعطني {num_distractors} كلمات عربية (كلمة واحدة لكل بديل) "
        f"تكون بدائل خاطئة لكلمة '{correct_entity}' في الجملة التالية: {sentence}. "
        f"كن محترمًا وابتعد عن أي كلمات جارحة أو مسيئة أو عنصرية أو جنسية. "
        f"أجب فقط بالكلمات مفصولة بفاصلة."
    )

    gpt_failed = False

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "أنت مساعد لإنشاء خيارات متعددة الاختيار باللغة العربية."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            temperature=0.7
        )

        generated_text = response['choices'][0]['message']['content'].strip()

        # Extract distractors
        for option in generated_text.split("،"):
            option_clean = option.strip()
            if option_clean and option_clean != correct_entity:
                distractors.append(option_clean)

    except Exception as e:
        print("Error generating distractors:", e)
        gpt_failed = True

    # Only use fallback if GPT failed completely
    if gpt_failed:
        distractors = [f"خيار{i+1}" for i in range(num_distractors)]

    # Limit to requested number of distractors
    distractors = distractors[:num_distractors]

    # Combine correct answer + distractors and shuffle
    options = [correct_entity] + distractors
    random.shuffle(options)

    return options


# In[9]:


def generate_MCQ_question(entity_word, sentence, entity_type="LOC", full_text=None):
    """
    Generate MCQ using entity as correct answer and GPT-3.5-generated distractors.
    """
    # Generate the question text
    question_text = generate_question_text(entity_type, "noun", entity_word, sentence)
    
    # Generate distractors using GPT-3.5
    options = make_mq_options(entity_type, entity_word, sentence, num_distractors=2)
    
    return {"type": "MCQ", "question": question_text, "options": options, "answer": entity_word}


# In[10]:


from tqdm import tqdm
import pyarabic.araby as araby

def make_quiz(text, level="medium"):
    settings = difficulty_settings.get(level, difficulty_settings["medium"])
    num_questions = settings["num_questions"]
    tf_ratio = settings["tf_ratio"]

    questions = []
    seen = set()
    sentences = arabic_sentence_split(text, max_words=200)

    # Batch NER + POS once since camel bert only takes 512 token at once
    ner_tags_all = ner(sentences, batch_size=32)
    pos_tags_all = pos(sentences, batch_size=32)

    for i, sentence in enumerate(tqdm(sentences)):
        if len(questions) >= num_questions:
            break
        ner_tags = ner_tags_all[i]
        pos_tags = pos_tags_all[i]
        if not ner_tags:
            continue

        entity = ner_tags[0]
        word = entity['word']
        entity_type = entity['entity']
        pos_tag = next((p['entity'] for p in pos_tags if p['word'] == word), "noun")

        # TF vs MCQ decision
        if random.random() < tf_ratio:
            q = generate_TF_question(sentence, word, entity_type, ner_tags)
        else:
            q = generate_MCQ_question(word, sentence, entity_type, full_text=text)

        # ensure no repeats
        sig = (q["type"], q.get("question"), q.get("answer"))
        if sig not in seen:
            questions.append(q)
            seen.add(sig)

    return questions


# In[11]:


import re

"""
Pre-process Arabic text while keeping relevant content for NLP:

- Skips empty lines
- Skips page numbers (English 1, 2… and Arabic-Indic ١, ٢…)
- Skips figure/table captions starting with شكل / جدول
- Removes inline references [1], (2) etc. (English + Arabic-Indic)
- Stops processing at references section
- Skips initial metadata/intros until first Arabic paragraph
- Skips first irrelevant pages in textbooks
- force arabic parsing back as it messed up the reading,tho it messes the logic compariosn in models above
"""

def is_textbook_front(line):
    """Return True if the line is likely textbook front matter."""
    keywords = ["الطبعة", "المؤلف", "المحرر", "الحقوق محفوظة", "نسخة", "جامعة", "الكتاب", "المقدمة"]
    return any(k in line for k in keywords)

    
def clean_arabic_text(text):
    lines = text.splitlines()
    cleaned_lines = []

    skip_intro = True
    arabic_re = re.compile(r'[\u0600-\u06FF]')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue

        if skip_intro:
            if is_textbook_front(line) or (len(arabic_re.findall(line)) < 5):
                continue
            else:
                skip_intro = False

        # Skip page numbers (English or Arabic-Indic)
        if re.match(r'^([0-9]+|[٠-٩]+)$', line):
            continue

        # Skip figure/table captions
        if re.match(r'^(شكل|جدول)\s*([0-9]+|[٠-٩]+)', line):
            continue

        # Stop at references section
        if line.startswith(("المراجع", "قائمة المراجع")):
            break

        # Remove inline references like [1], (2), [١], (٢)
        line = re.sub(r'\[[0-9٠-٩]+\]|\([0-9٠-٩]+\)', '', line)

        cleaned_lines.append(line)

    # Join all cleaned lines
    cleaned_text = "\n".join(cleaned_lines)

   

    return cleaned_text


# In[12]:


# Strategy for 30 max we do less pages we will need to chunck the request to model 
from concurrent.futures import ThreadPoolExecutor

def make_quiz_from_large_text(text, level="medium"):
   # cleaned_text=clean_arabic_text(text)
    sentences =arabic_sentence_split(text, max_words=200)
    chunks = [sentences[i:i+200] for i in range(0, len(sentences), 200)]
    results = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(make_quiz, " ".join(chunk), level) for chunk in chunks]
        for f in futures:
            results.extend(f.result())

    return results



# #### Since we don’t have a gold-standard labeled dataset for the 30 pages of Arabic text uploaded by users (our use case), we will evaluate our model on the dataset used in the original paper.

# ![image.png](attachment:95cb5ff5-9de9-450f-9d4c-c050aaa86129.png)
# #### Due to computational limitations on my CPU, I could not evaluate on the full dataset as done in the original paper. Instead, I randomly sampled approximately 500 words for evaluation. Despite the reduced sample size, the model achieved an F1 score of 84%, which is comparable to the 82.6% F1 reported in the original study. This suggests that even a small, random sample provides a reasonably accurate estimate of the model’s performance.

# ## evaluating the POS model based on thier evaluation data set
