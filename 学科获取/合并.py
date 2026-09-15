import json
import pandas as pd
from collections import Counter

# ============================================================
# 旧版 OpenAlex Concept → Field 映射表
# ============================================================
CONCEPT_TO_FIELD = {
    # ── Computer Science ─────────────────────────────────────
    'Computer science':                  'Computer Science',
    'Artificial intelligence':           'Computer Science',
    'Machine learning':                  'Computer Science',
    'Deep learning':                     'Computer Science',
    'Natural language processing':       'Computer Science',
    'Computer vision':                   'Computer Science',
    'Reinforcement learning':            'Computer Science',
    'Transfer learning':                 'Computer Science',
    'Convolutional neural network':      'Computer Science',
    'Recurrent neural network':          'Computer Science',
    'Speech recognition':                'Computer Science',
    'Information retrieval':             'Computer Science',
    'Human–computer interaction':        'Computer Science',
    'Robotics':                          'Computer Science',
    'Computer security':                 'Computer Science',
    'Cryptography':                      'Computer Science',
    'Software engineering':              'Computer Science',
    'Operating system':                  'Computer Science',
    'Computer network':                  'Computer Science',
    'Distributed computing':             'Computer Science',
    'Database':                          'Computer Science',
    'Data mining':                       'Computer Science',
    'Data science':                      'Computer Science',
    'Computer graphics':                 'Computer Science',
    'Algorithm':                         'Computer Science',
    'Theoretical computer science':      'Computer Science',
    'Parallel computing':                'Computer Science',
    'Embedded system':                   'Computer Science',
    'Programming language':              'Computer Science',
    'Knowledge representation and reasoning': 'Computer Science',
    'Expert system':                     'Computer Science',
    'Recommender system':                'Computer Science',
    'Computer hardware':                 'Computer Science',
    'World Wide Web':                    'Computer Science',
    'Simulation':                        'Computer Science',
    'Pattern recognition':               'Computer Science',
    'Semantic Web':                      'Computer Science',
    'Internet of things':                'Computer Science',
    'Cloud computing':                   'Computer Science',
    'Cybersecurity':                     'Computer Science',
    'Multimedia':                        'Computer Science',

    # ── Mathematics / Statistics ─────────────────────────────
    'Mathematics':                       'Mathematics',
    'Statistics':                        'Mathematics',
    'Applied mathematics':               'Mathematics',
    'Mathematical optimization':         'Mathematics',
    'Combinatorics':                     'Mathematics',
    'Mathematical analysis':             'Mathematics',
    'Algebra':                           'Mathematics',
    'Geometry':                          'Mathematics',
    'Probability theory':                'Mathematics',
    'Bayesian probability':              'Mathematics',
    'Linear algebra':                    'Mathematics',
    'Calculus':                          'Mathematics',
    'Graph theory':                      'Mathematics',
    'Topology':                          'Mathematics',
    'Numerical analysis':                'Mathematics',
    'Stochastic process':                'Mathematics',
    'Discrete mathematics':              'Mathematics',

    # ── Physics ──────────────────────────────────────────────
    'Physics':                           'Physics',
    'Quantum mechanics':                 'Physics',
    'Condensed matter physics':          'Physics',
    'Optics':                            'Physics',
    'Astrophysics':                      'Physics',
    'Nuclear physics':                   'Physics',
    'Thermodynamics':                    'Physics',
    'Particle physics':                  'Physics',
    'Quantum computing':                 'Physics',
    'Quantum information':               'Physics',
    'Electromagnetism':                  'Physics',
    'Statistical mechanics':             'Physics',

    # ── Engineering ──────────────────────────────────────────
    'Engineering':                       'Engineering',
    'Materials science':                 'Engineering',
    'Mechanical engineering':            'Engineering',
    'Electrical engineering':            'Engineering',
    'Electronic engineering':            'Engineering',
    'Chemical engineering':              'Engineering',
    'Signal processing':                 'Engineering',
    'Control theory':                    'Engineering',
    'Telecommunications':                'Engineering',
    'Remote sensing':                    'Engineering',
    'Structural engineering':            'Engineering',
    'Civil engineering':                 'Engineering',
    'Biomedical engineering':            'Engineering',

    # ── Biology / Life Science ───────────────────────────────
    'Biology':                           'Biology',
    'Genetics':                          'Biology',
    'Molecular biology':                 'Biology',
    'Cell biology':                      'Biology',
    'Ecology':                           'Biology',
    'Evolutionary biology':              'Biology',
    'Neuroscience':                      'Biology',
    'Bioinformatics':                    'Biology',
    'Biochemistry':                      'Biology',
    'Microbiology':                      'Biology',
    'Proteomics':                        'Biology',
    'Genomics':                          'Biology',
    'Systems biology':                   'Biology',
    'Zoology':                           'Biology',
    'Botany':                            'Biology',

    # ── Medicine ─────────────────────────────────────────────
    'Medicine':                          'Medicine',
    'Pharmacology':                      'Medicine',
    'Oncology':                          'Medicine',
    'Immunology':                        'Medicine',
    'Pathology':                         'Medicine',
    'Psychiatry':                        'Medicine',
    'Epidemiology':                      'Medicine',
    'Surgery':                           'Medicine',
    'Radiology':                         'Medicine',
    'Cardiology':                        'Medicine',
    'Neurology':                         'Medicine',
    'Clinical psychology':               'Medicine',
    'Internal medicine':                 'Medicine',
    'Nursing':                           'Medicine',
    'Public health':                     'Medicine',

    # ── Chemistry ────────────────────────────────────────────
    'Chemistry':                         'Chemistry',
    'Organic chemistry':                 'Chemistry',
    'Inorganic chemistry':               'Chemistry',
    'Analytical chemistry':              'Chemistry',
    'Polymer chemistry':                 'Chemistry',
    'Physical chemistry':                'Chemistry',
    'Catalysis':                         'Chemistry',

    # ── Economics / Business ─────────────────────────────────
    'Economics':                         'Economics',
    'Microeconomics':                    'Economics',
    'Macroeconomics':                    'Economics',
    'Econometrics':                      'Economics',
    'Finance':                           'Economics',
    'Business':                          'Economics',
    'Management':                        'Economics',
    'Marketing':                         'Economics',
    'Accounting':                        'Economics',
    'Labor economics':                   'Economics',

    # ── Psychology / Social Science ──────────────────────────
    'Psychology':                        'Psychology',
    'Cognitive psychology':              'Psychology',
    'Social psychology':                 'Psychology',
    'Sociology':                         'Sociology',
    'Demography':                        'Sociology',
    'Social network':                    'Sociology',
    'Political science':                 'Political Science',
    'Law':                               'Political Science',

    # ── Humanities ───────────────────────────────────────────
    'Linguistics':                       'Humanities',
    'Philosophy':                        'Humanities',
    'History':                           'Humanities',
    'Art':                               'Humanities',
    'Education':                         'Humanities',

    # ── Environmental Science ────────────────────────────────
    'Environmental science':             'Environmental Science',
    'Climate change':                    'Environmental Science',
    'Atmospheric science':               'Environmental Science',
    'Oceanography':                      'Environmental Science',
    'Hydrology':                         'Environmental Science',
    'Geology':                           'Environmental Science',

# 在 CONCEPT_TO_FIELD 里追加以下几行即可
    'Geography':               'Environmental Science',
    'Archaeology':             'Humanities',
    'Regional science':        'Economics',
    'Economic geography':      'Economics',
    'Meteorology':             'Environmental Science',
    'Environmental planning':  'Environmental Science',

}

# ============================================================
# 提取函数（格式B，字符串列表）
# ============================================================
def extract_primary_field(concepts):
    if not concepts or not isinstance(concepts, list):
        return 'Other'
    for concept in concepts:
        if isinstance(concept, str) and concept in CONCEPT_TO_FIELD:
            return CONCEPT_TO_FIELD[concept]
    return 'Other'

# ============================================================
# 三份数据统一处理
# ============================================================
FILES = {
    'stage_1': r'E:\PythonProject\GenAI\data_scrap_v2\stage_1\phase1_hybrid_cleaned.jsonl',
    'stage_2': r'E:\PythonProject\GenAI\data_scrap_v2\stage_2\phase2_hybrid_cleaned.jsonl',
    'stage_3': r'E:\PythonProject\GenAI\data_scrap_v2\stage_3\phase3_hybrid_cleaned.jsonl',
}

all_records  = []
miss_counter = Counter()

for stage, filepath in FILES.items():
    print(f"\n📖 处理 {stage}...")
    hit, miss = 0, 0

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            rec      = json.loads(line)
            concepts = rec.get('concepts', [])
            field    = extract_primary_field(concepts)

            all_records.append({
                'work_id':       rec['work_id'],
                'stage':         stage,
                'primary_field': field,
            })

            if field != 'Other':
                hit += 1
            else:
                miss += 1
                for c in concepts:
                    if isinstance(c, str):
                        miss_counter[c] += 1

    total = hit + miss
    print(f"  ✅ 命中 {hit:,} / {total:,} = {hit/total:.1%}")

# ── 汇总保存 ─────────────────────────────────────────────────
field_df = pd.DataFrame(all_records)
field_df.to_csv('work_id_to_field_all_stages.csv', index=False)

print(f"\n{'='*50}")
print(f"📊 三份合计：{len(field_df):,} 篇")
print(f"\n学科分布：")
print(field_df['primary_field'].value_counts())

print(f"\n⚠️  仍未命中的 Top 20（需补充）：")
for name, cnt in miss_counter.most_common(20):
    print(f"  {name:45s}  {cnt:,} 次")
