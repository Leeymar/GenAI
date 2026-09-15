import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from collections import defaultdict


def calculate_full_dynamic_diffusion():
    print("🚀 启动【全量保留·动态加权扩散引擎 (绝对路径修复版)】...\n")

    # 你的粉丝学科数据 (假设和脚本在同一个目录，如果不在，也请改成绝对路径)
    CITING_DATA_FILE = 'citing_disciplines_master.jsonl'

    # 🌟 已经为你替换成了最安全的 Raw String 绝对路径
    SOURCE_PHASE_FILES = [
        r'E:\PythonProject\GenAI\data_scrap_v2\stage_1\phase1_hybrid_cleaned.jsonl',
        r'E:\PythonProject\GenAI\data_scrap_v2\stage_2\phase2_hybrid_cleaned.jsonl',
        r'E:\PythonProject\GenAI\data_scrap_v2\stage_3\phase3_hybrid_cleaned.jsonl'
    ]
    OUTPUT_CSV = 'dynamic_diffusion_full_metrics.csv'

    print("🧬 正在提取源论文的自身学科基因...")
    source_concepts_map = {}
    source_phase_map = {}

    for phase_file in SOURCE_PHASE_FILES:
        if not os.path.exists(phase_file):
            print(f"⚠️ 找不到源文件：{phase_file}")
            continue

        # 🛡️ 核心修复：无视多长的路径，只提取最后的文件名！
        file_name = os.path.basename(phase_file).lower()

        # 极其严谨的正则化命名
        if 'phase1' in file_name:
            phase_name = 'Phase1'
        elif 'phase2' in file_name:
            phase_name = 'Phase2'
        elif 'phase3' in file_name:
            phase_name = 'Phase3'
        else:
            phase_name = 'Unknown'

        with open(phase_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                    raw_id = data.get("work_id", "")
                    clean_id = str(raw_id).split('/')[-1]
                    concepts = data.get("concepts", [])
                    source_concepts_map[clean_id] = set(concepts)
                    # 贴上极其干净的阶段标签 (Phase1, Phase2, Phase3)
                    source_phase_map[clean_id] = phase_name
                except:
                    pass

    phase_ood_flow = {
        "Phase1": defaultdict(int),
        "Phase2": defaultdict(int),
        "Phase3": defaultdict(int)
    }

    records = []
    print(f"✅ 成功提取了 {len(source_concepts_map)} 篇源论文的动态主阵地。\n")
    print("🧠 正在进行全量比对与加权计算...")

    if not os.path.exists(CITING_DATA_FILE):
        print(f"❌ 找不到粉丝数据：{CITING_DATA_FILE}，请确认它在当前文件夹！")
        return

    with open(CITING_DATA_FILE, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="处理引用数据"):
            if not line.strip(): continue
            try:
                data = json.loads(line)
                raw_id = data.get("work_id", "")
                clean_id = str(raw_id).split('/')[-1]
                disciplines_counts = data.get("citing_disciplines", {})

                if not disciplines_counts or clean_id not in source_concepts_map:
                    continue

                host_concepts = source_concepts_map[clean_id]
                phase = source_phase_map.get(clean_id, "Unknown")

                total_occurrences = 0
                host_count = 0
                ood_count = 0
                ood_disciplines = {}

                for citing_discipline, count in disciplines_counts.items():
                    total_occurrences += count

                    if citing_discipline in host_concepts:
                        host_count += count
                    else:
                        ood_count += count
                        ood_disciplines[citing_discipline] = count

                        if phase in phase_ood_flow:
                            phase_ood_flow[phase][citing_discipline] += count

                if total_occurrences == 0: continue

                dynamic_ood_rate = ood_count / total_occurrences

                entropy = 0
                for count in disciplines_counts.values():
                    p_i = count / total_occurrences
                    if p_i > 0:
                        entropy -= p_i * np.log(p_i)

                records.append({
                    "work_id": clean_id,
                    "stage": phase,  # 这里现在存入的绝对是干干净净的 Phase1, Phase2...
                    "total_occurrences": total_occurrences,
                    "host_count": host_count,
                    "ood_count": ood_count,
                    "dynamic_ood_rate": round(dynamic_ood_rate, 4),
                    "shannon_entropy": round(entropy, 4),
                    "ood_distribution": json.dumps(ood_disciplines, ensure_ascii=False)
                })
            except Exception as e:
                pass

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"\n✨ 全量计算完成！已生成全新的干净宽表：{OUTPUT_CSV}")


if __name__ == "__main__":
    calculate_full_dynamic_diffusion()