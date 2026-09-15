import json
import os


def check_phase3_leakage():
    print("🔍 启动【绝对隔离度测试】：排查 Phase 3 新兵营是否混入老兵...\n")

    # 1. 历史档案库 (只需这两个文件，它们代表了 2022 年之前所有的绝对入局者)
    # ⚠️ 请确保这里的文件名与你本地跑出来的一致
    historical_files = [
        'E:\PythonProject\GenAI\data_scrap_v2\stage_1\phase1_newcomers_scopus_pwfc.jsonl',
        'E:\PythonProject\GenAI\data_scrap_v2\stage_2\phase2_newcomers_scopus_pwfc.jsonl'
    ]

    # 2. 待检查的 Phase 3 目标文件
    phase3_file = 'phase3_newcomers_scopus_pwfc.jsonl'

    # 3. 输出文件
    clean_phase3_file = 'phase3_TRUE_newcomers_scopus_pwfc.jsonl'  # 绝对纯血的 Phase 3 新兵
    leaked_veterans_file = 'phase3_LEAKED_veterans.jsonl'  # 被揪出来的“伪新兵”

    if not os.path.exists(phase3_file):
        print(f"❌ 找不到待检查的文件: {phase3_file}")
        return

    # ---------------------------------------------------------
    # 步骤一：提取全量历史老兵 ID 构建“防火墙”
    # ---------------------------------------------------------
    print("🛡️ 正在构建一、二阶段历史老兵防火墙...")
    historical_ids = set()

    for h_file in historical_files:
        if os.path.exists(h_file):
            with open(h_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        data = json.loads(line)
                        # 注意：兼容有些文件叫 author_id，有些叫 author_ids
                        a_id = data.get('author_ids') or data.get('author_id')
                        if a_id:
                            historical_ids.add(str(a_id).strip())
                    except:
                        pass
        else:
            print(f"   ⚠️ 警告: 找不到历史文件 {h_file}，请确认文件名！")

    print(f"   ➤ 防火墙构建完毕！共录入历史老兵: {len(historical_ids):,} 人。\n")

    # ---------------------------------------------------------
    # 步骤二：扫描 Phase 3 新兵库，揪出内鬼
    # ---------------------------------------------------------
    print("🔬 正在用显微镜逐个排查 Phase 3 新兵库...")

    true_newcomers_count = 0
    leaked_veterans_count = 0

    with open(phase3_file, 'r', encoding='utf-8') as f_in, \
            open(clean_phase3_file, 'w', encoding='utf-8') as f_clean, \
            open(leaked_veterans_file, 'w', encoding='utf-8') as f_leak:

        for line in f_in:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                a_id = str(data.get('author_ids')).strip()

                if a_id in historical_ids:
                    # 抓到内鬼！他其实在 Phase 1 或 Phase 2 就存在了
                    leaked_veterans_count += 1
                    f_leak.write(line)
                else:
                    # 绝对的纯血新兵，放行写入 Clean 文件
                    true_newcomers_count += 1
                    f_clean.write(line)
            except Exception:
                pass

    print("================================================================")
    print(" 🎯 Phase 3 新兵纯度质检终极战报")
    print("================================================================")
    print(f"👶 绝对纯净的 Phase 3 新兵: {true_newcomers_count:>9,} 人")
    print(f"🕵️ 被揪出的混入老兵 (伪新兵): {leaked_veterans_count:>9,} 人")
    print("================================================================\n")

    print(f"💾 【100% 纯血新兵库】已保存至: {clean_phase3_file}")
    if leaked_veterans_count > 0:
        print(f"🗑️ 建议：以后做 Phase 3 统计时，请严格使用 TRUE_newcomers 文件！")
    else:
        print("🎉 完美！你的 Phase 3 库极其纯净，没有混入任何前两个阶段的人！")


# 启动排查！
check_phase3_leakage()