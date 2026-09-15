import pandas as pd

# ── 读取已有的 topic hierarchy ────────────────────────────────
df = pd.read_csv('openalex_topic_hierarchy.csv')
print(f"原始 topic 数量：{len(df)}")

# ── 提取 subfield（Level1） → field（Level2） 唯一映射 ─────────
subfield_to_field = (
    df[['subfield_id', 'subfield_name', 'field_id', 'field_name', 'domain_id', 'domain_name']]
    .drop_duplicates()
    .sort_values(['domain_name', 'field_name', 'subfield_name'])
    .reset_index(drop=True)
)

print(f"\n✅ 共 {len(subfield_to_field)} 个唯一 subfield（Level 1）")
print("\n📊 完整映射表：")
print(subfield_to_field.to_string(index=False))

subfield_to_field.to_csv('subfield_to_field_mapping.csv', index=False)
print("\n✅ 已保存至 subfield_to_field_mapping.csv")

# ── 同时生成简洁的字典，方便后续使用 ─────────────────────────
l1_to_l2 = dict(zip(subfield_to_field['subfield_name'], subfield_to_field['field_name']))
l1_to_domain = dict(zip(subfield_to_field['subfield_name'], subfield_to_field['domain_name']))

print(f"\n📊 各 Field（Level 2）下的 Subfield 数量：")
print(subfield_to_field.groupby('field_name')['subfield_name'].count().sort_values(ascending=False))

print(f"\n📊 各 Domain 下的 Field 数量：")
print(subfield_to_field.groupby('domain_name')['field_name'].nunique().sort_values(ascending=False))
