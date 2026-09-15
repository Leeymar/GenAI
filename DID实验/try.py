from csdid.att_gt import ATTgt

obj = ATTgt(
    yname='avg_logRCR', tname='year', idname='id_num', gname='g',
    data=raw_cs, control_group='nevertreated', anticipation=0,
    panel=True, allow_unbalanced_panel=True, biters=999, alp=0.05
)
if hasattr(obj, 'fit'):
    obj.fit()

print("=== ATTgt 所有属性 ===")
print([a for a in dir(obj) if not a.startswith('__')])

print("\n=== aggregate('event') 返回对象属性 ===")
try:
    agg = obj.aggregate('event')
    print(type(agg))
    print([a for a in dir(agg) if not a.startswith('__')])
    if hasattr(agg, '__dict__'):
        print("\n__dict__ keys:", list(agg.__dict__.keys()))
except Exception as e:
    print(f"aggregate 报错：{e}")
