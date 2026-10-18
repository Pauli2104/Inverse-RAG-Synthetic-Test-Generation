"""Check the reference answers of the native Ragas suite against the hidden fact table.

For every Ragas test case whose question names a room and an attribute, the reference answer is compared with
the current value in llmstudy/ground_truth.json; a wrong reference that matches the value in a superseded
archive manual is reported as "stale" (Sect. 5.9 of the article).

Usage:  python -m llmstudy.check_ragas_references [suite.jsonl]     (default: llmstudy/native_suites/ragas.jsonl)
"""
import json, re, sys, glob, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from llmstudy.study import same_answer, norm
SUITE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "native_suites", "ragas.jsonl")
truth=json.load(open(os.path.join(HERE, 'ground_truth.json'), encoding='utf-8'))['facts']
arch={}
for p in glob.glob(os.path.join(HERE, 'facility_docs', 'archive_*.md')):
    b=os.path.basename(p).split('_')[1].capitalize()
    for line in open(p,encoding='utf-8'):
        c=[x.strip() for x in line.strip().strip('|').split('|')]
        if len(c)==3 and re.match(r'(Hall|Lab|Studio|Lounge) ',c[0]):
            arch[f"{b}|{c[0]}|occupancy"]=c[1]; arch[f"{b}|{c[0]}|co2"]=c[2]
PAT=[(r'occupan|headcount|capacity|people allowed','occupancy'),(r'\bco[₂2]\b|alarm threshold','co2'),(r'set-?point|temperature','setpoint'),(r'badge|access level|colou?r','access'),(r'maintenan|serviced|maintained','maintenance'),(r'escalat|handles incidents|responsible for incidents','escalation')]
subjects=sorted({k.split('|')[1] for k in truth}, key=len, reverse=True)
blds=sorted({k.split('|')[0] for k in truth})
rows=[json.loads(l) for l in open(SUITE,encoding='utf-8')]
res=[]
for r in rows:
    q=r['query']; ref=str(r['ground_truth']); ql=q.lower()
    subs=[s for s in subjects if re.search(r'\b'+re.escape(s.lower())+r'\b', ql)]
    subs=[s for s in subs if not any(s!=t and s.lower() in t.lower() for t in subs)]
    bs=[b for b in blds if b.lower() in ql]
    fts=[ft for pat,ft in PAT if re.search(pat, ql)]
    if not subs or not fts: res.append((r['id'],'unchecked',r['subtype'][:20])); continue
    keys=[]
    for s in subs:
        for ft in fts:
            c=[k for k in truth if k.split('|')[1]==s and k.endswith('|'+ft) and (not bs or k.split('|')[0] in bs)]
            if len(c)==1: keys.append(c[0])
    if not keys: res.append((r['id'],'unchecked',r['subtype'][:20])); continue
    oks=[(norm(truth[k]) in norm(ref) or same_answer(truth[k], ref)) for k in keys]
    stale=any((not ok) and k in arch and norm(arch[k]) in norm(ref) for k,ok in zip(keys,oks))
    res.append((r['id'],'valid' if all(oks) else ('stale' if stale else 'wrong'),r['subtype'][:20], [(k,truth[k],arch.get(k)) for k,ok in zip(keys,oks) if not ok], ref[:100]))
from collections import Counter
print(Counter(x[1] for x in res))
for x in res:
    if x[1] in ('stale','wrong'): print('  ',x)
