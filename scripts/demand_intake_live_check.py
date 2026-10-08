# Live check for the demand intake: posts the recorded SD-WAN request to the running procwise and
# prints each field with its MEASURED confidence. Needs the model resident (AgentNick:unified, ~19.6 GiB).
# Run: python3 scripts/demand_intake_live_check.py   (failed=True in <1s means the model never ran)
import json, time, urllib.request
text = ('We need SD-WAN connectivity for 42 UK branch sites, live by 31 March 2027. '
        'Budget is about £240,000 over three years on cost centre CC-4120. Today the '
        'MPLS circuits cost us £95k a year and drop out weekly.')
paths = ['title','cat','problem.cur','problem.want','finance.cc','finance.budget_amount','finance.tco','finance.saving','finance.type',
 'finance.phasing','benefit.target','benefit.baseline','benefit.owner','pillar','alignment','party',
 'intake.go_live','intake.needed_by','intake.existing_contract','intake.po_required','intake.delivery_location','criteria','value']
ctx = {'categories':['IT Services','Facilities','Professional Services'],
       'fields':'\n'.join(p+': text' for p in paths), 'known':'profile.entity', 'asked_field':'',
       'text':text, 'today':'2026-10-07', 'currency':'GBP'}
t=time.time()
r=urllib.request.Request('http://localhost:8000/demand/intake/extract', data=json.dumps({'context':ctx}).encode(),
   headers={'Content-Type':'application/json'})
try:
    d=json.load(urllib.request.urlopen(r, timeout=280))
except urllib.error.HTTPError as e:
    print(e.code, e.read()[:300]); raise SystemExit
print('%.1fs governed=%s failed=%s fields=%d' % (time.time()-t, d.get('governed'), d.get('failed'), len(d.get('fields',{}))))
for k,v in d.get('fields',{}).items(): print(' %-26s %-7s %s' % (k, v['confidence'], json.dumps(v['value'], ensure_ascii=False)[:70]))
