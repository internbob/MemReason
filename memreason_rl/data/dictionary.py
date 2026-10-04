"""Conservative item catalog. Candidate semantics are NOT dictionary verification.

Unknown encodings remain excluded from confirmatory runs. RAVLT '*Int' fields
must not be silently reinterpreted as word recall scores.
"""
import re
import yaml
from .common import ROOT, DOMAINS

TOTALS = {'adas_totscore', 'adas_total13', 'mmse_mmscore', 'moca_moca', 'cdr_cdglobal', 'cdr_cdrsb'}

def build_dictionary(columns):
    items = {}
    for c in columns:
        if c in {'subject_id', 'rid', 'visit_month', 'visit_code', 'scale_source_count'} or c.endswith(('_visit_date', '_userdate', '_present')):
            continue
        inst = c.split('_')[0].upper()
        if c.startswith('LogMem'):
            inst = 'LOGMEM'
        e = dict(instrument=inst, domain='Unverified', construct='Unverified',
                 task_description=f'Unverified source variable {c}; do not infer task or coding.',
                 value_type='continuous', range=None, valid_values=None, direction='verify',
                 core_memory=False, use_for_state=False, use_for_aux=False,
                 use_for_outcome=False, verified_from_dictionary=False,
                 source=None, verification_note='Requires source dictionary and phase harmonization.',
                 missing_codes=[-4, -1, 999, 9999], default_drop=False)
        if c.startswith('dxsum_'):
            e.update(domain='ClinicalOutcome', construct='ClinicalOutcome', use_for_outcome=True, value_type='categorical')
        elif c.startswith('faq_') or c in {'cdr_cdcommun', 'cdr_cdhome', 'cdr_cdcare'}:
            e.update(domain='Function', construct='DailyFunction', use_for_outcome=True, range=[0, 30] if c.endswith('total') else [0, 3])
        elif c in TOTALS:
            bounds = {'adas_totscore':[0,70], 'adas_total13':[0,85], 'mmse_mmscore':[0,30], 'moca_moca':[0,30], 'cdr_cdglobal':[0,3], 'cdr_cdrsb':[0,18]}
            e.update(domain='GlobalOutcome', construct='GlobalScore', use_for_outcome=True, range=bounds[c])
        elif c.startswith('adas_q'):
            q = int(re.search(r'q(\d+)', c)[1])
            mapping = {1:('Memory','WordRecall',10), 2:('Language','Commands',5), 3:('Executive','ConstructionalPraxis',5), 4:('Memory','DelayedRecall',10), 5:('Language','Naming',5), 6:('Executive','IdeationalPraxis',5), 7:('Orientation','Orientation',8), 8:('Memory','WordRecognition',12), 9:('Memory','RememberingInstructions',5)}
            if q in mapping and not c.endswith('_cue'):
                d, construct, hi = mapping[q]
                e.update(domain=d, construct=construct, range=[0, hi], direction='higher=worse', core_memory=q in [1,4,8])
            # q10/q12 conflict across descriptions and q13 harmonization are deliberately pending.
            if q == 13:
                e.update(domain='Attention', construct='NumberCancellation', range=[0,5], direction='higher=worse', source='https://adni.loni.usc.edu/wp-content/uploads/2008/07/adas-number-cancellation-instructions.pdf', verification_note='Task documented; merged CSV phase-specific coding not verified. Excluded from final main experiment.')
        elif c.startswith('mmse_'):
            k = c[5:]
            if k.startswith('mmltr'):
                e.update(default_drop=True, value_type='categorical')
            elif k.startswith('word'):
                e.update(domain='Memory', construct='DelayedRecall' if k.endswith('dl') else 'Registration', valid_values=[0,1], direction='higher=better', value_type='binary', core_memory=True)
            elif k == 'mmtrials':
                e.update(domain='Memory', construct='LearningTrials', range=[1,6], direction='higher=worse')
            elif k in ['mmyear','mmmonth','mmday','mmseason','mmhospit','mmfloor','mmcity','mmarea','mmstate']:
                e.update(domain='Orientation', construct='Orientation', valid_values=[0,1], direction='higher=better', value_type='binary')
            elif k in ['mmd','mml','mmr','mmo','mmw','worldscore']:
                e.update(domain='Attention', construct='SpellingBackward', range=[0,5] if k=='worldscore' else [0,1], direction='higher=better')
            else:
                e.update(domain='Language', construct='LanguagePraxis', valid_values=[0,1], direction='higher=better', value_type='binary')
        elif c.startswith('moca_'):
            k = c[5:]
            if k.startswith('delw'):
                e.update(domain='Memory', construct='DelayedRecallCueLevel', value_type='categorical', valid_values=[0,1,2,3], core_memory=True, verification_note='Raw 0/1/2/3 codes; ordinal order and response labels are unverified. No binary conversion.')
            elif k in ['trails','cube','clockcon','clockno','clockhan','abstran','absmeas']:
                e.update(domain='Executive', construct='ExecutiveVisuospatial', valid_values=[0,1], value_type='binary', direction='higher=better')
            elif k in ['date','month','year','day','place','city']:
                e.update(domain='Orientation', construct='Orientation', valid_values=[0,1], value_type='binary', direction='higher=better')
            elif k.startswith('serial'):
                e.update(domain='Attention', construct='SerialSubtractionRawResponse', range=None, direction='verify')
            elif k == 'letters':
                e.update(domain='Attention', construct='LetterVigilanceRawResponse', range=None, direction='verify', verification_note='Observed values exceed binary range; raw response/count coding requires dictionary verification.')
            elif k in ['digfor','digback']:
                e.update(domain='Attention', construct='AttentionWorkingMemory', valid_values=[0,1], value_type='binary', direction='higher=better')
            else:
                e.update(domain='Language', construct='Language', range=[0,60] if k=='ffluency' else [0,1], direction='higher=better')
        elif c == 'cdr_cdmemory':
            e.update(domain='Memory', construct='InformantMemoryRating', valid_values=[0,0.5,1,2,3], value_type='ordinal', direction='higher=worse', core_memory=True)
        elif c in ['cdr_cdorient','cdr_cdjudge']:
            e.update(domain='Orientation' if c.endswith('cdorient') else 'Executive', construct='InformantRating', valid_values=[0,0.5,1,2,3], value_type='ordinal', direction='higher=worse')
        elif c.startswith('AVLT_'):
            e.update(domain='Memory', construct='UnverifiedAVLT', range=[0,100], verification_note='Int may encode intrusions, not correct recall. Source ITEM dictionary needed.', default_drop=True)
        elif c.startswith('LogMem'):
            e.update(domain='Memory', construct='DelayedRecall' if 'Delayed' in c else 'ImmediateRecall', range=[0,25], core_memory=True)
            if 'Reminder' in c:
                e.update(construct='ReminderMetadata', domain='Protocol', core_memory=False, default_drop=True, range=None)
        if e['domain'] in DOMAINS and not e['default_drop']:
            e['use_for_state'] = True
            e['use_for_aux'] = e['domain'] != 'Memory'
            e['task_description'] = f"Candidate task: {e['construct']} ({inst}); raw code for {c}. Pending dictionary verification."
        items[c] = e
    return items

def load_dictionary(path=None):
    return yaml.safe_load((ROOT / 'configs/cognitive_item_dictionary.yaml' if path is None else path).read_text())['items']

def state_items(dictionary, exploratory=False):
    return [k for k,e in dictionary.items() if e['use_for_state'] and not e['default_drop'] and (exploratory or e['verified_from_dictionary'])]
