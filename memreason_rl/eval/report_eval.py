from ..models.reasoning_llm import parse_report
from ..rl.evidence_verifier import verify_evidence
from ..rl.domain_verifier import domain_reward

def evaluate_report(text,event_store,dictionary,subject_id,cutoff):
    try:
        report=parse_report(text)
    except (ValueError,TypeError):
        return {'format_valid':False,'evidence_precision':0.,'numeric_citation_accuracy':0.,'hallucinated_evidence_rate':1.}
    score,details=verify_evidence(report['evidence'],event_store,subject_id,cutoff)
    n=len(details)
    exists=sum(x['exists_in_history'] for x in details)/n if n else 0.
    return dict(format_valid=True,evidence_score=score,evidence_precision=exists,
        numeric_citation_accuracy=sum(x['value_correct'] for x in details)/n if n else 0.,
        hallucinated_evidence_rate=1-exists if n else 0.,domain_accuracy=domain_reward(report['evidence'],dictionary),
        evidence_recall=None,recall_note='Requires a prespecified relevant-evidence reference set; not all history is necessarily relevant.')
