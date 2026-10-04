import math

def verify_evidence(citations,event_store,subject_id,cutoff,tolerance=1e-5):
    results=[]
    for citation in citations:
        event=event_store.get(citation.get('event_id'))
        key=citation.get('item')
        valid=bool(event and event['subject_id']==subject_id and event['time_month']<=cutoff and key in event['payload'])
        value_ok=False
        time_ok=False
        if valid:
            try:
                value_ok=math.isclose(float(event['payload'][key]),float(citation.get('value')),abs_tol=tolerance)
                time_ok=math.isclose(float(event['time_month']),float(citation.get('time_month')),abs_tol=tolerance)
            except (ValueError,TypeError):
                pass
        results.append({'exists_in_history':valid,'value_correct':value_ok,'time_correct':time_ok})
    if not results:
        return 0.,results
    score=sum(.5*r['exists_in_history']+.3*r['value_correct']+.2*r['time_correct'] for r in results)/len(results)
    return score,results
