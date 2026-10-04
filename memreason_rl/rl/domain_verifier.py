def domain_reward(citations,dictionary):
    # Unverified domain assignments cannot furnish a supposedly verifiable reward.
    applicable=[c for c in citations if c.get('item') in dictionary and dictionary[c['item']]['verified_from_dictionary']]
    if not applicable:
        return None
    return sum(c.get('domain')==dictionary[c['item']]['domain'] for c in applicable)/len(applicable)
