"""Review physical episode logs, including unfinished cells, without refitting."""
import csv
import json
import math
from pathlib import Path


def summarize(root):
    out = root / 'runs/physical-evaluation'
    table = []
    episodes = []
    for path in sorted(out.glob('*/episodes.jsonl')):
        method, family, condition = path.parent.name.split('-', 2)
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not rows:
            continue
        episodes.extend(dict(r, method=method, family=family, condition=condition) for r in rows)
        n = len(rows)
        successes = sum(r['success'] for r in rows)
        p, z = successes / n, 1.96
        centre = (p + z*z/(2*n)) / (1 + z*z/n)
        radius = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
        table.append(dict(
            method=method, family=family, condition=condition,
            successes=successes, trials=n, wilson_low=round(centre-radius, 4),
            wilson_high=round(centre+radius, 4),
            correct_unavailable=sum(r['reason']=='required_part_unavailable' and r['mode']=='unavailable' for r in rows),
            false_completion=sum(r['false_completion'] for r in rows),
            duplicate=sum(r['duplicate'] for r in rows),
            retries=sum(r['retries'] for r in rows),
            controller_failures=sum(r['controller_failures'] for r in rows),
            perception_verification_errors=sum(r['perception_verification_errors'] for r in rows),
            stops=sum(r['stops'] for r in rows),
            unsafe_continuation=sum(r['unsafe_continuation'] for r in rows),
            resets=sum(r['resets'] for r in rows),
            assistance=sum(r['assistance'] for r in rows),
            time_limits=sum(r['reason']=='time_limit' for r in rows),
            mean_simulated_seconds=round(sum(r['simulated_seconds'] for r in rows)/n, 2),
            mean_wall_seconds=round(sum(r['wall_seconds'] for r in rows)/n, 2),
        ))
    if table:
        with (out/'review-summary.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=table[0])
            writer.writeheader()
            writer.writerows(table)
        failure_groups = {}
        for row in episodes:
            key = '/'.join(row[k] for k in ('method', 'family', 'condition', 'mode'))
            group = failure_groups.setdefault(key, dict(trials=0, successes=0, false_completion=0,
                                                       controller_failures=0, perception_verification_errors=0))
            group['trials'] += 1
            group['successes'] += row['success']
            for field in ('false_completion', 'controller_failures', 'perception_verification_errors'):
                group[field] += row[field]
        (out/'review-by-failure-mode.json').write_text(json.dumps(failure_groups, indent=2))
        reference = {(r['family'], r['condition'], r['seed']): r for r in episodes if r['method']=='rule'}
        pairs = {}
        for row in episodes:
            if row['method']=='rule':
                continue
            ref = reference.get((row['family'], row['condition'], row['seed']))
            if ref is None:
                continue
            group = pairs.setdefault(row['method'], dict(wins=0, losses=0, ties=0))
            label = 'ties' if row['success']==ref['success'] else ('wins' if row['success'] else 'losses')
            group[label] += 1
        (out/'review-paired-vs-rule.json').write_text(json.dumps(pairs, indent=2))
    return table


def integration_summary(root):
    path = root/'runs/physical-heldout-integration-pilot/episodes.jsonl'
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    result = dict(trials=len(rows), task_successes=sum(r['success'] for r in rows),
                  correct_incomplete=sum(r['reason']=='required_part_unavailable' and r['mode']=='unavailable' for r in rows),
                  incorrect_unavailable=sum(r['reason']=='required_part_unavailable' and r['mode']!='unavailable' for r in rows),
                  time_limits=sum(r['reason']=='time_limit' for r in rows),
                  all_attempts_lift_verified=sum(r['success'] and r['verified_grasps'] for r in rows))
    (path.parent/'review-summary.json').write_text(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    table = summarize(root)
    integration_summary(root)
    print(f'{sum(r["trials"] for r in table)} logged physical episodes; unfinished cells included')
