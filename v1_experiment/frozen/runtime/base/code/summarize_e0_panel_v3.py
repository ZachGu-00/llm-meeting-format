"""Summarize synced E0 progress; never label pilots or E0 as E0-E3 completion."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

OPS = ('option_mass_median', 'first_token_format', 'R2_concordance')
TIERS = ('R0_primary', 'R1_primary_R0_operational', 'R1_only')
DEFAULT_ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def local_path(value, root, remote_root='/workspace/voices_nonadditivity_v3'):
    """Map remote receipts into a synced tree without rewriting those receipts."""
    text = str(value).replace('\\', '/')
    for prefix in (str(remote_root).replace('\\', '/').rstrip('/'), str(root).replace('\\', '/').rstrip('/')):
        if text.startswith(prefix + '/'):
            return root / text[len(prefix) + 1:]
    marker = '/voices_nonadditivity_v3_20261030/'
    if marker in text:
        return root / text.split(marker, 1)[1]
    path = Path(text)
    return path if path.is_absolute() else root / path


def overall(gates):
    statuses = [gates.get(key, {}).get('status', 'incomplete') for key in OPS]
    return 'incomplete' if 'incomplete' in statuses else 'pass' if all(s == 'pass' for s in statuses) else 'failed'


def compact_gates(gates):
    return {k: {x: v[x] for x in ('value', 'status', 'threshold', 'n', 'successes', 'reason') if x in v}
            for k, v in gates.items() if isinstance(v, dict)}


def verify_analysis(path, analysis, root, remote_root, cache):
    missing, mismatch, snapshots = [], [], []
    sources = dict(analysis.get('file_sha256', {}))
    receipt = analysis.get('execution_integrity', {}).get('R0', {})
    suffix = '_E0_conditional_analysis.json' if path.name.endswith('_E0_conditional_analysis.json') else '_E0_analysis.json'
    raw = path.with_name(path.name.replace(suffix, '_R0_R1.jsonl'))
    if receipt.get('status') != 'complete':
        mismatch.append('analysis does not contain a completed R0 run receipt')
    for suffix, key in (('.run.json', 'run_sha256'), ('.summary.json', 'summary_sha256')):
        if receipt.get(key):
            sources[str(raw) + suffix] = receipt[key]
    if not analysis.get('file_sha256'):
        mismatch.append('analysis has no source hashes')
    for source, expected in sources.items():
        p = local_path(source, root, remote_root)
        if not p.is_file():
            missing.append(str(p))
            continue
        stamp = (str(p), p.stat().st_size, p.stat().st_mtime_ns)
        if stamp not in cache:
            cache[stamp] = sha(p)
        if cache[stamp] != expected:
            archived = root / 'provenance/execution_protocol_E0_original.txt'
            if p == root / 'execution_protocol.txt' and archived.is_file() and sha(archived) == expected:
                snapshots.append({'receipt_source': source, 'local_snapshot': str(archived.relative_to(root)),
                                  'sha256': expected, 'reason': 'explicit archived E0 protocol; current protocol later amended'})
                continue
            mismatch.append(str(p))
    return {'status': 'mismatch' if mismatch else 'partial_local_sync' if missing else 'verified_local',
            'missing_local_files': missing, 'mismatched_files': mismatch,
            'source_snapshot_substitutions': snapshots}


def summarize_model(repo, folder, queue_state, root, remote_root, cache):
    result = {'repo_id': repo, 'folder': str(folder.relative_to(root)),
              'status': 'not_started_or_not_synced', 'closed_development100': False,
              'runtime_error': bool(queue_state.get('exit_code', 0)) or
                  queue_state.get('status') == 'execution_error_not_model_disqualification',
              'queue_status': queue_state.get('status')}
    state_path = folder / 'e0_driver_state.json'
    if not state_path.is_file():
        if result['runtime_error']:
            result['status'] = 'execution_error'
        elif queue_state.get('status') == 'checkpoint_unavailable':
            result['status'] = 'checkpoint_unavailable'
        return result
    state = read(state_path)
    result['driver_status'] = state.get('status')
    result['driver_schema'] = state.get('schema')
    result['state_sha256'] = sha(state_path)
    result['initial_queue_runtime_error'] = result['runtime_error']
    if state.get('schema') in ('voices_v3_E0_taskpilot_driver_v1', 'voices_v3_E0_conditional_driver_v1') and state.get('status') == 'complete':
        result['runtime_error'] = False
    result['selection_provenance'] = {k: state.get(k) for k in ('amendment_file', 'amendment_sha256',
        'previous_state_snapshot', 'previous_state_sha256', 'supersedes_selection', 'original_calibration_only_skips') if k in state}
    result['attempts'] = [{k: a[k] for k in ('remedy', 'calibration_q_median',
        'calibration_argmax_format_proxy', 'calibration_screen_pass', 'decision') if k in a}
        for a in state.get('attempts', [])]
    chosen = state.get('selected_analysis')
    remedy = state.get('selected_remedy')
    if not chosen:
        for attempt in reversed(state.get('attempts', [])):
            stage = next((stage for stage in ('development', 'pilot') if attempt.get(stage, {}).get('analysis')), None)
            if stage:
                chosen = attempt[stage]['analysis']; remedy = attempt['remedy']; break
    result.update(remedy=remedy, status='execution_error' if result['runtime_error'] else 'in_progress')
    if not chosen:
        return result
    path = local_path(chosen, root, remote_root)
    if not path.is_file():
        result.update(status='analysis_not_synced', analysis_file=str(path))
        return result
    analysis = read(path)
    integrity = verify_analysis(path, analysis, root, remote_root, cache)
    qualification = analysis.get('qualification', {})
    closed = (state.get('status') == 'complete' and analysis.get('n_items') == 100 and
              analysis.get('bootstrap_draws') == 10000 and qualification.get('development100_complete') is True and
              qualification.get('final_qualification') in TIERS and not result['runtime_error'] and
              integrity['status'] != 'mismatch')
    result.update(analysis_file=str(path.relative_to(root)), analysis_sha256=sha(path), integrity=integrity,
        n_items=analysis.get('n_items'), n_topics=analysis.get('n_topics'), n_rows=analysis.get('n_rows'), closed_development100=closed,
        stage='development100_closed' if closed else 'pilot_only' if analysis.get('n_items') != 100 else 'development100_not_finalized',
        operational_gates=compact_gates(analysis.get('operational_gates', {})),
        strict_operational_gates={k: compact_gates(analysis.get('strict_original_gates', {})).get(k, {}) for k in OPS},
        relaxed_operational_status=qualification.get('R0_operational', overall(analysis.get('operational_gates', {}))),
        prior_relaxed_operational_status=overall(analysis.get('operational_gates', {})),
        operational_policy=analysis.get('operational_policy'),
        conditional_option_mass_admission=qualification.get('conditional_option_mass_admission', False),
        strict_operational_status=overall(analysis.get('strict_original_gates', {})),
        strict_original_diagnostic_gates=compact_gates(analysis.get('strict_original_gates', {})),
        qualification=qualification, applicability=analysis.get('applicability_diagnostics', {}))
    if integrity['status'] == 'mismatch':
        result['status'] = 'integrity_error'
    elif closed:
        result['status'] = 'complete'
    elif not result['runtime_error']:
        result['status'] = 'pilot_closed' if result['stage'] == 'pilot_only' else 'in_progress'
    return result


def build(root, remote_root='/workspace/voices_nonadditivity_v3'):
    root = Path(root).resolve()
    queue_path = root / 'setup/download_queue.json'
    models = [x['repo_id'] for x in read(queue_path)] if queue_path.is_file() else []
    results = root / 'results'
    for folder in sorted(results.glob('*/e0')):
        state_path = folder / 'e0_driver_state.json'
        repo = folder.parent.name.replace('--', '/', 1)
        if state_path.is_file():
            try:
                repo = read(state_path).get('repo_id', repo)
            except (ValueError, OSError):
                pass  # Per-model handling below reports a partially synced state without hiding other models.
        if repo and repo not in models:
            models.append(repo)
    qs = root / 'setup/e0_queue_status.json'
    queue = {r.get('model', r.get('repo_id')): r for r in read(qs)} if qs.is_file() else {}
    cache, rows = {}, []
    for repo in models:
        folder = results / repo.replace('/', '--') / 'e0'
        try:
            rows.append(summarize_model(repo, folder, queue.get(repo, {}), root, remote_root, cache))
        except (ValueError, KeyError, TypeError, OSError) as error:
            rows.append({'repo_id': repo, 'status': 'local_summary_error', 'closed_development100': False,
                         'runtime_error': False, 'error': str(error)})
    closed = [r for r in rows if r['closed_development100']]
    decision_path = root / 'setup/runtime_decision.json'
    decision = read(decision_path) if decision_path.is_file() else {'status': 'not_synced'}
    protocol = root / 'execution_protocol.txt'
    archived = root / 'provenance/execution_protocol_E0_original.txt'
    protocol_text = protocol.read_text(encoding='utf-8-sig') if protocol.is_file() else ''
    marker = '2026-10-31 pre-E1 clarification:'
    amendment = protocol_text[protocol_text.index(marker):].strip() if marker in protocol_text else None
    return {'schema': 'voices_v3_E0_progress_v1', 'title': 'E0 progress / E0 阶段进度',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'experiment_root': str(root),
        'scope': 'Intermediate E0 only. E1/E2/E3 and E4 completion are not assessed here.',
        'counts': {'models_listed': len(rows), 'closed_development100': len(closed),
            'closed_development100_locally_verified': sum(r['integrity']['status'] == 'verified_local' for r in closed),
            'runtime_errors': sum(r.get('runtime_error', False) for r in rows),
            'local_integrity_or_summary_errors': sum(r['status'] in ('integrity_error', 'local_summary_error') for r in rows),
            'closed_R0_operational_failed': sum(r['relaxed_operational_status'] == 'failed' for r in closed),
            'closed_tiers': dict(Counter(r['qualification']['tier'] for r in closed)),
            'statuses': dict(Counter(r['status'] for r in rows))},
        'counting_policy': 'Runtime errors, missing artifacts and pilots are not model instruction failures. '
            'closed_development100 counts completed remote receipts; locally_verified additionally checks all synced source hashes.',
        'runtime_decision': decision,
        'runtime_decision_sha256': sha(decision_path) if decision_path.is_file() else None,
        'protocol_versions': {'current_sha256': sha(protocol) if protocol.is_file() else None,
            'archived_E0_path': str(archived.relative_to(root)) if archived.is_file() else None,
            'archived_E0_sha256': sha(archived) if archived.is_file() else None,
            'pre_E1_clarification': amendment,
            'validation_policy': 'E0 receipts may resolve to the explicit archived E0 protocol only on an exact hash match.'},
        'models': rows, 'summarizer_sha256': sha(__file__)}


def fmt(value):
    return '—' if value is None else f'{value:.3f}' if isinstance(value, (int, float)) else str(value).replace('|', '/')


def markdown(report):
    c = report['counts']; d = report['runtime_decision']
    lines = ['# E0 progress / E0 阶段进度', '',
        '这是 E0 中间汇总，不代表 E0–E3 前置实验或 E4 已完成。', '',
        f"已登记 {c['models_listed']} 个模型；完整 development100 已闭合 {c['closed_development100']} 个"
        f"（本地源文件完整校验 {c['closed_development100_locally_verified']} 个）；运行错误 {c['runtime_errors']} 个；"
        f"本地文件/汇总错误 {c['local_integrity_or_summary_errors']} 个。", '',
        '运行错误与材料未同步单独记录，不计为模型不服从指令；pilot 不计入完整验收。', '',
        '| 模型 | 进度 / n | 补救 | q 中位 | 格式率 | R2 | 原严格 / 当前操作门 | 读数分层 | 适用性标记 |',
        '|---|---|---|---:|---:|---:|---|---|---|']
    for row in report['models']:
        gates = row.get('operational_gates', {}); app = row.get('applicability', {})
        flags = [k + ':' + v['status'] for k, v in app.items() if isinstance(v, dict) and
                 'status' in v and v['status'] != 'pass']
        flags += ['weight_' + k + ':' + v['status'] for k, v in app.get('weight_directions', {}).items()
                  if v.get('status') != 'consistent']
        if row.get('integrity', {}).get('status') == 'partial_local_sync':
            flags.append('本地材料未全部同步')
        if row.get('conditional_option_mass_admission'):
            flags.append('conditional-only q≥.50/格式≥.95/R2≥.90')
        cells = [row['repo_id'], row['status'] + '/' + str(row.get('n_items', '—')), row.get('remedy'),
                 *[gates.get(k, {}).get('value') for k in OPS],
                 row.get('strict_operational_status', '—') + ' / ' + row.get('relaxed_operational_status', '—'),
                 row.get('qualification', {}).get('tier'), '; '.join(flags) or '—']
        lines.append('| ' + ' | '.join(fmt(x) for x in cells) + ' |')
    lines += ['', '原严格操作门：q≥0.90、格式率≥0.95、R2≥0.90；此前放宽门：q≥0.70、格式率≥0.90、R2≥0.85。'
        '新 conditional 分析接受此前放宽门，或 q≥0.50、格式率≥0.95、R2≥0.90；两分支与原门均保留。'
        '尚未生成 conditional 分析的旧状态仍按其原分析报告。'
        '其他原始门保留在 JSON 中，不作为本轮自动淘汰条件。', '',
        'slot、ASI 精度、饱和及读数差异属于适用性诊断；接近零的方向翻转不会自动淘汰模型。', '',
        '运行配置及数值修订：', '',
        f"- 状态：{d.get('status')}；batch={d.get('batch_size', '—')}；attention={d.get('attn_implementation', '—')}；backend={d.get('backend', '—')}。",
        '- ' + d.get('rationale', '运行决策材料尚未同步。'),
        '- 限制：' + d.get('limitations', '尚无可用说明。'), '',
        '逐模型运行错误、未完成状态、补救历史及源文件校验信息见同名 JSON。']
    amendment = report.get('protocol_versions', {}).get('pre_E1_clarification')
    if amendment:
        lines += ['', '协议版本说明：E0 结果依照原始 E0 协议快照校验，原分析文件不变；当前协议在 E1 之前有以下澄清：',
                  '', amendment]
    return '\n'.join(lines) + '\n'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    p.add_argument('--remote-root', default='/workspace/voices_nonadditivity_v3')
    p.add_argument('--output-prefix', type=Path, required=True)
    args = p.parse_args()
    report = build(args.root, args.remote_root)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix, text in (('.json', json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n'),
                         ('.md', markdown(report))):
        path = Path(str(args.output_prefix) + suffix)
        temp = path.with_suffix(path.suffix + '.tmp')
        temp.write_text(text, encoding='utf-8'); temp.replace(path)
    print(json.dumps(report['counts'], ensure_ascii=False))


if __name__ == '__main__':
    main()
