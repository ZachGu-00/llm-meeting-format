"""Write a short Chinese E0/E1/E2 report from verified artifacts; E3 is skipped.

This performs only source/closure verification. It never scores models, reruns
bootstrap analyses, selects models, or evaluates E4 completion. Paired E2 reports
and a frozen E4 selection are accepted only through explicit input paths.
"""
import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path

import compare_e2_v3 as e2
import summarize_e0_panel_v3 as e0
import summarize_select_e1_v3 as e1

ROOT = Path(__file__).resolve().parents[1]
RULES = ('additive', 'present_mean', 'side_mean', 'side_max')
E2_FILES = [*(('E2a_neutral'+str(n), f'e2a_neutral{n}_R0_analysis.json', 'E2a', n) for n in range(4)),
            ('E2b', 'e2b_R0_analysis.json', 'E2b', None), ('E2c', 'e2c_R0_analysis.json', 'E2c', None),
            ('E2d', 'e2d_R0_analysis.json', 'E2d', None), ('E2e', 'e2e_R0_analysis.json', 'E2e', None)]


def verify_sources(report, verifier, required=()):
    bindings = report.get('file_sha256') or report.get('verified_source_sha256')
    if not bindings:
        raise ValueError('Report lacks source hashes')
    mapped = {}
    for path, digest in bindings.items():
        local = verifier.path(path); verifier.check(local, digest); mapped[str(local.resolve())] = digest
    if any(str(Path(path).resolve()) not in mapped for path in required):
        raise ValueError('Report does not bind every required source')
    return mapped


def paired_summary(cohort):
    named = cohort['paired_named_peer_differences']; counts = cohort['paired_count_grid_differences']
    return {
        'neutral0_minus_E1_C': named['E2a_neutral0_minus_E1']['C'],
        'neutral3_minus_neutral0_C': named['E2a_neutral3_minus_neutral0']['C'],
        'present_only_minus_E1_C': named['E2b_minus_E1']['C'],
        'anonymous_minus_E1_C': named['E2c_minus_E1']['C'],
        'neutral3_minus_neutral0_single_opponent_increment': named['E2a_neutral3_minus_neutral0']['count_delta_1_0'],
        'anonymous_minus_E1_single_opponent_increment': named['E2c_minus_E1']['count_delta_1_0'],
        'stance_only_minus_E1_power_t_common_1_to_3': counts['E2d_minus_E1_matched_prefix']['power_t_common_1_to_3'],
        'stance_only_minus_E1_prefix_count_contrast': counts['E2d_minus_E1_matched_prefix']['count_grouping_contrast'],
        'extended_minus_E1_prefix_count_contrast': counts['E2e_minus_E1_matched_prefix']['count_grouping_contrast'],
        'extended_power_t_common_1_to_3': cohort['count_grid_results']['E2e']['metrics']['power_t_common_1_to_3'],
        'extended_power_t_native_1_to_6': cohort['count_grid_results']['E2e']['metrics']['power_t_native_range'],
    }


def summarize_e2(root, repo, e1row, e0_verified, paired_path, verifier):
    result = {'status': 'pending_E0', 'complete_verified': False, 'analyses': {}, 'paired': None}
    if not e0_verified:
        return result
    if e1row.get('R0_operational') is False:
        result['status'] = 'not_applicable_R1_only'; return result
    if e1row.get('R0_operational') is not True:
        result['status'] = 'pending_verified_E1_identity'; return result
    semantic = e1row.get('analyses', {}).get('semantic64_R0_analysis')
    if not semantic:
        result['status'] = 'pending_verified_E1_semantic64'; return result
    core = e0.read(root/'data/core_data_manifest.json'); manifest = e0.read(root/'data/e2_manifest.json')
    for path in (root/'data/core_data_manifest.json', root/'data/e2_manifest.json'):
        verifier.check(path, verifier.digest(path))
    spec = core['outputs']['e1_holdout200_prompts.jsonl']; source = root/spec['path']
    verifier.check(source, spec['sha256'])
    semantic_path = verifier.path(semantic['path']); verifier.check(semantic_path, semantic['sha256'])
    semantic_data, identity = e2.closed_analysis(semantic_path, verifier, source, spec['n_rows'])
    expected_ids = e1.item_ids(root/'data/e1_semantic64_items.json')
    if len(expected_ids) != 64 or set(semantic_data['item_ids']) != expected_ids:
        raise ValueError('E1 matched semantic64 cohort differs')
    all_closed = True; analysis_paths = []
    for name, filename, key, neutral in E2_FILES:
        path = root/'results'/repo.replace('/', '--')/'e2'/filename
        item = {'path': str(path), 'status': 'pending_or_not_synced'}
        try:
            spec = manifest['outputs'][key]; source = root/spec['path']; verifier.check(source, spec['sha256'])
            analysis, current = e2.closed_analysis(path, verifier, source, spec['n_rows'])
            if current != identity or len(analysis['item_ids']) != 64 or set(analysis['item_ids']) != expected_ids:
                raise ValueError('E2 execution or matched64 item set differs from E1')
            if key in ('E2d', 'E2e'):
                expected = set(itertools.product(range(4 if key == 'E2d' else 7), range(3 if key == 'E2d' else 4)))
                if set(map(tuple, analysis['conditions'])) != expected or len(analysis['conditions']) != len(expected):
                    raise ValueError('E2 count grid changed')
            else:
                masks = e2.C_MASKS if neutral is not None else set(range(32))
                if set(analysis['masks']) != masks:
                    raise ValueError('E2 named-mask support changed')
                if neutral is not None and analysis.get('selection', {}).get('filters') != {'neutral_count': neutral}:
                    raise ValueError('E2 neutral-condition selection changed')
            item.update(status='complete_verified', sha256=verifier.digest(path)); analysis_paths.append(path)
        except (OSError, ValueError, KeyError, TypeError) as error:
            all_closed = False; item['error'] = str(error)
            if not isinstance(error, FileNotFoundError): item['status'] = 'invalid_or_incomplete'
        result['analyses'][name] = item
    result['analyses_verified_count'] = len(analysis_paths)
    result['status'] = 'stage_analyses_complete_paired_pending' if all_closed else 'stage_analyses_pending'
    if paired_path is None:
        return result
    result['paired'] = {'path': str(paired_path), 'status': 'pending_source_validation'}
    try:
        paired = e0.read(paired_path)
        if (paired.get('schema') != 'voices_v3_paired_E1_E2_diagnostics_v1'
                or paired.get('status') != 'complete_paired_diagnostics' or paired.get('repo_id') != repo
                or paired.get('readout') != 'R0' or paired['model_identity'] != semantic['execution_identity']
                or paired.get('independent_CI_subtraction_used') is not False):
            raise ValueError('Paired report model/scope or estimation identity differs')
        if not all_closed:
            raise ValueError('Paired report cannot close a stage with missing or invalid raw analyses')
        verify_sources(paired, verifier, [semantic_path, *analysis_paths, Path(e2.__file__).resolve()])
        cohorts = paired['cohorts']; main = cohorts['semantic64_exploratory']
        if set(main['item_ids']) != expected_ids or main['n_items'] != 64:
            raise ValueError('Paired semantic64 cohort differs')
        for cohort in cohorts.values():
            bootstrap = cohort['bootstrap']
            if bootstrap.get('draws') != 10000 or bootstrap.get('unit') != 'topic' or bootstrap.get('paired_all_conditions') is not True:
                raise ValueError('Paired report does not use the planned 10000 topic bootstrap')
        fresh = None
        if repo in e1.MANDATORY:
            data = cohorts['fresh44_sensitivity']; keep = e1.item_ids(root/'data/e1_semantic_remaining44_items.json')
            if len(keep) != 44 or set(data['item_ids']) != keep or data['n_items'] != 44:
                raise ValueError('Paired fresh44 sensitivity cohort differs')
            fresh = paired_summary(data)
        verifier.check(paired_path, verifier.digest(paired_path))
        result.update(status='complete_verified', complete_verified=True, major_comparisons=paired_summary(main),
                      fresh44_sensitivity=fresh, paired={'path': str(paired_path), 'sha256': verifier.digest(paired_path),
                                                       'status': 'complete_verified'})
    except (OSError, ValueError, KeyError, TypeError) as error:
        result['paired'].update(status='invalid_or_incomplete', error=str(error))
    return result


def selection_summary(path, verifier, rows):
    if path is None:
        return {'status': 'not_provided', 'verified_frozen': False, 'selected_repos': []}
    result = {'status': 'invalid_or_incomplete', 'verified_frozen': False, 'path': str(path), 'selected_repos': []}
    try:
        frozen = e0.read(path); repos = frozen['selected_repos']
        if (frozen.get('schema') != 'voices_v3_frozen_E4_model_selection_v1' or frozen.get('status') != 'frozen'
                or frozen.get('performed') is not True or len(repos) != 6 or len(set(repos)) != 6):
            raise ValueError('Not an explicit frozen six-model selection')
        panel_path = verifier.path(frozen['panel_path']); verifier.check(panel_path, frozen['panel_sha256'])
        verifier.check(Path(e1.__file__).resolve(), frozen['selector_sha256'])
        verifier.check(verifier.root/'execution_protocol.txt', frozen['protocol_sha256'])
        panel = e0.read(panel_path)
        if panel.get('all_requested_E1_complete') is not True or any(frozen.get(k) != v for k, v in panel['E4_selection'].items()):
            raise ValueError('Frozen selection differs from its verified E1 panel')
        verify_sources(panel, verifier)
        byrepo = {row['repo_id']: row for row in rows}; panel_rows = {row['repo_id']: row for row in panel['models']}
        for repo in repos:
            current = byrepo[repo]['E1']; original = panel_rows[repo]
            if not current.get('E1_complete') or not current.get('E4_eligible') or current['primary']['sha256'] != original['primary']['sha256']:
                raise ValueError('Selected E1 result is not currently verified: '+repo)
        verifier.check(path, verifier.digest(path))
        result.update(status='frozen_verified', verified_frozen=True, selected_repos=repos, sha256=verifier.digest(path))
    except (OSError, ValueError, KeyError, TypeError) as error:
        result['error'] = str(error)
    return result


def build(root=ROOT, remote_root='/workspace/voices_nonadditivity_v3', models=None, paired_paths=(), selection=None):
    root = Path(root).resolve(); verifier = e1.Verifier(root, remote_root)
    E0 = e0.build(root, remote_root); byrepo = {row['repo_id']: row for row in E0['models']}
    requested = list(models) if models else list(byrepo)
    if not requested or len(requested) != len(set(requested)):
        raise ValueError('A nonempty unique report model scope is required')
    paired = {}; input_errors = []
    for path in paired_paths:
        path = Path(path).resolve()
        try:
            repo = e0.read(path)['repo_id']
            if repo not in requested or repo in paired:
                raise ValueError('Paired report model is outside scope or appears twice')
            paired[repo] = path
        except (OSError, ValueError, KeyError, TypeError) as error:
            input_errors.append({'path': str(path), 'error': str(error)})
    rows = []
    for repo in requested:
        E0row = byrepo.get(repo, {'repo_id': repo, 'status': 'not_found', 'closed_development100': False})
        verified = E0row.get('closed_development100') is True and E0row.get('integrity', {}).get('status') == 'verified_local'
        try:
            E1row = e1.summarize_model(root, repo, E0row, verifier)
        except (OSError, ValueError, KeyError, TypeError) as error:
            E1row = {'repo_id': repo, 'E1_complete': False, 'E4_eligible': False, 'E1_status': 'invalid_or_incomplete', 'error': str(error)}
        try:
            E2row = summarize_e2(root, repo, E1row, verified, paired.get(repo), verifier)
        except (OSError, ValueError, KeyError, TypeError) as error:
            E2row = {'status': 'invalid_or_incomplete', 'complete_verified': False, 'error': str(error)}
        rows.append({'repo_id': repo, 'E0_verified': verified,
                     'conditional_option_mass_admission': bool(E0row.get('qualification', {}).get('conditional_option_mass_admission', False)),
                     'operational_admission_branch': E0row.get('qualification', {}).get('operational_admission_branch'),
                     'E0': E0row, 'E1': E1row, 'E2': E2row})
    counts = {'requested_models': len(rows), 'E0_verified': sum(row['E0_verified'] for row in rows),
              'E1_verified': sum(bool(row['E1'].get('E1_complete')) for row in rows),
              'E2_verified': sum(row['E2']['complete_verified'] for row in rows),
              'E2_not_applicable_R1_only': sum(row['E2']['status'] == 'not_applicable_R1_only' for row in rows)}
    complete = (counts['E0_verified'] == counts['E1_verified'] == len(rows)
                and counts['E2_verified'] > 0
                and counts['E2_verified']+counts['E2_not_applicable_R1_only'] == len(rows) and not input_errors)
    selected = selection_summary(Path(selection).resolve() if selection else None, verifier, rows)
    for path in (Path(__file__).resolve(), Path(e0.__file__).resolve(), Path(e1.__file__).resolve(), Path(e2.__file__).resolve()):
        verifier.check(path, verifier.digest(path))
    manifest = root/'data/e2_manifest.json'; amendments = []
    if manifest.is_file():
        verifier.check(manifest, verifier.digest(manifest)); amendments = e0.read(manifest).get('amendments', [])
    return {'schema': 'voices_v3_prerequisite_report_v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'complete_verified_for_requested_scope' if complete else 'intermediate_pending',
        'scope': 'E0/E1/E2 only; E3 explicitly skipped; no E4 completion claim', 'requested_models': requested,
        'counts': counts, 'models': rows, 'input_errors': input_errors,
        'E3': {'status': 'skipped_by_user', 'counts_as_passed': False, 'reason': 'D2事实的个体影响核按用户指示跳过'},
        'E4_selection': selected, 'E4_completion_assessed': False,
        'H1': e1.h1_family([row['E1'] for row in rows]), 'E0_runtime': E0['runtime_decision'],
        'protocol_versions': E0['protocol_versions'], 'E2_design_amendments': amendments,
        'limitations': [
            'D1输入来自真实人写的ArgKP论点；当前测量的是小模型对受控同伴信息的响应。E3按请求跳过，因此不能声称事实正确率、真人讨论或自由生成对话已获验证，也不能直接推断真实人群因果机制。',
            '操作门对全部模型统一采用原门(q≥0.70、格式率≥0.90、R2一致率≥0.85)或条件门(q≥0.50、格式率≥0.95、R2一致率≥0.90)。仅条件门通过者单独标记；原值与读数分层保留。',
            '执行失败、未同步和源文件不匹配不等于模型不服从指令；条件准入是读数质量规则，不是按E1效应方向选模。',
            'Qwen4/LFM的E1主结论仅用未见180条；匹配读数为44条。E2主诊断为64条探索性比较，另报44条敏感性。',
            'E2配对区间按议题共同重采样；多项消融未做多重比较校正。计数网格对比不等于命名成员C。',
            'E2e采用9个槽位容纳6反对者+3同盟；六次循环不构成完全位置平衡；非正增量的幂指数保留为未定义。',
            'E3已跳过。本报告不评估E4是否完成；前置实验完成本身不等于论文假设或高阶网络效应已成立。'],
        'verified_source_sha256': verifier.verified}


def number(value):
    return '—' if value is None else f'{value:.3g}' if isinstance(value, (int, float)) else str(value).replace('|', '/')


def estimate(value, interval):
    return number(value)+(' ['+', '.join(number(x) for x in interval)+']' if interval is not None else ' [区间未定义]' if value is not None else '')


def metric(value):
    return estimate(value['estimate'], value['interval95']) if value else '—'


def markdown(report):
    c = report['counts']; done = report['status'] == 'complete_verified_for_requested_scope'
    lines = ['# 前置实验简报（v3）', '',
        ('所列范围的 E0/E1/E2 已通过产物闭合与源文件校验。' if done else '这是中间报告，前置实验尚未在所列范围全部闭合。'),
        f"E0 已核验 {c['E0_verified']}/{c['requested_models']}；E1 {c['E1_verified']}/{c['requested_models']}；E2 配对诊断 {c['E2_verified']}/{c['requested_models']-c['E2_not_applicable_R1_only']} 个；另有 {c['E2_not_applicable_R1_only']} 个 R1-only 不适用 E2。",
        'E3（D2 事实个体影响核）按用户要求跳过，未计为通过。', '',
        '| 模型 | E0 核验 | 补救 | q / 格式率 / R2一致率 | 原严格 / 当前操作门 | 读数层级 |',
        '|---|---|---|---|---|---|']
    for row in report['models']:
        r = row['E0']; verified = row['E0_verified']; gates = r.get('operational_gates', {})
        values = ' / '.join(number(gates.get(k, {}).get('value')) for k in e0.OPS) if verified else '—'
        tier = r.get('qualification', {}).get('final_qualification') if verified else '—'
        if verified and row.get('conditional_option_mass_admission'): tier = str(tier)+'（仅条件门通过）'
        cells = [row['repo_id'], '完成' if verified else '待核验：'+r.get('status', 'unknown'),
                 r.get('remedy'), values,
                 r.get('strict_operational_status', '—')+' / '+r.get('relaxed_operational_status', '—'),
                 tier]
        lines.append('| '+' | '.join(number(x) for x in cells)+' |')
    lines += ['', '操作门仅用于读数可用性；R1-primary/R0-operational 保留 R0 资格并标记，R1-only 只作探索性读数。', '',
        '| 模型 | E1 范围/状态 | C [95%区间] | 加和R² | 全体均值R² | 阵营均值R² | 阵营极值R² |',
        '|---|---|---|---:|---:|---:|---:|']
    for row in report['models']:
        r = row['E1']; primary = r.get('primary') or r.get('R1_only_exploratory') or {}
        verified = bool(primary)  # Each retained primary already passed the E1 source verifier.
        scope = f"{primary.get('readout')}/{primary.get('n_items')}" if verified and primary else '待核验：'+r.get('E1_status', 'unknown')
        if verified and not r.get('E1_complete'): scope += '（其他分析待闭合）'
        rules = primary.get('rules', {}) if verified else {}
        cells = [row['repo_id'], scope, estimate(primary.get('C'), primary.get('C_interval95')) if verified else '—',
                 *[rules.get(rule, {}).get('R2_within_item') for rule in RULES]]
        lines.append('| '+' | '.join(number(x) for x in cells)+' |')
    lines += ['', 'C 是归一化二阶对比，区间跨零时方向未稳定区分；四规则均使用实测单人效应。表中 R² 为题内 R²，负值原样保留；阵营极值取各阵营绝对值最大的有符号单人效应。', '',
        '| 模型 | E2 状态 | 5槽无中性背景−E1 ΔC | 加3中性−0中性 ΔC | 仅出现者−E1 ΔC | 去身份−E1 ΔC |',
        '|---|---|---|---|---|---|']
    for row in report['models']:
        r = row['E2']; m = r.get('major_comparisons', {})
        cells = [row['repo_id'], r['status'], *[metric(m.get(key)) for key in (
            'neutral0_minus_E1_C', 'neutral3_minus_neutral0_C', 'present_only_minus_E1_C', 'anonymous_minus_E1_C')]]
        lines.append('| '+' | '.join(number(x) for x in cells)+' |')
    lines += ['', 'E2a 固定5个槽位、10种排列；每个中性人数条件使用16个空集合/单人/双人子集，中性评论替换未发言槽位。']
    ready = [row for row in report['models'] if row['E2']['complete_verified']]
    if ready:
        lines += ['', '| 模型 | 加3中性−0中性：单反对者增量差 | 去作者标记−E1：单反对者增量差 | 纯立场−E1：Δt（共同1–3人） |',
                  '|---|---|---|---|']
        for row in ready:
            m = row['E2']['major_comparisons']
            lines.append('| '+row['repo_id']+' | '+' | '.join(metric(m[key]) for key in (
                'neutral3_minus_neutral0_single_opponent_increment', 'anonymous_minus_E1_single_opponent_increment',
                'stance_only_minus_E1_power_t_common_1_to_3'))+' |')
        lines += ['', '单反对者增量指无同盟时，单名反对者相对同一条件空集合基线的 R0 对数几率读数增量，并在三名反对者间取均值；表中为该增量的条件间配对差，不是概率差。C 近似不变不代表影响强度不变；ΔC 区间跨零也不等于已证实 C 不变。此表统一展示所有 E2 已完成模型的现成探索性配对估计和95%区间，未重估或作多重比较校正。', '',
                  '| 模型 | 仅立场−E1 计数对比 | 扩展−E1 计数对比 | 扩展t（1–3人） | 扩展t（1–6人） |',
                  '|---|---|---|---|---|']
        for row in ready:
            m = row['E2']['major_comparisons']
            lines.append('| '+row['repo_id']+' | '+' | '.join(metric(m[key]) for key in (
                'stance_only_minus_E1_prefix_count_contrast', 'extended_minus_E1_prefix_count_contrast',
                'extended_power_t_common_1_to_3', 'extended_power_t_native_1_to_6'))+' |')
        lines += ['', 'E2 均为点估计 [配对95%区间]，主诊断使用相同64条；计数对比匹配相同原始成员前缀，不能称为命名成员 C。']
        for row in ready:
            fresh = row['E2'].get('fresh44_sensitivity')
            if fresh:
                lines.append(f"{row['repo_id']} 未见44条：中性增量 ΔC={metric(fresh['neutral3_minus_neutral0_C'])}；仅出现者 ΔC={metric(fresh['present_only_minus_E1_C'])}；去身份 ΔC={metric(fresh['anonymous_minus_E1_C'])}。")
                lines.append(f"同一未见44条的影响强度：加3中性−0中性单反对者增量差={metric(fresh['neutral3_minus_neutral0_single_opponent_increment'])}；去作者标记−E1单反对者增量差={metric(fresh['anonymous_minus_E1_single_opponent_increment'])}；纯立场−E1 Δt（共同1–3人）={metric(fresh['stance_only_minus_E1_power_t_common_1_to_3'])}。")
    selected = report['E4_selection']
    lines += ['', 'E4 六模型冻结名单：'+('、'.join(selected['selected_repos']) if selected['verified_frozen'] else '尚未提供或未通过当前文件校验；本报告不选模。'), '',
              '偏离与限制：', '']
    lines += ['- '+text for text in report['limitations']]
    lines += ['', '逐模型缺失/错误、配对报告路径、协议版本、E2设计修订和校验 SHA 见同名 JSON。']
    return '\n'.join(lines)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--remote-root', default='/workspace/voices_nonadditivity_v3')
    parser.add_argument('--models', nargs='+')
    parser.add_argument('--e2-report', type=Path, action='append', default=[])
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--output-prefix', required=True, type=Path)
    args = parser.parse_args()
    paths = [Path(str(args.output_prefix)+suffix) for suffix in ('.json', '.md')]
    if any(path.exists() for path in paths):
        raise FileExistsError('Use a new report output prefix; existing reports are retained')
    report = build(args.root, args.remote_root, args.models, args.e2_report, args.selection)
    paths[0].parent.mkdir(parents=True, exist_ok=True)
    for path, text in zip(paths, [json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', markdown(report)]):
        with path.open('x', encoding='utf-8') as stream: stream.write(text)
    print(json.dumps({'status': report['status'], 'counts': report['counts'], 'markdown': str(paths[1])}, ensure_ascii=False))


if __name__ == '__main__':
    main()
