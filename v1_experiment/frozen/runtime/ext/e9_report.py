"""Collect complete full E9 diagnostics with an explicit Holm family."""
from pathlib import Path
import sys
from common import load, sha, verify_bindings, write_new
from e9_materials import VARIANTS

def report(plan_path):
    plan = load(plan_path); root = Path(plan['extension'])
    verify_bindings(plan['frozen_file_sha256'])
    sys.path.insert(0, str(Path(plan['base']) / 'code'))
    from analyze_v3 import holm
    models, family = [], {}
    bindings = {str(Path(plan_path)): sha(plan_path)}
    for model in plan['models']:
        path = root / 'results/e9_full' / model['slug'] / 'analysis.json'
        record = load(path)
        if (record.get('status') != 'complete_verified' or record['repo_id'] != model['repo_id']
            or record['revision'] != model['revision'] or record['original_E0_tier'] != model['tier']):
            raise ValueError('Missing or mismatched model analysis')
        verify_bindings(record['source_sha256']); bindings[str(path)] = sha(path)
        bindings.update(record['source_sha256'])
        for variant in VARIANTS[1:]:
            for readout in ('R0', 'R1'):
                family['|'.join((model['repo_id'], variant, readout))] = record['paired_template_comparisons'][variant][readout]['centered_two_sided_bootstrap_p']
        models.append(record)
    position = load(plan['position_report'])
    if (position.get('schema') != 'voices_v3_E9_affine_panel_report_v1'
        or position.get('status') != 'complete_requested_diagnostic_panel'
        or position.get('n_verified_models') != len(plan['models'])
        or set(position['requested_models']) != {m['repo_id'] for m in plan['models']}):
        raise ValueError('Position diagnostic does not cover the complete registered panel')
    verify_bindings(position['verified_source_sha256'])
    bindings.update(position['verified_source_sha256'])
    bindings[plan['position_report']] = sha(plan['position_report'])
    result = {'schema': 'voices_v3_E9_full_initial10_report_v1', 'status': 'complete_verified_initial10',
        'models': models, 'n_models': len(models), 'position_diagnostic_report': plan['position_report'],
        'position_report_status': position.get('status'), 'template_C_difference_Holm': holm(family),
        'Holm_family_size': len(family), 'family_scope': 'all ten models x three variants x R0/R1; diagnostic/exploratory',
        'expanded_core_models_complete': False, 'full_v3_goal_complete': False,
        'source_sha256': bindings,
        'limits': ['Fixed previously used 20 items are not a new confirmatory sample.',
                   'R1-only and readout-disagreement qualifications remain in all comparisons.',
                   'R2 is an independently generated categorical first token, never a probability report.',
                   'Undefined bootstrap draws are retained as unresolved intervals.']}
    target = root / 'reports/E9_initial10.json'; write_new(target, result)
    lines = ['# 完整 E9：首批十模型', '',
             '固定的既有 20 题诊断；不作为新的确证样本。全部结果保留原 E0 质量标记。', '',
             '| 模型 | 原读出资格 | 原模板 C（R0） | 原模板 C（R1） |', '| --- | --- | --- | --- |']
    for record in models:
        values = record['variants']['baseline']['readouts']
        lines.append('| ' + ' | '.join([record['repo_id'], record['original_E0_tier'],
            str(values['R0']['point']['C']), str(values['R1']['point']['C'])]) + ' |')
    lines += ['', '三个模板变化的配对区间及全部 60 项 Holm 校正在 JSON 报告中。',
              '数值诊断使用相同 1,464 条输入，比较 batch1 与 batch4、batch16。',
              '槽位诊断沿用已核验的 affine 位置报告。扩展核心模型尚需单独完成。', '']
    markdown = target.with_suffix('.md'); content = '\n'.join(lines)
    if markdown.exists() and markdown.read_text() != content: raise ValueError('Existing report differs')
    if not markdown.exists():
        with markdown.open('x') as f: f.write(content)
    return {'json': str(target), 'sha256': sha(target), 'markdown': str(markdown), 'markdown_sha256': sha(markdown)}
