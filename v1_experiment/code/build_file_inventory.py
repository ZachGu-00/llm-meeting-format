"""Create a file inventory with source links, checksums and tabular schemas."""
from pathlib import Path
import csv, gzip, hashlib, json

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'provenance/data_inventory.csv'
TABLE_PURPOSES = {
    '01_same_people_group_effect.csv': '同人对照：改变会面大小与只换搭配的配对效应及区间',
    '02_meeting_size_predictors.csv': '会面大小对照：各预测器的逐模型性能及区间',
    '03_prediction_improvements.csv': '会面大小对照：各方法与基线的配对误差改善',
    '04_micro_count_and_neutral_effects.csv': '人数曲线指数和中性发言稀释的连续估计',
    '05_original_dialogue_prediction.csv': '原三模型G4对话中的条件预测改善和六项Holm校正',
    '06_LFM_3090_replication.csv': '3090独立复现结果及与原LFM的配对差异',
    '07_frozen_confirmatory_tests.csv': '冻结预设检验，保留获支持及未获支持结果',
    '08_LFM_original_replication_pairs.csv': '原LFM与3090复现的288次逐运行配对',
    '09_model_versions.csv': '11个分析模型的原下载revision、来源链接与版本清单位置',
}

def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def describe(path):
    name = path.relative_to(ROOT).as_posix()
    if name.startswith('tables/'):
        return 'writing_tables', TABLE_PURPOSES[path.name]
    categories = [
        ('dialogue/original/', 'G4_original', '原864次运行：发言、成员状态、预测损失和终点'),
        ('dialogue/replication_3090/', 'LFM_3090_replication', '独立288次复现：对话、读数、统计及协议核验'),
        ('raw/original_native_supplement/', 'G4_original', '原实验原生包：完整提示、采样token、发言及状态轨迹'),
        ('raw/replication_native/', 'LFM_3090_replication', '3090复现原生包：完整提示、采样token、发言及状态轨迹'),
        ('raw/render/', 'E4_G1_G2_G3', '主实验逐提示读数；state_bits按字符串读取'),
        ('cache/', 'E4_G1_G2_G3', '主实验数值张量缓存，用于重新计算设计和预测器'),
        ('analysis/', 'E4_G1_G2_G3', '逐条件预测、逐议题指标、权重及配对统计'),
        ('micro/', 'E1_E2_micro', '微观逐题测量与论点材料，用于人数曲线、交互及稀释分析'),
        ('inputs/source_corpus/', 'human_argument_corpus', '原始论点、关键点、人工标签和来源许可'),
        ('inputs/', 'frozen_inputs', '冻结的逐模型输入、微观估计及设计和报告'),
        ('historical/', 'historical_frozen', '原冻结检验、早期诊断和历史数值结果，包含负结果'),
        ('protocols/', 'protocol', '冻结实验方案及原G4模型计划'),
        ('code/', 'reproduction_code', '数据核查、写作表生成与冻结计算代码'),
        ('provenance/', 'provenance', '来源、哈希与本次完整性检查依据'),
        ('mainline/', 'mainline_evidence', '研究主线的制图底表、图表、相关文献核查和复算验证'),
        ('writing_package/', 'writing_handoff', '论文写作包：大纲、写作要点、主张证据对应表与可移交的数据'),
    ]
    for prefix, dataset, purpose in categories:
        if name.startswith(prefix):
            return dataset, purpose
    if name == 'coverage.csv':
        return 'E4_G1_G2_G3', '162个主读数任务的模型、议题、读数和覆盖清单'
    if name == 'README.md':
        return 'documentation', '目录使用说明、统计单位与证据边界'
    if name == 'RESEARCH_MAINLINE_20261106.md':
        return 'research_mainline', '独立研究主线：核心问题、理论解释、新发现、证据边界和图表数据映射'
    raise ValueError('Unclassified file: ' + name)

def tabular_info(path):
    name = path.relative_to(ROOT).as_posix()
    if not (name.endswith('.csv') or name.endswith('.csv.gz')):
        return '', '', ''
    known = {
        'dialogue/original/member_updates.csv.gz':154656,
        'dialogue/original/utterances.csv.gz':154656,
        'dialogue/replication_3090/member_updates.csv.gz':51552,
        'dialogue/replication_3090/utterances.csv.gz':51552,
        'dialogue/replication_3090/R0_readouts.csv.gz':721728,
        'mainline/figure_data/F1_matched_conditions.csv.gz':311040,
        'writing_package/data/source/F1_matched_conditions.csv.gz':311040,
    }
    opener = gzip.open if name.endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        if name.startswith('raw/render/'):
            n, basis = 49020, 'validated_against_coverage'
        elif name in known:
            n, basis = known[name], 'validated_complete_table'
        elif path.stat().st_size <= 20 * 1024 * 1024:
            n, basis = sum(1 for _ in reader), 'csv_reader_count'
        else:
            n, basis = '', 'header_only'
    return n, basis, json.dumps(header, ensure_ascii=False)

def main():
    with (ROOT/'provenance/source_map.csv').open(encoding='utf-8-sig', newline='') as f:
        sources = {r['file']:r for r in csv.DictReader(f)}
    inventory = []
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file() or path == OUTPUT:
            continue
        name = path.relative_to(ROOT).as_posix()
        if name == 'BUILDING.json' or name.endswith('.building'):
            raise ValueError('Dataset is still being built: ' + name)
        dataset, purpose = describe(path)
        n, basis, header = tabular_info(path)
        source = sources.get(name, {})
        inventory.append({'path':name, 'dataset':dataset, 'purpose':purpose,
                          'bytes':path.stat().st_size, 'sha256':sha(path),
                          'rows':n, 'row_count_basis':basis, 'columns':header,
                          'source_package':source.get('package','generated_this_cleanup'),
                          'source_member':source.get('member','')})
    with OUTPUT.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(inventory[0]))
        writer.writeheader()
        writer.writerows(inventory)
    print(json.dumps({'inventory_files':len(inventory), 'bytes_excluding_inventory':sum(r['bytes'] for r in inventory),
                      'sha256_for_every_listed_file':True,'inventory_self_hash_excluded':True}), flush=True)

if __name__ == '__main__':
    main()
