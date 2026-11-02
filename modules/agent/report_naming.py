"""网页报告沿用原始图片名，安全化只影响文件名，不改写来源信息。"""
from pathlib import Path
import re
import hashlib


def safe_stem(name, limit=48):
    value = re.sub(r'[\\/:*?"<>|\x00-\x1f]', '_', Path(name).stem).strip(' .') or 'image'
    if value.upper() in {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1, 10)), *(f'LPT{i}' for i in range(1, 10))}:
        value = 'image_' + value
    if len(value) > limit:
        value = value[:limit - 9] + '_' + hashlib.sha256(value.encode()).hexdigest()[:8]
    return value


def build_report_name(source_names, image_ids, report_format):
    if not source_names or len(source_names) != len(image_ids) or report_format not in {'html', 'markdown'}:
        raise ValueError('报告命名需要匹配的图片清单及支持的格式')
    stems = [safe_stem(name) for name in source_names]
    # 安全化后重名用稳定图片ID区分，正文仍显示完整原始名。
    folded = [stem.casefold() for stem in stems]
    parts = [stem + '_' + image_ids[i] if folded.count(stem.casefold()) > 1 else stem for i, stem in enumerate(stems)]
    originals = [Path(name).stem for name in source_names]
    count = len(parts)
    if count == 1:
        base, display = parts[0] + '_report', originals[0] + ' 灾情分析报告'
    elif count == 2:
        base, display = '_'.join(parts) + '_multi_report', '、'.join(originals) + ' 多图灾情分析报告'
    else:
        base = '_'.join(parts[:2]) + f'_etc_{count}_images_report'
        display = '、'.join(originals[:2]) + f' 等 {count} 张图像综合报告'
    return {'display_name': display, 'filename': base + ('.html' if report_format == 'html' else '.md')}


def build_individual_report_name(source_name, image_id, report_format, peer_names):
    """组内独立报告也按安全化后的名字检测碰撞，兼容Windows大小写语义。"""
    result = build_report_name([source_name], [image_id], report_format)
    if sum(safe_stem(name).casefold() == safe_stem(source_name).casefold() for name in peer_names) > 1:
        result['filename'] = safe_stem(source_name) + '_' + image_id + '_report' + ('.html' if report_format == 'html' else '.md')
    return result
