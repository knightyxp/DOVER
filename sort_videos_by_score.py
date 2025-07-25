#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
from pathlib import Path
import re

def parse_csv_file(csv_path):
    """
    解析CSV文件，返回视频路径到分数的映射
    """
    score_dict = {}
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 跳过标题行
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        
        # 分割CSV行，注意处理路径中可能包含逗号的情况
        parts = [part.strip() for part in line.split(',')]
        if len(parts) >= 4:
            video_path = parts[0]
            try:
                final_score = float(parts[3])
                # 使用文件名作为键
                filename = os.path.basename(video_path)
                score_dict[filename] = final_score
            except ValueError:
                print(f"无法解析分数: {line}")
                continue
    
    return score_dict

def extract_video_info(filename):
    """
    从文件名中提取视频信息
    例如: 0001_source_052b2df0812ccd0e8a0da839683d2032_org.mp4 -> (0001, source, 052b2df0812ccd0e8a0da839683d2032)
    """
    # 匹配格式：数字_类型_ID_后缀.mp4
    pattern = r'(\d+)_(source|target)_([a-f0-9]+)(?:_org)?\.mp4'
    match = re.match(pattern, filename)
    
    if match:
        number = match.group(1)
        video_type = match.group(2)
        video_id = match.group(3)
        return number, video_type, video_id
    
    return None, None, None

def scan_video_directory(video_dir):
    """
    扫描视频目录，返回所有视频文件
    """
    video_files = []
    for filename in os.listdir(video_dir):
        if filename.endswith('.mp4'):
            video_files.append(filename)
    
    return sorted(video_files)

def group_video_pairs(video_files, score_dict):
    """
    将视频文件按source/target配对分组
    """
    pairs = {}
    
    for filename in video_files:
        number, video_type, video_id = extract_video_info(filename)
        
        if number and video_type and video_id:
            pair_key = f"{number}_{video_id}"
            
            if pair_key not in pairs:
                pairs[pair_key] = {
                    'number': number,
                    'video_id': video_id,
                    'source_file': None,
                    'target_file': None,
                    'source_score': 0.0,
                    'target_score': 0.0
                }
            
            pairs[pair_key][f'{video_type}_file'] = filename
            
            # 从分数字典中获取分数
            if filename in score_dict:
                pairs[pair_key][f'{video_type}_score'] = score_dict[filename]
            else:
                print(f"警告: 未找到文件 {filename} 的分数")
    
    return pairs

def sort_and_copy_video_pairs(pairs, source_dir, output_dir):
    """
    按总分排序并复制视频对
    """
    # 计算总分并过滤完整的配对
    valid_pairs = []
    
    for pair_key, pair_data in pairs.items():
        if pair_data['source_file'] and pair_data['target_file']:
            total_score = pair_data['source_score'] + pair_data['target_score']
            pair_data['total_score'] = total_score
            valid_pairs.append(pair_data)
        else:
            print(f"警告: 配对不完整 {pair_key} - Source: {pair_data['source_file']}, Target: {pair_data['target_file']}")
    
    # 按总分降序排序
    valid_pairs.sort(key=lambda x: x['total_score'], reverse=True)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 复制文件
    copied_pairs = []
    
    print(f"开始复制 {len(valid_pairs)} 对视频...")
    
    for rank, pair in enumerate(valid_pairs, 1):
        source_file = pair['source_file']
        target_file = pair['target_file']
        
        source_path = os.path.join(source_dir, source_file)
        target_path = os.path.join(source_dir, target_file)
        
        # 生成新的文件名
        source_ext = Path(source_file).suffix
        target_ext = Path(target_file).suffix
        
        new_source_name = f"{rank:04d}_source_{pair['video_id']}_score{pair['source_score']:.2f}{source_ext}"
        new_target_name = f"{rank:04d}_target_{pair['video_id']}_score{pair['target_score']:.2f}{target_ext}"
        
        source_dest = os.path.join(output_dir, new_source_name)
        target_dest = os.path.join(output_dir, new_target_name)
        
        # 复制文件
        success = True
        
        try:
            if os.path.exists(source_path):
                shutil.copy2(source_path, source_dest)
                print(f"已复制 source: {source_file} -> {new_source_name}")
            else:
                print(f"源文件不存在: {source_path}")
                success = False
        except Exception as e:
            print(f"复制源文件失败: {source_path} - 错误: {e}")
            success = False
        
        try:
            if os.path.exists(target_path):
                shutil.copy2(target_path, target_dest)
                print(f"已复制 target: {target_file} -> {new_target_name}")
            else:
                print(f"目标文件不存在: {target_path}")
                success = False
        except Exception as e:
            print(f"复制目标文件失败: {target_path} - 错误: {e}")
            success = False
        
        if success:
            copied_pairs.append({
                'rank': rank,
                'video_id': pair['video_id'],
                'source_file': new_source_name,
                'target_file': new_target_name,
                'source_score': pair['source_score'],
                'target_score': pair['target_score'],
                'total_score': pair['total_score']
            })
    
    return copied_pairs

def save_results(copied_pairs, output_dir):
    """
    保存排序结果到文件
    """
    result_file = os.path.join(output_dir, "sorted_video_pairs_ranking.txt")
    
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write("排名\t视频ID\t源视频分数\t目标视频分数\t总分\t源视频文件\t目标视频文件\n")
        f.write("-" * 100 + "\n")
        
        for pair in copied_pairs:
            f.write(f"{pair['rank']}\t{pair['video_id']}\t{pair['source_score']:.2f}\t{pair['target_score']:.2f}\t{pair['total_score']:.2f}\t{pair['source_file']}\t{pair['target_file']}\n")
    
    print(f"排序结果已保存到: {result_file}")

def main():
    # 配置路径
    csv_file = "/projects/D2DCRC/xiangpeng/DOVER/obj_removal_1k_dove_score.csv"
    source_directory = "/projects/D2DCRC/xiangpeng/Senorita/obj_removal_videos_vie_top1k"
    output_directory = "/projects/D2DCRC/xiangpeng/Senorita/sorted_obj_removal_video_pairs_by_dover_score"
    
    # 检查输入文件和目录
    if not os.path.exists(csv_file):
        print(f"错误: CSV文件不存在: {csv_file}")
        return
    
    if not os.path.exists(source_directory):
        print(f"错误: 源目录不存在: {source_directory}")
        return
    
    print("步骤1: 解析CSV文件...")
    score_dict = parse_csv_file(csv_file)
    print(f"从CSV中读取了 {len(score_dict)} 个视频的分数")
    
    print("步骤2: 扫描视频目录...")
    video_files = scan_video_directory(source_directory)
    print(f"在目录中找到 {len(video_files)} 个视频文件")
    
    print("步骤3: 配对视频...")
    pairs = group_video_pairs(video_files, score_dict)
    print(f"识别出 {len(pairs)} 对视频")
    
    print("步骤4: 排序并复制视频...")
    copied_pairs = sort_and_copy_video_pairs(pairs, source_directory, output_directory)
    
    print("步骤5: 保存结果...")
    save_results(copied_pairs, output_directory)
    
    print(f"\n排序完成！")
    print(f"共处理了 {len(copied_pairs)} 对视频文件")
    print(f"视频文件已保存到: {output_directory}")
    
    # 显示前10名
    print(f"\n前10名视频对:")
    for i, pair in enumerate(copied_pairs[:10]):
        print(f"{i+1:2d}. 总分: {pair['total_score']:7.2f} (Source: {pair['source_score']:.2f} + Target: {pair['target_score']:.2f}) - ID: {pair['video_id']}")

if __name__ == "__main__":
    main() 