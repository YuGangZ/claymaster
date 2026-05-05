# prepare_training_data.py
import pandas as pd
import numpy as np
import json
import os
import glob
import sys

sys.path.append(r"C:\Users\23142\Desktop\github_upload")


def load_custom_format_data(data_file):
    """加载自定义格式的数据文件"""
    try:
        df = pd.read_csv(data_file)
        required_columns = [
            'time', 'scale_a1', 'scale_a2', 'scale_a3',
            'shape_epsilon1', 'shape_epsilon2',
            'translation_x', 'translation_y', 'translation_z',
            'euler_rx', 'euler_ry', 'euler_rz',
            'volume', 'elongation', 'flatness', 'smoothness', 'convexity',
            'Δx_elastic', 'Δy_elastic', 'Δz_elastic'
        ]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"警告: 缺少以下列: {missing_columns}")
            return None
        print(f"成功加载数据，总记录数: {len(df)}")
        return df
    except Exception as e:
        print(f"加载数据文件失败: {e}")
        return None


def load_simulation_data_from_json(json_directory):
    """从JSON文件目录加载仿真数据，添加session_id标识"""
    json_files = glob.glob(os.path.join(json_directory, "superquadric_*.json"))
    records = []

    # 从目录路径提取session_id（使用父目录名）
    parent_dir = os.path.dirname(json_directory)
    session_id = os.path.basename(parent_dir)  # 取目录名作为session_id

    print(f"加载Session: {session_id}, 发现 {len(json_files)} 个JSON文件")

    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)

            contact_detected = data.get('contact_info', {}).get('contact_detected', False)
            if not contact_detected:
                continue

            record = {
                'session_id': session_id,  # ✅ 关键：添加session标识
                'step': data.get('step', 0),
                'time': data.get('time', 0),
                'scale_a1': data['parameters_11d']['scale_a1'],
                'scale_a2': data['parameters_11d']['scale_a2'],
                'scale_a3': data['parameters_11d']['scale_a3'],
                'shape_epsilon1': data['parameters_11d']['shape_epsilon1'],
                'shape_epsilon2': data['parameters_11d']['shape_epsilon2'],
                'translation_x': data['parameters_11d']['translation_x'],
                'translation_y': data['parameters_11d']['translation_y'],
                'translation_z': data['parameters_11d']['translation_z'],
                'euler_rx': data['parameters_11d']['euler_rx'],
                'euler_ry': data['parameters_11d']['euler_ry'],
                'euler_rz': data['parameters_11d']['euler_rz'],
                'volume': data['geometric_features'].get('volume', 0),
                'elongation': data['geometric_features'].get('elongation', 1.0),
                'flatness': data['geometric_features'].get('flatness', 0.0),
                'smoothness': data['geometric_features'].get('smoothness', 0.5),
                'convexity': data['geometric_features'].get('convexity', 1.0),
                'Δx_elastic': data.get('delta_elastic', {}).get('delta_x', 0),
                'Δy_elastic': data.get('delta_elastic', {}).get('delta_y', 0),
                'Δz_elastic': data.get('delta_elastic', {}).get('delta_z', 0),
                'contact_detected': contact_detected,
                'vel_x': data.get('control_velocity', {}).get('vel_x', 0),
                'vel_y': data.get('control_velocity', {}).get('vel_y', 0),
                'vel_z': data.get('control_velocity', {}).get('vel_z', 0)
            }
            records.append(record)
        except Exception as e:
            print(f"  加载失败 {os.path.basename(json_file)}: {e}")
            continue

    if not records:
        print(f"  警告: Session {session_id} 未找到有效接触数据")
        return None

    df = pd.DataFrame(records)
    print(f"  ✅ Session {session_id} 加载成功: {len(df)} 条记录")
    return df


def create_training_pairs(df_contact, target_dt=0.05, target_step_gap=25):
    """
    严格版本：仅创建时间间隔恰好为target_dt且step连续的数据对
    能正确处理因contact过滤导致的连续性问题

    参数:
        df_contact: DataFrame, 必须包含 'session_id', 'step', 'time' 列
        target_dt: 目标时间间隔(秒), 默认0.05
        target_step_gap: 目标step间隔, 默认25

    返回:
        X_array: (N, 19) 输入数组
        y_array: (N, 16) 输出数组
        df_metadata: DataFrame, 包含配对元数据
    """
    # 定义状态参数（16维）
    shape_params = [
        'scale_a1', 'scale_a2', 'scale_a3',
        'shape_epsilon1', 'shape_epsilon2',
        'translation_x', 'translation_y', 'translation_z',
        'euler_rx', 'euler_ry', 'euler_rz',
        'volume', 'elongation', 'flatness', 'smoothness', 'convexity'
    ]

    # 验证必需列存在
    required_cols = ['session_id', 'step', 'time'] + shape_params
    missing_cols = [col for col in required_cols if col not in df_contact.columns]
    if missing_cols:
        raise ValueError(f"缺少必需列: {missing_cols}")

    X_list, y_list, metadata_list = [], [], []
    total_attempts = 0
    valid_pairs = 0
    skipped_reasons = {
        'time_mismatch': 0,  # 时间间隔不匹配
        'step_mismatch': 0,  # step间隔不匹配
        'missing_next_frame': 0,  # 找不到下一帧
        'duplicate_timestamp': 0,  # 时间戳重复
    }

    # 按session分组处理
    for session_id, group in df_contact.groupby('session_id'):
        print(f"\n处理Session: {session_id}, 总帧数: {len(group)}")

        # 按时间排序并建立快速查找索引
        group = group.sort_values('time').reset_index(drop=True)

        # 建立时间戳到行索引的映射
        time_to_rows = {}
        for idx, row in group.iterrows():
            t = row['time']
            if t in time_to_rows:
                # 时间戳重复，记录警告
                if skipped_reasons['duplicate_timestamp'] == 0:
                    print(f"  警告: 时间戳 {t:.4f}s 在session {session_id} 中重复出现")
                skipped_reasons['duplicate_timestamp'] += 1
            time_to_rows[t] = idx

        # 遍历每一帧作为起点
        for i, current_row in group.iterrows():
            total_attempts += 1

            # 计算目标下一帧的时间
            target_next_time = current_row['time'] + target_dt
            candidates = []
            for t, idx in time_to_rows.items():
                if abs(t - target_next_time) <= 1e-6:  # ✅ 查找时就用容差
                    candidates.append((t, idx))
            if not candidates:
                skipped_reasons['missing_next_frame'] += 1
                continue

            best_time, j = min(candidates, key=lambda x: abs(x[0] - target_next_time))
            next_row = group.iloc[j]

            # 验证step连续性
            actual_step_gap = next_row['step'] - current_row['step']
            if actual_step_gap != target_step_gap:
                skipped_reasons['step_mismatch'] += 1
                # 只显示前几个示例，避免刷屏
                if skipped_reasons['step_mismatch'] <= 3:
                    print(
                        f"  Step不连续: {current_row['step']} -> {next_row['step']} (gap={actual_step_gap}, 期望={target_step_gap})")
                continue

            # 验证时间间隔（二次确认）
            actual_dt = next_row['time'] - current_row['time']
            if not np.isclose(actual_dt, target_dt, atol=1e-6):
                skipped_reasons['time_mismatch'] += 1
                continue

            # 所有检查通过，创建训练对
            valid_pairs += 1

            # 构建输入向量（16维状态 + 3维速度）
            current_state = [current_row[p] for p in shape_params]

            # # 计算实际速度（通过位置差分）
            dt_inv = 1.0 / actual_dt  # 预计算倒数提高性能
            # actual_vel_x = (next_row['translation_x'] - current_row['translation_x']) * dt_inv
            # actual_vel_y = (next_row['translation_y'] - current_row['translation_y']) * dt_inv
            # actual_vel_z = (next_row['translation_z'] - current_row['translation_z']) * dt_inv
            control_vel_x = current_row.get('vel_x', 0)
            control_vel_y = current_row.get('vel_y', 0)
            control_vel_z = current_row.get('vel_z', 0)

            input_vector = current_state + [control_vel_x, control_vel_y, control_vel_z]

            # 构建输出向量（下一时刻的16维状态）
            next_state = [next_row[p] for p in shape_params]

            X_list.append(input_vector)
            y_list.append(next_state)

            # 记录元数据（用于后续验证）
            metadata_list.append({
                'session_id': session_id,
                'current_step': current_row['step'],
                'next_step': next_row['step'],
                'current_time': current_row['time'],
                'next_time': next_row['time'],
                'time_gap': actual_dt,
                'step_gap': actual_step_gap,
                'input_vector': input_vector,
                'output_vector': next_state,
                # 同时记录两种速度用于对比
                'control_vel_x': control_vel_x,
                'control_vel_y': control_vel_y,
                'control_vel_z': control_vel_z,
                'computed_vel_x': (next_row['translation_x'] - current_row['translation_x']) * dt_inv,
                'computed_vel_y': (next_row['translation_y'] - current_row['translation_y']) * dt_inv,
                'computed_vel_z': (next_row['translation_z'] - current_row['translation_z']) * dt_inv,
            })

    # ========== 统计与验证 ==========
    print("\n" + "=" * 60)
    print("数据配对统计")
    print("=" * 60)
    print(f"总尝试配对数: {total_attempts}")
    print(f"有效数据对: {valid_pairs}")
    if total_attempts > 0:
        print(f"有效率: {valid_pairs / total_attempts * 100:.2f}%")
    print("\n跳过原因统计:")
    for reason, count in skipped_reasons.items():
        if count > 0:
            print(f"  {reason}: {count}")

    if valid_pairs == 0:
        print("\n❌ 错误: 没有生成任何有效训练对！请检查数据")
        print("建议:")
        print("  1. 检查时间戳是否为0.05s的整数倍")
        print("  2. 检查step是否从0开始，间隔为1")
        print("  3. 确认无接触帧没有被过早过滤")
        return None, None, None

    # 转换为numpy数组（使用float32减少内存）
    X_array = np.array(X_list, dtype=np.float32)
    y_array = np.array(y_list, dtype=np.float32)
    df_metadata = pd.DataFrame(metadata_list)

    # 验证生成的数据
    print("\n" + "=" * 60)
    print("数据完整性验证")
    print("=" * 60)
    print(f"X形状: {X_array.shape}")
    print(f"y形状: {y_array.shape}")
    print(f"元数据记录数: {len(df_metadata)}")

    # 验证时间间隔一致性
    dt_unique = np.unique(np.round(df_metadata['time_gap'].values, 6))
    print(f"唯一时间间隔值: {dt_unique}")

    if len(dt_unique) == 1 and np.isclose(dt_unique[0], target_dt):
        print(f"✅ 所有数据对时间间隔严格一致: {dt_unique[0]}s")
    else:
        print("⚠️  警告：存在不同的时间间隔！")
        print(f"  目标间隔: {target_dt}s")

    # 验证step间隔一致性
    step_gaps = df_metadata['step_gap'].unique()
    if len(step_gaps) == 1 and step_gaps[0] == target_step_gap:
        print(f"✅ 所有step间隔严格一致: {step_gaps[0]}")
    else:
        print("⚠️  警告：step间隔不一致！")
        print(f"  发现间隔: {step_gaps}")

    print("=" * 60)

    return X_array, y_array, df_metadata


def simple_validation(X, y, df_metadata):
    """简单验证：检查数据一致性"""
    print("=" * 60)
    print("数据验证")
    print("=" * 60)

    # 检查1: 形状匹配
    print(f"X形状: {X.shape}, y形状: {y.shape}")
    print(f"元数据记录数: {len(df_metadata)}")

    if len(X) != len(y) or len(X) != len(df_metadata):
        print(f"错误: 数据长度不匹配!")
        return False

    # 检查2: 随机抽取几个样本验证
    n_check = min(5, len(X))
    print(f"\n随机检查 {n_check} 个样本:")

    for i in range(n_check):
        idx = np.random.randint(0, len(X))

        # 从X中获取输入
        x_input = X[idx]
        y_output = y[idx]

        # 从元数据中获取原始数据
        meta = df_metadata.iloc[idx]

        # 注意：元数据中的input_vector和output_vector是列表
        expected_input = meta['input_vector']
        expected_output = meta['output_vector']

        # 检查匹配
        x_match = np.allclose(x_input, expected_input, rtol=1e-4)
        y_match = np.allclose(y_output, expected_output, rtol=1e-4)

        print(f"样本 {idx}: X匹配: {x_match}, y匹配: {y_match}")
        if not x_match or not y_match:
            print(f"  时间: {meta['time_current']} -> {meta['time_next']}")
            print(f"  X前3维: {x_input[:3]} vs {expected_input[:3]}")
            print(f"  y前3维: {y_output[:3]} vs {expected_output[:3]}")

    # 检查3: 时间连续性
    print(f"\n时间连续性检查:")
    time_diffs = []
    for _, meta in df_metadata.iterrows():
        time_diff = meta['time_next'] - meta['time_current']
        if time_diff > 0:
            time_diffs.append(time_diff)

    if time_diffs:
        avg_time_diff = np.mean(time_diffs)
        print(f"平均时间间隔: {avg_time_diff:.6f}")
        print(f"最小时间间隔: {min(time_diffs):.6f}")
        print(f"最大时间间隔: {max(time_diffs):.6f}")
    else:
        print("警告: 没有有效的时间间隔!")

    print("=" * 60)
    return True


def prepare_training_data(data_source, source_type='json'):
    """简化的数据准备流程（使用严格配对）"""
    # 加载数据
    if isinstance(data_source, (list, tuple)):
        dfs = []
        for src in data_source:
            if source_type == 'csv':
                df_i = load_custom_format_data(src)
            elif source_type == 'json':
                df_i = load_simulation_data_from_json(src)  # 现在包含session_id
            else:
                raise ValueError("source_type必须是'csv'或'json'")

            if df_i is not None and len(df_i) > 0:
                dfs.append(df_i)

        if not dfs:
            print("所有数据源加载失败")
            return None, None, None

        df_raw = pd.concat(dfs, ignore_index=True)
        print(f"\n数据合并完成，总记录数: {len(df_raw)}")
    else:
        if source_type == 'csv':
            df_raw = load_custom_format_data(data_source)
        elif source_type == 'json':
            df_raw = load_simulation_data_from_json(data_source)
        else:
            raise ValueError("source_type必须是'csv'或'json'")

    if df_raw is None or len(df_raw) == 0:
        print("错误: 数据加载失败")
        return None, None, None

    print(f"原始数据记录数: {len(df_raw)}")
    print(f"Session数量: {df_raw['session_id'].nunique()}")
    print(f"Session列表: {df_raw['session_id'].unique()}")

    # 创建训练对（使用严格版本）
    X, y, df_metadata = create_training_pairs(df_raw)

    if X is None:
        return None, None, None

    return X, y, df_metadata


# 主训练流程
if __name__ == "__main__":
    script_dir = os.path.dirname(__file__)
    data_collect_dir = os.path.abspath(os.path.join(script_dir, '..', 'data_collect'))

    # 查找数据
    json_dirs = glob.glob(os.path.join(data_collect_dir, '*', 'superquadric_params'))
    csv_files = glob.glob(os.path.join(data_collect_dir, '*', 'custom_format_data.csv'))
    print(f"发现JSON目录: {len(json_dirs)} 个, CSV文件: {len(csv_files)} 个")

    X, y, df_clean = None, None, None

    if len(json_dirs) > 0:
        print("使用JSON数据源")
        X, y, df_clean = prepare_training_data(json_dirs, source_type='json')
    elif len(csv_files) > 0:
        print("使用CSV数据源")
        X, y, df_clean = prepare_training_data(csv_files, source_type='csv')
    else:
        fallback_csv = os.path.join(script_dir, 'training_data.csv')
        if os.path.exists(fallback_csv):
            print("使用备用CSV文件")
            X, y, df_clean = prepare_training_data(fallback_csv, source_type='csv')

    if X is not None and len(X) > 0:
        # 保存数据
        np.save('training_data_X.npy', X)
        np.save('training_data_y.npy', y)
        if df_clean is not None:
            df_clean.to_csv('training_data_meta.csv', index=False)

        print("\n" + "=" * 60)
        print("数据准备完成!")
        print(f"训练数据已保存:")
        print(f"  X: training_data_X.npy ({X.shape})")
        print(f"  y: training_data_y.npy ({y.shape})")
        print(f"  元数据: training_data_meta.csv")
        print("=" * 60)
    else:
        print("错误: 无法生成训练数据")