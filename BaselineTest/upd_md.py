def update_mesh_column(md_text, column_name="Mesh", divisor=1937):
    lines = md_text.strip().splitlines()

    # 提取表头
    header_line = lines[0].strip('|').split('|')
    col_names = [h.strip() for h in header_line]
    if column_name not in col_names:
        raise ValueError(f"列 '{column_name}' 不存在")

    mesh_idx = col_names.index(column_name)

    # 收集所有行（保持格式）
    updated_lines = []
    for i, line in enumerate(lines):
        if i < 2:
            updated_lines.append(line.rstrip())
            continue

        parts = line.strip().strip('|').split('|')
        parts = [p.strip() for p in parts]

        # 确保列数对齐
        while len(parts) < len(col_names):
            parts.append('')

        try:
            value = float(parts[mesh_idx])
            parts[mesh_idx] = f"{value / divisor:.6f}"
        except ValueError:
            pass  # 非数字或空白保持不变

        # 格式化成 markdown 表格行
        formatted_line = '| ' + ' | '.join(parts) + ' |'
        updated_lines.append(formatted_line)

    return '\n'.join(updated_lines)

import re

def delete_columns(markdown_table, columns_to_delete):
    """
    删除Markdown表格中指定的列，返回删除后的表格
    
    参数:
        markdown_table (str): 原始Markdown表格字符串
        columns_to_delete (list): 要删除的列名列表
        
    返回:
        str: 删除指定列后的Markdown表格
    """
    lines = markdown_table.strip().split('\n')
    if len(lines) < 2:
        return markdown_table
    
    # 解析表头
    header_line = lines[0]
    header_parts = [part.strip() for part in header_line.split('|')[1:-1]]
    
    # 解析分隔线
    separator_line = lines[1]
    separator_parts = [part.strip() for part in separator_line.split('|')[1:-1]]
    
    # 确定要保留的列索引
    columns_to_keep_indices = []
    for i, col_name in enumerate(header_parts):
        if col_name not in columns_to_delete:
            columns_to_keep_indices.append(i)
    
    # 处理每一行
    new_table = []
    for line in lines:
        parts = [part.strip() for part in line.split('|')[1:-1]]
        if len(parts) != len(header_parts):
            continue  # 跳过格式不正确的行
        
        # 只保留需要的列
        new_parts = [parts[i] for i in columns_to_keep_indices]
        new_line = '| ' + ' | '.join(new_parts) + ' |'
        new_table.append(new_line)
    
    return '\n'.join(new_table)

def extract_columns(markdown_table, columns_to_extract):
    """
    从Markdown表格中提取指定的列，返回提取的列内容
    
    参数:
        markdown_table (str): 原始Markdown表格字符串
        columns_to_extract (list): 要提取的列名列表
        
    返回:
        str: 包含提取列的Markdown表格
    """
    lines = markdown_table.strip().split('\n')
    if len(lines) < 2:
        return markdown_table
    
    # 解析表头
    header_line = lines[0]
    header_parts = [part.strip() for part in header_line.split('|')[1:-1]]
    
    # 解析分隔线
    separator_line = lines[1]
    separator_parts = [part.strip() for part in separator_line.split('|')[1:-1]]
    
    # 确定要提取的列索引
    columns_to_extract_indices = []
    for i, col_name in enumerate(header_parts):
        if col_name in columns_to_extract:
            columns_to_extract_indices.append(i)
    
    # 处理每一行
    new_table = []
    for line in lines:
        parts = [part.strip() for part in line.split('|')[1:-1]]
        if len(parts) != len(header_parts):
            continue  # 跳过格式不正确的行
        
        # 只提取需要的列
        new_parts = [parts[i] for i in columns_to_extract_indices]
        new_line = '| ' + ' | '.join(new_parts) + ' |'
        new_table.append(new_line)
    
    return '\n'.join(new_table)

# # 示例用法
# markdown_table = """|          |    Mesh    |   bca    |   bco    |   sbm    | Hmob  | T-Drive |   IoT    |
# | -------- | :--------: | :------: | :------: | :------: | :---: | :-----: | :------: |
# | TMF      | 86.019119  | 0.000012 | 0.022221 |          |       |         | 0.018949 |
# | LIST     | 35.048849  | 0.006052 | 2.086790 | 2.425441 |       |         | 2.083085 |
# | E-LSTM-D | 25.193612  | 0.000001 | 0.004360 | 0.492358 |       |         | 0.014699 |
# | D2V      | 122.438914 | 0.095795 | 0.141738 | 0.494501 |       |         | 0.035756 |
# | DDNE     | 35.635798  | 0.014720 | 0.008427 | 0.498243 |       |         | 0.015196 |
# | STGSN    | 35.511632  | 0.000968 | 0.001406 | 0.489512 |       |         | 0.021034 |
# | GCN-GAN  | 25.339260  |          | 0.002218 | 0.494954 |       |         | 0.015069 |"""

# # 测试删除列
# print("删除bca和Hmob列后的表格:")
# print(delete_columns(markdown_table, ['bca', 'Hmob']))
# print("\n")

# # 测试提取列
# print("提取Mesh和IoT列:")
# print(extract_columns(markdown_table, ['Mesh', 'IoT']))


# updated_table = update_mesh_column(markdown_table)
# print(updated_table)

if __name__ == "__main__":
    # 示例用法
    markdown_table = """
|          |   Mesh   |   bca    |   bco    |   sbm    |   Hmob   | T-Drive  |   IoT    |
| -------- | :------: | :------: | :------: | :------: | :------: | :------: | :------: |
| TMF      | 0.473089 | 0.000184 | 0.360850 | 0.002605 | 0.009824 | 0.306800 | 0.314543 |
| LIST     | 0.467354 | 0.282907 | 0.493202 | 0.435350 | 0.014660 | 0.210680 | 0.496118 |
| E-LSTM-D | 0.724432 | 0.000001 | 0.165788 | 0.008948 | 0.212060 | 0.691420 | 0.053604 |
| D2V      | 0.772770 | 0.954572 | 0.988328 | 0.008948 | 0.013300 | 0.208460 | 0.201824 |
| DDNE     | 0.756981 | 0.672164 | 0.462276 | 0.008948 | 0.294400 | 0.000980 | 0.077780 |
| STGSN    | 0.779972 | 0.009600 | 0.005972 | 0.008948 | 0.000980 | 0.146324 | 0.216008 |
| GCN-GAN  | 0.034709 | 0.001088 | 0.003656 | 0.008948 | 0.009340 | 0.018040 | 0.019418 |

    """
    print(delete_columns(markdown_table, ['T-Drive', 'Hmob']))
    