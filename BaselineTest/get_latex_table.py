import re

def markdown_to_latex_table(markdown):
    lines = [line.strip() for line in markdown.strip().splitlines() if line.strip()]
    
    # 表头
    header = lines[1].strip('|').split('|')
    header = [cell.strip() for cell in header]

    # 表体
    data_lines = lines[3:]  # 去掉表头和分隔线

    # 列数
    num_cols = len(header)

    # LaTeX 表格列格式，居中对齐
    col_format = '|'.join(['c'] * num_cols)
    latex = [f'\\begin{{tabular}}{{|{col_format}|}}', '\\hline']

    # 添加表头
    latex.append(' & '.join(header) + ' \\\\')
    latex.append('\\hline')

    # 添加每行数据
    for line in data_lines:
        cells = [cell.strip() for cell in line.strip('|').split('|')]

        formatted_cells = []
        for cell in cells:
            # 检查粗体
            if '**' in cell:
                cell = re.sub(r'\*\*(.*?)\*\*', r'\\textbf{\1}', cell)
            formatted_cells.append(cell if cell else '')  # 保留空白单元格

        latex.append(' & '.join(formatted_cells) + ' \\\\')
        latex.append('\\hline')

    latex.append('\\end{tabular}')
    return '\n'.join(latex)

# 示例用法
markdown_table = """
|          |   Mesh   |   bca    |   bco    |   sbm    |   IoT    |
| -------- | :------: | :------: | :------: | :------: | :------: |
| TMF      | 0.473089 | 0.000184 | 0.360850 | 0.002605 | 0.314543 |
| LIST     | 0.467354 | 0.282907 | 0.493202 | 0.435350 | 0.496118 |
| E-LSTM-D | 0.724432 | 0.000001 | 0.165788 | 0.008948 | 0.053604 |
| D2V      | 0.772770 | 0.954572 | 0.988328 | 0.008948 | 0.201824 |
| DDNE     | 0.756981 | 0.672164 | 0.462276 | 0.008948 | 0.077780 |
| STGSN    | 0.779972 | 0.009600 | 0.005972 | 0.008948 | 0.216008 |
| GCN-GAN  | 0.034709 | 0.001088 | 0.003656 | 0.008948 | 0.019418 |
| random   | 0.93545  | 0.981085 | 0.980060 | 0.005255 | 0.965358 |
| HLGNN2   | 0.039072 | 0.035885 | 0.039809 | 0.005255 |          |
| epoch x3 |          | 0.012661 |          |          |          |
"""

latex_code = markdown_to_latex_table(markdown_table)
print(latex_code)
