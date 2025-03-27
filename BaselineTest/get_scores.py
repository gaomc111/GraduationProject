from collections import defaultdict

def get_scores(datafile):
    # 定义数据字符串
    # 从txt文件读取数据
    with open(datafile, 'r') as file:
        data = file.read().strip()


    # 初始化字典以存储每种指标的值
    scores = defaultdict(list)

    # 数据处理
    for line in data.strip().split('\n'):
        parts = line.split()
        for i in range(2, len(parts), 2):  # 从第3列开始，每次步进2列
            score_type = parts[i]
            score_value = float(parts[i + 1])
            scores[score_type].append(score_value)

    # 计算并输出每个指标的平均值
    for score_type, values in scores.items():
        avg_value = sum(values) / len(values)
        print(f"{score_type} 平均值: {avg_value:.6f}")

if __name__ == '__main__':
    get_scores('D:\school\毕设\GraduationProject\BaselineTest\list_demo2.txt')