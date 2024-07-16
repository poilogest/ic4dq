import subprocess
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# 定义参数取值范围
ws = range(11)  # argv[2]的取值范围为0-10
epsilons = range(11)  # argv[2]的取值范围为0-8
function_types = ['l', 'q']  # argv[3]的取值为'l'或'q' (linear or quadratic)
indices = [3]  # argv[4]的取值为0或1 (different datasets)

# 定义脚本名称列表
script_names = [
    'simulation_SWM.py',
    'simulation_NE.py',
    'simulation_Distributed_Algorithm_CSFL.py'

]

def run_experiment(script_name, variable, value, function_type, index):
    cmd = ['python', script_name, variable, str(value), function_type, str(index)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # 假设脚本的输出格式为 "Acc avg: <value>\nSW avg: <value>"
    acc_avg = None
    sw_avg = None
    for line in result.stdout.splitlines():
        print(line)
        if line.startswith("Acc avg:"):
            acc_avg = float(line.split(":")[1].strip())
        elif line.startswith("SW avg:"):
            sw_avg = float(line.split(":")[1].strip())
    
    return script_name, variable, value, function_type, index, acc_avg, sw_avg

# 使用线程池并行运行实验
results = []

with ThreadPoolExecutor() as executor:
    futures = []

    # 遍历所有可能的参数组合并运行实验
    for script_name in script_names:
        for w_index in ws:
            for function_type in function_types:
                for index in indices:
                    futures.append(executor.submit(run_experiment, script_name, 'w', w_index, function_type, index))

        for epsilon_index in epsilons:
            for function_type in function_types:
                for index in indices:
                    futures.append(executor.submit(run_experiment, script_name, 'epsilon', epsilon_index, function_type, index))

    # 收集所有任务的结果
    for future in as_completed(futures):
        results.append(future.result())

# 处理结果
acc_results = []
sw_results = []

for result in results:
    script_name, variable, value, function_type, index, acc_avg, sw_avg = result
    if acc_avg is not None:
        acc_results.append([script_name, variable, value, function_type, index, acc_avg])
    if sw_avg is not None:
        sw_results.append([script_name, variable, value, function_type, index, sw_avg])

# 转换为DataFrame并分组计算结果
acc_df = pd.DataFrame(acc_results, columns=['script_name', 'variable', 'value', 'function_type', 'index', 'acc_avg'])
sw_df = pd.DataFrame(sw_results, columns=['script_name', 'variable', 'value', 'function_type', 'index', 'sw_avg'])


# 在分组前按 value 排序
acc_df_sorted = acc_df.sort_values(by='value')
sw_df_sorted = sw_df.sort_values(by='value')


acc_grouped = acc_df_sorted.groupby(['script_name', 'variable', 'function_type', 'index'])['acc_avg'].apply(list).reset_index()
sw_grouped = sw_df_sorted.groupby(['script_name', 'variable', 'function_type', 'index'])['sw_avg'].apply(list).reset_index()

# 将结果保存为 CSV 文件
acc_grouped.to_csv("grouped_results_acc.csv", index=False)
sw_grouped.to_csv("grouped_results_sw.csv", index=False)

# 打印结果
print("Grouped Accuracy Results:")
print(acc_grouped)

print("Grouped Social Welfare Results:")
print(sw_grouped)
