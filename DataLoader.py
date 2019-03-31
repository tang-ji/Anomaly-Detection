import numpy as np
import pandas as pd
import glob as g
import gc

def DataLoader(cpu_contention_path, input_rate_path, node_failure_path):
	cpu_contention = g.glob(cpu_contention_path + '*.csv')
	input_rate = g.glob(input_rate_path + '*.csv')
	node_failure = g.glob(node_failure_path + '*.csv')

	header = pd.read_csv(cpu_contention[0]).iloc[:, 1:].columns.values

	abnormal_cpu_contention = pd.DataFrame(columns=header)
	normal_cpu_contention = pd.DataFrame(columns=header)

	abnormal_input_rate = pd.DataFrame(columns=header)
	normal_input_rate = pd.DataFrame(columns=header)

	abnormal_node_failure = pd.DataFrame(columns=header)
	normal_node_failure = pd.DataFrame(columns=header)

	for i in cpu_contention:
		if i.split('/')[-1].split('_')[0] == 'abnormal':
			abnormal_cpu_contention = pd.concat((abnormal_cpu_contention, pd.read_csv(i).iloc[:, 1:].fillna(value = 0)))
		else: 
			df1 = pd.read_csv(i).sample(frac=1)
			normal_cpu_contention = pd.concat([normal_cpu_contention, df1.iloc[:min(1000, df1.shape[0]), 1:].fillna(value = 0)])
			del df1
			gc.collect()

	abnormal_cpu_contention = abnormal_cpu_contention.reindex_axis(header, axis=1)
	abnormal_cpu_contention['label'] = 'cpu_contention'
	normal_cpu_contention = normal_cpu_contention.reindex_axis(header, axis=1)
	normal_cpu_contention['label'] = 'normal'

	for i in input_rate:
		if i.split('/')[-1].split('_')[0] == 'abnormal':
			abnormal_input_rate = pd.concat((abnormal_input_rate, pd.read_csv(i).iloc[:, 1:].fillna(value = 0)))
		else:
			df2 = pd.read_csv(i).sample(frac=1)
			normal_input_rate = pd.concat((normal_input_rate, df2.iloc[:min(1000,df2.shape[0]), 1:].fillna(value = 0)))
			del df2
			gc.collect()

	abnormal_input_rate = abnormal_input_rate.reindex_axis(header, axis=1)
	abnormal_input_rate['label'] = 'input_rate'
	normal_input_rate = normal_input_rate.reindex_axis(header, axis=1)
	normal_input_rate['label'] = 'normal'

	for i in node_failure:
		if i.split('/')[-1].split('_')[0] == 'abnormal':
			abnormal_node_failure = pd.concat((abnormal_node_failure, pd.read_csv(i).iloc[:, 1:].fillna(value = 0)))
		else: 
			df3 = pd.read_csv(i).sample(frac=1)
			normal_node_failure = pd.concat((normal_node_failure, df3.iloc[:min(1000,df3.shape[0]), 1:].fillna(value = 0)))
			del df3
			gc.collect()

	abnormal_node_failure = abnormal_node_failure.reindex_axis(header, axis=1)
	abnormal_node_failure['label'] = 'node_failure'
	normal_node_failure = normal_node_failure.reindex_axis(header, axis=1)
	normal_node_failure['label'] = 'normal'


	return abnormal_cpu_contention, abnormal_input_rate, abnormal_node_failure, normal_cpu_contention, normal_input_rate, normal_node_failure

